"""Execution backends for redteam generation and evaluation flows."""

from __future__ import annotations

import atexit
import base64
import fnmatch
import getpass
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
from uuid import uuid4
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

DEFAULT_EXECUTION_BACKEND = "modal"
EXECUTION_BACKEND_ENV = "AGENTLAB_REDTEAM_EXECUTION_BACKEND"
UNSAFE_LOCAL_EXECUTION_ENV = "AGENTLAB_REDTEAM_UNSAFE_LOCAL_EXECUTION"
CLAUDE_SKIP_PERMISSIONS_ENV = "AGENTLAB_REDTEAM_CLAUDE_SKIP_PERMISSIONS"
TRUSTED_AUTHORING_WARM_START_ENV = "AGENTLAB_REDTEAM_TRUSTED_AUTHORING_WARM_START"
TRUSTED_AUTHORING_MODAL_APP_ENV = "AGENTLAB_REDTEAM_TRUSTED_AUTHORING_MODAL_APP"
SANDBOX_APP_ROOT = PurePosixPath("/workspace/app")
SANDBOX_WORKER_MODULE = "agentlab.benchmarks.redteam.sandbox_worker"
MODAL_APP_NAME = "browser-sim-redteam"
CLAUDE_ALLOWED_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
)
AUTHORING_SESSION_MIN_TIMEOUT = 900
AUTHORING_SESSION_IDLE_TIMEOUT = 600
AUTHORING_SESSION_BUSY_WAIT_SECONDS = 1.0
AUTHORING_IMAGE_REVISION = "2026-04-01.1"
PYTHON_WORKER_IMAGE_REVISION = "2026-04-01.1"
BROWSER_WORKER_IMAGE_REVISION = "2026-04-01.1"
AUTHORING_OWNER_SCOPE_ENV = "AGENTLAB_REDTEAM_AUTHORING_OWNER_SCOPE"
EXECUTION_PROVENANCE_FILENAME = ".redteam_execution_provenance.json"
TRUSTED_AUTHORING_RUNTIME_DIR = PurePosixPath(".redteam_authoring")
TRUSTED_AUTHORING_WARM_START_MARKER = TRUSTED_AUTHORING_RUNTIME_DIR / "warm_start.json"
TRUSTED_AUTHORING_BUSY_LOCK = TRUSTED_AUTHORING_RUNTIME_DIR / "busy.lock"
RESERVED_RUNTIME_FILES = ("server.py",)
WORKSPACE_FINGERPRINT_IGNORE_PATTERNS = (
    "__pycache__/**",
    "*.pyc",
    ".claude_prompt.md",
    ".claude_output.log",
    "app_manifest.json",
    "functional_results.json",
    "server_events.log",
    "results/**",
    "audit_report.json",
    "audit_summary.md",
    EXECUTION_PROVENANCE_FILENAME,
)
IMAGE_FAMILY_REVISIONS = {
    "authoring-image": AUTHORING_IMAGE_REVISION,
    "python-worker-image": PYTHON_WORKER_IMAGE_REVISION,
    "browser-worker-image": BROWSER_WORKER_IMAGE_REVISION,
}


@dataclass(frozen=True)
class ExecutionPolicy:
    """Resolved execution backend policy."""

    backend: str
    unsafe_local: bool = False


@dataclass(frozen=True)
class ExecutionOperationProfile:
    """Execution policy for a concrete operation class."""

    name: str
    reuse_workspace: bool
    use_pty: bool
    default_block_network: bool
    image_family: str
    secret_set_name: str | None = None
    local_nonexecuting: bool = False


SandboxExecutionProfile = ExecutionOperationProfile

TRUSTED_AUTHORING_PROFILE = ExecutionOperationProfile(
    name="trusted-authoring",
    reuse_workspace=True,
    use_pty=True,
    default_block_network=False,
    image_family="authoring-image",
    secret_set_name="model-provider-env",
)
FRESH_PYTHON_WORKER_PROFILE = ExecutionOperationProfile(
    name="fresh-python-worker",
    reuse_workspace=False,
    use_pty=False,
    default_block_network=True,
    image_family="python-worker-image",
)
FRESH_BROWSER_WORKER_PROFILE = ExecutionOperationProfile(
    name="fresh-browser-worker",
    reuse_workspace=False,
    use_pty=False,
    default_block_network=True,
    image_family="browser-worker-image",
)
HOST_LOCAL_NONEXECUTING_PROFILE = ExecutionOperationProfile(
    name="host-local-nonexecuting",
    reuse_workspace=False,
    use_pty=False,
    default_block_network=True,
    image_family="host-local",
    local_nonexecuting=True,
)


@dataclass(frozen=True)
class AllowlistSyncResult:
    """Summary of allowlisted files materialized back to the host."""

    returned_files: tuple[str, ...] = ()
    digests: dict[str, str] = field(default_factory=dict)
    removed: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkspaceIngress:
    """Explicit sandbox input staged separately from the base image."""

    staged_dir: Path
    remote_path: PurePosixPath = SANDBOX_APP_ROOT


@dataclass
class SandboxCommandResult:
    """Result of a backend-executed command."""

    returncode: int
    stdout: str
    stderr: str
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)


_AUTHORING_SESSION_LOCK = threading.Lock()
_AUTHORING_SESSIONS: dict[str, "ModalAuthoringSession"] = {}


def _is_ignored_workspace_path(relative_path: PurePosixPath, ignore_patterns: Sequence[str]) -> bool:
    rel_str = relative_path.as_posix()
    if not rel_str:
        return False
    return any(fnmatch.fnmatch(rel_str, pattern) for pattern in ignore_patterns)


def copy_workspace_tree(
    source: str | Path,
    target: str | Path,
    *,
    ignore_patterns: Sequence[str] = (),
) -> None:
    source_root = Path(source).resolve()
    target_root = Path(target)
    if target_root.exists():
        raise RuntimeError(f"Refusing to overwrite existing staged workspace: {target_root}")
    target_root.mkdir(parents=True, exist_ok=False)

    def _copy_dir(current_source: Path, current_target: Path) -> None:
        for child in sorted(current_source.iterdir()):
            relative_path = child.relative_to(source_root)
            relative_posix = PurePosixPath(relative_path.as_posix())
            if _is_ignored_workspace_path(relative_posix, ignore_patterns):
                continue
            if child.is_symlink():
                raise RuntimeError(f"Refusing to stage symlinked workspace path: {relative_path.as_posix()}")
            resolved_child = child.resolve()
            try:
                resolved_child.relative_to(source_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"Refusing to stage workspace path outside root: {relative_path.as_posix()}"
                ) from exc
            destination = current_target / child.name
            if child.is_dir():
                destination.mkdir(parents=True, exist_ok=False)
                _copy_dir(child, destination)
            elif child.is_file():
                shutil.copy2(child, destination)

    _copy_dir(source_root, target_root)


def resolve_execution_policy(backend: str | None = None) -> ExecutionPolicy:
    """Return the selected execution backend."""
    resolved = (backend or os.getenv(EXECUTION_BACKEND_ENV, DEFAULT_EXECUTION_BACKEND)).strip()
    if resolved == "modal":
        return ExecutionPolicy(backend="modal")
    if resolved == "local":
        if os.getenv(UNSAFE_LOCAL_EXECUTION_ENV) != "1":
            raise RuntimeError(
                "Local redteam execution is disabled by default. "
                f"Set {UNSAFE_LOCAL_EXECUTION_ENV}=1 to opt into unsafe host execution."
            )
        logger.warning(
            "Using unsafe local redteam execution because %s=1",
            UNSAFE_LOCAL_EXECUTION_ENV,
        )
        return ExecutionPolicy(backend="local", unsafe_local=True)
    raise ValueError(f"Unsupported redteam execution backend: {resolved}")


def execution_backend_metadata(backend: str | None = None) -> dict[str, Any]:
    """Return manifest-safe metadata describing the selected backend."""
    policy = resolve_execution_policy(backend)
    return {
        "default_backend": DEFAULT_EXECUTION_BACKEND,
        "selected_backend": policy.backend,
        "unsafe_local_override": policy.unsafe_local,
    }


def build_claude_command(prompt: str) -> list[str]:
    """Return the Claude Code command for the hardened pipeline."""
    command = ["claude", "--print"]
    if os.getenv(CLAUDE_SKIP_PERMISSIONS_ENV) == "1":
        logger.warning(
            "Using claude --dangerously-skip-permissions because %s=1",
            CLAUDE_SKIP_PERMISSIONS_ENV,
        )
        command.append("--dangerously-skip-permissions")
    command.append(prompt)
    return command


def build_claude_env() -> dict[str, str]:
    """Return the minimal set of credentials forwarded to Claude authoring runs.

    When ``CLAUDE_CODE_OAUTH_TOKEN`` is set, ``ANTHROPIC_API_KEY`` is excluded
    so the CLI uses the subscription token.  Claude Code's auth precedence puts
    API keys (#3) above subscription OAuth (#5), so forwarding both causes the
    API key to silently win.
    """
    env = {
        key: value
        for key in CLAUDE_ALLOWED_ENV_VARS
        if (value := os.getenv(key))
    }
    if "CLAUDE_CODE_OAUTH_TOKEN" in env and "ANTHROPIC_API_KEY" in env:
        logger.info(
            "CLAUDE_CODE_OAUTH_TOKEN is set — suppressing ANTHROPIC_API_KEY "
            "so Claude Code uses subscription auth instead of API key billing"
        )
        del env["ANTHROPIC_API_KEY"]
    return env


def execute_workspace_command(
    working_dir: str | Path,
    argv: Sequence[str],
    *,
    timeout: int,
    sync_allowlist: Sequence[str] = (),
    backend: str | None = None,
    env: Mapping[str, str] | None = None,
    block_network: bool = True,
) -> SandboxCommandResult:
    """Execute *argv* in a fresh stateless sandbox or on the opted-in local host."""
    policy = resolve_execution_policy(backend)
    working_dir = Path(working_dir)

    if policy.backend == "local":
        return _run_local_command(
            argv=argv,
            cwd=working_dir,
            timeout=timeout,
            env=env,
        )

    runner = ModalSandboxRunner(working_dir)
    return runner.run_command(
        argv=argv,
        timeout=timeout,
        sync_allowlist=sync_allowlist,
        env=env,
        block_network=block_network,
    )


def execute_authoring_command(
    working_dir: str | Path,
    argv: Sequence[str],
    *,
    timeout: int,
    sync_allowlist: Sequence[str] = (),
    backend: str | None = None,
    env: Mapping[str, str] | None = None,
) -> SandboxCommandResult:
    """Execute a trusted authoring command against a reusable workspace session."""
    policy = resolve_execution_policy(backend)
    working_dir = Path(working_dir)

    if policy.backend == "local":
        return _run_local_command(
            argv=argv,
            cwd=working_dir,
            timeout=timeout,
            env=env,
        )

    session = _get_or_create_authoring_session(working_dir)
    return session.run_command(
        argv=argv,
        timeout=timeout,
        sync_allowlist=sync_allowlist,
        env=env,
    )


def run_python_script(
    working_dir: str | Path,
    script_path: str | Path,
    *,
    timeout: int,
    argv: Sequence[str] = (),
    sync_allowlist: Sequence[str] = (),
    backend: str | None = None,
    env: Mapping[str, str] | None = None,
    block_network: bool = True,
) -> SandboxCommandResult:
    """Execute a Python script inside the selected backend."""
    working_dir = Path(working_dir)
    script_path = Path(script_path)
    try:
        rel_script = script_path.relative_to(working_dir)
    except ValueError:
        raise ValueError(f"Script path {script_path} must be inside {working_dir}") from None

    policy = resolve_execution_policy(backend)
    if policy.backend == "local":
        return _run_local_command(
            argv=[sys.executable or "python3", str(script_path), *argv],
            cwd=working_dir,
            timeout=timeout,
            env=env,
        )

    runner = ModalSandboxRunner(working_dir)
    remote_script = SANDBOX_APP_ROOT / rel_script.as_posix()
    return runner.run_command(
        argv=["python", str(remote_script), *argv],
        timeout=timeout,
        sync_allowlist=sync_allowlist,
        env=env,
        block_network=block_network,
    )


def run_trusted_worker(
    working_dir: str | Path,
    subcommand: str,
    *,
    timeout: int,
    argv: Sequence[str] = (),
    backend: str | None = None,
    env: Mapping[str, str] | None = None,
    block_network: bool = True,
    profile: ExecutionOperationProfile = FRESH_PYTHON_WORKER_PROFILE,
) -> SandboxCommandResult:
    """Run a trusted repo-owned helper module inside the selected backend."""
    working_dir = Path(working_dir)
    policy = resolve_execution_policy(backend)
    command = [
        sys.executable if policy.backend == "local" else "python",
        "-m",
        SANDBOX_WORKER_MODULE,
        subcommand,
        *argv,
    ]
    if policy.backend == "local":
        return _run_local_command(
            argv=command,
            cwd=working_dir,
            timeout=timeout,
            env=env,
        )

    runner = ModalSandboxRunner(working_dir, profile=profile)
    return runner.run_command(
        argv=command,
        timeout=timeout,
        sync_allowlist=(),
        env=env,
        block_network=block_network,
    )


def read_remote_file_sha256(
    working_dir: str | Path,
    relative_path: str,
    *,
    backend: str | None = None,
) -> str | None:
    """Return the sha256 of *relative_path* inside the selected backend workspace."""
    working_dir = Path(working_dir)
    policy = resolve_execution_policy(backend)
    rel_path = str(relative_path)

    if policy.backend == "local":
        target = working_dir / rel_path
        if not target.exists():
            return None
        import hashlib

        return hashlib.sha256(target.read_bytes()).hexdigest()

    runner = ModalSandboxRunner(working_dir)
    return runner.file_sha256(rel_path)


def close_authoring_session(working_dir: str | Path) -> None:
    """Terminate any cached authoring sandbox for *working_dir*."""
    working_path = _workspace_identity(Path(working_dir))
    with _AUTHORING_SESSION_LOCK:
        sessions = [
            session
            for key, session in list(_AUTHORING_SESSIONS.items())
            if session.workspace_identity == working_path
        ]
        for session in sessions:
            _AUTHORING_SESSIONS.pop(session.resource_key, None)
    for session in sessions:
        session.close()


def close_all_authoring_sessions() -> None:
    """Terminate all cached authoring sandboxes."""
    with _AUTHORING_SESSION_LOCK:
        sessions = list(_AUTHORING_SESSIONS.values())
        _AUTHORING_SESSIONS.clear()
    for session in sessions:
        session.close()


class _ModalWorkspaceRuntime:
    """Shared Modal sandbox helpers."""

    def __init__(self, working_dir: str | Path, *, profile: ExecutionOperationProfile):
        self.working_dir = Path(working_dir)
        self.profile = profile

    @property
    def image_revision(self) -> str:
        return _image_revision_for_family(self.profile.image_family)

    def _build_base_image(self, modal_module):
        image = modal_module.Image.debian_slim(python_version="3.11").add_local_python_source(
            "agentlab",
            copy=False,
        )
        if self.profile.image_family == TRUSTED_AUTHORING_PROFILE.image_family:
            image = image.apt_install("nodejs", "npm").run_commands(
                "npm install -g @anthropic-ai/claude-code"
            )
        elif self.profile.image_family == FRESH_BROWSER_WORKER_PROFILE.image_family:
            image = image.pip_install("playwright==1.44.0").run_commands(
                "playwright install --with-deps chromium"
            )
        return image

    def _stage_workspace_ingress(self, staging_root: Path) -> WorkspaceIngress:
        staged_dir = staging_root / "app"
        self._copy_workspace(self.working_dir, staged_dir)
        return WorkspaceIngress(staged_dir=staged_dir)

    def _build_workspace_ingress(self, modal_module, workspace_ingress: WorkspaceIngress) -> dict[str, Any]:
        mount_factory = getattr(getattr(modal_module, "Mount", None), "from_local_dir", None)
        if not callable(mount_factory):
            raise RuntimeError(
                "Modal backend requires modal.Mount.from_local_dir for redteam workspace ingress."
            )
        return {
            "mounts": [
                mount_factory(
                    str(workspace_ingress.staged_dir),
                    remote_path=str(workspace_ingress.remote_path),
                )
            ]
        }

    @staticmethod
    def _copy_workspace(source: Path, target: Path) -> None:
        copy_workspace_tree(
            source,
            target,
            ignore_patterns=("__pycache__", "*.pyc"),
        )

    def _create_sandbox(
        self,
        workspace_ingress: WorkspaceIngress,
        *,
        timeout: int,
        block_network: bool,
        idle_timeout: int | None = None,
    ):
        try:
            import modal
        except ImportError as exc:
            raise RuntimeError(
                "Modal backend selected but the 'modal' package is not installed. "
                "Install Modal and configure credentials before using the hardened redteam pipeline."
            ) from exc

        image = self._build_base_image(modal)
        app = modal.App(MODAL_APP_NAME)
        sandbox_kwargs: dict[str, Any] = {
            "app": app,
            "image": image,
            "timeout": timeout,
            "workdir": str(workspace_ingress.remote_path),
            "block_network": block_network,
        }
        sandbox_kwargs.update(self._build_workspace_ingress(modal, workspace_ingress))
        if idle_timeout is not None:
            sandbox_kwargs["idle_timeout"] = idle_timeout
        return modal.Sandbox.create(
            **sandbox_kwargs,
        )

    @staticmethod
    def _execute_in_sandbox(
        sandbox,
        *,
        argv: Sequence[str],
        timeout: int,
        env: Mapping[str, str] | None = None,
        pty: bool = False,
    ) -> dict[str, Any]:
        proc = sandbox.exec(*argv, timeout=timeout, env=dict(env or {}), pty=pty)
        proc.wait()
        return {
            "returncode": getattr(proc, "returncode", 1),
            "stdout": _read_process_stream(getattr(proc, "stdout", None)),
            "stderr": _read_process_stream(getattr(proc, "stderr", None)),
        }

    def _worker_json(self, sandbox, subcommand: str, argv: Sequence[str]) -> dict[str, Any]:
        proc = sandbox.exec(
            "python",
            "-m",
            SANDBOX_WORKER_MODULE,
            subcommand,
            *argv,
            pty=False,
        )
        proc.wait()
        stdout = _read_process_stream(getattr(proc, "stdout", None))
        stderr = _read_process_stream(getattr(proc, "stderr", None))
        if getattr(proc, "returncode", 1) != 0:
            raise RuntimeError(
                f"Modal sandbox helper {subcommand} failed with rc={proc.returncode}: {stderr or stdout}"
            )
        return json.loads(stdout or "{}")

    @staticmethod
    def _bundle_pattern_args(patterns: Sequence[str]) -> list[str]:
        args: list[str] = []
        for pattern in patterns:
            args.extend(["--pattern", pattern])
        return args

    @staticmethod
    def _sync_allowlist_bundle(
        working_dir: Path,
        patterns: Sequence[str],
        bundle_payload: dict[str, Any],
    ) -> AllowlistSyncResult:
        returned_files = {path for path in bundle_payload.get("files", []) if path}
        removed: list[str] = []
        for existing in _list_allowlisted_files(working_dir, patterns):
            if existing not in returned_files:
                target = working_dir / existing
                if target.exists():
                    target.unlink()
                    removed.append(existing.as_posix())

        tarball_b64 = bundle_payload.get("tar_gz_base64", "")
        if not tarball_b64:
            return AllowlistSyncResult(digests={}, removed=tuple(sorted(removed)))

        raw = base64.b64decode(tarball_b64.encode("ascii"))
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
            for member in archive.getmembers():
                destination = (working_dir / member.name).resolve()
                if working_dir.resolve() not in destination.parents and destination != working_dir.resolve():
                    raise RuntimeError(f"Refusing to extract path outside workspace: {member.name}")
            extract_kwargs = {}
            if sys.version_info >= (3, 12):
                extract_kwargs["filter"] = "data"
            archive.extractall(working_dir, **extract_kwargs)
        digests = {
            path: _sha256_path(working_dir / path)
            for path in sorted(returned_files)
            if (working_dir / path).is_file()
        }
        return AllowlistSyncResult(
            returned_files=tuple(sorted(returned_files)),
            digests=digests,
            removed=tuple(sorted(removed)),
        )

    def _egress_metadata(
        self,
        *,
        sync_allowlist: Sequence[str],
        sync_summary: AllowlistSyncResult,
        command_succeeded: bool,
    ) -> dict[str, Any]:
        return {
            "requested_allowlist": list(sync_allowlist),
            "returned_files": list(sync_summary.returned_files),
            "synced_files": sync_summary.digests,
            "removed_files": list(sync_summary.removed),
            "applied": bool(sync_allowlist) and command_succeeded,
            "command_succeeded": command_succeeded,
        }

    def _command_metadata(
        self,
        *,
        argv: Sequence[str],
        sandbox_id: str | None,
        returncode: int,
        block_network: bool,
        sync_allowlist: Sequence[str],
        sync_summary: AllowlistSyncResult,
        reserved_runtime_mutations: Sequence[str] = (),
        resource_key: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        command_succeeded = returncode == 0
        egress = self._egress_metadata(
            sync_allowlist=sync_allowlist,
            sync_summary=sync_summary,
            command_succeeded=command_succeeded,
        )
        metadata: dict[str, Any] = {
            "profile": self.profile.name,
            "operation_class": self.profile.name,
            "image_family": self.profile.image_family,
            "image_revision": self.image_revision,
            "sandbox_id": sandbox_id,
            "command": _summarize_argv(argv),
            "command_summary": _summarize_argv(argv),
            "egress": egress,
            "synced_files": egress["synced_files"],
            "removed_files": egress["removed_files"],
            "returned_files": egress["returned_files"],
            "block_network": block_network,
            "network_policy": _describe_network_policy(block_network=block_network),
            "resource_key": resource_key,
            "secret_set_name": self.profile.secret_set_name,
            "reserved_runtime_mutations": list(reserved_runtime_mutations),
        }
        if extra:
            metadata.update(dict(extra))
        return metadata

    def _append_provenance_record(
        self,
        *,
        argv: Sequence[str],
        returncode: int,
        sandbox_id: str | None,
        block_network: bool,
        sync_allowlist: Sequence[str],
        sync_summary: AllowlistSyncResult,
        reserved_runtime_mutations: Sequence[str] = (),
        resource_key: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = self._command_metadata(
            argv=argv,
            sandbox_id=sandbox_id,
            returncode=returncode,
            block_network=block_network,
            sync_allowlist=sync_allowlist,
            sync_summary=sync_summary,
            reserved_runtime_mutations=reserved_runtime_mutations,
            resource_key=resource_key,
            extra=extra,
        )
        record = {
            "operation_id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": "modal",
            "operation_class": self.profile.name,
            "image_family": self.profile.image_family,
            "image_revision": self.image_revision,
            "sandbox_id": sandbox_id,
            "resource_key": resource_key,
            "network_policy": metadata["network_policy"],
            "secret_set_name": self.profile.secret_set_name,
            "command_summary": metadata["command_summary"],
            "workspace_identity": metadata.get("workspace_identity", _workspace_identity(self.working_dir)),
            "workspace_fingerprint": metadata.get(
                "workspace_fingerprint",
                _workspace_fingerprint(self.working_dir),
            ),
            "owner_scope": metadata.get("owner_scope"),
            "egress": metadata["egress"],
            "returncode": returncode,
            "reserved_runtime_mutations": list(reserved_runtime_mutations),
            "behavior_id": metadata.get("behavior_id", _infer_behavior_id(self.working_dir)),
            "run_id": None,
        }
        for key in (
            "warm_start_enabled",
            "warm_start_attempted",
            "warm_start_hit",
            "warm_start_mode",
            "warm_start_name",
            "warm_start_lookup_app",
        ):
            if key in metadata:
                record[key] = metadata[key]
        _append_execution_provenance(self.working_dir, record)
        return metadata


class ModalSandboxRunner(_ModalWorkspaceRuntime):
    """Run redteam commands in a fresh Modal sandbox."""

    def __init__(
        self,
        working_dir: str | Path,
        *,
        profile: ExecutionOperationProfile = FRESH_PYTHON_WORKER_PROFILE,
    ):
        super().__init__(working_dir, profile=profile)

    def run_command(
        self,
        *,
        argv: Sequence[str],
        timeout: int,
        sync_allowlist: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        block_network: bool = True,
    ) -> SandboxCommandResult:
        with tempfile.TemporaryDirectory(prefix="redteam-modal-stage-") as tmpdir:
            workspace_ingress = self._stage_workspace_ingress(Path(tmpdir))
            sandbox = self._create_sandbox(workspace_ingress, timeout=timeout, block_network=block_network)
            sandbox_id = getattr(sandbox, "object_id", None)
            try:
                result = self._execute_in_sandbox(
                    sandbox,
                    argv=argv,
                    timeout=timeout,
                    env=env,
                )
                returncode = int(result.get("returncode", 1))
                sync_summary = AllowlistSyncResult()
                if sync_allowlist and returncode == 0:
                    bundle = self._worker_json(
                        sandbox,
                        "bundle-allowlist",
                        ["--root", str(SANDBOX_APP_ROOT), *self._bundle_pattern_args(sync_allowlist)],
                    )
                    sync_summary = self._sync_allowlist_bundle(self.working_dir, sync_allowlist, bundle)
                metadata = self._append_provenance_record(
                    argv=argv,
                    returncode=returncode,
                    sandbox_id=sandbox_id,
                    block_network=block_network,
                    sync_allowlist=sync_allowlist,
                    sync_summary=sync_summary,
                )
                return SandboxCommandResult(
                    returncode=returncode,
                    stdout=str(result.get("stdout", "")),
                    stderr=str(result.get("stderr", "")),
                    backend="modal",
                    metadata=metadata,
                )
            finally:
                terminate = getattr(sandbox, "terminate", None)
                if callable(terminate):
                    terminate()

    def file_sha256(self, relative_path: str) -> str | None:
        with tempfile.TemporaryDirectory(prefix="redteam-modal-stage-") as tmpdir:
            workspace_ingress = self._stage_workspace_ingress(Path(tmpdir))
            sandbox = self._create_sandbox(workspace_ingress, timeout=60, block_network=True)
            try:
                payload = self._worker_json(
                    sandbox,
                    "sha256-file",
                    [
                        "--root",
                        str(SANDBOX_APP_ROOT),
                        "--path",
                        relative_path,
                    ],
                )
                return payload.get("sha256")
            finally:
                terminate = getattr(sandbox, "terminate", None)
                if callable(terminate):
                    terminate()


class ModalAuthoringSession(_ModalWorkspaceRuntime):
    """Reuse a single Modal sandbox across trusted authoring commands."""

    def __init__(
        self,
        working_dir: str | Path,
        *,
        owner_scope: str,
        workspace_identity: str,
        workspace_fingerprint: str,
        behavior_id: str | None,
    ):
        super().__init__(working_dir, profile=TRUSTED_AUTHORING_PROFILE)
        self._sandbox = None
        self._sandbox_timeout = 0
        self._workspace_fingerprint = workspace_fingerprint
        self._idle_timeout = 0
        self._busy_lock = threading.Lock()
        self._session_id = str(uuid4())
        self.owner_scope = owner_scope
        self.workspace_identity = workspace_identity
        self.behavior_id = behavior_id
        self.command_count = 0
        self.last_used_at = datetime.now(timezone.utc)
        self.resource_key = _trusted_authoring_resource_key(
            profile=self.profile,
            workspace_identity=self.workspace_identity,
            behavior_id=self.behavior_id,
            owner_scope=self.owner_scope,
            workspace_fingerprint=self._workspace_fingerprint,
            image_revision=self.image_revision,
        )
        self._warm_start_enabled = _trusted_authoring_warm_start_enabled()
        self._warm_start_lookup_app = _trusted_authoring_modal_app_name()
        self._warm_start_name = _trusted_authoring_warm_start_lookup_name(
            profile=self.profile,
            workspace_identity=self.workspace_identity,
            behavior_id=self.behavior_id,
            owner_scope=self.owner_scope,
            image_revision=self.image_revision,
        )
        self._warm_start_attempted = False
        self._warm_start_hit = False
        self._warm_start_mode = "cold-start"
        self._uses_named_sandbox = False

    def run_command(
        self,
        *,
        argv: Sequence[str],
        timeout: int,
        sync_allowlist: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
    ) -> SandboxCommandResult:
        if not self._busy_lock.acquire(timeout=AUTHORING_SESSION_BUSY_WAIT_SECONDS):
            raise RuntimeError(
                f"Trusted authoring sandbox is busy for resource {self.resource_key}; failing closed."
            )

        try:
            self._ensure_sandbox(timeout=timeout)
            if self._sandbox is None:
                raise RuntimeError("Authoring sandbox was not initialized")
            self._acquire_remote_busy_lock()

            try:
                result = self._execute_in_sandbox(
                    self._sandbox,
                    argv=argv,
                    timeout=timeout,
                    env=env,
                    pty=True,
                )
            except Exception:
                self.close()
                raise

            returncode = int(result.get("returncode", 1))
            sync_summary = AllowlistSyncResult()
            if sync_allowlist and returncode == 0:
                bundle = self._worker_json(
                    self._sandbox,
                    "bundle-allowlist",
                    ["--root", str(SANDBOX_APP_ROOT), *self._bundle_pattern_args(sync_allowlist)],
                )
                sync_summary = self._sync_allowlist_bundle(self.working_dir, sync_allowlist, bundle)

            reserved_runtime_mutations = self._detect_reserved_runtime_mutations()
            self.command_count += 1
            self.last_used_at = datetime.now(timezone.utc)

            if returncode == 0 and not reserved_runtime_mutations:
                self._refresh_resource_identity(
                    workspace_fingerprint=_workspace_fingerprint(self.working_dir),
                    behavior_id=_infer_behavior_id(self.working_dir),
                )
                self._refresh_remote_warm_start_marker()

            metadata = self._append_provenance_record(
                argv=argv,
                returncode=returncode,
                sandbox_id=getattr(self._sandbox, "object_id", None),
                block_network=False,
                sync_allowlist=sync_allowlist,
                sync_summary=sync_summary,
                reserved_runtime_mutations=reserved_runtime_mutations,
                resource_key=self.resource_key,
                extra={
                    "command_count": self.command_count,
                    "owner_scope": self.owner_scope,
                    "workspace_identity": self.workspace_identity,
                    "workspace_fingerprint": self._workspace_fingerprint,
                    "behavior_id": self.behavior_id,
                    "busy": True,
                    **self._warm_start_metadata(),
                },
            )

            if reserved_runtime_mutations or returncode != 0:
                self.close()

            return SandboxCommandResult(
                returncode=returncode,
                stdout=str(result.get("stdout", "")),
                stderr=str(result.get("stderr", "")),
                backend="modal",
                metadata=metadata,
            )
        finally:
            self._release_remote_busy_lock()
            self._busy_lock.release()

    def close(self) -> None:
        self._terminate_sandbox(unregister=True)

    def _terminate_sandbox(self, *, unregister: bool) -> None:
        if unregister:
            _unregister_authoring_session(self)
        sandbox = self._sandbox
        self._sandbox = None
        self._sandbox_timeout = 0
        self._idle_timeout = 0
        if sandbox is None:
            return
        terminate = getattr(sandbox, "terminate", None)
        if callable(terminate):
            terminate()

    def _ensure_sandbox(self, *, timeout: int) -> None:
        current_fingerprint = _workspace_fingerprint(self.working_dir)
        session_timeout = max(timeout, AUTHORING_SESSION_MIN_TIMEOUT)
        should_reseed = (
            self._sandbox is None
            or self._workspace_fingerprint != current_fingerprint
            or session_timeout > self._sandbox_timeout
        )
        if not should_reseed:
            return

        self._terminate_sandbox(unregister=False)
        with tempfile.TemporaryDirectory(prefix="redteam-modal-authoring-stage-") as tmpdir:
            workspace_ingress = self._stage_workspace_ingress(Path(tmpdir))
            self._sandbox = self._create_or_attach_sandbox(
                workspace_ingress,
                timeout=session_timeout,
            )
        self._sandbox_timeout = session_timeout
        self._idle_timeout = min(AUTHORING_SESSION_IDLE_TIMEOUT, session_timeout)
        self._workspace_fingerprint = current_fingerprint
        self.last_used_at = datetime.now(timezone.utc)
        _register_authoring_session(self)

    def _detect_reserved_runtime_mutations(self) -> list[str]:
        if self._sandbox is None:
            return []
        mutated: list[str] = []
        for relative_path in RESERVED_RUNTIME_FILES:
            local_path = self.working_dir / relative_path
            if not local_path.exists():
                continue
            payload = self._worker_json(
                self._sandbox,
                "sha256-file",
                [
                    "--root",
                    str(SANDBOX_APP_ROOT),
                    "--path",
                    relative_path,
                ],
            )
            remote_sha = payload.get("sha256")
            if remote_sha != _sha256_path(local_path):
                mutated.append(relative_path)
        return mutated

    @property
    def in_use(self) -> bool:
        return self._busy_lock.locked()

    def is_stale(self, now: datetime | None = None) -> bool:
        if self._sandbox is None:
            return True
        now = now or datetime.now(timezone.utc)
        idle_limit = self._idle_timeout or AUTHORING_SESSION_IDLE_TIMEOUT
        return (now - self.last_used_at).total_seconds() > idle_limit

    def _refresh_resource_identity(
        self,
        *,
        workspace_fingerprint: str,
        behavior_id: str | None,
    ) -> None:
        previous_key = self.resource_key
        self._workspace_fingerprint = workspace_fingerprint
        self.behavior_id = behavior_id
        self.resource_key = _trusted_authoring_resource_key(
            profile=self.profile,
            workspace_identity=self.workspace_identity,
            behavior_id=self.behavior_id,
            owner_scope=self.owner_scope,
            workspace_fingerprint=self._workspace_fingerprint,
            image_revision=self.image_revision,
        )
        self._warm_start_name = _trusted_authoring_warm_start_lookup_name(
            profile=self.profile,
            workspace_identity=self.workspace_identity,
            behavior_id=self.behavior_id,
            owner_scope=self.owner_scope,
            image_revision=self.image_revision,
        )
        if previous_key != self.resource_key:
            _reindex_authoring_session(self, previous_key=previous_key)

    def _create_or_attach_sandbox(
        self,
        workspace_ingress: WorkspaceIngress,
        *,
        timeout: int,
    ):
        idle_timeout = min(AUTHORING_SESSION_IDLE_TIMEOUT, timeout)
        self._warm_start_attempted = False
        self._warm_start_hit = False
        self._warm_start_mode = "cold-start"
        self._uses_named_sandbox = False

        if self._warm_start_enabled and self._warm_start_lookup_app:
            self._warm_start_attempted = True
            attached = self._attach_named_sandbox()
            if attached is not None:
                self._warm_start_hit = True
                self._warm_start_mode = "named-sandbox"
                self._uses_named_sandbox = True
                return attached

            created = self._create_named_sandbox(
                workspace_ingress,
                timeout=timeout,
                idle_timeout=idle_timeout,
                modal_app_name=self._warm_start_lookup_app,
                sandbox_name=self._warm_start_name,
            )
            if created is not None:
                self._uses_named_sandbox = True
                return created

        return self._create_sandbox(
            workspace_ingress,
            timeout=timeout,
            idle_timeout=idle_timeout,
            block_network=False,
        )

    def _attach_named_sandbox(self):
        if not self._warm_start_enabled or not self._warm_start_lookup_app:
            return None
        try:
            import modal
        except ImportError:
            return None

        try:
            sandbox = modal.Sandbox.from_name(
                self._warm_start_lookup_app,
                self._warm_start_name,
            )
        except Exception as exc:
            if _is_modal_not_found(exc):
                return None
            logger.warning(
                "Trusted authoring warm-start lookup failed for %s via app %s: %s",
                self._warm_start_name,
                self._warm_start_lookup_app,
                exc,
            )
            return None

        marker = self._read_remote_warm_start_marker(sandbox)
        if marker.get("resource_key") == self.resource_key:
            return sandbox

        terminate = getattr(sandbox, "terminate", None)
        if callable(terminate):
            terminate()
        return None

    def _create_named_sandbox(
        self,
        workspace_ingress: WorkspaceIngress,
        *,
        timeout: int,
        idle_timeout: int,
        modal_app_name: str,
        sandbox_name: str,
    ):
        try:
            import modal
        except ImportError:
            return None

        image = self._build_base_image(modal)
        app = modal.App(modal_app_name)
        sandbox_kwargs: dict[str, Any] = {
            "app": app,
            "image": image,
            "timeout": timeout,
            "idle_timeout": idle_timeout,
            "workdir": str(workspace_ingress.remote_path),
            "block_network": False,
            "name": sandbox_name,
        }
        sandbox_kwargs.update(self._build_workspace_ingress(modal, workspace_ingress))
        try:
            sandbox = modal.Sandbox.create(**sandbox_kwargs)
        except Exception as exc:
            logger.warning(
                "Trusted authoring warm-start create failed for %s via app %s; falling back to cold start: %s",
                sandbox_name,
                modal_app_name,
                exc,
            )
            return None

        if not self._write_remote_warm_start_marker(sandbox):
            terminate = getattr(sandbox, "terminate", None)
            if callable(terminate):
                terminate()
            logger.warning(
                "Trusted authoring warm-start marker initialization failed for %s; falling back to cold start.",
                sandbox_name,
            )
            return None
        return sandbox

    def _warm_start_marker_payload(self) -> dict[str, Any]:
        return {
            "resource_key": self.resource_key,
            "workspace_identity": self.workspace_identity,
            "workspace_fingerprint": self._workspace_fingerprint,
            "behavior_id": self.behavior_id,
            "owner_scope": self.owner_scope,
            "image_family": self.profile.image_family,
            "image_revision": self.image_revision,
        }

    def _warm_start_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "warm_start_enabled": self._warm_start_enabled,
            "warm_start_attempted": self._warm_start_attempted,
            "warm_start_hit": self._warm_start_hit,
            "warm_start_mode": self._warm_start_mode,
        }
        if self._warm_start_lookup_app:
            metadata["warm_start_lookup_app"] = self._warm_start_lookup_app
        if self._warm_start_enabled:
            metadata["warm_start_name"] = self._warm_start_name
        return metadata

    def _read_remote_warm_start_marker(self, sandbox) -> dict[str, Any]:
        result = self._execute_in_sandbox(
            sandbox,
            argv=[
                "python",
                "-c",
                (
                    "import json\n"
                    "from pathlib import Path\n"
                    f"path = Path({str(TRUSTED_AUTHORING_WARM_START_MARKER)!r})\n"
                    "if not path.exists():\n"
                    "    print('{}')\n"
                    "else:\n"
                    "    print(path.read_text(encoding='utf-8'))\n"
                ),
            ],
            timeout=10,
            pty=False,
        )
        if int(result.get("returncode", 1)) != 0:
            return {}
        try:
            return json.loads(str(result.get("stdout", "{}")) or "{}")
        except json.JSONDecodeError:
            return {}

    def _write_remote_warm_start_marker(self, sandbox) -> bool:
        payload = json.dumps(self._warm_start_marker_payload(), sort_keys=True)
        result = self._execute_in_sandbox(
            sandbox,
            argv=[
                "python",
                "-c",
                (
                    "from pathlib import Path\n"
                    f"runtime_dir = Path({str(TRUSTED_AUTHORING_RUNTIME_DIR)!r})\n"
                    f"marker_path = Path({str(TRUSTED_AUTHORING_WARM_START_MARKER)!r})\n"
                    "runtime_dir.mkdir(parents=True, exist_ok=True)\n"
                    f"marker_path.write_text({payload!r}, encoding='utf-8')\n"
                ),
            ],
            timeout=10,
            pty=False,
        )
        return int(result.get("returncode", 1)) == 0

    def _refresh_remote_warm_start_marker(self) -> None:
        if not self._uses_named_sandbox or self._sandbox is None:
            return
        if not self._write_remote_warm_start_marker(self._sandbox):
            logger.warning(
                "Trusted authoring warm-start marker refresh failed for %s; terminating named sandbox.",
                self._warm_start_name,
            )
            self.close()

    def _acquire_remote_busy_lock(self) -> None:
        if not self._uses_named_sandbox or self._sandbox is None:
            return
        result = self._execute_in_sandbox(
            self._sandbox,
            argv=[
                "python",
                "-c",
                (
                    "import json, os, sys\n"
                    "from pathlib import Path\n"
                    f"runtime_dir = Path({str(TRUSTED_AUTHORING_RUNTIME_DIR)!r})\n"
                    f"lock_path = Path({str(TRUSTED_AUTHORING_BUSY_LOCK)!r})\n"
                    "runtime_dir.mkdir(parents=True, exist_ok=True)\n"
                    "flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY\n"
                    "try:\n"
                    "    fd = os.open(str(lock_path), flags)\n"
                    "except FileExistsError:\n"
                    "    raise SystemExit(17)\n"
                    "with os.fdopen(fd, 'w', encoding='utf-8') as handle:\n"
                    "    handle.write(json.dumps({'session_id': sys.argv[1]}))\n"
                ),
                self._session_id,
            ],
            timeout=10,
            pty=False,
        )
        if int(result.get("returncode", 1)) != 0:
            raise RuntimeError(
                f"Trusted authoring sandbox is busy for resource {self.resource_key}; failing closed."
            )

    def _release_remote_busy_lock(self) -> None:
        if not self._uses_named_sandbox or self._sandbox is None:
            return
        self._execute_in_sandbox(
            self._sandbox,
            argv=[
                "python",
                "-c",
                (
                    "import json, sys\n"
                    "from pathlib import Path\n"
                    f"lock_path = Path({str(TRUSTED_AUTHORING_BUSY_LOCK)!r})\n"
                    "if not lock_path.exists():\n"
                    "    raise SystemExit(0)\n"
                    "try:\n"
                    "    payload = json.loads(lock_path.read_text(encoding='utf-8') or '{}')\n"
                    "except json.JSONDecodeError:\n"
                    "    payload = {}\n"
                    "if payload.get('session_id') == sys.argv[1]:\n"
                    "    lock_path.unlink(missing_ok=True)\n"
                ),
                self._session_id,
            ],
            timeout=10,
            pty=False,
        )


def _get_or_create_authoring_session(working_dir: Path) -> ModalAuthoringSession:
    owner_scope = _authoring_owner_scope()
    _cleanup_stale_authoring_sessions(owner_scope)
    workspace_identity = _workspace_identity(working_dir)
    behavior_id = _infer_behavior_id(working_dir)
    workspace_fingerprint = _workspace_fingerprint(working_dir)
    resource_key = _trusted_authoring_resource_key(
        profile=TRUSTED_AUTHORING_PROFILE,
        workspace_identity=workspace_identity,
        behavior_id=behavior_id,
        owner_scope=owner_scope,
        workspace_fingerprint=workspace_fingerprint,
        image_revision=_image_revision_for_family(TRUSTED_AUTHORING_PROFILE.image_family),
    )
    with _AUTHORING_SESSION_LOCK:
        session = _AUTHORING_SESSIONS.get(resource_key)
        if session is None:
            session = ModalAuthoringSession(
                working_dir,
                owner_scope=owner_scope,
                workspace_identity=workspace_identity,
                workspace_fingerprint=workspace_fingerprint,
                behavior_id=behavior_id,
            )
            _AUTHORING_SESSIONS[session.resource_key] = session
        return session


def _run_local_command(
    *,
    argv: Sequence[str],
    cwd: Path,
    timeout: int,
    env: Mapping[str, str] | None = None,
) -> SandboxCommandResult:
    env_vars = os.environ.copy()
    if env:
        env_vars.update(env)
    result = subprocess.run(
        list(argv),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env_vars,
    )
    return SandboxCommandResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        backend="local",
    )


def _read_process_stream(stream) -> str:
    if stream is None:
        return ""
    reader = getattr(stream, "read", None)
    if not callable(reader):
        return ""
    payload = reader()
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    return str(payload)


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workspace_identity(root: Path) -> str:
    return str(root.resolve())


def _workspace_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(_workspace_identity(root).encode("utf-8"))
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        rel_posix = rel.as_posix()
        if any(fnmatch.fnmatch(rel_posix, pattern) for pattern in WORKSPACE_FINGERPRINT_IGNORE_PATTERNS):
            continue
        stat = path.stat()
        digest.update(rel_posix.encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def _summarize_argv(argv: Sequence[str]) -> list[str]:
    if len(argv) <= 3:
        return [str(arg) for arg in argv]
    return [str(argv[0]), str(argv[1]), f"... ({len(argv) - 2} more args)"]


def _describe_network_policy(*, block_network: bool) -> str:
    return "blocked" if block_network else "unrestricted"


def _image_revision_for_family(image_family: str) -> str:
    return IMAGE_FAMILY_REVISIONS.get(image_family, "unversioned")


def _authoring_owner_scope() -> str:
    explicit = os.getenv(AUTHORING_OWNER_SCOPE_ENV)
    if explicit:
        return explicit
    if os.getenv("CI") == "1":
        ci_parts = [
            os.getenv("GITHUB_RUN_ID"),
            os.getenv("GITHUB_JOB"),
            os.getenv("BUILD_BUILDID"),
            os.getenv("BUILDKITE_BUILD_ID"),
            os.getenv("CI_PIPELINE_ID"),
        ]
        resolved = next((part for part in ci_parts if part), None)
        return f"ci:{resolved or os.getpid()}"
    return f"local:{getpass.getuser()}:{os.getpid()}"


def _trusted_authoring_warm_start_enabled() -> bool:
    return os.getenv(TRUSTED_AUTHORING_WARM_START_ENV) == "1"


def _trusted_authoring_modal_app_name() -> str | None:
    value = os.getenv(TRUSTED_AUTHORING_MODAL_APP_ENV, "").strip()
    return value or None


def _trusted_authoring_resource_key(
    *,
    profile: ExecutionOperationProfile,
    workspace_identity: str,
    behavior_id: str | None,
    owner_scope: str,
    workspace_fingerprint: str,
    image_revision: str,
) -> str:
    payload = {
        "trust_profile": profile.name,
        "workspace_identity": workspace_identity,
        "behavior_id": behavior_id,
        "image_family": profile.image_family,
        "image_revision": image_revision,
        "workspace_fingerprint": workspace_fingerprint,
        "owner_scope": owner_scope,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"{profile.name}:{digest}"


def _trusted_authoring_warm_start_lookup_name(
    *,
    profile: ExecutionOperationProfile,
    workspace_identity: str,
    behavior_id: str | None,
    owner_scope: str,
    image_revision: str,
) -> str:
    payload = {
        "trust_profile": profile.name,
        "workspace_identity": workspace_identity,
        "behavior_id": behavior_id,
        "owner_scope": owner_scope,
        "image_family": profile.image_family,
        "image_revision": image_revision,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"redteam-authoring-{hashlib.sha256(encoded).hexdigest()[:32]}"


def _is_modal_not_found(exc: Exception) -> bool:
    return exc.__class__.__name__ == "NotFoundError"


def _infer_behavior_id(working_dir: Path) -> str | None:
    manifest_path = working_dir / "app_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    behavior_id = payload.get("behavior_id") or payload.get("app_id")
    return str(behavior_id) if behavior_id is not None else None


def _append_execution_provenance(working_dir: Path, record: Mapping[str, Any]) -> None:
    provenance_path = working_dir / EXECUTION_PROVENANCE_FILENAME
    document: dict[str, Any]
    if provenance_path.exists():
        try:
            loaded = json.loads(provenance_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = {}
        document = loaded if isinstance(loaded, dict) else {}
    else:
        document = {}

    existing_records = document.get("records")
    if isinstance(existing_records, list):
        records = existing_records
    else:
        records = []
        document["records"] = records
    records.append(dict(record))
    provenance_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _register_authoring_session(session: ModalAuthoringSession) -> None:
    with _AUTHORING_SESSION_LOCK:
        _AUTHORING_SESSIONS[session.resource_key] = session


def _unregister_authoring_session(session: ModalAuthoringSession) -> None:
    with _AUTHORING_SESSION_LOCK:
        current = _AUTHORING_SESSIONS.get(session.resource_key)
        if current is session:
            _AUTHORING_SESSIONS.pop(session.resource_key, None)


def _reindex_authoring_session(session: ModalAuthoringSession, *, previous_key: str) -> None:
    with _AUTHORING_SESSION_LOCK:
        current = _AUTHORING_SESSIONS.get(previous_key)
        if current is session:
            _AUTHORING_SESSIONS.pop(previous_key, None)
        _AUTHORING_SESSIONS[session.resource_key] = session


def _cleanup_stale_authoring_sessions(owner_scope: str) -> None:
    now = datetime.now(timezone.utc)
    with _AUTHORING_SESSION_LOCK:
        stale_sessions = []
        for key, session in list(_AUTHORING_SESSIONS.items()):
            if session.owner_scope != owner_scope:
                continue
            if session.in_use:
                continue
            if session.is_stale(now):
                _AUTHORING_SESSIONS.pop(key, None)
                stale_sessions.append(session)
    for session in stale_sessions:
        session.close()


def _list_allowlisted_files(root: Path, patterns: Sequence[str]) -> set[Path]:
    matches: set[Path] = set()
    if not root.exists():
        return matches
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        rel_posix = rel.as_posix()
        if any(fnmatch.fnmatch(rel_posix, pattern) for pattern in patterns):
            matches.add(rel)
    return matches


atexit.register(close_all_authoring_sessions)
