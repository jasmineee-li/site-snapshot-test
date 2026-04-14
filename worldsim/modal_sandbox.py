"""Modal sandbox for running Claude Code with file-routed inputs.

Each pipeline step gets a fresh ``modal.Sandbox`` whose image contains only
the files that step needs (inclusion-based scoping, not ignore-based).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import modal

logger = logging.getLogger(__name__)

APP_NAME = "worldsim-v5"

CLAUDE_ALLOWED_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN",
)

NAMED_SECRET_ENV_VAR = "WORLDSIM_CLAUDE_MODAL_SECRET"

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "git", "jq")
    .env(
        {
            # IS_SANDBOX=1 lets Claude Code accept bypassPermissions as root
            # (github.com/anthropics/claude-code/issues/3490).
            "IS_SANDBOX": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    # claude-agent-sdk bundles the Claude Code CLI + Node.js runtime,
    # so no separate nodejs/npm/npm-install needed.
    .pip_install("requests", "browser-use>=0.12.6", "claude-agent-sdk")
    .run_commands(
        "mkdir -p /workspace /root/.claude",
        # Pre-accept trust dialog so Claude Code runs non-interactively.
        'python3 -c "'
        "import json; from pathlib import Path; "
        "Path('/root/.claude.json').write_text("
        "json.dumps({'projects': {'/workspace': {'hasTrustDialogAccepted': True}}})); "
        "Path('/root/.claude/settings.json').write_text("
        "json.dumps({'skipDangerousModePermissionPrompt': True}))"
        '"',
    )
)


def _build_claude_secrets() -> list[modal.Secret]:
    """Build Modal secrets for Claude Code auth.

    Raises RuntimeError if no credentials are available.

    Note: empty-string env vars are treated as unset. Shell environments
    (especially Claude Code sessions) often pre-set auth vars to ``""``
    which would otherwise pass ``os.environ.get(key)`` truthiness checks.
    """
    named = os.environ.get(NAMED_SECRET_ENV_VAR, "").strip()
    if named:
        logger.info("Using named Modal secret %r for Claude Code auth", named)
        return [modal.Secret.from_name(named)]

    env: dict[str, str] = {}
    for key in CLAUDE_ALLOWED_ENV_VARS:
        value = os.environ.get(key, "").strip()
        if value:
            env[key] = value
        elif os.environ.get(key) is not None:
            # Key is present but empty — log so the user knows.
            logger.debug(
                "Env var %s is set but empty (treating as unset)",
                key,
            )

    has_creds = (
        "ANTHROPIC_API_KEY" in env
        or "ANTHROPIC_AUTH_TOKEN" in env
        or "CLAUDE_CODE_OAUTH_TOKEN" in env
    )
    if not has_creds:
        # Build a diagnostic showing what was actually in the environment.
        present = {
            k: ("set" if os.environ.get(k, "").strip() else "empty")
            for k in CLAUDE_ALLOWED_ENV_VARS
            if os.environ.get(k) is not None
        }
        raise RuntimeError(
            "No Claude Code auth available for the Modal sandbox.\n"
            f"  Environment check: {present or '(none of the auth vars are set)'}\n"
            "  Ensure load_dotenv(override=True) ran, or export one of:\n"
            "    CLAUDE_CODE_OAUTH_TOKEN=...              # Pro/Max subscription (preferred)\n"
            "    ANTHROPIC_AUTH_TOKEN=sk-or-v1-...         # OpenRouter\n"
            "      ANTHROPIC_BASE_URL=https://openrouter.ai/api\n"
            "    ANTHROPIC_API_KEY=sk-ant-...              # direct Anthropic API\n"
            "  Or point at an existing named Modal secret:\n"
            f"    {NAMED_SECRET_ENV_VAR}=<secret-name>"
        )

    # OAuth wins: Claude Code's own precedence puts API key above OAuth,
    # so forwarding both would silently bill API credits instead of the subscription.
    if "CLAUDE_CODE_OAUTH_TOKEN" in env:
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
        env.pop("ANTHROPIC_BASE_URL", None)
        return [modal.Secret.from_dict(env)]

    # OpenRouter path: ANTHROPIC_API_KEY must be explicitly empty so Claude
    # Code doesn't prefer it over the auth token (per openrouter.ai/docs).
    if "ANTHROPIC_AUTH_TOKEN" in env:
        env["ANTHROPIC_API_KEY"] = ""
        return [modal.Secret.from_dict(env)]

    return [modal.Secret.from_dict(env)]


def preflight_auth_check() -> None:
    """Verify Claude Code auth is available before launching sandboxes.

    Call this early in any phase that uses ``run_claude_in_sandbox`` so
    failures surface immediately instead of after minutes of sandbox setup.
    Raises RuntimeError with diagnostics if auth is missing.
    """
    _build_claude_secrets()
    logger.info("Pre-flight auth check passed")


_RUNNER_PATH = str(Path(__file__).with_name("_sandbox_runner.py"))
_VALIDATOR_PATH = str(Path(__file__).with_name("_sandbox_validator.py"))


async def _get_app() -> modal.App:
    """Look up (or create) the Modal App for this pipeline.

    Called per-sandbox. Modal handles server-side idempotency on
    create_if_missing, so concurrent lookups are safe.
    """
    return await modal.App.lookup.aio(APP_NAME, create_if_missing=True)


async def run_claude_in_sandbox(
    site_files: dict[str, str],
    prompt: str,
    output_paths: list[str],
    timeout: int = 14400,
    model: str = "claude-opus-4-6",
    volumes: dict[str, modal.Volume] | None = None,
    label: str = "",
) -> dict[str, str | None]:
    """Run Claude Code in an isolated Modal Sandbox with only the specified files.

    Uses the ``claude-agent-sdk`` Python package for typed observability (cost
    tracking, token usage, session IDs, tool-call logging) while preserving the
    file-based output contract that all callers depend on.

    Args:
        site_files: Mapping from sandbox-remote path (``/workspace/...``) to a
            local source path. Each entry is added to the sandbox image via
            ``add_local_file`` (for files) or ``add_local_dir`` (for
            directories). This is the file-routing primitive that gives each
            phase true filesystem isolation.
        prompt: Full Claude Code prompt text (typically the contents of a
            ``worldsim/prompts/*.md`` file).
        output_paths: Absolute paths inside the sandbox to read back after
            Claude exits. Missing files are returned as ``None``.
        timeout: Wall-clock timeout for the sandbox in seconds.
        model: Model identifier passed to Claude Agent SDK (default:
            ``claude-opus-4-6``).
        volumes: Optional dict mapping mount paths to ``modal.Volume`` objects.
            Use for large, stable file sets (e.g., benchmark codebases) that
            should be uploaded once and mounted read-only, instead of being
            re-hashed and re-uploaded as mounts on every call.
        label: Optional human-readable label for log lines (e.g. phase name
            or site name). Prefixed to all sandbox log output.

    Returns:
        Dict mapping each entry in ``output_paths`` to the file's text
        contents, or ``None`` if the file could not be read. An additional
        ``"_summary"`` key contains a JSON string with cost, token usage,
        session ID, and tool-call metadata from the SDK.
    """
    # -- Build sandbox image with site files + runner --------------------------
    image = base_image
    for remote_path, local_path in site_files.items():
        local = Path(local_path)
        if local.is_file():
            image = image.add_local_file(local_path, remote_path=remote_path)
        elif local.is_dir():
            parent = str(Path(remote_path).parent)
            image = image.add_local_dir(local_path, remote_path=parent)
        else:
            logger.warning("skipping %s -> %s: path does not exist", local_path, remote_path)

    # Stage the SDK runner script and output validator into the sandbox.
    image = image.add_local_file(_RUNNER_PATH, remote_path="/workspace/_sdk_runner.py")
    image = image.add_local_file(_VALIDATOR_PATH, remote_path="/workspace/_validate.py")

    # -- Create sandbox --------------------------------------------------------
    app = await _get_app()
    sandbox_kwargs: dict = {"app": app, "image": image, "timeout": timeout}
    if volumes:
        sandbox_kwargs["volumes"] = volumes
    sandbox = await modal.Sandbox.create.aio(**sandbox_kwargs)

    try:
        # Write the prompt directly to the sandbox filesystem. This avoids
        # creating a mount object for a small, per-call file.
        await sandbox.filesystem.write_text.aio(prompt, "/workspace/_prompt.txt")

        claude_ps = await sandbox.exec.aio(
            "python",
            "/workspace/_sdk_runner.py",
            model,
            secrets=_build_claude_secrets(),
            workdir="/workspace",
        )

        # Stream NDJSON events from the runner for live observability.
        tag = f"[{label}] " if label else ""
        summary_data: dict | None = None
        turn_count = 0
        sandbox_start = time.monotonic()
        async for line in claude_ps.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("%snon-JSON stdout line from sandbox runner: %s", tag, line[:200])
                continue

            etype = event.get("type")
            if etype == "tool_call":
                turn_count += 1
                logger.info("  %s[sandbox] tool_call #%d: %s", tag, turn_count, event.get("tool"))
                if turn_count % 10 == 0:
                    elapsed = time.monotonic() - sandbox_start
                    logger.info(
                        "  %s[sandbox] progress: %d tool calls in %.1fs",
                        tag,
                        turn_count,
                        elapsed,
                    )
            elif etype == "text":
                preview = (event.get("preview", "") or "")[:100]
                logger.info("  %s[sandbox] text: %s", tag, preview)
            elif etype == "thinking":
                preview = (event.get("preview", "") or "")[:100]
                logger.debug("  %s[sandbox] thinking: %s", tag, preview)
            elif etype == "rate_limit":
                logger.warning(
                    "  %s[sandbox] rate limited, retry after %ss",
                    tag,
                    event.get("retry_after_seconds"),
                )
            elif etype == "error":
                logger.warning("  %s[sandbox] SDK error: %s", tag, event.get("message"))
            elif etype == "summary":
                summary_data = event
                _log_summary(event, label=label)

        await claude_ps.wait.aio()

        # Surface stderr so import failures, SDK errors, and tracebacks
        # are visible instead of silently lost.
        stderr_lines: list[str] = []
        async for line in claude_ps.stderr:
            line = line.strip()
            if line:
                stderr_lines.append(line)
        if stderr_lines:
            logger.warning(
                "%sSandbox stderr (%d lines):\n%s",
                tag,
                len(stderr_lines),
                "\n".join(stderr_lines[:20]),
            )

        if claude_ps.returncode != 0:
            logger.warning(
                "%sSandbox runner exited with rc=%d (%d tool calls)",
                tag,
                claude_ps.returncode,
                turn_count,
            )
        else:
            logger.info("%sSandbox runner exited (rc=0, %d tool calls)", tag, turn_count)

        # -- Read output files -------------------------------------------------
        outputs: dict[str, str | None] = {}
        for path in output_paths:
            try:
                outputs[path] = await sandbox.filesystem.read_text.aio(path)
            except Exception as e:
                outputs[path] = None
                logger.warning("could not read %s from sandbox: %s", path, e)

        # Attach summary metadata under a reserved key that callers can ignore.
        outputs["_summary"] = json.dumps(summary_data) if summary_data else None

        return outputs
    finally:
        await sandbox.terminate.aio()


async def upload_to_volume(
    local_dir: Path,
    volume_name: str | None = None,
    remote_prefix: str = "/",
) -> modal.Volume:
    """Upload a local directory to a Modal Volume keyed by directory contents.

    If the volume already has files under ``remote_prefix``, the upload is
    skipped. Returns the Volume object ready for mounting.
    """
    local_dir = Path(local_dir).resolve()
    if volume_name is None:
        volume_name = _content_addressed_volume_name(local_dir)
    vol = modal.Volume.from_name(volume_name, create_if_missing=True)
    try:
        entries = list(await vol.listdir.aio(remote_prefix))
        if entries:
            logger.info(
                "Volume %r already populated (%d entries at %s), skipping upload",
                volume_name,
                len(entries),
                remote_prefix,
            )
            return vol
    except Exception:
        pass  # Volume empty or path doesn't exist

    logger.info("Uploading %s to volume %r at %s ...", local_dir, volume_name, remote_prefix)
    async with vol.batch_upload.aio(force=True) as batch:
        batch.put_directory(str(local_dir), remote_prefix)
    logger.info("Upload complete")
    return vol


def _content_addressed_volume_name(local_dir: Path) -> str:
    """Build a stable Modal volume name from a directory content hash."""
    stem = (
        "".join(char if char.isalnum() else "-" for char in local_dir.name.lower()).strip("-")
        or "benchmark"
    )
    digest = _hash_directory(local_dir)[:12]
    return f"worldsim-{stem[:24]}-{digest}"


def _hash_directory(local_dir: Path) -> str:
    """Hash relative paths and file contents for a directory tree."""
    hasher = hashlib.sha256()
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                hasher.update(chunk)
        hasher.update(b"\0")
    return hasher.hexdigest()


def _log_summary(summary: dict, *, label: str = "") -> None:
    """Log key metrics from the runner's summary event."""
    tag = f"[{label}] " if label else ""
    parts = [f"elapsed={summary.get('elapsed_s')}s"]
    if summary.get("total_cost_usd") is not None:
        parts.append(f"cost=${summary['total_cost_usd']:.4f}")
    if summary.get("num_turns") is not None:
        parts.append(f"turns={summary['num_turns']}")
    parts.append(f"tool_calls={summary.get('num_tool_calls', 0)}")
    if summary.get("session_id"):
        parts.append(f"session={summary['session_id']}")
    if summary.get("is_error"):
        parts.append("ERROR")
    logger.info("  %s[sandbox] summary: %s", tag, ", ".join(parts))
