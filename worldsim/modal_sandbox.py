"""Modal sandbox for running Claude Code with file-routed inputs.

Each pipeline step gets a fresh ``modal.Sandbox`` whose image contains only
the files that step needs (inclusion-based scoping, not ignore-based).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
SANDBOX_WATCHDOG_SILENCE_SECONDS = 20 * 60
SANDBOX_WATCHDOG_POLL_SECONDS = 15
SANDBOX_RATE_LIMIT_GRACE_SECONDS = 90
SANDBOX_LIVENESS_LOG_SECONDS = 2 * 60

_RUNNER_PATH = str(Path(__file__).with_name("_sandbox_runner.py"))
_VALIDATOR_PATH = str(Path(__file__).with_name("_sandbox_validator.py"))
_PHASE_0C_VERIFY_HTTP_PATH = str(Path(__file__).with_name("phase_0c_verify_http.py"))


def _write_registry_snapshot() -> str:
    """Serialize the editor-method contract registry to a tempfile.

    Shipped alongside ``_sandbox_validator.py`` in the sandbox payload so
    the stdlib-only validator can consult the same contract data that
    the host-side :func:`worldsim.seeding._assert_benign_tokens_bound`
    and :func:`worldsim.phases.phase_2_injections._validate_option_a_placement_registry`
    read. Written once per process; the path is stable for the Modal
    image cache.
    """
    import tempfile as _tempfile

    from worldsim.editors._registry import serialize_registry

    tmp = _tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_editor_registry.json",
        delete=False,
        encoding="utf-8",
    )
    json.dump(serialize_registry(), tmp, sort_keys=True, indent=2)
    tmp.flush()
    tmp.close()
    return tmp.name


_EDITOR_REGISTRY_JSON_PATH = _write_registry_snapshot()

_APP_CACHE: modal.App | None = None
_APP_LOOKUP_LOCK: asyncio.Lock | None = None
_BASE_IMAGE_BUILT = False
_BASE_IMAGE_BUILD_LOCK: asyncio.Lock | None = None


def _app_lookup_lock() -> asyncio.Lock:
    global _APP_LOOKUP_LOCK
    if _APP_LOOKUP_LOCK is None:
        _APP_LOOKUP_LOCK = asyncio.Lock()
    return _APP_LOOKUP_LOCK


def _base_image_build_lock() -> asyncio.Lock:
    global _BASE_IMAGE_BUILD_LOCK
    if _BASE_IMAGE_BUILD_LOCK is None:
        _BASE_IMAGE_BUILD_LOCK = asyncio.Lock()
    return _BASE_IMAGE_BUILD_LOCK


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
    .pip_install("requests", "browser-use>=0.12.6", "claude-agent-sdk>=0.1.71,<0.2")
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
    # These harness files are used by every sandbox, so baking them into the
    # static image removes two per-launch local-file injections.
    .add_local_file(_RUNNER_PATH, remote_path="/workspace/_sdk_runner.py", copy=True)
    .add_local_file(_VALIDATOR_PATH, remote_path="/workspace/_validate.py", copy=True)
    .add_local_file(
        _PHASE_0C_VERIFY_HTTP_PATH,
        remote_path="/workspace/verify_http.py",
        copy=True,
    )
    .add_local_file(
        _EDITOR_REGISTRY_JSON_PATH,
        remote_path="/workspace/_editor_registry.json",
        copy=True,
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


async def preflight_sandbox_environment() -> None:
    """Verify auth and eagerly build the shared base image once per process."""
    preflight_auth_check()
    try:
        await _ensure_base_image_built()
    except Exception as exc:
        raise RuntimeError(f"Modal sandbox base-image prebuild failed: {exc}") from exc


class SandboxRetriableTimeoutError(TimeoutError):
    """Raised when the sandbox appears stalled but is safe to retry."""

    retriable = True

    def __init__(self, message: str, *, label: str = "") -> None:
        super().__init__(message)
        self.label = label


@dataclass
class SandboxWatchdogState:
    """Track shard liveness across normal progress and rate-limit backoff."""

    last_event_at: float
    last_non_rate_limit_progress_at: float
    last_liveness_log_at: float
    blocked_until: float | None = None
    consecutive_rejected_rate_limits: int = 0


def _coerce_float(value: Any) -> float | None:
    """Best-effort float coercion for JSON event fields."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _update_watchdog_state(
    state: SandboxWatchdogState,
    event: dict[str, Any],
    *,
    now_monotonic: float,
    now_wall: float,
) -> None:
    """Record liveness state from a sandbox NDJSON event."""
    state.last_event_at = now_monotonic
    if event.get("type") != "rate_limit":
        state.last_non_rate_limit_progress_at = now_monotonic
        state.blocked_until = None
        state.consecutive_rejected_rate_limits = 0
        return

    if event.get("status") == "rejected":
        state.consecutive_rejected_rate_limits += 1
        resets_at = _coerce_float(event.get("resets_at"))
        if resets_at is not None:
            state.blocked_until = max(resets_at, now_wall)
            return
        retry_after_seconds = _coerce_float(event.get("retry_after_seconds"))
        if retry_after_seconds is not None:
            state.blocked_until = now_wall + max(0.0, retry_after_seconds)
            return
        state.blocked_until = None
        return

    state.blocked_until = None
    state.consecutive_rejected_rate_limits = 0


def _watchdog_timeout_reason(
    state: SandboxWatchdogState,
    *,
    now_monotonic: float,
    now_wall: float,
    silence_seconds: int = SANDBOX_WATCHDOG_SILENCE_SECONDS,
    rate_limit_grace_seconds: int = SANDBOX_RATE_LIMIT_GRACE_SECONDS,
) -> str | None:
    """Return a timeout reason when the shard appears stalled."""
    if state.consecutive_rejected_rate_limits > 0 and state.blocked_until is not None:
        if now_wall > state.blocked_until + rate_limit_grace_seconds:
            quiet_for = now_monotonic - state.last_non_rate_limit_progress_at
            return (
                "no non-rate-limit progress after rejected rate limit backoff "
                f"(quiet={quiet_for:.0f}s, blocked_until={state.blocked_until:.0f}, "
                f"rejected_events={state.consecutive_rejected_rate_limits})"
            )

    quiet_for = now_monotonic - state.last_event_at
    if quiet_for > silence_seconds:
        return f"no sandbox events for {quiet_for:.0f}s"
    return None


def _watchdog_liveness_message(
    state: SandboxWatchdogState,
    *,
    now_monotonic: float,
    now_wall: float,
    tool_calls: int,
    liveness_log_seconds: int = SANDBOX_LIVENESS_LOG_SECONDS,
) -> str | None:
    """Return a periodic liveness message while a sandbox is quiet but not stalled."""
    if now_monotonic - state.last_liveness_log_at < liveness_log_seconds:
        return None

    state.last_liveness_log_at = now_monotonic
    quiet_for = now_monotonic - state.last_event_at
    non_rate_limit_quiet_for = now_monotonic - state.last_non_rate_limit_progress_at
    parts = [
        f"waiting for sandbox event (quiet={quiet_for:.0f}s",
        f"non_rate_limit_quiet={non_rate_limit_quiet_for:.0f}s",
        f"tool_calls={tool_calls}",
    ]
    if state.consecutive_rejected_rate_limits > 0 and state.blocked_until is not None:
        remaining = max(0.0, state.blocked_until - now_wall)
        parts.append(
            f"rate_limit_backoff_remaining={remaining:.0f}s "
            f"rejected_events={state.consecutive_rejected_rate_limits}"
        )
    parts.append(")")
    return ", ".join(parts)


async def _get_app() -> modal.App:
    """Look up (or create) the Modal App for this pipeline.

    The app identity is stable for the process lifetime, so cache the first
    successful lookup and reuse it on every sandbox launch.
    """
    global _APP_CACHE
    if _APP_CACHE is not None:
        return _APP_CACHE

    async with _app_lookup_lock():
        if _APP_CACHE is None:
            _APP_CACHE = await modal.App.lookup.aio(APP_NAME, create_if_missing=True)
    return _APP_CACHE


async def _ensure_base_image_built() -> None:
    """Eagerly build the shared base image once so first sandbox avoids it."""
    global _BASE_IMAGE_BUILT
    if _BASE_IMAGE_BUILT:
        return

    async with _base_image_build_lock():
        if _BASE_IMAGE_BUILT:
            return
        app = await _get_app()
        start = time.monotonic()
        await base_image.build.aio(app)
        elapsed = time.monotonic() - start
        _BASE_IMAGE_BUILT = True
        logger.info("Pre-built Modal sandbox base image in %.2fs", elapsed)


async def run_claude_in_sandbox(
    site_files: dict[str, str],
    prompt: str,
    output_paths: list[str],
    timeout: int = 14400,
    model: str = "claude-sonnet-4-6",
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
            ``claude-sonnet-4-6``). In practice, we run Sonnet first for
            long smokes and cost-sensitive passes, then move to Opus for
            confirmation runs when needed.
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
    launch_start = time.monotonic()
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

    # -- Create sandbox --------------------------------------------------------
    app_lookup_start = time.monotonic()
    app = await _get_app()
    app_lookup_done = time.monotonic()
    sandbox_kwargs: dict = {"app": app, "image": image, "timeout": timeout}
    if volumes:
        sandbox_kwargs["volumes"] = volumes
    sandbox_create_start = time.monotonic()
    sandbox = await modal.Sandbox.create.aio(**sandbox_kwargs)
    sandbox_create_done = time.monotonic()

    try:
        # Write the prompt directly to the sandbox filesystem. This avoids
        # creating a mount object for a small, per-call file.
        prompt_write_start = time.monotonic()
        await sandbox.filesystem.write_text.aio(prompt, "/workspace/_prompt.txt")
        prompt_write_done = time.monotonic()

        exec_start = time.monotonic()
        claude_ps = await sandbox.exec.aio(
            "python",
            "/workspace/_sdk_runner.py",
            model,
            secrets=_build_claude_secrets(),
            workdir="/workspace",
            timeout=timeout,
            bufsize=1,
        )
        exec_done = time.monotonic()

        tag = f"[{label}] " if label else ""
        logger.info(
            "%sSandbox startup timings: app=%.3fs create=%.3fs prompt_write=%.3fs exec_start=%.3fs",
            tag,
            app_lookup_done - app_lookup_start,
            sandbox_create_done - sandbox_create_start,
            prompt_write_done - prompt_write_start,
            exec_done - exec_start,
        )

        # Stream NDJSON events from the runner for live observability.
        # IMPORTANT, stdout and stderr MUST be drained concurrently. Modal's
        # _StreamReaderThroughServer creates a background asyncio.Task per
        # stream at construction that drains the gRPC stream into an in-memory
        # buffer with 10 retry credits. If we drain serially, an unread
        # stream's retry budget can exhaust during long runs (server recycles
        # its 55s window repeatedly), the background task silently dies WITHOUT
        # writing the `None` EOF sentinel, and the subsequent async-for polls
        # the buffer forever. This wedged 5/43 shards in our first Phase 2 run.
        summary_data: dict | None = None
        turn_count = 0
        sandbox_start = time.monotonic()
        watchdog_state = SandboxWatchdogState(
            last_event_at=sandbox_start,
            last_non_rate_limit_progress_at=sandbox_start,
            last_liveness_log_at=sandbox_start,
        )
        stderr_lines: list[str] = []
        watchdog_done = asyncio.Event()
        watchdog_reason: str | None = None
        first_event_at: float | None = None
        event_counts: dict[str, int] = {}
        rate_limit_events = 0
        non_json_stdout_lines = 0

        async def _drain_stdout() -> None:
            nonlocal summary_data, turn_count, first_event_at, rate_limit_events
            nonlocal non_json_stdout_lines
            try:
                async for line in claude_ps.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug(
                            "%snon-JSON stdout line from sandbox runner: %s",
                            tag,
                            line[:200],
                        )
                        non_json_stdout_lines += 1
                        continue

                    now_monotonic = time.monotonic()
                    if first_event_at is None:
                        first_event_at = now_monotonic
                        logger.info(
                            "%sSandbox first runner event after %.3fs (%.3fs since launch)",
                            tag,
                            first_event_at - exec_done,
                            first_event_at - launch_start,
                        )
                    _update_watchdog_state(
                        watchdog_state,
                        event,
                        now_monotonic=now_monotonic,
                        now_wall=time.time(),
                    )
                    watchdog_state.last_liveness_log_at = now_monotonic
                    etype = event.get("type")
                    event_key = str(etype or "unknown")
                    event_counts[event_key] = event_counts.get(event_key, 0) + 1
                    if etype == "tool_call":
                        turn_count += 1
                        logger.info(
                            "  %s[sandbox] tool_call #%d: %s",
                            tag,
                            turn_count,
                            event.get("tool"),
                        )
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
                        rate_limit_events += 1
                        status = event.get("status")
                        rate_limit_type = event.get("rate_limit_type")
                        utilization = event.get("utilization")
                        logger.warning(
                            "  %s[sandbox] rate limited status=%s type=%s retry_after=%ss resets_at=%s utilization=%s",
                            tag,
                            status,
                            rate_limit_type,
                            event.get("retry_after_seconds"),
                            event.get("resets_at"),
                            utilization,
                        )
                    elif etype == "error":
                        logger.warning("  %s[sandbox] SDK error: %s", tag, event.get("message"))
                    elif etype == "summary":
                        summary_data = event
                        _log_summary(event, label=label)
            finally:
                # Release the SDK's stream generator so its background
                # _consume_container_process_task can clean up and not leak.
                try:
                    await claude_ps.stdout.aclose()
                except Exception:
                    pass

        async def _drain_stderr() -> None:
            try:
                async for line in claude_ps.stderr:
                    line = line.strip()
                    if line:
                        stderr_lines.append(line)
            finally:
                try:
                    await claude_ps.stderr.aclose()
                except Exception:
                    pass

        async def _watchdog() -> None:
            nonlocal watchdog_reason
            while not watchdog_done.is_set():
                await asyncio.sleep(SANDBOX_WATCHDOG_POLL_SECONDS)
                reason = _watchdog_timeout_reason(
                    watchdog_state,
                    now_monotonic=time.monotonic(),
                    now_wall=time.time(),
                )
                if reason is None:
                    message = _watchdog_liveness_message(
                        watchdog_state,
                        now_monotonic=time.monotonic(),
                        now_wall=time.time(),
                        tool_calls=turn_count,
                    )
                    if message is not None:
                        logger.info("  %s[sandbox] liveness: %s", tag, message)
                    continue
                watchdog_reason = reason
                logger.error("  %s[sandbox] watchdog timeout: %s", tag, reason)
                try:
                    await sandbox.terminate.aio()
                except Exception as exc:
                    logger.warning("%sSandbox watchdog terminate failed: %s", tag, exc)
                return

        watchdog_task = asyncio.create_task(_watchdog())
        await asyncio.gather(_drain_stdout(), _drain_stderr())
        watchdog_done.set()
        try:
            await claude_ps.wait.aio()
        finally:
            watchdog_task.cancel()
            try:
                await watchdog_task
            except asyncio.CancelledError:
                pass

        # Surface stderr so import failures, SDK errors, and tracebacks
        # are visible instead of silently lost.
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

        if watchdog_reason is not None:
            raise SandboxRetriableTimeoutError(
                f"{label or 'sandbox'} stalled: {watchdog_reason}",
                label=label,
            )

        # -- Read output files -------------------------------------------------
        outputs: dict[str, str | None] = {}
        missing_output_paths: list[str] = []
        for path in output_paths:
            try:
                outputs[path] = await sandbox.filesystem.read_text.aio(path)
            except Exception as e:
                outputs[path] = None
                missing_output_paths.append(path)
                logger.warning("could not read %s from sandbox: %s", path, e)

        # Attach summary metadata under a reserved key that callers can ignore.
        outputs["_summary"] = json.dumps(summary_data) if summary_data else None
        outputs["_telemetry"] = json.dumps(
            {
                "schema_version": 1,
                "label": label,
                "model": model,
                "timeout_seconds": timeout,
                "startup_seconds": {
                    "app_lookup": app_lookup_done - app_lookup_start,
                    "sandbox_create": sandbox_create_done - sandbox_create_start,
                    "prompt_write": prompt_write_done - prompt_write_start,
                    "exec_start": exec_done - exec_start,
                    "launch_to_exec": exec_done - launch_start,
                    "first_event_from_exec": (
                        first_event_at - exec_done if first_event_at is not None else None
                    ),
                    "first_event_from_launch": (
                        first_event_at - launch_start if first_event_at is not None else None
                    ),
                },
                "event_counts": dict(sorted(event_counts.items())),
                "tool_calls": turn_count,
                "rate_limit_events": rate_limit_events,
                "non_json_stdout_lines": non_json_stdout_lines,
                "stderr_line_count": len(stderr_lines),
                "returncode": claude_ps.returncode,
                "watchdog_reason": watchdog_reason,
                "requested_output_paths": output_paths,
                "missing_output_paths": missing_output_paths,
            },
            sort_keys=True,
        )

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
