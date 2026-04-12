"""Modal sandbox for running Claude Code with file-routed inputs.

Each pipeline step gets a fresh ``modal.Sandbox`` whose image contains only
the files that step needs (inclusion-based scoping, not ignore-based).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
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
    .apt_install("curl", "git", "jq", "nodejs", "npm")
    .run_commands(
        "npm install -g @anthropic-ai/claude-code",
        "mkdir -p /workspace /root/.claude",
    )
    .env({
        # IS_SANDBOX=1 lets Claude Code accept --dangerously-skip-permissions
        # as root (github.com/anthropics/claude-code/issues/3490). This is
        # Modal's own recommended pattern for running Claude Code in sandboxes.
        "IS_SANDBOX": "1",
    })
    # Pre-accept trust dialog so Claude Code runs non-interactively.
    .run_commands(
        'python3 -c "'
        "import json; from pathlib import Path; "
        "Path('/root/.claude.json').write_text("
        "json.dumps({'projects': {'/workspace': {'hasTrustDialogAccepted': True}}})); "
        "Path('/root/.claude/settings.json').write_text("
        "json.dumps({'skipDangerousModePermissionPrompt': True}))"
        '"',
    )
    .pip_install("requests", "browser-use>=0.12.6", "claude-agent-sdk")
)


def _build_claude_secrets() -> list[modal.Secret]:
    """Build Modal secrets for Claude Code auth.

    Raises RuntimeError if no credentials are available.
    """
    named = os.environ.get(NAMED_SECRET_ENV_VAR, "").strip()
    if named:
        logger.info("Using named Modal secret %r for Claude Code auth", named)
        return [modal.Secret.from_name(named)]

    env: dict[str, str] = {}
    for key in CLAUDE_ALLOWED_ENV_VARS:
        value = os.environ.get(key)
        if value:
            env[key] = value

    has_creds = (
        "ANTHROPIC_API_KEY" in env
        or "ANTHROPIC_AUTH_TOKEN" in env
        or "CLAUDE_CODE_OAUTH_TOKEN" in env
    )
    if not has_creds:
        raise RuntimeError(
            "No Claude Code auth available for the Modal sandbox. Either:\n"
            "  export CLAUDE_CODE_OAUTH_TOKEN=...              # Pro/Max subscription (preferred)\n"
            "  export ANTHROPIC_AUTH_TOKEN=sk-or-v1-...         # OpenRouter\n"
            "         ANTHROPIC_BASE_URL=https://openrouter.ai/api\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...              # direct Anthropic API\n"
            "Or point at an existing named Modal secret:\n"
            f"  export {NAMED_SECRET_ENV_VAR}=<secret-name>"
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


_RUNNER_PATH = str(Path(__file__).with_name("_sandbox_runner.py"))


async def run_claude_in_sandbox(
    site_files: dict[str, str],
    prompt: str,
    output_paths: list[str],
    timeout: int = 3600,
    model: str = "claude-opus-4-6",
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

    Returns:
        Dict mapping each entry in ``output_paths`` to the file's text
        contents, or ``None`` if the file could not be read. An additional
        ``"_summary"`` key contains a JSON string with cost, token usage,
        session ID, and tool-call metadata from the SDK.
    """
    # -- Build sandbox image with site files + prompt + runner ----------------
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

    # Stage the SDK runner script into the sandbox.
    image = image.add_local_file(_RUNNER_PATH, remote_path="/workspace/_sdk_runner.py")

    # Stage the prompt as a file so it doesn't need shell escaping.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="worldsim_prompt_", delete=False,
    ) as tmp:
        tmp.write(prompt)
        prompt_tmp_path = tmp.name
    try:
        image = image.add_local_file(prompt_tmp_path, remote_path="/workspace/_prompt.txt")
    finally:
        # Clean up promptly; Modal has already read the file into the image.
        Path(prompt_tmp_path).unlink(missing_ok=True)

    # -- Create sandbox and execute the runner --------------------------------
    app = await modal.App.lookup.aio(APP_NAME, create_if_missing=True)
    sandbox = await modal.Sandbox.create.aio(app=app, image=image, timeout=timeout)
    try:
        claude_ps = await sandbox.exec.aio(
            "python", "/workspace/_sdk_runner.py", model,
            secrets=_build_claude_secrets(),
            workdir="/workspace",
        )

        # Stream NDJSON events from the runner for live observability.
        summary_data: dict | None = None
        async for line in claude_ps.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("non-JSON stdout line from sandbox runner: %s", line[:200])
                continue

            etype = event.get("type")
            if etype == "tool_call":
                logger.info("  [sandbox] tool_call: %s", event.get("tool"))
            elif etype == "text":
                logger.debug("  [sandbox] text: %s", event.get("preview", "")[:120])
            elif etype == "error":
                logger.warning("  [sandbox] SDK error: %s", event.get("message"))
            elif etype == "summary":
                summary_data = event
                _log_summary(event)

        await claude_ps.wait.aio()

        if claude_ps.returncode != 0:
            logger.warning(
                "Sandbox runner exited with rc=%d",
                claude_ps.returncode,
            )
        else:
            logger.info("Sandbox runner finished (rc=0)")

        # -- Read output files ------------------------------------------------
        outputs: dict[str, str | None] = {}
        for path in output_paths:
            try:
                outputs[path] = await sandbox.filesystem.read_text.aio(path)
            except Exception as e:  # noqa: BLE001 -- tolerate missing files
                outputs[path] = None
                logger.warning("could not read %s from sandbox: %s", path, e)

        # Attach summary metadata under a reserved key that callers can ignore.
        outputs["_summary"] = json.dumps(summary_data) if summary_data else None

        return outputs
    finally:
        await sandbox.terminate.aio()


def _log_summary(summary: dict) -> None:
    """Log key metrics from the runner's summary event."""
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
    logger.info("  [sandbox] summary: %s", ", ".join(parts))
