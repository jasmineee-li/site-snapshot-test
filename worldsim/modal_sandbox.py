"""Modal sandbox for running Claude Code with file-routed inputs.

Each pipeline step gets a fresh ``modal.Sandbox`` whose image contains only
the files that step needs (inclusion-based scoping, not ignore-based).
"""

from __future__ import annotations

import logging
import os
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
    .env({"PATH": "/root/.local/bin:$PATH"})  # Modal expands $PATH at container build time, not at Python parse time
    .run_commands("curl -fsSL https://claude.ai/install.sh | bash")
    .pip_install("requests", "browser-use>=0.12.6")
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


async def run_claude_in_sandbox(
    site_files: dict[str, str],
    prompt: str,
    output_paths: list[str],
    timeout: int = 3600,
) -> dict[str, str | None]:
    """Run Claude Code in an isolated Modal Sandbox with only the specified files.

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

    Returns:
        Dict mapping each entry in ``output_paths`` to the file's text
        contents, or ``None`` if the file could not be read.
    """
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

    app = await modal.App.lookup.aio(APP_NAME, create_if_missing=True)
    sandbox = await modal.Sandbox.create.aio(app=app, image=image, timeout=timeout)
    try:
        claude_ps = await sandbox.exec.aio(
            "claude",
            "-p",
            prompt,
            "--dangerously-skip-permissions",
            "--verbose",
            "--effort",
            "high",
            pty=True,
            secrets=_build_claude_secrets(),
            workdir="/workspace",
        )
        await claude_ps.wait.aio()

        # With pty=True, stderr is multiplexed into stdout (Modal SDK).
        stdout = await claude_ps.stdout.read.aio()
        if claude_ps.returncode != 0:
            logger.warning(
                "Claude Code exited with rc=%d. Output tail:\n%s",
                claude_ps.returncode,
                stdout[-2000:] if stdout else "(empty)",
            )
        else:
            logger.info("Claude Code finished (rc=0, output=%d chars)", len(stdout))

        outputs: dict[str, str | None] = {}
        for path in output_paths:
            try:
                outputs[path] = sandbox.filesystem.read_text(path)
            except Exception as e:  # noqa: BLE001 — we want to tolerate any missing file
                outputs[path] = None
                logger.warning("could not read %s from sandbox: %s", path, e)
        return outputs
    finally:
        sandbox.terminate()
