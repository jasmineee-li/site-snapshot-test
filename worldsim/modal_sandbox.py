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
    "CLAUDE_CODE_OAUTH_TOKEN",
)

NAMED_SECRET_ENV_VAR = "WORLDSIM_CLAUDE_MODAL_SECRET"

app = modal.App(APP_NAME)

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "git", "jq")
    .env({"PATH": "/root/.local/bin:$PATH"})
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

    if not env:
        raise RuntimeError(
            "No Claude Code auth available for the Modal sandbox. Either:\n"
            "  export CLAUDE_CODE_OAUTH_TOKEN=...   # Claude Pro/Max subscription (preferred)\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...  # API credit billing\n"
            "Or point at an existing named Modal secret:\n"
            f"  export {NAMED_SECRET_ENV_VAR}=<secret-name>"
        )

    # OAuth wins: Claude Code's own precedence puts API key above OAuth,
    # so forwarding both would silently bill API credits instead of the subscription.
    if "CLAUDE_CODE_OAUTH_TOKEN" in env:
        env.pop("ANTHROPIC_API_KEY", None)

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
            local source path. For each entry, the sandbox image is extended
            with ``image.add_local_dir(local_path, remote_path=parent)``, where
            ``parent`` is the parent directory of ``remote_path``. This is the
            file-routing primitive that gives each phase true filesystem
            isolation.
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
        parent = str(Path(remote_path).parent)
        image = image.add_local_dir(local_path, remote_path=parent)

    sandbox = modal.Sandbox.create(app=app, image=image, timeout=timeout)
    try:
        claude_ps = sandbox.exec(
            "claude",
            "-p",
            prompt,
            "--dangerously-skip-permissions",
            "--permission-mode",
            "plan",
            "--verbose",
            "--effort",
            "high",
            pty=True,
            secrets=_build_claude_secrets(),
            workdir="/workspace",
        )
        claude_ps.wait()

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
