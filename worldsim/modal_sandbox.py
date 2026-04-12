"""Modal sandbox primitive for running Claude Code with file-routed inputs.

Every Claude Code step in the WorldSim v5 pipeline runs inside a fresh
``modal.Sandbox`` created by :func:`run_claude_in_sandbox`. The sandbox image
is built *per call* by adding only the files that step needs via
``modal.Image.add_local_dir`` — inclusion-based scoping, not ignore-based.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md``, "Modal Infrastructure" section.

The two AgentLab reference files may be consulted for auth / secret wiring
mechanics but are NOT imported:

- ``AgentLab/src/agentlab/benchmarks/redteam/execution.py``
- ``AgentLab/src/agentlab/benchmarks/redteam/claude_code.py``

Claude Code authentication
--------------------------

Two auth methods are supported in parallel:

- ``ANTHROPIC_API_KEY`` — API credit billing (traditional Anthropic API key).
- ``CLAUDE_CODE_OAUTH_TOKEN`` — Claude Pro / Claude Max subscription auth.

**OAuth wins when both are set.** Claude Code's internal auth precedence puts
API keys *above* OAuth tokens, so if we forwarded both, the CLI would silently
bill against API credits instead of the subscription. We therefore drop
``ANTHROPIC_API_KEY`` from the forwarded env when ``CLAUDE_CODE_OAUTH_TOKEN``
is also present.

Two delivery mechanisms are supported:

1. **Local shell forwarding (default).** Export one or both of the env vars
   above in your shell. ``_build_claude_secrets`` reads them and packages
   them into an ephemeral ``modal.Secret.from_dict`` per sandbox invocation.
   The values live only in your shell, in-memory, and in the sandbox — not
   in Modal's named-secret store.

2. **Named Modal secret (opt-in, CI-friendly).** Set
   ``WORLDSIM_CLAUDE_MODAL_SECRET=<name>`` and we'll use
   ``modal.Secret.from_name(<name>)`` instead. Use this when a shared
   Modal workspace has a curated named secret (e.g. for CI). Whatever env
   vars the named secret contains are injected as-is, *without* the
   OAuth-wins fixup — managing that is your responsibility.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import modal

logger = logging.getLogger(__name__)

APP_NAME = "worldsim-v5"

#: Env vars forwarded from the local shell into the sandbox's Claude Code
#: process. See module docstring for priority rules.
CLAUDE_ALLOWED_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
)

#: If set, :func:`_build_claude_secrets` uses ``Secret.from_name(<value>)``
#: instead of building one from the local shell. Opt-in for CI / shared
#: workspaces.
NAMED_SECRET_ENV_VAR = "WORLDSIM_CLAUDE_MODAL_SECRET"

app = modal.App(APP_NAME)

#: Base image used by every Claude Code sandbox in the pipeline.
#:
#: Matches the v5 spec "Modal Infrastructure" section verbatim.
base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "git", "jq")
    .env({"PATH": "/root/.local/bin:$PATH"})
    .run_commands("curl -fsSL https://claude.ai/install.sh | bash")
    .pip_install("requests", "browser-use>=0.12.6")
)


def _build_claude_secrets() -> list[modal.Secret]:
    """Build the list of Modal secrets to attach to a Claude Code sandbox.

    Priority rules (from the feat/app-generation-infra branch's
    ``build_claude_env`` function, whose comment explains the rationale):

    1. If ``WORLDSIM_CLAUDE_MODAL_SECRET`` is set, attach that named secret
       and return. Managing its contents is the user's responsibility.
    2. Otherwise, read ``ANTHROPIC_API_KEY`` and ``CLAUDE_CODE_OAUTH_TOKEN``
       from the local shell. If both are present, drop
       ``ANTHROPIC_API_KEY`` — Claude Code's auth precedence would otherwise
       silently bill against API credits instead of the Pro/Max subscription.
    3. Package the remaining env vars into an ephemeral
       ``modal.Secret.from_dict``.

    Raises:
        RuntimeError: If neither delivery mechanism yields any credentials.
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

    if "CLAUDE_CODE_OAUTH_TOKEN" in env and "ANTHROPIC_API_KEY" in env:
        logger.info(
            "Both CLAUDE_CODE_OAUTH_TOKEN and ANTHROPIC_API_KEY are set; "
            "dropping ANTHROPIC_API_KEY so Claude Code uses the subscription "
            "auth path (otherwise API key silently wins and you get billed "
            "for API credits instead of your Pro/Max subscription)."
        )
        del env["ANTHROPIC_API_KEY"]
    elif "CLAUDE_CODE_OAUTH_TOKEN" in env:
        logger.debug("Using CLAUDE_CODE_OAUTH_TOKEN for Claude Code auth")
    else:
        logger.debug("Using ANTHROPIC_API_KEY for Claude Code auth")

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
