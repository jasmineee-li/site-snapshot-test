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
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import modal

logger = logging.getLogger(__name__)

APP_NAME = "worldsim-v5"
CLAUDE_SECRET_NAME = os.environ.get("WORLDSIM_CLAUDE_MODAL_SECRET", "anthropic-secret")

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
            secrets=[modal.Secret.from_name(CLAUDE_SECRET_NAME)],
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
