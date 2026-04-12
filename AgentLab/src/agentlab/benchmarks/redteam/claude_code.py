"""Claude Code CLI invocation helpers.

Thin wrapper around the Claude Code CLI binary.  Builds the subprocess
command, sets up the environment, invokes the configured execution backend,
and persists prompt / output logs for reproducibility.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from agentlab.benchmarks.redteam.execution import (
    build_claude_command,
    build_claude_env,
    execute_authoring_command,
)
from agentlab.benchmarks.redteam.utils import write_text

logger = logging.getLogger(__name__)

# Glob patterns forwarded to the execution backend so only these files are
# synced back from a remote sandbox after Claude Code finishes.
_CLAUDE_SYNC_ALLOWLIST = (
    "index.html",
    "css/**",
    "js/**",
    "benign/**",
    "adversarial_*/**",
    "behaviors/**",
    "function-tasks.json",
    "function-tasks/**",
    "real-tasks.json",
    "real-tasks/**",
    "sanity_check*.py",
    "APP_DESCRIPTION.md",
    "app_manifest.json",
    "functional_results.json",
    "results/**",
)


def run_claude_code(
    prompt: str,
    working_dir: str | Path,
    timeout: int = 300,
) -> tuple[int, str, str]:
    """Invoke Claude Code CLI with the given prompt.

    Runs Claude Code through the configured execution backend,
    passing *prompt* as a positional argument.  Logs the prompt and output to
    ``.claude_prompt.md`` and ``.claude_output.log`` inside *working_dir*.

    Args:
        prompt: The full prompt text to pass to Claude Code.
        working_dir: Directory in which Claude Code will execute.
        timeout: Maximum wall-clock seconds before the subprocess is killed.

    Returns:
        Tuple of (return_code, stdout, stderr).

    """
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    # Persist the prompt for reproducibility / debugging
    prompt_log = working_dir / ".claude_prompt.md"
    write_text(prompt_log, prompt)

    cmd = build_claude_command(prompt)

    logger.info(
        "Running Claude Code (cwd=%s, timeout=%ds, prompt_len=%d)",
        working_dir,
        timeout,
        len(prompt),
    )

    try:
        result = execute_authoring_command(
            working_dir=working_dir,
            argv=cmd,
            timeout=timeout,
            sync_allowlist=_CLAUDE_SYNC_ALLOWLIST,
            env=build_claude_env(),
        )
    except FileNotFoundError:
        msg = (
            "claude CLI not found on PATH. "
            "Install it with: npm install -g @anthropic-ai/claude-code"
        )
        logger.error(msg)
        raise FileNotFoundError(msg)
    except subprocess.TimeoutExpired:
        logger.error("Claude Code timed out after %ds", timeout)
        output_log = working_dir / ".claude_output.log"
        write_text(output_log, f"TIMEOUT after {timeout}s\n")
        return (1, "", f"Timeout after {timeout}s")

    # Persist output for debugging
    output_log = working_dir / ".claude_output.log"
    log_content = result.stdout or ""
    if result.stderr:
        log_content += "\n--- stderr ---\n" + result.stderr
    write_text(output_log, log_content)

    reserved_runtime_mutations = result.metadata.get("reserved_runtime_mutations", [])
    if reserved_runtime_mutations:
        violation = (
            "Reserved runtime file mutation detected and rejected: "
            + ", ".join(sorted(str(path) for path in reserved_runtime_mutations))
        )
        output_log.write_text(log_content + "\n--- policy ---\n" + violation, encoding="utf-8")
        logger.error(violation)
        stderr = result.stderr
        if stderr:
            stderr = f"{stderr}\n{violation}"
        else:
            stderr = violation
        return (1, result.stdout, stderr)

    if result.returncode != 0:
        logger.error(
            "Claude Code failed (rc=%d): %s",
            result.returncode,
            (result.stderr or "")[-500:],
        )

    return (result.returncode, result.stdout, result.stderr)
