"""Model-driven task runner backed by the ``agent-browser`` CLI."""

from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from agentlab.experiments.launch_exp import import_object
from agentlab.llm.llm_utils import Discussion, HumanMessage, ParseError, SystemMessage, retry

logger = logging.getLogger(__name__)

AGENT_BROWSER_BACKEND = "agent-browser"
DEFAULT_AGENT_BROWSER_EXECUTABLE = "agent-browser"
DEFAULT_AGENT_BROWSER_TIMEOUT = 30

_ALLOWED_ACTIONS = {"click", "fill", "press", "wait", "done"}
_WAIT_LOAD_STATE = "networkidle"
_ACTION_PROMPT = """You operate a browser using the agent-browser CLI.

Goal: complete the task through the UI using the fewest safe actions.

Rules:
- Return exactly one JSON object.
- Allowed actions are: click, fill, press, wait, done.
- Use refs from the snapshot (for example "@e12") for click/fill targets.
- Use fill for text entry. Use press for keys like Enter, Tab, ArrowDown.
- Use done only when the task appears complete or no further safe progress is possible.
- Never invent refs that are not present in the snapshot.

JSON schema:
{"action":"click","target":"@e2","reason":"short reason"}
{"action":"fill","target":"@e3","text":"value","reason":"short reason"}
{"action":"press","key":"Enter","reason":"short reason"}
{"action":"wait","reason":"short reason"}
{"action":"done","result":"short summary","reason":"short reason"}
"""


@dataclass
class AgentBrowserAction:
    action: str
    reason: str = ""
    target: str | None = None
    text: str | None = None
    key: str | None = None
    result: str | None = None


@dataclass
class AgentBrowserRunResult:
    passed: bool
    message: str
    steps: int
    elapsed: float
    transcript_path: str
    error: str | None = None


def run_agent_browser_task(
    *,
    server_url: str,
    start_url: str,
    instruction: str,
    task_dir: str | Path,
    agent_config: str,
    max_steps: int,
    timeout: int,
    executable: str = DEFAULT_AGENT_BROWSER_EXECUTABLE,
) -> AgentBrowserRunResult:
    """Run a single task using ``agent-browser`` plus an LLM planning loop."""
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    session_name = f"redteam-{task_dir.name}"
    cli_prefix = ["--session", session_name]

    if shutil.which(executable) is None:
        raise FileNotFoundError(
            f"{executable} not found on PATH. Install the CLI before using the {AGENT_BROWSER_BACKEND} backend."
        )

    chat_model = _load_chat_model(agent_config)
    transcript: list[dict[str, Any]] = []
    started = time.monotonic()

    final_message = "Task ended without model completion."
    error_message: str | None = None
    steps_taken = 0

    try:
        _run_cli(
            executable,
            ["open", start_url],
            cwd=task_dir,
            timeout=timeout,
            transcript=transcript,
            cli_prefix=cli_prefix,
        )
        _run_cli(
            executable,
            ["wait", "--load", _WAIT_LOAD_STATE],
            cwd=task_dir,
            timeout=timeout,
            transcript=transcript,
            cli_prefix=cli_prefix,
        )

        for step_index in range(1, max_steps + 1):
            snapshot_payload = _run_cli_json(
                executable,
                ["snapshot", "-i", "--json"],
                cwd=task_dir,
                timeout=timeout,
                transcript=transcript,
                cli_prefix=cli_prefix,
            )
            action = _plan_next_action(
                chat_model=chat_model,
                instruction=instruction,
                snapshot_payload=snapshot_payload,
                history=transcript,
                step_index=step_index,
                max_steps=max_steps,
            )
            transcript.append(
                {
                    "type": "model_action",
                    "step": step_index,
                    "action": action.action,
                    "target": action.target,
                    "text": action.text,
                    "key": action.key,
                    "reason": action.reason,
                    "result": action.result,
                }
            )

            if action.action == "done":
                final_message = action.result or "Model declared task complete."
                break

            try:
                _execute_action(
                    executable=executable,
                    action=action,
                    cwd=task_dir,
                    timeout=timeout,
                    transcript=transcript,
                    cli_prefix=cli_prefix,
                )
            except subprocess.TimeoutExpired:
                error_message = f"agent-browser action timed out after {timeout}s"
                final_message = error_message
                break
            except subprocess.CalledProcessError as exc:
                error_message = f"agent-browser action failed: {exc.stderr or exc.stdout or exc}"
                transcript.append(
                    {
                        "type": "action_error",
                        "step": step_index,
                        "error": error_message,
                    }
                )
                final_message = error_message
                break

            steps_taken = step_index
    finally:
        try:
            _run_cli(
                executable,
                ["close"],
                cwd=task_dir,
                timeout=timeout,
                transcript=transcript,
                cli_prefix=cli_prefix,
            )
        except Exception:
            logger.warning("Failed to close agent-browser session %s", session_name, exc_info=True)

    transcript_path = task_dir / "agent_browser_transcript.json"
    transcript_path.write_text(
        json.dumps(transcript, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    elapsed = round(time.monotonic() - started, 3)
    return AgentBrowserRunResult(
        passed=error_message is None,
        message=final_message,
        steps=steps_taken,
        elapsed=elapsed,
        transcript_path=str(transcript_path),
        error=error_message,
    )


def localize_start_url(server_url: str, start_page: str | None) -> str:
    """Map an external app URL onto the local per-task server URL."""
    if not start_page:
        return server_url

    parsed = urlparse(start_page)
    if not parsed.scheme and not parsed.netloc:
        if start_page.startswith("#"):
            return f"{server_url.rstrip('/')}/{start_page}"
        path = start_page if start_page.startswith("/") else f"/{start_page}"
        return f"{server_url.rstrip('/')}{path}"

    local = urlparse(server_url)
    return urlunparse(
        (
            local.scheme,
            local.netloc,
            parsed.path or "/",
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def _load_chat_model(agent_config: str):
    agent_args = import_object(agent_config)
    chat_model_args = getattr(agent_args, "chat_model_args", None)
    if chat_model_args is None:
        raise ValueError(
            f"Agent config {agent_config!r} does not expose chat_model_args; cannot drive agent-browser backend."
        )
    return chat_model_args.make_model()


def _extract_json_object(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ParseError("Return a single JSON object and nothing else.")
    return raw[start : end + 1]


def _parse_model_action(raw: str) -> AgentBrowserAction:
    payload = json.loads(_extract_json_object(raw))
    action = payload.get("action")
    if action not in _ALLOWED_ACTIONS:
        raise ParseError(f"Unsupported action {action!r}. Allowed: {sorted(_ALLOWED_ACTIONS)}")
    reason = str(payload.get("reason", "") or "")
    parsed = AgentBrowserAction(action=action, reason=reason)
    if action in {"click", "fill"}:
        target = payload.get("target")
        if not isinstance(target, str) or not target.startswith("@"):
            raise ParseError("click/fill actions require a ref target like @e12.")
        parsed.target = target
    if action == "fill":
        text = payload.get("text")
        if not isinstance(text, str):
            raise ParseError("fill actions require a string text value.")
        parsed.text = text
    if action == "press":
        key = payload.get("key")
        if not isinstance(key, str) or not key:
            raise ParseError("press actions require a non-empty key.")
        parsed.key = key
    if action == "done":
        parsed.result = str(payload.get("result", "") or "")
    return parsed


def _plan_next_action(
    *,
    chat_model,
    instruction: str,
    snapshot_payload: dict[str, Any],
    history: list[dict[str, Any]],
    step_index: int,
    max_steps: int,
) -> AgentBrowserAction:
    snapshot_text = _snapshot_text(snapshot_payload)
    recent_history = json.dumps(history[-8:], indent=2, ensure_ascii=False)
    prompt = HumanMessage(
        "\n".join(
            [
                f"Task: {instruction}",
                f"Step: {step_index}/{max_steps}",
                "Recent history:",
                recent_history,
                "Current snapshot:",
                snapshot_text,
            ]
        )
    )
    discussion = Discussion([SystemMessage(_ACTION_PROMPT), prompt])
    payload = retry(chat_model, discussion, n_retry=2, parser=_parse_model_action, log=False)
    if isinstance(payload, AgentBrowserAction):
        return payload
    raise ParseError("Model parser did not return an AgentBrowserAction.")


def _snapshot_text(snapshot_payload: dict[str, Any]) -> str:
    if not isinstance(snapshot_payload, dict):
        return json.dumps(snapshot_payload, ensure_ascii=False)
    data = snapshot_payload.get("data")
    if isinstance(data, dict):
        snapshot = data.get("snapshot")
        if isinstance(snapshot, str) and snapshot.strip():
            return snapshot
    return json.dumps(snapshot_payload, indent=2, ensure_ascii=False)


def _execute_action(
    *,
    executable: str,
    action: AgentBrowserAction,
    cwd: Path,
    timeout: int,
    transcript: list[dict[str, Any]],
    cli_prefix: list[str],
) -> None:
    if action.action == "click":
        _run_cli(
            executable,
            ["click", action.target or ""],
            cwd=cwd,
            timeout=timeout,
            transcript=transcript,
            cli_prefix=cli_prefix,
        )
        _run_cli(
            executable,
            ["wait", "--load", _WAIT_LOAD_STATE],
            cwd=cwd,
            timeout=timeout,
            transcript=transcript,
            cli_prefix=cli_prefix,
        )
        return
    if action.action == "fill":
        _run_cli(
            executable,
            ["fill", action.target or "", action.text or ""],
            cwd=cwd,
            timeout=timeout,
            transcript=transcript,
            cli_prefix=cli_prefix,
        )
        return
    if action.action == "press":
        _run_cli(
            executable,
            ["press", action.key or ""],
            cwd=cwd,
            timeout=timeout,
            transcript=transcript,
            cli_prefix=cli_prefix,
        )
        _run_cli(
            executable,
            ["wait", "--load", _WAIT_LOAD_STATE],
            cwd=cwd,
            timeout=timeout,
            transcript=transcript,
            cli_prefix=cli_prefix,
        )
        return
    if action.action == "wait":
        _run_cli(
            executable,
            ["wait", "--load", _WAIT_LOAD_STATE],
            cwd=cwd,
            timeout=timeout,
            transcript=transcript,
            cli_prefix=cli_prefix,
        )
        return
    raise ValueError(f"Unsupported executable action: {action.action}")


def _run_cli_json(
    executable: str,
    args: list[str],
    *,
    cwd: Path,
    timeout: int,
    transcript: list[dict[str, Any]],
    cli_prefix: list[str],
) -> dict[str, Any]:
    completed = _run_cli(
        executable,
        args,
        cwd=cwd,
        timeout=timeout,
        transcript=transcript,
        cli_prefix=cli_prefix,
    )
    try:
        return json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"agent-browser returned invalid JSON for {' '.join(args)}: {exc}"
        ) from exc


def _run_cli(
    executable: str,
    args: list[str],
    *,
    cwd: Path,
    timeout: int,
    transcript: list[dict[str, Any]],
    cli_prefix: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = [executable, *(cli_prefix or []), *args]
    logger.info("Running %s", shlex.join(cmd))
    started = time.monotonic()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        transcript.append(
            {
                "type": "command",
                "command": cmd,
                "elapsed": round(time.monotonic() - started, 3),
                "stdout": exc.stdout,
                "stderr": exc.stderr,
                "returncode": exc.returncode,
            }
        )
        raise

    transcript.append(
        {
            "type": "command",
            "command": cmd,
            "elapsed": round(time.monotonic() - started, 3),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
        }
    )
    return completed
