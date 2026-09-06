"""Phase 4 AgentLab artifact clearing, recovery, manifests, and timeline reads."""

from __future__ import annotations

import html
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from warp_taskgen.har_converter import minimal_har_placeholder_entry
from warp_taskgen.resume_metadata import RESULT_FINGERPRINT_KEY
from warp_taskgen.runners.agentlab_sidecar_redaction import _redact_sidecar_text

logger = logging.getLogger(__name__)

_AGENTLAB_TIMELINE_ARTIFACT = "agentlab_step_timeline.jsonl"
_AGENTLAB_EVENTS_ARTIFACT = "agentlab_events.jsonl"


def _clear_phase4_sidecar_artifacts(task_dir: Path) -> None:
    """Remove stale sidecar-owned artifacts before a fresh AgentLab attempt."""

    files = (
        "summary_info.json",
        "history.json",
        "final_response.json",
        "needham_trace.json",
        "needham_trace.xml",
        "network_trace.json",
        "network.har",
        "network_evidence.json",
        "navigation_trace.json",
        "browser_runtime.json",
        "agentlab_sidecar_result.json",
        "agentlab_sidecar_status.json",
        "agentlab_sidecar_stdout.log",
        "agentlab_sidecar_stderr.log",
        _AGENTLAB_TIMELINE_ARTIFACT,
        _AGENTLAB_EVENTS_ARTIFACT,
        "phase4_sidecar_request.json",
        "agentlab_native_exp_args.pkl",
    )
    dirs = ("pvpo", "screenshots", "reward_private")
    patterns = ("screenshot_step_*", "step_*.pkl.gz")
    for name in files:
        path = task_dir / name
        try:
            if path.exists() or path.is_symlink():
                path.unlink()
        except OSError:
            logger.warning("could not remove stale AgentLab sidecar artifact %s", path)
    for name in dirs:
        path = task_dir / name
        try:
            if path.is_symlink():
                path.unlink()
            elif path.exists():
                shutil.rmtree(path)
        except OSError:
            logger.warning("could not remove stale AgentLab sidecar artifact dir %s", path)
    for pattern in patterns:
        for path in task_dir.glob(pattern):
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except OSError:
                logger.warning("could not remove stale AgentLab sidecar artifact %s", path)


def _phase4_sidecar_error_result(
    request: dict[str, Any],
    task_dir: Path,
    message: str,
) -> dict[str, Any]:
    task_dir.mkdir(parents=True, exist_ok=True)
    existing_runtime = _load_phase4_runtime_artifact(task_dir)
    fatal_capture = _load_phase4_fatal_capture(task_dir)
    _write_minimal_timeout_artifacts(task_dir, request, message, status="error")
    runtime = {
        "runner": "agentlab",
        "mode": "phase4",
        "browser_instance_scope": "agent_run",
        "sidecar_error": True,
        "cdp_url": None,
    }
    if existing_runtime:
        runtime.update(existing_runtime)
        runtime["sidecar_error"] = True
    runtime["runtime_artifact_status"] = "sidecar_error"
    if fatal_capture:
        runtime["pvpo_capture_fatal"] = True
        runtime["pvpo_capture_fatal_details"] = fatal_capture
    (task_dir / "browser_runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "mode": "phase4",
        "task_id": request.get("task_id"),
        "status": "error",
        "passed": None,
        "reward": 0.0,
        "agentlab_reward": 0.0,
        "steps": 0,
        "is_done": False,
        "final_result": None,
        "elapsed": 0.0,
        "errors": [message],
        "error": message,
        "network_trace": [],
        "summary_info": {
            "n_steps": 0,
            "cum_reward": 0.0,
            "cum_raw_reward": 0.0,
            "err_msg": message,
            "terminated": False,
            "truncated": True,
        },
        "artifacts": _phase4_artifact_manifest(task_dir),
        "evidence_status": "sidecar_error_partial_artifacts"
        if fatal_capture
        else "sidecar_error_placeholder",
        "browser_runtime": runtime,
    }
    fingerprint = request.get(RESULT_FINGERPRINT_KEY)
    if isinstance(fingerprint, str) and fingerprint.strip():
        payload[RESULT_FINGERPRINT_KEY] = fingerprint
    return payload


def _load_phase4_runtime_artifact(task_dir: Path) -> dict[str, Any]:
    try:
        payload = json.loads((task_dir / "browser_runtime.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_phase4_fatal_capture(task_dir: Path) -> dict[str, Any]:
    try:
        payload = json.loads((task_dir / "pvpo" / "fatal_capture.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _phase4_timeout_result(
    request: dict[str, Any],
    task_dir: Path,
    timeout: int | None,
    timeout_payload: dict[str, str],
) -> dict[str, Any]:
    message = f"AgentLab sidecar exceeded task timeout {timeout}s"
    task_dir.mkdir(parents=True, exist_ok=True)
    _write_minimal_timeout_artifacts(task_dir, request, message, status="timeout")
    recovered = _recover_phase4_timeout_artifacts(task_dir)
    runtime = {
        "runner": "agentlab",
        "mode": "phase4",
        "browser_instance_scope": "agent_run",
        "timeout": timeout,
        "timeout_expired": True,
        "cdp_url": None,
    }
    (task_dir / "browser_runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "mode": "phase4",
        "task_id": request.get("task_id"),
        "status": "timeout",
        "passed": None,
        "reward": 0.0,
        "agentlab_reward": 0.0,
        "steps": recovered["steps"],
        "is_done": False,
        "final_result": recovered["final_result"],
        "elapsed": float(timeout or 0),
        "errors": [message],
        "error": message,
        "network_trace": recovered["network_trace"],
        "summary_info": {
            "n_steps": recovered["steps"],
            "cum_reward": 0.0,
            "cum_raw_reward": 0.0,
            "err_msg": message,
            "terminated": False,
            "truncated": True,
        },
        "artifacts": _phase4_artifact_manifest(task_dir),
        "evidence_status": recovered["evidence_status"],
        "browser_runtime": runtime,
        "timeout_stdout": _redact_sidecar_text(
            (timeout_payload.get("stdout") or "")[-1000:],
            request,
        ),
        "timeout_stderr": _redact_sidecar_text(
            (timeout_payload.get("stderr") or "")[-1000:],
            request,
        ),
    }
    fingerprint = request.get(RESULT_FINGERPRINT_KEY)
    if isinstance(fingerprint, str) and fingerprint.strip():
        payload[RESULT_FINGERPRINT_KEY] = fingerprint
    return payload


def _recover_phase4_timeout_artifacts(task_dir: Path) -> dict[str, Any]:
    """Read partial sidecar artifacts preserved after a parent timeout."""

    steps = 0
    final_result: str | None = None
    evidence_status = "timeout_placeholder"
    history_path = task_dir / "history.json"
    try:
        history_payload = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        history_payload = None
    if isinstance(history_payload, dict):
        history = history_payload.get("history")
        if isinstance(history, list):
            steps = max(0, len([item for item in history if isinstance(item, dict)]) - 1)
            if history:
                evidence_status = "timeout_partial_artifacts"
        for item in reversed(history if isinstance(history, list) else []):
            if not isinstance(item, dict):
                continue
            for result_item in item.get("result") or []:
                if not isinstance(result_item, dict):
                    continue
                text = result_item.get("extracted_content")
                if isinstance(text, str) and text.strip():
                    final_result = text.strip()
                    break
            if final_result is not None:
                break
    try:
        network_trace = json.loads((task_dir / "network_trace.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        network_trace = []
    if not isinstance(network_trace, list):
        network_trace = []
    if network_trace:
        evidence_status = "timeout_partial_artifacts"
    timeline = _load_agentlab_timeline(task_dir)
    if timeline:
        evidence_status = "timeout_partial_artifacts"
        steps = max(steps, _steps_from_agentlab_timeline(timeline))
    return {
        "steps": steps,
        "final_result": final_result,
        "network_trace": [event for event in network_trace if isinstance(event, dict)],
        "evidence_status": evidence_status,
    }


def _write_minimal_timeout_artifacts(
    task_dir: Path,
    request: dict[str, Any],
    message: str,
    *,
    status: str,
) -> None:
    _write_text_if_absent(
        task_dir / "history.json",
        json.dumps(
            {
                "history": [],
                "runner": "agentlab",
                "trajectory_format": "worldsim-agentlab-history-v1",
                "partial": True,
                "errors": [message],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _write_text_if_absent(
        task_dir / "final_response.json",
        json.dumps(
            {"status": status, "final_result": None, "errors": [message], "steps": 0}, indent=2
        )
        + "\n",
    )
    needham_messages = [
        {
            "role": "user",
            "text": str(request.get("task") or ""),
            "provenance": {"source": "agentlab_timeout_request"},
        },
        {
            "role": "assistant",
            "text": message,
            "tool_calls": None,
            "provenance": {"source": "agentlab_timeout"},
        },
    ]
    needham_xml = (
        '<transcript format="needham-xml-v1">\n'
        f'<message role="user">{html.escape(needham_messages[0]["text"])}</message>\n'
        f'<message role="assistant">{html.escape(message)}</message>\n'
        "</transcript>\n"
    )
    _write_text_if_absent(
        task_dir / "needham_trace.json",
        json.dumps(
            {
                "format": "needham-agentlab-timeout-v1",
                "transcript_format": "needham-xml-v1",
                "source": "agentlab_timeout",
                "messages": needham_messages,
                "xml": needham_xml,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _write_text_if_absent(task_dir / "needham_trace.xml", needham_xml)
    _write_text_if_absent(task_dir / "network_trace.json", "[]\n")
    _write_text_if_absent(
        task_dir / "network.har",
        json.dumps(
            {
                "log": {
                    "version": "1.2",
                    "creator": {"name": "worldsim-agentlab", "version": "timeout"},
                    "_worldsim_evidence_status": "timeout_placeholder",
                    "entries": [minimal_har_placeholder_entry()],
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _write_text_if_absent(
        task_dir / "network_evidence.json",
        json.dumps(
            {
                "public_trace": "timeout_placeholder",
                "private_reward_trace": "unavailable",
                "private_reward_trace_dir": None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    pvpo_dir = task_dir / "pvpo"
    pvpo_dir.mkdir(parents=True, exist_ok=True)
    _write_text_if_absent(
        pvpo_dir / "capture_summary.json",
        json.dumps(
            {
                "status": "timeout_no_artifacts",
                "payload_present": bool(
                    request.get("payload_text") or request.get("payload_witnesses")
                ),
                "steps_seen": 0,
                "steps_captured": 0,
                "issue_steps": 1,
                "first_issue_class": "sidecar_timeout",
                "first_issue_step": None,
                "first_issue_message": message,
                "last_issue_class": "sidecar_timeout",
                "last_issue_step": None,
                "last_issue_message": message,
                "issue_counts": {"sidecar_timeout": 1},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _write_text_if_absent(
        task_dir / "summary_info.json",
        json.dumps(
            {
                "n_steps": 0,
                "cum_reward": 0.0,
                "cum_raw_reward": 0.0,
                "err_msg": message,
                "terminated": False,
                "truncated": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _write_text_if_absent(task_dir / _AGENTLAB_TIMELINE_ARTIFACT, "")
    _write_text_if_absent(task_dir / _AGENTLAB_EVENTS_ARTIFACT, "")


def _write_text_if_absent(path: Path, text: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.write_text(text, encoding="utf-8")


def _phase4_artifact_manifest(output_dir: Path) -> dict[str, Any]:
    files = {
        "summary_info": output_dir / "summary_info.json",
        "history": output_dir / "history.json",
        "final_response": output_dir / "final_response.json",
        "network_trace": output_dir / "network_trace.json",
        "network_har": output_dir / "network.har",
        "network_evidence": output_dir / "network_evidence.json",
        "navigation_trace": output_dir / "navigation_trace.json",
        "browser_runtime": output_dir / "browser_runtime.json",
        "agentlab_request": output_dir / "agentlab_phase4_request.json",
        "agentlab_result": output_dir / "agentlab_sidecar_result.json",
        "agentlab_status": output_dir / "agentlab_sidecar_status.json",
        "agentlab_stdout": output_dir / "agentlab_sidecar_stdout.log",
        "agentlab_stderr": output_dir / "agentlab_sidecar_stderr.log",
        "agentlab_timeline": output_dir / _AGENTLAB_TIMELINE_ARTIFACT,
        "agentlab_events": output_dir / _AGENTLAB_EVENTS_ARTIFACT,
        "needham_trace": output_dir / "needham_trace.json",
        "needham_xml": output_dir / "needham_trace.xml",
        "pvpo_summary": output_dir / "pvpo" / "capture_summary.json",
        "pvpo_fatal_capture": output_dir / "pvpo" / "fatal_capture.json",
    }
    screenshots = sorted(
        {str(path) for path in output_dir.glob("screenshot_step_*")}
        | {str(path) for path in (output_dir / "screenshots").glob("step_*.png")}
    )
    pvpo_steps = sorted(str(path) for path in (output_dir / "pvpo").glob("step_*.json"))
    steps = sorted(str(path) for path in output_dir.glob("step_*.pkl.gz"))
    videos = sorted(str(path) for path in output_dir.glob("**/*.webm"))
    return {key: str(path) for key, path in files.items() if path.exists()} | {
        "screenshots": screenshots,
        "pvpo_steps": pvpo_steps,
        "steps": steps,
        "videos": videos,
    }


def _load_agentlab_timeline(task_dir: Path) -> list[dict[str, Any]]:
    path = task_dir / _AGENTLAB_TIMELINE_ARTIFACT
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _steps_from_agentlab_timeline(events: list[dict[str, Any]]) -> int:
    steps = 0
    for event in events:
        try:
            step = int(event.get("step"))
        except (TypeError, ValueError):
            continue
        if event.get("event") in {"browser_step", "agent_action", "reset"}:
            steps = max(steps, step)
    return steps
