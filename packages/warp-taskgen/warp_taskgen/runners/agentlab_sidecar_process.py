"""AgentLab sidecar process launch, log streaming, status, and result parsing."""

from __future__ import annotations

import ctypes
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from warp_taskgen.runners.agentlab_phase4_artifacts import _AGENTLAB_TIMELINE_ARTIFACT
from warp_taskgen.runners.agentlab_sidecar_redaction import (
    _redact_sidecar_payload,
    _redact_sidecar_text,
)

_SIDECAR_CMD_ENV = "WARP_TASKGEN_AGENTLAB_RUNNER_CMD"
_LEGACY_SIDECAR_CMD_ENV = "WORLDSIM_AGENTLAB_RUNNER_CMD"
_SIDECAR_TERMINATE_GRACE_S = 5.0


@dataclass
class _SidecarProcessResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    elapsed: float


def _run_sidecar_process_streaming(
    cmd: list[str],
    *,
    request: dict[str, Any],
    task_dir: Path,
    stdout_log_path: Path,
    stderr_log_path: Path,
    status_path: Path,
    subcommand: str,
    timeout: int | None,
) -> _SidecarProcessResult:
    """Run the isolated AgentLab sidecar while teeing redacted diagnostics."""

    started = time.monotonic()
    task_dir.mkdir(parents=True, exist_ok=True)
    stdout_log_path.write_text("", encoding="utf-8")
    stderr_log_path.write_text("", encoding="utf-8")
    _write_sidecar_status(
        status_path,
        request=request,
        subcommand=subcommand,
        status="sidecar_starting",
        timeout=timeout,
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
    )
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
        preexec_fn=_sidecar_preexec,
    )
    _write_sidecar_status(
        status_path,
        request=request,
        subcommand=subcommand,
        status="sidecar_running",
        timeout=timeout,
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
        pid=proc.pid,
        started_monotonic=started,
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def _reader(stream, chunks: list[str], log_path: Path) -> None:
        try:
            for chunk in iter(stream.readline, ""):
                if not chunk:
                    break
                chunks.append(chunk)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(_redact_sidecar_text(chunk, request))
                    handle.flush()
        finally:
            try:
                stream.close()
            except Exception:
                pass

    stdout_thread = threading.Thread(
        target=_reader,
        args=(proc.stdout, stdout_chunks, stdout_log_path),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_reader,
        args=(proc.stderr, stderr_chunks, stderr_log_path),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    stop_status = threading.Event()

    def _status_heartbeat() -> None:
        while not stop_status.wait(5.0):
            _write_sidecar_status(
                status_path,
                request=request,
                subcommand=subcommand,
                status="sidecar_running",
                timeout=timeout,
                stdout_log_path=stdout_log_path,
                stderr_log_path=stderr_log_path,
                pid=proc.pid,
                started_monotonic=started,
            )

    heartbeat_thread = threading.Thread(target=_status_heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        returncode = proc.wait(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        _append_redacted_sidecar_log(
            stderr_log_path,
            f"\nworldsim: AgentLab sidecar exceeded task timeout {timeout}s; terminating\n",
            request,
        )
        _terminate_sidecar_process(proc)
        returncode = proc.returncode if proc.returncode is not None else -signal.SIGKILL
    stop_status.set()
    heartbeat_thread.join(timeout=1.0)
    stdout_thread.join(timeout=1.0)
    stderr_thread.join(timeout=1.0)
    elapsed = time.monotonic() - started
    stdout = "".join(stdout_chunks)
    stderr = "".join(stderr_chunks)
    status = "sidecar_timeout" if timed_out else "sidecar_completed"
    _write_sidecar_status(
        status_path,
        request=request,
        subcommand=subcommand,
        status=status,
        timeout=timeout,
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
        pid=proc.pid,
        returncode=returncode,
        elapsed=elapsed,
        timed_out=timed_out,
        stdout_bytes=len(stdout.encode("utf-8")),
        stderr_bytes=len(stderr.encode("utf-8")),
    )
    if timed_out:
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout, output=stdout, stderr=stderr)
    return _SidecarProcessResult(
        returncode=int(returncode or 0),
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
        elapsed=elapsed,
    )


def _terminate_sidecar_process(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=_SIDECAR_TERMINATE_GRACE_S)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception:
        proc.kill()
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass


def _sidecar_preexec() -> None:
    """Make Linux tear down the sidecar if the worker process dies abruptly."""

    if os.name != "posix" or not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL(None)
        pr_set_pdeathsig = 1
        libc.prctl(pr_set_pdeathsig, signal.SIGTERM)
    except Exception:
        return
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGTERM)


def _append_redacted_sidecar_log(path: Path, text: str, request: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_redact_sidecar_text(text, request))
        handle.flush()


def _write_sidecar_status(
    path: Path,
    *,
    request: dict[str, Any],
    subcommand: str,
    status: str,
    timeout: int | None,
    stdout_log_path: Path,
    stderr_log_path: Path,
    pid: int | None = None,
    returncode: int | None = None,
    elapsed: float | None = None,
    timed_out: bool | None = None,
    stdout_bytes: int | None = None,
    stderr_bytes: int | None = None,
    started_monotonic: float | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "runner": "agentlab",
        "mode": "phase4" if subcommand == "phase4-run" else "comparison",
        "subcommand": subcommand,
        "status": status,
        "task_id": request.get("task_id"),
        "timeout_s": timeout,
        "stdout_log": str(stdout_log_path),
        "stderr_log": str(stderr_log_path),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if pid is not None:
        payload["pid"] = pid
    if returncode is not None:
        payload["returncode"] = returncode
    if elapsed is not None:
        payload["elapsed_s"] = round(max(0.0, elapsed), 3)
    elif started_monotonic is not None:
        payload["elapsed_s"] = round(max(0.0, time.monotonic() - started_monotonic), 3)
    if timed_out is not None:
        payload["timed_out"] = timed_out
    if stdout_bytes is not None:
        payload["stdout_bytes"] = stdout_bytes
    if stderr_bytes is not None:
        payload["stderr_bytes"] = stderr_bytes
    payload.update(_live_agentlab_status(path.parent))
    path.write_text(
        json.dumps(_redact_sidecar_payload(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _live_agentlab_status(task_dir: Path) -> dict[str, Any]:
    """Return compact sidecar-owned live fields for status/progress polling."""

    runtime = _load_json_dict(task_dir / "browser_runtime.json")
    timeline_tail = _tail_jsonl(task_dir / _AGENTLAB_TIMELINE_ARTIFACT)
    fields: dict[str, Any] = {}
    if runtime:
        for key in (
            "runtime_artifact_status",
            "browser_instance_scope",
            "current_phase",
            "current_step",
            "last_url",
            "last_title",
            "last_action",
            "last_screenshot",
            "last_network_event_count",
            "last_updated_at",
            "agent_browser_connect_count",
            "auxiliary_browser_connect_count",
            "recycle_status",
        ):
            if key in runtime:
                fields[key] = runtime[key]
    if timeline_tail:
        fields["last_timeline_event"] = timeline_tail.get("event")
        fields["last_timeline_step"] = timeline_tail.get("step")
        fields["last_timeline_timestamp"] = timeline_tail.get("timestamp")
        fields["timeline_path"] = str(task_dir / _AGENTLAB_TIMELINE_ARTIFACT)
    return fields


def _load_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _tail_jsonl(path: Path) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        return payload if isinstance(payload, dict) else {}
    return {}


def _sidecar_json_payload(stdout: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        for line in reversed(stdout.splitlines()):
            candidate = line.strip()
            if not candidate.startswith("{") or not candidate.endswith("}"):
                continue
            payload = json.loads(candidate)
            break
        else:
            payload = None
            for marker in ('{"agentlab_reward"', '{"status"', '{"artifacts"'):
                index = stdout.find(marker)
                if index < 0:
                    continue
                try:
                    candidate, end = decoder.raw_decode(stdout[index:])
                except json.JSONDecodeError:
                    continue
                if stdout[index + end :].strip():
                    continue
                payload = candidate
                break
            if payload is None:
                raise
    if not isinstance(payload, dict):
        raise RuntimeError("AgentLab sidecar returned a non-object JSON payload")
    return payload


def _default_sidecar_command(subcommand: str = "run") -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    project_dir = repo_root / "packages" / "worldsim-agentlab-runner"
    venv_bin = project_dir / ".venv" / "bin"
    for script_name in ("warp-taskgen-agentlab-runner", "worldsim-agentlab-runner"):
        venv_entrypoint = venv_bin / script_name
        if venv_entrypoint.is_file() and os.access(venv_entrypoint, os.X_OK):
            return [str(venv_entrypoint), subcommand]
    return [
        "uv",
        "run",
        "--project",
        str(project_dir),
        "warp-taskgen-agentlab-runner",
        subcommand,
    ]


def _sidecar_command(subcommand: str = "run") -> list[str]:
    raw = os.environ.get(_SIDECAR_CMD_ENV) or os.environ.get(_LEGACY_SIDECAR_CMD_ENV)
    if raw and raw.strip():
        parts = shlex.split(raw)
        if parts and parts[-1] in {"run", "phase4-run"}:
            return [*parts[:-1], subcommand]
        return [*parts, subcommand]
    return _default_sidecar_command(subcommand)


def _parse_sidecar_result(task_id: str, task_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    summary = result.get("summary_info")
    if not isinstance(summary, dict):
        summary = {}
    n_steps = int(_first_present(result.get("steps"), summary.get("n_steps"), 0) or 0)
    reward = float(_first_present(result.get("reward"), summary.get("cum_reward"), 0.0) or 0.0)
    err_msg = result.get("error") or summary.get("err_msg")
    raw_passed = result.get("passed")
    passed = bool(raw_passed) if raw_passed is not None else reward > 0 and err_msg is None
    status = str(
        result.get("status") or ("error" if err_msg else ("success" if passed else "failure"))
    )
    errors = [str(err_msg)] if err_msg else list(result.get("errors") or [])

    if err_msg:
        message = f"AgentLab error: {err_msg}"
    elif result.get("message"):
        message = str(result["message"])
    elif passed:
        message = f"passed (reward={reward:.2f}, steps={n_steps})"
    else:
        message = f"failed (reward={reward:.2f}, steps={n_steps})"
    is_done_default = bool(
        result.get("terminated")
        or result.get("truncated")
        or summary.get("terminated")
        or summary.get("truncated")
    )

    return {
        "task_id": task_id,
        "passed": passed,
        "message": message,
        "elapsed": float(result.get("elapsed", 0.0)),
        "steps": n_steps,
        "is_done": bool(result.get("is_done", is_done_default)),
        "status": status,
        "errors": errors,
        "outcome": "error" if status == "error" else None,
        "trajectory_dir": str(task_dir),
        "reward": reward,
        "agentlab_reward": reward,
        "agentlab_summary": summary,
        "agentlab_versions": result.get("versions", {}),
        "agentlab_native_artifacts": result.get("artifacts", {}),
        "agentlab_model": result.get("model", {}),
    }


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None
