"""AgentLab / BrowserGym sidecar adapter.

This module intentionally does not import AgentLab. Current AgentLab releases
depend on ``openai<2`` while Browser Use 0.12.6 depends on ``openai==2.16.0``.
The comparison runner therefore crosses a JSON subprocess boundary into
``packages/worldsim-agentlab-runner``.

The sidecar has two entrypoints: benchmark-native comparison runs and the
WARP Taskgen Phase 4 runtime. The Phase 4 path preserves WARP Taskgen-owned
admission, seeding, rewards, PVPO gates, judges, variants, and summaries while
delegating only the browser-agent episode to AgentLab/BrowserGym.
"""

from __future__ import annotations

import asyncio
import ctypes  # noqa: F401 - patched through this module by tests/test_agentlab_runner.py
import json
import logging
import os  # noqa: F401 - patched through this module by tests/test_agentlab_runner.py
import shutil
import subprocess
import sys  # noqa: F401 - patched through this module by tests/test_agentlab_runner.py
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from warp_taskgen.agent_config import execution_instance_dict, task_reset_endpoints
from warp_taskgen.agent_runtime import AgentResult
from warp_taskgen.benchmark_capabilities import get_benchmark_capabilities
from warp_taskgen.comparison_ingestion import (
    COMPARISON_RESULT_FILENAME,
    ingest_comparison_payload,
    write_comparison_result,
)
from warp_taskgen.resume_metadata import RESULT_FINGERPRINT_KEY
from warp_taskgen.runners.agentlab_phase4_artifacts import (
    _clear_phase4_sidecar_artifacts,
    _phase4_sidecar_error_result,
    _phase4_timeout_result,
)
from warp_taskgen.runners.agentlab_sidecar_process import (
    _live_agentlab_status,  # noqa: F401 - exercised through this module by tests
    _parse_sidecar_result,
    _run_sidecar_process_streaming,
    _sidecar_command,
    _sidecar_json_payload,
    _sidecar_preexec,  # noqa: F401 - exercised through this module by tests
    _SidecarProcessResult,  # noqa: F401 - constructed through this module by tests
    _write_sidecar_status,
)
from warp_taskgen.runners.agentlab_sidecar_redaction import (
    _redact_sidecar_payload,
    _redact_sidecar_text,
    _secret_strings_from_payload,
)
from warp_taskgen.runners.agentlab_sidecar_request import (
    _browsergym_env_overrides,  # noqa: F401 - imported through this module by tests
    _build_phase4_sidecar_request,
    _build_sidecar_request,
    _task_identity,
    _validate_task_benchmark,
)
from warp_taskgen.seeding import apply_data_seed_async
from warp_taskgen.trajectory import save_result_payload

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt52"
_DEFAULT_MAX_STEPS = 30
_RESET_TIMEOUT = 300
_RESET_MAX_RETRIES = 2
_RESET_RETRY_DELAY_S = 10
_SUPPORTED_ATTACK_MODES = frozenset({"comparison", "seeded_comparison"})


@dataclass
class AgentLabAgentWrapper:
    """Worker-pool lifecycle wrapper for AgentLab sidecar settings."""

    model: str = _DEFAULT_MODEL
    provider: str | None = None
    service_tier: str | None = None
    max_steps: int = _DEFAULT_MAX_STEPS
    timeout: int | None = None
    llm_timeout: int | None = None
    step_timeout: int | None = None

    async def setup(self, server_url: str) -> None:
        return None

    async def teardown(self) -> None:
        return None

    async def run(
        self,
        task: str,
        server_url: str,
        task_dir: Path,
        **kwargs: Any,
    ) -> AgentResult:
        request = _build_phase4_sidecar_request(
            task,
            server_url,
            task_dir,
            self,
            kwargs,
        )
        try:
            payload = await asyncio.to_thread(
                _run_sidecar_request,
                request,
                task_dir,
                "phase4-run",
                self.timeout,
            )
        except Exception as exc:
            return AgentResult(
                elapsed=0.0,
                steps=0,
                is_done=False,
                final_result=None,
                status="error",
                errors=[f"AgentLab sidecar failed: {type(exc).__name__}: {exc}"],
                network_trace=[],
            )
        return _agent_result_from_phase4_sidecar(payload)


def _post_reset(endpoint: str) -> None:
    last_exc: Exception | None = None
    for attempt in range(1, _RESET_MAX_RETRIES + 1):
        try:
            response = requests.post(endpoint, timeout=_RESET_TIMEOUT)
            response.raise_for_status()
            return
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last_exc = exc
            if attempt < _RESET_MAX_RETRIES:
                logger.warning(
                    "Reset %s attempt %d failed (%s), retrying in %ds",
                    endpoint,
                    attempt,
                    exc,
                    _RESET_RETRY_DELAY_S,
                )
                time.sleep(_RESET_RETRY_DELAY_S)
    raise RuntimeError(
        f"Reset endpoint {endpoint} failed after {_RESET_MAX_RETRIES} attempts"
    ) from last_exc


async def _reset_task_environment(task: dict[str, Any]) -> None:
    endpoints = task_reset_endpoints(task)
    if endpoints:
        await asyncio.gather(*(asyncio.to_thread(_post_reset, endpoint) for endpoint in endpoints))
        await asyncio.sleep(2)


def _agent_result_from_phase4_sidecar(payload: dict[str, Any]) -> AgentResult:
    network_trace = payload.get("network_trace")
    if not isinstance(network_trace, list):
        network_trace = []
    errors = payload.get("errors")
    if not isinstance(errors, list):
        errors = [str(payload["error"])] if payload.get("error") else []
    final_result = payload.get("final_result")
    return AgentResult(
        elapsed=float(payload.get("elapsed") or 0.0),
        steps=int(payload.get("steps") or 0),
        is_done=bool(payload.get("is_done", False)),
        final_result=final_result if isinstance(final_result, str) else None,
        status=str(payload.get("status") or "error"),
        errors=[str(error) for error in errors],
        network_trace=[event for event in network_trace if isinstance(event, dict)],
    )


def _run_sidecar_request(
    request: dict[str, Any],
    task_dir: Path,
    subcommand: str = "run",
    timeout: int | None = None,
) -> dict[str, Any]:
    task_dir.mkdir(parents=True, exist_ok=True)
    request_path = task_dir / (
        "agentlab_phase4_request.json" if subcommand == "phase4-run" else "agentlab_request.json"
    )
    response_path = task_dir / "agentlab_sidecar_result.json"
    stdout_log_path = task_dir / "agentlab_sidecar_stdout.log"
    stderr_log_path = task_dir / "agentlab_sidecar_stderr.log"
    status_path = task_dir / "agentlab_sidecar_status.json"
    request_path.write_text(
        json.dumps(_redact_sidecar_payload(request), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if subcommand == "phase4-run":
        _clear_phase4_sidecar_artifacts(task_dir)

    try:
        with tempfile.TemporaryDirectory(prefix="worldsim-agentlab-request-") as runtime_dir:
            runtime_request_path = Path(runtime_dir) / request_path.name
            runtime_request_path.write_text(
                json.dumps(request, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            runtime_request_path.chmod(0o600)
            proc = _run_sidecar_process_streaming(
                [*_sidecar_command(subcommand), str(runtime_request_path)],
                request=request,
                task_dir=task_dir,
                stdout_log_path=stdout_log_path,
                stderr_log_path=stderr_log_path,
                status_path=status_path,
                subcommand=subcommand,
                timeout=timeout,
            )
    except subprocess.TimeoutExpired as exc:
        timeout_payload = {
            "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
            "stderr": exc.stderr if isinstance(exc.stderr, str) else "",
        }
        if subcommand == "phase4-run":
            payload = _phase4_timeout_result(request, task_dir, timeout, timeout_payload)
            response_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
            return payload
        raise
    finally:
        storage_runtime_dir = request.get("storage_state_runtime_dir")
        if isinstance(storage_runtime_dir, str) and storage_runtime_dir:
            shutil.rmtree(storage_runtime_dir, ignore_errors=True)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        detail = _redact_sidecar_text(detail, request)
        _write_sidecar_status(
            status_path,
            request=request,
            subcommand=subcommand,
            status="sidecar_error",
            timeout=timeout,
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            returncode=proc.returncode,
            elapsed=proc.elapsed,
            timed_out=False,
            stdout_bytes=len(proc.stdout.encode("utf-8")),
            stderr_bytes=len(proc.stderr.encode("utf-8")),
        )
        if subcommand == "phase4-run":
            payload = _phase4_sidecar_error_result(
                request,
                task_dir,
                f"AgentLab sidecar failed with exit {proc.returncode}: {detail}",
            )
            response_path.write_text(
                json.dumps(
                    _redact_sidecar_payload(
                        payload, secret_values=_secret_strings_from_payload(request)
                    ),
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            return payload
        raise RuntimeError(f"AgentLab sidecar failed with exit {proc.returncode}: {detail}")
    try:
        payload = _sidecar_json_payload(proc.stdout)
    except json.JSONDecodeError as exc:
        _write_sidecar_status(
            status_path,
            request=request,
            subcommand=subcommand,
            status="sidecar_invalid_json",
            timeout=timeout,
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            returncode=proc.returncode,
            elapsed=proc.elapsed,
            timed_out=False,
            stdout_bytes=len(proc.stdout.encode("utf-8")),
            stderr_bytes=len(proc.stderr.encode("utf-8")),
        )
        if subcommand == "phase4-run":
            detail = _redact_sidecar_text(proc.stdout[:500], request)
            payload = _phase4_sidecar_error_result(
                request,
                task_dir,
                f"AgentLab sidecar returned invalid JSON: {detail}",
            )
            response_path.write_text(
                json.dumps(
                    _redact_sidecar_payload(
                        payload, secret_values=_secret_strings_from_payload(request)
                    ),
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            return payload
        raise RuntimeError(f"AgentLab sidecar returned invalid JSON: {proc.stdout[:500]}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("AgentLab sidecar returned a non-object JSON payload")
    response_path.write_text(
        json.dumps(
            _redact_sidecar_payload(payload, secret_values=_secret_strings_from_payload(request)),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return payload


def _persist_result_sentinel(
    task: dict[str, Any],
    task_dir: Path,
    parsed_result: dict[str, Any],
) -> None:
    metadata = {
        key: task[key]
        for key in ("benchmark", "benchmark_name", "benchmark_adapter", "agentlab_task_name")
        if key in task
    }
    resume_fingerprint = task.get(RESULT_FINGERPRINT_KEY)
    if isinstance(resume_fingerprint, str) and resume_fingerprint.strip():
        metadata[RESULT_FINGERPRINT_KEY] = resume_fingerprint
    for key in (
        "reward",
        "agentlab_reward",
        "agentlab_summary",
        "agentlab_versions",
        "agentlab_native_artifacts",
        "agentlab_model",
    ):
        if parsed_result.get(key) is not None:
            metadata[key] = parsed_result[key]

    save_result_payload(
        task_dir,
        {
            "task_id": task.get("id", task.get("task_id", "unknown")),
            "passed": bool(parsed_result.get("passed", False)),
            "message": str(parsed_result.get("message", "")),
            "status": str(parsed_result.get("status", "error")),
            "elapsed": float(parsed_result.get("elapsed", 0.0)),
            "steps": int(parsed_result.get("steps", 0)),
            "is_done": bool(parsed_result.get("is_done", False)),
            "final_result": parsed_result.get("message"),
            "errors": list(parsed_result.get("errors", [])),
            "trajectory_dir": str(task_dir),
            "reward": float(parsed_result.get("reward", 0.0)),
            "agentlab_reward": float(parsed_result.get("agentlab_reward", 0.0)),
            **metadata,
            **({"outcome": "error"} if parsed_result.get("outcome") == "error" else {}),
        },
    )


def make_task_runner(
    *,
    attack_mode: str = "comparison",
    benchmark_prefix: str = "webarena_verified",
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> Callable[..., Any]:
    """Return an async comparison task runner for AgentLab/BrowserGym."""

    if attack_mode not in _SUPPORTED_ATTACK_MODES:
        raise ValueError(
            f"unsupported AgentLab attack_mode {attack_mode!r}; "
            f"supported={sorted(_SUPPORTED_ATTACK_MODES)}"
        )

    async def run_task(
        task: dict[str, Any],
        agent: Any,
        instance: Any,
        task_dir: Path,
    ) -> dict[str, Any]:
        benchmark_name = _validate_task_benchmark(task)
        capabilities = get_benchmark_capabilities(benchmark_name)
        resolved_task_id = _task_identity(
            task,
            reject_conflicts=capabilities.supports("comparison_ingestion"),
            strict=capabilities.supports("comparison_ingestion"),
        )
        if capabilities.supports("comparison_ingestion") and resolved_task_id is None:
            raise ValueError("AgentLab task is missing id/task_id metadata")
        task_id = resolved_task_id or "unknown"
        instance_dict = execution_instance_dict(instance, task)
        agent_wrapper = agent if isinstance(agent, AgentLabAgentWrapper) else AgentLabAgentWrapper()
        request = _build_sidecar_request(
            task,
            instance_dict,
            agent_wrapper,
            task_dir,
            benchmark_name=benchmark_name,
            benchmark_prefix=benchmark_prefix,
            max_steps=max_steps,
        )

        comparison_result_path = task_dir / COMPARISON_RESULT_FILENAME
        comparison_result_path.unlink(missing_ok=True)
        if capabilities.supports("comparison_ingestion"):
            # A reused task directory must never present a stale WARP sentinel
            # beside the current benchmark-native comparison envelope.
            (task_dir / "result.json").unlink(missing_ok=True)
        await _reset_task_environment(task)
        if attack_mode == "seeded_comparison":
            seed = task.get("adversarial_data_seed", task.get("data_seed", {}))
            if isinstance(seed, dict) and seed.get("mechanism") not in (None, "none"):
                await apply_data_seed_async(seed, instance_dict)

        sidecar_result = await asyncio.to_thread(_run_sidecar_request, request, task_dir)
        if capabilities.supports("comparison_ingestion"):
            # Native comparison records require string identity.  Normalize
            # legacy numeric task IDs at this orchestration boundary while
            # keeping the ingestion module strict for direct callers.
            comparison_task = dict(task)
            comparison_task.pop("id", None)
            comparison_task.pop("task_id", None)
            comparison_task["task_id"] = task_id
            comparison_record = ingest_comparison_payload(
                comparison_task,
                sidecar_result,
                artifact_dir=task_dir,
            )
            write_comparison_result(comparison_result_path, comparison_record)
            return comparison_record.to_dict()
        parsed = _parse_sidecar_result(task_id, task_dir, sidecar_result)
        _persist_result_sentinel(task, task_dir, parsed)
        return parsed

    return run_task


def make_agent_factory(
    model: str = _DEFAULT_MODEL,
    provider: str | None = None,
    service_tier: str | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
    task_timeout: int | None = None,
    llm_timeout: int | None = None,
    step_timeout: int | None = None,
    **_: Any,
) -> Callable[[], AgentLabAgentWrapper]:
    """Return a factory for AgentLab sidecar settings."""

    def factory() -> AgentLabAgentWrapper:
        return AgentLabAgentWrapper(
            model=model,
            provider=provider,
            service_tier=service_tier,
            max_steps=max_steps,
            timeout=task_timeout,
            llm_timeout=llm_timeout,
            step_timeout=step_timeout,
        )

    return factory
