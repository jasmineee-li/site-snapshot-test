"""AgentLab / BrowserGym sidecar adapter.

This module intentionally does not import AgentLab. Current AgentLab releases
depend on ``openai<2`` while Browser Use 0.12.6 depends on ``openai==2.16.0``.
The comparison runner therefore crosses a JSON subprocess boundary into
``packages/worldsim-agentlab-runner``.

The sidecar path is for benchmark-native comparison runs. It is not a
WorldSim-v5 Phase 4 parity runtime until PVPO, HAR/network reward evidence,
auth, and trajectory artifact compatibility are implemented.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from worldsim.agent_config import execution_instance_dict, task_reset_endpoints
from worldsim.agent_models import resolve_agent_model_profile
from worldsim.agent_runtime import AgentResult
from worldsim.benchmark_capabilities import get_benchmark_capabilities
from worldsim.resume_metadata import RESULT_FINGERPRINT_KEY
from worldsim.seeding import apply_data_seed_async
from worldsim.trajectory import save_result_payload

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt52"
_DEFAULT_MAX_STEPS = 30
_RESET_TIMEOUT = 300
_RESET_MAX_RETRIES = 2
_RESET_RETRY_DELAY_S = 10
_SIDECAR_CMD_ENV = "WORLDSIM_AGENTLAB_RUNNER_CMD"
_SUPPORTED_ATTACK_MODES = frozenset({"comparison", "seeded_comparison"})


@dataclass
class AgentLabAgentWrapper:
    """Worker-pool lifecycle wrapper for AgentLab sidecar settings."""

    model: str = _DEFAULT_MODEL
    provider: str | None = None
    service_tier: str | None = None
    max_steps: int = _DEFAULT_MAX_STEPS

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


def _task_benchmark_name(task: dict[str, Any]) -> str:
    return str(
        task.get("benchmark_name") or task.get("benchmark_adapter") or task.get("benchmark") or ""
    ).strip()


def _validate_task_benchmark(task: dict[str, Any]) -> str:
    benchmark_name = _task_benchmark_name(task)
    if not benchmark_name:
        raise ValueError("AgentLab task is missing benchmark_name/benchmark_adapter metadata")
    return get_benchmark_capabilities(benchmark_name).canonical_name


def _resolve_browsergym_task_name(
    task: dict[str, Any],
    *,
    benchmark_name: str,
    benchmark_prefix: str,
) -> str:
    explicit = task.get("agentlab_task_name") or task.get("browsergym_task_name")
    if explicit:
        return str(explicit)
    task_id = task.get("id", task.get("task_id"))
    if task_id is None:
        raise ValueError("task has no id/task_id and no agentlab_task_name")
    if benchmark_name != "webarena_verified":
        raise ValueError(
            f"task {task_id!r} for benchmark {benchmark_name!r} is missing agentlab_task_name"
        )
    return f"{benchmark_prefix}.{task_id}"


def _storage_state_for_agentlab(instance_dict: dict[str, Any]) -> str | None:
    auth = instance_dict.get("agent_auth")
    if not isinstance(auth, dict) or auth.get("type") != "storage_state":
        return None
    path = auth.get("path")
    if isinstance(path, str) and path.strip():
        return path
    storage_state = auth.get("storage_state")
    if isinstance(storage_state, dict):
        path = storage_state.get("path")
        if isinstance(path, str) and path.strip():
            return path
    return None


def _browsergym_env_overrides(instance_dict: dict[str, Any]) -> dict[str, str]:
    """Map WorldSim instance URLs to BrowserGym WebArena env vars where possible."""

    overrides: dict[str, str] = {}
    site_to_env = {
        "gitlab": "WA_GITLAB",
        "reddit": "WA_REDDIT",
        "postmill": "WA_REDDIT",
        "shopping": "WA_SHOPPING",
        "shopping_admin": "WA_SHOPPING_ADMIN",
        "wikipedia": "WA_WIKIPEDIA",
        "map": "WA_MAP",
        "osm": "WA_MAP",
        "homepage": "WA_HOMEPAGE",
    }
    site_name = str(instance_dict.get("site_name") or "").strip().lower()
    site_url = str(instance_dict.get("site_url") or "").strip()
    env_name = site_to_env.get(site_name)
    if env_name and site_url:
        overrides[env_name] = site_url
    placeholders = instance_dict.get("url_placeholders")
    if isinstance(placeholders, dict):
        placeholder_to_env = {
            "__GITLAB__": "WA_GITLAB",
            "__REDDIT__": "WA_REDDIT",
            "__SHOPPING__": "WA_SHOPPING",
            "__SHOPPING_ADMIN__": "WA_SHOPPING_ADMIN",
            "__WIKIPEDIA__": "WA_WIKIPEDIA",
            "__MAP__": "WA_MAP",
            "__HOMEPAGE__": "WA_HOMEPAGE",
        }
        for placeholder, env_key in placeholder_to_env.items():
            value = placeholders.get(placeholder)
            if isinstance(value, str) and value.strip():
                overrides[env_key] = value
    return overrides


def _build_sidecar_request(
    task: dict[str, Any],
    instance_dict: dict[str, Any],
    agent: AgentLabAgentWrapper,
    task_dir: Path,
    *,
    benchmark_name: str,
    benchmark_prefix: str,
    max_steps: int,
) -> dict[str, Any]:
    task_id = str(task.get("id", task.get("task_id", "unknown")))
    model_profile = resolve_agent_model_profile(
        agent.model,
        agent.provider,
        service_tier=agent.service_tier,
    )
    return {
        "schema_version": 1,
        "task_id": task_id,
        "benchmark_name": benchmark_name,
        "browsergym_task_name": _resolve_browsergym_task_name(
            task,
            benchmark_name=benchmark_name,
            benchmark_prefix=benchmark_prefix,
        ),
        "output_dir": str(task_dir),
        "requested_model": agent.model,
        "requested_provider": agent.provider,
        "model": model_profile.transport_model,
        "provider": model_profile.provider,
        "service_tier": agent.service_tier,
        "model_profile": model_profile.to_sidecar_dict(),
        "model_metadata_path": str(task_dir / "worldsim_model_calls.jsonl"),
        "max_steps": max_steps,
        "headless": True,
        "vision_support": model_profile.vision_support,
        "storage_state": _storage_state_for_agentlab(instance_dict),
        "env_overrides": _browsergym_env_overrides(instance_dict),
        "task_seed": task.get("agentlab_task_seed", task.get("task_seed")),
    }


def _build_phase4_sidecar_request(
    task: str,
    server_url: str,
    task_dir: Path,
    agent: AgentLabAgentWrapper,
    run_kwargs: dict[str, Any],
) -> dict[str, Any]:
    model_profile = resolve_agent_model_profile(
        agent.model,
        agent.provider,
        service_tier=agent.service_tier,
    )
    auth_mechanism = _dict_or_none(run_kwargs.get("auth_mechanism"))
    return {
        "schema_version": 1,
        "mode": "phase4",
        "task_id": task_dir.name,
        "benchmark_name": "webarena_verified",
        "output_dir": str(task_dir),
        "task": task,
        "server_url": server_url,
        "start_urls": _string_list(run_kwargs.get("start_urls")),
        "site_prompt": run_kwargs.get("site_prompt"),
        "task_site": run_kwargs.get("task_site"),
        "instance_id": run_kwargs.get("instance_id"),
        "auth_mechanism": auth_mechanism,
        "benchmark_root": str(run_kwargs["benchmark_root"])
        if run_kwargs.get("benchmark_root") is not None
        else None,
        "payload_text": run_kwargs.get("payload_text"),
        "payload_witnesses": run_kwargs.get("payload_witnesses") or [],
        "pvpo_cdp_url": run_kwargs.get("pvpo_cdp_url"),
        "url_origin_rewrites": run_kwargs.get("url_origin_rewrites") or {},
        "worldsim_repo_root": str(Path(__file__).resolve().parents[2]),
        "requested_model": agent.model,
        "requested_provider": agent.provider,
        "model": model_profile.transport_model,
        "provider": model_profile.provider,
        "service_tier": agent.service_tier,
        "model_profile": model_profile.to_sidecar_dict(),
        "model_metadata_path": str(task_dir / "worldsim_model_calls.jsonl"),
        "max_steps": agent.max_steps,
        "headless": True,
        "vision_support": model_profile.vision_support,
        "storage_state": _storage_state_for_agentlab({"agent_auth": auth_mechanism}),
        "env_overrides": _phase4_env_overrides(server_url, run_kwargs),
        "task_seed": None,
    }


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


def _default_sidecar_command(subcommand: str = "run") -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    project_dir = repo_root / "packages" / "worldsim-agentlab-runner"
    return [
        "uv",
        "run",
        "--project",
        str(project_dir),
        "worldsim-agentlab-runner",
        subcommand,
    ]


def _sidecar_command(subcommand: str = "run") -> list[str]:
    raw = os.environ.get(_SIDECAR_CMD_ENV)
    if raw and raw.strip():
        return shlex.split(raw)
    return _default_sidecar_command(subcommand)


def _run_sidecar_request(
    request: dict[str, Any],
    task_dir: Path,
    subcommand: str = "run",
) -> dict[str, Any]:
    task_dir.mkdir(parents=True, exist_ok=True)
    request_path = task_dir / (
        "agentlab_phase4_request.json" if subcommand == "phase4-run" else "agentlab_request.json"
    )
    response_path = task_dir / "agentlab_sidecar_result.json"
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")

    proc = subprocess.run(
        [*_sidecar_command(subcommand), str(request_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"AgentLab sidecar failed with exit {proc.returncode}: {detail}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"AgentLab sidecar returned invalid JSON: {proc.stdout[:500]}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("AgentLab sidecar returned a non-object JSON payload")
    response_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _parse_sidecar_result(task_id: str, task_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    summary = result.get("summary_info")
    if not isinstance(summary, dict):
        summary = {}
    n_steps = int(_first_present(result.get("steps"), summary.get("n_steps"), 0) or 0)
    reward = float(_first_present(result.get("reward"), summary.get("cum_reward"), 0.0) or 0.0)
    err_msg = result.get("error") or summary.get("err_msg")
    raw_passed = result.get("passed")
    passed = bool(raw_passed) if raw_passed is not None else reward > 0 and err_msg is None
    status = str(result.get("status") or ("error" if err_msg else ("success" if passed else "failure")))
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


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _phase4_env_overrides(server_url: str, run_kwargs: dict[str, Any]) -> dict[str, str]:
    task_site = str(run_kwargs.get("task_site") or "").strip().lower()
    site_to_env = {
        "gitlab": "WA_GITLAB",
        "reddit": "WA_REDDIT",
        "postmill": "WA_REDDIT",
    }
    env_key = site_to_env.get(task_site)
    if env_key:
        return {env_key: server_url}
    return {}


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
        task_id = str(task.get("id", task.get("task_id", "unknown")))
        benchmark_name = _validate_task_benchmark(task)
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

        await _reset_task_environment(task)
        if attack_mode == "seeded_comparison":
            seed = task.get("adversarial_data_seed", task.get("data_seed", {}))
            if isinstance(seed, dict) and seed.get("mechanism") not in (None, "none"):
                await apply_data_seed_async(seed, instance_dict)

        sidecar_result = await asyncio.to_thread(_run_sidecar_request, request, task_dir)
        parsed = _parse_sidecar_result(task_id, task_dir, sidecar_result)
        _persist_result_sentinel(task, task_dir, parsed)
        return parsed

    return run_task


def make_agent_factory(
    model: str = _DEFAULT_MODEL,
    provider: str | None = None,
    service_tier: str | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
    **_: Any,
) -> Callable[[], AgentLabAgentWrapper]:
    """Return a factory for AgentLab sidecar settings."""

    def factory() -> AgentLabAgentWrapper:
        return AgentLabAgentWrapper(
            model=model,
            provider=provider,
            service_tier=service_tier,
            max_steps=max_steps,
        )

    return factory
