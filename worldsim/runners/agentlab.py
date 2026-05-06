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
import base64
import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests

from worldsim.agent_auth import _resolve_declared_storage_state_path, resolve_agent_auth_headers
from worldsim.agent_config import execution_instance_dict, task_reset_endpoints
from worldsim.agent_models import resolve_agent_model_profile
from worldsim.agent_runtime import AgentResult
from worldsim.benchmark_capabilities import get_benchmark_capabilities
from worldsim.browser_use_agent import (
    AuthArtifactMissingError,
    _augment_storage_state_origin_aliases,
    _storage_state_context_value,
    _storage_state_site_error,
)
from worldsim.har_converter import minimal_har_placeholder_entry
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
    timeout: int | None = None

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


def _resolved_storage_state_for_phase4(
    auth: dict[str, Any] | None,
    *,
    benchmark_root: Path | None,
    task_site: str | None,
    instance_id: str | None,
    site_url: str | None,
    runtime_dir: Path,
    url_origin_rewrites: dict[str, str] | None,
) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(auth, dict) or auth.get("type") != "storage_state":
        return None
    storage = auth.get("storage_state") if isinstance(auth.get("storage_state"), dict) else {}
    raw_path = storage.get("path") or auth.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    site_name = str(task_site or "").strip()
    path, error = _resolve_declared_storage_state_path(
        raw_path.strip(),
        benchmark_root=benchmark_root,
        site_name=site_name,
        instance_id=instance_id,
    )
    if error is not None or path is None:
        raise RuntimeError(error or "storage_state path could not be resolved")
    site_error = _storage_state_site_error(path, site_url)
    if site_error is not None:
        raise RuntimeError(site_error)
    try:
        runtime_path = _storage_state_context_value(path, runtime_dir=runtime_dir)
        alias_summary = _augment_storage_state_origin_aliases(runtime_path, url_origin_rewrites)
    except AuthArtifactMissingError as exc:
        raise RuntimeError(str(exc)) from exc
    Path(runtime_path).chmod(0o600)
    return runtime_path, alias_summary


def _same_scheme_origin_rewrites(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    rewrites: dict[str, str] = {}
    for source, target in value.items():
        source_origin = _origin_from_url(str(source))
        target_origin = _origin_from_url(str(target))
        if not source_origin or not target_origin or source_origin == target_origin:
            continue
        if urlparse(source_origin).scheme != urlparse(target_origin).scheme:
            continue
        rewrites[source_origin] = target_origin
    return rewrites


def _scoped_auth_for_phase4(
    auth: dict[str, Any] | None,
    *,
    server_url: str,
) -> dict[str, Any]:
    if not isinstance(auth, dict):
        return {}
    auth_type = str(auth.get("type") or "").strip()
    if auth_type == "http_basic":
        block = auth.get("http_basic") if isinstance(auth.get("http_basic"), dict) else auth
        username = block.get("username")
        password = block.get("password")
        if not isinstance(username, str) or not isinstance(password, str):
            raise RuntimeError("http_basic auth requires username/password")
        token = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
        return {
            "origin": _origin_from_url(server_url),
            "headers": {"Authorization": f"Basic {token}"},
        }
    if auth_type == "http_headers":
        return {"origin": _origin_from_url(server_url), "headers": resolve_agent_auth_headers(auth)}
    return {}


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
    benchmark_root = (
        Path(run_kwargs["benchmark_root"]) if run_kwargs.get("benchmark_root") is not None else None
    )
    task_site = run_kwargs.get("task_site")
    instance_id = run_kwargs.get("instance_id")
    url_origin_rewrites = _same_scheme_origin_rewrites(run_kwargs.get("url_origin_rewrites"))
    storage_state: str | None = None
    storage_state_aliases: dict[str, Any] = {}
    storage_runtime_dir: str | None = None
    if isinstance(auth_mechanism, dict) and auth_mechanism.get("type") == "storage_state":
        storage_runtime_path = Path(
            tempfile.mkdtemp(prefix=f"worldsim-agentlab-storage-{task_dir.name}-{uuid.uuid4().hex}-")
        )
        storage_runtime_dir = str(storage_runtime_path)
        resolved_storage = _resolved_storage_state_for_phase4(
            auth_mechanism,
            benchmark_root=benchmark_root,
            task_site=str(task_site) if task_site is not None else None,
            instance_id=str(instance_id) if instance_id is not None else None,
            site_url=server_url,
            runtime_dir=storage_runtime_path,
            url_origin_rewrites=url_origin_rewrites,
        )
        if resolved_storage is not None:
            storage_state, storage_state_aliases = resolved_storage
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
        "benchmark_root": str(benchmark_root) if benchmark_root is not None else None,
        "payload_text": run_kwargs.get("payload_text"),
        "payload_witnesses": run_kwargs.get("payload_witnesses") or [],
        "pvpo_cdp_url": run_kwargs.get("pvpo_cdp_url"),
        "url_origin_rewrites": url_origin_rewrites,
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
        "storage_state": storage_state,
        "storage_state_runtime_dir": storage_runtime_dir,
        "storage_state_aliases": storage_state_aliases,
        "scoped_auth": _scoped_auth_for_phase4(auth_mechanism, server_url=server_url),
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
        parts = shlex.split(raw)
        if parts and parts[-1] in {"run", "phase4-run"}:
            return [*parts[:-1], subcommand]
        return [*parts, subcommand]
    return _default_sidecar_command(subcommand)


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
    request_path.write_text(
        json.dumps(_redact_sidecar_payload(request), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    try:
        with tempfile.TemporaryDirectory(prefix="worldsim-agentlab-request-") as runtime_dir:
            runtime_request_path = Path(runtime_dir) / request_path.name
            runtime_request_path.write_text(
                json.dumps(request, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            runtime_request_path.chmod(0o600)
            proc = subprocess.run(
                [*_sidecar_command(subcommand), str(runtime_request_path)],
                text=True,
                capture_output=True,
                check=False,
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
        raise RuntimeError(f"AgentLab sidecar failed with exit {proc.returncode}: {detail}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
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


def _phase4_timeout_result(
    request: dict[str, Any],
    task_dir: Path,
    timeout: int | None,
    timeout_payload: dict[str, str],
) -> dict[str, Any]:
    message = f"AgentLab sidecar exceeded task timeout {timeout}s"
    task_dir.mkdir(parents=True, exist_ok=True)
    _write_minimal_timeout_artifacts(task_dir, request, message)
    runtime = {
        "runner": "agentlab",
        "mode": "phase4",
        "timeout": timeout,
        "timeout_expired": True,
        "cdp_url": request.get("pvpo_cdp_url"),
    }
    runtime.update(
        _recycle_pvpo_browser_after_parent_timeout(str(request.get("pvpo_cdp_url") or ""))
    )
    (task_dir / "browser_runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
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
        "elapsed": float(timeout or 0),
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
        "evidence_status": "timeout_placeholder",
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


def _write_minimal_timeout_artifacts(task_dir: Path, request: dict[str, Any], message: str) -> None:
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
            {"status": "error", "final_result": None, "errors": [message], "steps": 0}, indent=2
        )
        + "\n",
    )
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
        "needham_trace": output_dir / "needham_trace.json",
        "needham_xml": output_dir / "needham_trace.xml",
        "pvpo_summary": output_dir / "pvpo" / "capture_summary.json",
    }
    screenshots = sorted(
        {str(path) for path in output_dir.glob("screenshot_step_*")}
        | {str(path) for path in (output_dir / "screenshots").glob("step_*.png")}
    )
    pvpo_steps = sorted(str(path) for path in (output_dir / "pvpo").glob("step_*.json"))
    steps = sorted(str(path) for path in output_dir.glob("step_*.pkl.gz"))
    videos = sorted(str(path) for path in output_dir.glob("**/*.webm"))
    return {
        key: str(path) for key, path in files.items() if path.exists()
    } | {"screenshots": screenshots, "pvpo_steps": pvpo_steps, "steps": steps, "videos": videos}


def _recycle_pvpo_browser_after_parent_timeout(cdp_url: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "recycle_enabled": True,
        "recycle_status": "disabled",
        "recycle_method": None,
        "recycle_reason": "parent_timeout",
    }
    if not cdp_url:
        return payload
    from urllib.parse import urlparse

    parsed = urlparse(cdp_url)
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"} or parsed.port is None:
        payload["recycle_status"] = "unmanaged_endpoint"
        return payload
    container = f"pvpo-chrome-{parsed.port}"
    payload["recycle_container"] = container
    if not shutil.which("docker"):
        payload["recycle_status"] = "docker_unavailable"
        return payload
    try:
        proc = subprocess.run(
            ["docker", "restart", container],
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except Exception as recycle_exc:
        payload["recycle_status"] = "failed"
        payload["recycle_method"] = "docker_restart"
        payload["recycle_failure"] = f"{type(recycle_exc).__name__}: {recycle_exc}"
        return payload
    payload["recycle_method"] = "docker_restart"
    payload["recycle_status"] = "recycled" if proc.returncode == 0 else "failed"
    if proc.returncode != 0:
        payload["recycle_failure"] = (proc.stderr or proc.stdout).strip()
    return payload


def _redact_sidecar_payload(value: Any, *, secret_values: set[str] | None = None) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            lower = str(key).lower()
            if lower == "storage_state":
                redacted[key] = {"present": bool(item), "runtime_only": True}
            elif lower == "storage_state_runtime_dir":
                redacted[key] = "<runtime-only>"
            elif lower == "network_trace" and isinstance(item, list):
                redacted[key] = [
                    _redact_network_event(event, secret_values=secret_values) for event in item
                ]
            elif lower in {"authorization", "cookie", "set-cookie", "proxy-authorization"} or any(
                marker in lower
                for marker in ("token", "secret", "password", "auth", "cookie", "csrf", "key")
            ):
                redacted[key] = "<redacted>"
            elif lower == "headers" and isinstance(item, dict):
                redacted[key] = _redact_sidecar_headers(item)
            else:
                redacted[key] = _redact_sidecar_payload(item, secret_values=secret_values)
        return redacted
    if isinstance(value, list):
        return [_redact_sidecar_payload(item, secret_values=secret_values) for item in value]
    if isinstance(value, str) and secret_values:
        return _redact_text_values(value, secret_values)
    return value


def _redact_network_event(
    value: Any,
    *,
    secret_values: set[str] | None = None,
) -> Any:
    if not isinstance(value, dict):
        return _redact_sidecar_payload(value, secret_values=secret_values)
    event: dict[str, Any] = {}
    for key, item in value.items():
        lower = str(key).lower()
        if lower in {"url"} and isinstance(item, str):
            event[key] = _redact_url_value(item, secret_values=secret_values)
        elif lower == "query_params" and isinstance(item, dict):
            event[key] = {
                str(param): [
                    _redact_network_scalar(str(param), child, secret_values=secret_values)
                    for child in values
                ]
                if isinstance(values, list)
                else _redact_network_scalar(str(param), values, secret_values=secret_values)
                for param, values in item.items()
            }
        elif lower in {"post_data", "response_content"}:
            event[key] = _redact_network_body(item, secret_values=secret_values)
        elif lower in {"request_headers", "headers", "response_headers"} and isinstance(item, dict):
            event[key] = _redact_sidecar_headers(item)
        elif lower == "response_cookies" and isinstance(item, list):
            event[key] = [
                {"name": str(cookie.get("name") or ""), "value": "<redacted>"}
                for cookie in item
                if isinstance(cookie, dict)
            ]
        else:
            event[key] = _redact_sidecar_payload(item, secret_values=secret_values)
    return event


def _redact_url_value(value: str, *, secret_values: set[str] | None = None) -> str:
    parsed = urlparse(value)
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    if not pairs:
        return _redact_text_values(value, secret_values or set())
    redacted_pairs = [
        (
            key,
            "<redacted>"
            if _is_sensitive_network_field(key)
            else _redact_text_values(val, secret_values or set()),
        )
        for key, val in pairs
    ]
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            urlencode(redacted_pairs, doseq=True),
            parsed.fragment,
        )
    )


def _redact_network_body(value: Any, *, secret_values: set[str] | None = None) -> Any:
    if not isinstance(value, str) or not value:
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        redacted = _redact_network_json(parsed, secret_values=secret_values)
        return json.dumps(redacted, sort_keys=True, separators=(",", ":"))
    pairs = parse_qsl(value, keep_blank_values=True)
    if pairs and urlencode(pairs) == value.replace(" ", "+"):
        return urlencode(
            [
                (
                    key,
                    _redact_network_scalar(key, item, secret_values=secret_values),
                )
                for key, item in pairs
            ]
        )
    return _redact_text_values(value, secret_values or set())


def _redact_network_json(value: Any, *, secret_values: set[str] | None = None) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _redact_network_scalar(str(key), item, secret_values=secret_values)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_network_json(item, secret_values=secret_values) for item in value]
    if isinstance(value, str):
        return _redact_text_values(value, secret_values or set())
    return value


def _redact_network_scalar(
    field_name: str, value: Any, *, secret_values: set[str] | None = None
) -> Any:
    if _is_sensitive_network_field(field_name):
        return "<redacted>"
    if isinstance(value, str):
        return _redact_text_values(value, secret_values or set())
    return _redact_network_json(value, secret_values=secret_values)


def _is_sensitive_network_field(name: str) -> bool:
    normalized = name.strip().lower().replace("-", "_")
    return normalized in {
        "password",
        "passwd",
        "secret",
        "csrf",
        "csrf_token",
        "authenticity_token",
        "access_token",
        "refresh_token",
        "id_token",
        "api_key",
        "apikey",
        "session",
        "session_id",
        "_session",
        "cookie",
    } or any(
        marker in normalized
        for marker in (
            "password",
            "passwd",
            "secret",
            "csrf",
            "authenticity_token",
            "access_token",
            "refresh_token",
            "id_token",
            "api_key",
            "session",
        )
    )


def _secret_strings_from_payload(value: Any) -> set[str]:
    secrets: set[str] = set()

    def visit(item: Any, *, sensitive: bool = False) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                lower = str(key).lower()
                child_sensitive = (
                    sensitive
                    or lower
                    in {
                        "authorization",
                        "cookie",
                        "set-cookie",
                        "proxy-authorization",
                    }
                    or any(
                        marker in lower
                        for marker in (
                            "token",
                            "secret",
                            "password",
                            "auth",
                            "cookie",
                            "csrf",
                            "key",
                        )
                    )
                )
                visit(child, sensitive=child_sensitive)
            return
        if isinstance(item, list):
            for child in item:
                visit(child, sensitive=sensitive)
            return
        if sensitive and isinstance(item, str) and item:
            secrets.add(item)

    visit(value)
    return secrets


def _redact_sidecar_text(text: str, request: dict[str, Any]) -> str:
    return _redact_text_values(text, _secret_strings_from_payload(request))


def _redact_text_values(text: str, secret_values: set[str]) -> str:
    redacted = text
    for secret in sorted(secret_values, key=len, reverse=True):
        if len(secret) >= 4:
            redacted = redacted.replace(secret, "<redacted>")
    return redacted


def _redact_sidecar_headers(headers: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, item in headers.items():
        lower = str(key).lower()
        if lower in {"authorization", "cookie", "set-cookie", "proxy-authorization"} or any(
            marker in lower
            for marker in ("token", "secret", "session", "auth", "cookie", "csrf", "key")
        ):
            out[str(key)] = "<redacted>"
        else:
            out[str(key)] = item
    return out


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


def _origin_from_url(raw_url: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    default_port = 80 if parsed.scheme == "http" else 443
    if parsed.port and parsed.port != default_port:
        return f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
    return f"{parsed.scheme}://{parsed.hostname}"


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
    task_timeout: int | None = None,
    llm_timeout: int | None = None,
    step_timeout: int | None = None,
    **_: Any,
) -> Callable[[], AgentLabAgentWrapper]:
    """Return a factory for AgentLab sidecar settings."""
    if llm_timeout is not None or step_timeout is not None:
        raise ValueError(
            "AgentLab runner does not yet support --agent-llm-timeout or --agent-step-timeout; "
            "omit them or use Browser Use until sidecar step/model deadlines are implemented"
        )

    def factory() -> AgentLabAgentWrapper:
        return AgentLabAgentWrapper(
            model=model,
            provider=provider,
            service_tier=service_tier,
            max_steps=max_steps,
            timeout=task_timeout,
        )

    return factory
