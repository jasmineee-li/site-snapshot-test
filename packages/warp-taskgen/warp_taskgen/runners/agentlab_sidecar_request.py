"""Sidecar request construction for AgentLab comparison and Phase 4 runs."""

from __future__ import annotations

import base64
import os
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from warp_taskgen.agent_auth import _resolve_declared_storage_state_path, resolve_agent_auth_headers
from warp_taskgen.agent_models import resolve_agent_model_profile
from warp_taskgen.benchmark_capabilities import (
    get_benchmark_capabilities,
    infer_benchmark_from_metadata,
)
from warp_taskgen.browser_use_agent import (
    AuthArtifactMissingError,
    _augment_storage_state_origin_aliases,
    _storage_state_context_value,
    _storage_state_site_error,
)
from warp_taskgen.config import has_effective_agent_auth
from warp_taskgen.resume_metadata import RESULT_FINGERPRINT_KEY

if TYPE_CHECKING:
    from warp_taskgen.runners.agentlab import AgentLabAgentWrapper


def _task_benchmark_name(task: dict[str, Any]) -> str:
    benchmark_name = infer_benchmark_from_metadata((task,))
    return benchmark_name or ""


def _task_identity(
    task: dict[str, Any],
    *,
    reject_conflicts: bool = False,
    strict: bool = False,
) -> str | None:
    values: list[str] = []
    for field in ("id", "task_id"):
        value = task.get(field)
        if value is None:
            continue
        if strict and (isinstance(value, bool) or not isinstance(value, (str, int))):
            raise ValueError(f"AgentLab task {field} must be a non-empty string or integer")
        normalized = str(value).strip()
        if normalized:
            values.append(normalized)
        elif strict and value != "":
            raise ValueError(f"AgentLab task {field} must be a non-empty string or integer")
    if reject_conflicts and len(set(values)) > 1:
        raise ValueError("AgentLab task has conflicting id/task_id metadata")
    if not values:
        return None
    return values[0]


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


def _validate_phase4_authenticated_start_urls(
    *,
    auth_mechanism: dict[str, Any] | None,
    server_url: str,
    start_urls: list[str],
    url_origin_rewrites: dict[str, str],
) -> None:
    if not has_effective_agent_auth(auth_mechanism):
        return
    trusted_origin = _origin_from_url(server_url)
    if not trusted_origin:
        return
    off_origin: list[str] = []
    for url in start_urls:
        origin = _origin_from_url(url)
        if not origin:
            continue
        effective_origin = url_origin_rewrites.get(origin, origin)
        if effective_origin != trusted_origin:
            off_origin.append(origin)
    if off_origin:
        raise AuthArtifactMissingError(
            "authenticated AgentLab Phase 4 run received off-origin start_urls: "
            f"{sorted(set(off_origin))}"
        )


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
    task_id = _task_identity(task) or "unknown"
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
        "llm_timeout": agent.llm_timeout,
        "step_timeout": agent.step_timeout,
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
    start_urls = _string_list(run_kwargs.get("start_urls"))
    _validate_phase4_authenticated_start_urls(
        auth_mechanism=auth_mechanism,
        server_url=server_url,
        start_urls=start_urls,
        url_origin_rewrites=url_origin_rewrites,
    )
    storage_state: str | None = None
    storage_state_aliases: dict[str, Any] = {}
    storage_runtime_dir: str | None = None
    if isinstance(auth_mechanism, dict) and auth_mechanism.get("type") == "storage_state":
        storage_runtime_path = Path(
            tempfile.mkdtemp(
                prefix=f"worldsim-agentlab-storage-{task_dir.name}-{uuid.uuid4().hex}-"
            )
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
        "start_urls": start_urls,
        "site_prompt": run_kwargs.get("site_prompt"),
        "task_site": run_kwargs.get("task_site"),
        "instance_id": run_kwargs.get("instance_id"),
        "auth_mechanism": auth_mechanism,
        "benchmark_root": str(benchmark_root) if benchmark_root is not None else None,
        "payload_text": run_kwargs.get("payload_text"),
        "payload_witnesses": run_kwargs.get("payload_witnesses") or [],
        "pvpo_capture_backend": "page-surface-stable",
        "pvpo_cdp_url": None,
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
        "llm_timeout": agent.llm_timeout,
        "step_timeout": agent.step_timeout,
        "headless": True,
        "vision_support": model_profile.vision_support,
        "storage_state": storage_state,
        "storage_state_runtime_dir": storage_runtime_dir,
        "storage_state_aliases": storage_state_aliases,
        "scoped_auth": _scoped_auth_for_phase4(auth_mechanism, server_url=server_url),
        "env_overrides": _phase4_env_overrides(server_url, run_kwargs),
        RESULT_FINGERPRINT_KEY: run_kwargs.get(RESULT_FINGERPRINT_KEY),
        "task_seed": run_kwargs.get("agentlab_task_seed", run_kwargs.get("task_seed")),
    }


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


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
