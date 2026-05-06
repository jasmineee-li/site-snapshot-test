from __future__ import annotations

import json
import logging
import pickle
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from worldsim_agentlab_runner.cdp_browser import patched_chromium_launch
from worldsim_agentlab_runner.model_args import model_args_from_request
from worldsim_agentlab_runner.network_trace import NetworkTraceRecorder
from worldsim_agentlab_runner.sync_pvpo import SyncPvpoRecorder
from worldsim_agentlab_runner.trajectory_projection import (
    final_result_from_env,
    write_worldsim_artifacts,
)
from worldsim_agentlab_runner.worldsim_task import make_worldsim_browsergym_env

logger = logging.getLogger(__name__)


def run_phase4_request_path(path: Path) -> dict[str, Any]:
    request = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise ValueError("request JSON must be an object")
    return run_phase4_request(request)


def run_phase4_request(request: dict[str, Any]) -> dict[str, Any]:
    from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
    from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
    from agentlab.experiments.loop import StepInfo, save_package_versions

    from worldsim_agentlab_runner.cli import (
        _apply_benchmark_config,
        _artifact_manifest,
        _default_action_flags,
        _default_obs_flags,
        _model_result,
        _patched_env,
        _string_env,
        _versions,
    )

    output_dir = Path(_required_str(request, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    errors: list[str] = []
    status = "success"
    episode_info: list[Any] = []
    env = None
    network = NetworkTraceRecorder(output_dir)
    runtime: dict[str, Any] = {
        "runner": "agentlab",
        "mode": "phase4",
        "cdp_url": request.get("pvpo_cdp_url"),
        "storage_state_aliases": request.get("storage_state_aliases") or {},
    }
    steps_taken = 0
    final_result: str | None = None
    chat_model_args = model_args_from_request(request)
    agent_args = GenericAgentArgs(
        chat_model_args=chat_model_args,
        flags=GenericPromptFlags(
            obs=_default_obs_flags(),
            action=_default_action_flags(),
            use_plan=True,
            use_thinking=True,
            use_memory=True,
            be_cautious=True,
        ),
    )
    benchmark_config: dict[str, Any] = {}
    pvpo = SyncPvpoRecorder(
        output_dir,
        payload_text=str(request.get("payload_text") or ""),
        witness_texts=_witnesses(request.get("payload_witnesses")),
        repo_root=_optional_path(request.get("worldsim_repo_root")),
        cdp_url=str(request.get("pvpo_cdp_url") or ""),
    )

    try:
        save_package_versions(output_dir)
        with _patched_env(_string_env(request.get("env_overrides"))):
            benchmark_config = _apply_benchmark_config(agent_args, request)
            agent_args.prepare()
            agent = agent_args.make_agent()
            if hasattr(agent, "set_task_name"):
                agent.set_task_name("worldsim.phase4")
            env = make_worldsim_browsergym_env(
                request,
                action_mapping=agent.action_set.to_python_code,
                exp_dir=output_dir,
                network_recorder=network,
                runtime=runtime,
            )
            with patched_chromium_launch(str(request.get("pvpo_cdp_url") or ""), runtime):
                step_info = StepInfo(step=0)
                step_info.from_reset(
                    env,
                    seed=_task_seed(request),
                    obs_preprocessor=agent.obs_preprocessor,
                )
            episode_info.append(step_info)
            pvpo.capture_step(_pvpo_capture_page(env), step_info.step)

            while not step_info.is_done and steps_taken < int(request.get("max_steps") or 30):
                action = step_info.from_action(agent)
                step_info.save_step_info(
                    output_dir,
                    save_screenshot=True,
                    save_som=False,
                )
                if action is None:
                    step_info.truncated = True
                    break

                next_step = StepInfo(step=step_info.step + 1)
                next_step.from_step(env, action, obs_preprocessor=agent.obs_preprocessor)
                steps_taken += 1
                episode_info.append(next_step)
                pvpo.capture_step(_pvpo_capture_page(env), next_step.step)
                step_info = next_step

            final_result = final_result_from_env(env)
    except Exception as exc:
        status = "error"
        errors.append(f"{type(exc).__name__}: {exc}")
        logger.exception("AgentLab Phase 4 sidecar failed")
    finally:
        try:
            if episode_info:
                episode_info[-1].save_step_info(
                    output_dir,
                    save_screenshot=True,
                    save_som=False,
                )
        except Exception as exc:
            errors.append(f"save_step_info failed: {type(exc).__name__}: {exc}")
            status = "error"
        try:
            network.persist()
        except Exception as exc:
            errors.append(f"network trace persist failed: {type(exc).__name__}: {exc}")
            status = "error"
        try:
            if env is not None:
                env.close()
        except Exception as exc:
            errors.append(f"env close failed: {type(exc).__name__}: {exc}")
            status = "error"
        try:
            agent_args.close()
        except Exception:
            pass
        try:
            pvpo.close()
        except Exception:
            pass
        runtime.update(_recycle_cdp_browser(str(request.get("pvpo_cdp_url") or "")))
        (output_dir / "browser_runtime.json").write_text(
            json.dumps(runtime, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    if final_result is None:
        final_result = _final_result_from_episode(episode_info)
    elapsed = time.monotonic() - start
    summary = {
        "n_steps": steps_taken,
        "cum_reward": sum(float(getattr(step, "reward", 0) or 0) for step in episode_info),
        "cum_raw_reward": sum(float(getattr(step, "raw_reward", 0) or 0) for step in episode_info),
        "err_msg": errors[-1] if errors else None,
        "terminated": bool(episode_info and getattr(episode_info[-1], "terminated", False)),
        "truncated": bool(episode_info and getattr(episode_info[-1], "truncated", False)),
    }
    (output_dir / "summary_info.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_worldsim_artifacts(
        output_dir,
        episode_info=episode_info,
        final_result=final_result,
        status=status,
        errors=errors,
        task_instruction=str(request.get("task") or ""),
    )
    _write_phase4_request_copy(output_dir, request, agent_args)
    return {
        "schema_version": 1,
        "mode": "phase4",
        "task_id": request.get("task_id"),
        "status": status,
        "passed": None,
        "reward": summary["cum_reward"],
        "agentlab_reward": summary["cum_reward"],
        "steps": steps_taken,
        "is_done": summary["terminated"] or summary["truncated"] or bool(final_result),
        "final_result": final_result,
        "elapsed": elapsed,
        "errors": errors,
        "error": errors[-1] if errors else None,
        "network_trace": network.events,
        "summary_info": summary,
        "artifacts": _artifact_manifest(output_dir),
        "versions": _versions(),
        "benchmark_config": benchmark_config,
        "browser_runtime": runtime,
        "model": _model_result(
            request,
            str(request.get("model") or ""),
            chat_model_args,
        ),
    }


def _write_phase4_request_copy(output_dir: Path, request: dict[str, Any], agent_args: Any) -> None:
    (output_dir / "phase4_sidecar_request.json").write_text(
        json.dumps(_redact_sidecar_payload(request), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    try:
        with (output_dir / "agentlab_native_exp_args.pkl").open("wb") as handle:
            pickle.dump(
                {"request": _redact_sidecar_payload(request), "agent_args": agent_args}, handle
            )
    except Exception:
        pass


def _redact_sidecar_payload(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            lower = str(key).lower()
            if lower == "storage_state":
                out[key] = {"present": bool(item), "runtime_only": True}
            elif lower == "storage_state_runtime_dir":
                out[key] = "<runtime-only>"
            elif lower in {"authorization", "cookie", "set-cookie", "proxy-authorization"} or any(
                marker in lower
                for marker in ("token", "secret", "password", "auth", "cookie", "csrf", "key")
            ):
                out[key] = "<redacted>"
            elif lower == "headers" and isinstance(item, dict):
                out[key] = _redact_headers(item)
            else:
                out[key] = _redact_sidecar_payload(item)
        return out
    if isinstance(value, list):
        return [_redact_sidecar_payload(item) for item in value]
    return value


def _redact_headers(headers: dict[str, Any]) -> dict[str, Any]:
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


def _final_result_from_episode(episode_info: list[Any]) -> str | None:
    for step in reversed(episode_info):
        task_info = getattr(step, "task_info", None)
        if isinstance(task_info, dict) and isinstance(task_info.get("worldsim_final_result"), str):
            return task_info["worldsim_final_result"]
    return None


def _pvpo_capture_page(env: Any) -> Any:
    unwrapped = getattr(env, "unwrapped", env)
    page = getattr(unwrapped, "page", None)
    context = getattr(page, "context", None)
    pages = getattr(context, "pages", None)
    if isinstance(pages, (list, tuple)):
        for candidate in reversed(pages):
            if _page_is_open(candidate):
                return candidate
    return page


def _page_is_open(page: Any) -> bool:
    is_closed = getattr(page, "is_closed", None)
    if callable(is_closed):
        try:
            return not bool(is_closed())
        except Exception:
            return True
    return True


def _task_seed(request: dict[str, Any]) -> int:
    raw = request.get("task_seed")
    if raw in (None, ""):
        return 0
    return int(raw)


def _witnesses(value: Any) -> list[str | dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, (str, dict))]


def _optional_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value)


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"request missing required string field {key!r}")
    return value


def _recycle_cdp_browser(cdp_url: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "recycle_enabled": _recycle_enabled(),
        "recycle_status": "disabled",
        "recycle_method": None,
    }
    if not cdp_url or not payload["recycle_enabled"]:
        return payload
    container = _managed_container_name(cdp_url)
    payload["recycle_container"] = container
    if not container:
        payload["recycle_status"] = "unmanaged_endpoint"
        return payload
    if shutil.which("docker") is None:
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
    except Exception as exc:
        payload["recycle_status"] = "failed"
        payload["recycle_failure"] = f"{type(exc).__name__}: {exc}"
        payload["recycle_method"] = "docker_restart"
        return payload
    payload["recycle_method"] = "docker_restart"
    payload["recycle_status"] = "recycled" if proc.returncode == 0 else "failed"
    if proc.returncode != 0:
        payload["recycle_failure"] = (proc.stderr or proc.stdout).strip()
    return payload


def _recycle_enabled() -> bool:
    import os

    return os.environ.get("WORLDSIM_PVPO_BROWSER_RECYCLE", "").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _managed_container_name(cdp_url: str) -> str | None:
    parsed = urlparse(cdp_url)
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"} or parsed.port is None:
        return None
    return f"pvpo-chrome-{parsed.port}"
