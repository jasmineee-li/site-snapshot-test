from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import pickle
import signal
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from worldsim_agentlab_runner.cdp_browser import patched_chromium_launch
from worldsim_agentlab_runner.model_args import model_args_from_request
from worldsim_agentlab_runner.network_trace import NetworkTraceRecorder
from worldsim_agentlab_runner.pvpo_screenshot_patch import patched_browsergym_screenshot_for_pvpo
from worldsim_agentlab_runner.sync_pvpo import FatalPvpoCaptureError, SyncPvpoRecorder
from worldsim_agentlab_runner.trajectory_projection import (
    final_result_from_env,
    write_worldsim_artifacts,
)
from worldsim_agentlab_runner.worldsim_task import make_worldsim_browsergym_env

logger = logging.getLogger(__name__)
_TIMELINE_ARTIFACT = "agentlab_step_timeline.jsonl"
_EVENTS_ARTIFACT = "agentlab_events.jsonl"


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
    network_mark = 0
    runtime: dict[str, Any] = {
        "runner": "agentlab",
        "mode": "phase4",
        "cdp_url": request.get("pvpo_cdp_url"),
        "storage_state_aliases": request.get("storage_state_aliases") or {},
        "browser_instance_scope": "agent_run",
        "lifecycle_events": [],
        "runtime_artifact_status": "running",
        "current_phase": "initializing",
        "current_step": None,
        "last_updated_at": _utc_now(),
    }
    steps_taken = 0
    final_result: str | None = None
    deadline_hit = False
    skip_env_close_reason: str | None = None
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
    _write_browser_runtime(output_dir, runtime)
    _append_agentlab_event(
        output_dir,
        "sidecar.start",
        runtime=runtime,
        task_id=request.get("task_id"),
        message="AgentLab Phase 4 sidecar started",
    )

    try:
        save_package_versions(output_dir)
        with _patched_env(_string_env(request.get("env_overrides"))):
            cdp_url = str(request.get("pvpo_cdp_url") or "")
            with patched_chromium_launch(cdp_url, runtime):
                with patched_browsergym_screenshot_for_pvpo(cdp_url, runtime):
                    with _step_deadline(request, "setup and reset"):
                        _update_runtime_progress(
                            output_dir,
                            runtime,
                            phase="setup",
                            step=0,
                            network=network,
                        )
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
                        step_info = StepInfo(step=0)
                        step_info.from_reset(
                            env,
                            seed=_task_seed(request),
                            obs_preprocessor=agent.obs_preprocessor,
                        )
                        _update_runtime_progress(
                            output_dir,
                            runtime,
                            phase="reset_observed",
                            step_info=step_info,
                            network=network,
                        )
                    episode_info.append(step_info)
                    pvpo.capture_step(_pvpo_capture_page(env), step_info.step)
                    _append_step_timeline(
                        output_dir,
                        event="reset",
                        step_info=step_info,
                        network=network,
                        runtime=runtime,
                        network_mark=network_mark,
                    )
                    network_mark = network.mark()
                    _update_runtime_progress(
                        output_dir,
                        runtime,
                        phase="reset_captured",
                        step_info=step_info,
                        network=network,
                    )

                    while not step_info.is_done and steps_taken < int(
                        request.get("max_steps") or 30
                    ):
                        with _step_deadline(request, f"action step {step_info.step}"):
                            _update_runtime_progress(
                                output_dir,
                                runtime,
                                phase="agent_action",
                                step_info=step_info,
                                network=network,
                            )
                            action = step_info.from_action(agent)
                            _update_runtime_progress(
                                output_dir,
                                runtime,
                                phase="agent_action_done",
                                step_info=step_info,
                                action=action,
                                network=network,
                            )
                        step_info.save_step_info(
                            output_dir,
                            save_screenshot=True,
                            save_som=False,
                        )
                        _append_step_timeline(
                            output_dir,
                            event="agent_action",
                            step_info=step_info,
                            action=action,
                            network=network,
                            runtime=runtime,
                            network_mark=network_mark,
                        )
                        network_mark = network.mark()
                        if action is None:
                            step_info.truncated = True
                            break

                        next_step = StepInfo(step=step_info.step + 1)
                        with _step_deadline(request, f"browser step {next_step.step}"):
                            _update_runtime_progress(
                                output_dir,
                                runtime,
                                phase="browser_step",
                                step=next_step.step,
                                action=action,
                                network=network,
                            )
                            next_step.from_step(
                                env, action, obs_preprocessor=agent.obs_preprocessor
                            )
                            _update_runtime_progress(
                                output_dir,
                                runtime,
                                phase="browser_step_done",
                                step_info=next_step,
                                action=action,
                                network=network,
                            )
                        steps_taken += 1
                        episode_info.append(next_step)
                        pvpo.capture_step(_pvpo_capture_page(env), next_step.step)
                        _append_step_timeline(
                            output_dir,
                            event="browser_step",
                            step_info=next_step,
                            action=action,
                            network=network,
                            runtime=runtime,
                            network_mark=network_mark,
                        )
                        network_mark = network.mark()
                        _update_runtime_progress(
                            output_dir,
                            runtime,
                            phase="pvpo_captured",
                            step_info=next_step,
                            action=action,
                            network=network,
                        )
                        step_info = next_step

                    final_result = final_result_from_env(env)
    except Exception as exc:
        deadline_hit = isinstance(exc, TimeoutError)
        if isinstance(exc, FatalPvpoCaptureError):
            skip_env_close_reason = "pvpo_capture_failed"
            runtime["pvpo_capture_fatal"] = True
            runtime["pvpo_capture_fatal_error"] = f"{type(exc).__name__}: {exc}"
        status = "timeout" if deadline_hit else "error"
        errors.append(f"{type(exc).__name__}: {exc}")
        _append_agentlab_event(
            output_dir,
            "sidecar.error",
            runtime=runtime,
            task_id=request.get("task_id"),
            error=f"{type(exc).__name__}: {exc}",
        )
        logger.exception("AgentLab Phase 4 sidecar failed")
    finally:
        try:
            if episode_info:
                runtime["lifecycle_events"].append("save_final_step")
                episode_info[-1].save_step_info(
                    output_dir,
                    save_screenshot=True,
                    save_som=False,
                )
                runtime["final_step_save_status"] = "saved"
        except Exception as exc:
            errors.append(f"save_step_info failed: {type(exc).__name__}: {exc}")
            status = "error"
            runtime["final_step_save_status"] = "failed"
            runtime["final_step_save_error"] = f"{type(exc).__name__}: {exc}"
        try:
            runtime["lifecycle_events"].append("persist_network")
            network.persist()
            runtime["network_persist_status"] = "persisted"
        except Exception as exc:
            errors.append(f"network trace persist failed: {type(exc).__name__}: {exc}")
            status = "error"
            runtime["network_persist_status"] = "failed"
            runtime["network_persist_error"] = f"{type(exc).__name__}: {exc}"
        try:
            if env is not None and not deadline_hit and skip_env_close_reason is None:
                runtime["lifecycle_events"].append("env_close")
                runtime["env_close_attempted"] = True
                env.close()
                runtime["env_close_status"] = "closed"
            elif env is not None:
                runtime["env_close_attempted"] = False
                runtime["env_close_status"] = "skipped"
                reason = skip_env_close_reason or "sidecar_deadline"
                runtime["env_close_skipped_reason"] = reason
                errors.append(f"env close skipped after {reason}")
        except Exception as exc:
            errors.append(f"env close failed: {type(exc).__name__}: {exc}")
            status = "error"
            runtime["env_close_status"] = "failed"
            runtime["env_close_error"] = f"{type(exc).__name__}: {exc}"
        try:
            runtime["lifecycle_events"].append("agent_args_close")
            runtime["agent_args_close_attempted"] = True
            agent_args.close()
            runtime["agent_args_close_status"] = "closed"
        except Exception as exc:
            runtime["agent_args_close_status"] = "failed"
            runtime["agent_args_close_error"] = f"{type(exc).__name__}: {exc}"
        try:
            runtime["lifecycle_events"].append("pvpo_close")
            runtime["pvpo_close_attempted"] = True
            pvpo.close()
            runtime["pvpo_close_status"] = "closed"
        except Exception as exc:
            runtime["pvpo_close_status"] = "failed"
            runtime["pvpo_close_error"] = f"{type(exc).__name__}: {exc}"
        runtime["lifecycle_events"].append("recycle_cdp_browser")
        _update_runtime_progress(
            output_dir,
            runtime,
            phase="recycle_cdp_browser",
            step=steps_taken,
            network=network,
        )
        runtime.update(_recycle_cdp_browser(str(request.get("pvpo_cdp_url") or "")))
        runtime["runtime_artifact_status"] = "complete"
        _write_browser_runtime(output_dir, runtime)

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


def _write_browser_runtime(output_dir: Path, runtime: dict[str, Any]) -> None:
    runtime["last_updated_at"] = _utc_now()
    (output_dir / "browser_runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _update_runtime_progress(
    output_dir: Path,
    runtime: dict[str, Any],
    *,
    phase: str,
    step_info: Any | None = None,
    step: int | None = None,
    action: Any = None,
    network: NetworkTraceRecorder | None = None,
) -> None:
    current_step = step if step is not None else getattr(step_info, "step", None)
    runtime["current_phase"] = phase
    runtime["current_step"] = current_step
    if step_info is not None:
        obs = getattr(step_info, "obs", None)
        if isinstance(obs, dict):
            runtime["last_url"] = _string_or_none(obs.get("url"))
            runtime["last_title"] = _active_title_from_obs(obs)
        runtime["last_screenshot"] = f"screenshots/step_{current_step}.png"
        runtime["last_reward"] = getattr(step_info, "reward", None)
        runtime["last_raw_reward"] = getattr(step_info, "raw_reward", None)
        runtime["last_terminated"] = getattr(step_info, "terminated", None)
        runtime["last_truncated"] = getattr(step_info, "truncated", None)
    if action is not None:
        runtime["last_action"] = str(action)
    if network is not None:
        runtime["last_network_event_count"] = len(network.events)
    _write_browser_runtime(output_dir, runtime)


def _append_step_timeline(
    output_dir: Path,
    *,
    event: str,
    step_info: Any,
    network: NetworkTraceRecorder,
    runtime: dict[str, Any],
    action: Any = None,
    network_mark: int | None = None,
) -> None:
    obs = getattr(step_info, "obs", None)
    if not isinstance(obs, dict):
        obs = {}
    step = getattr(step_info, "step", None)
    network_summary = _network_delta_summary(network, network_mark)
    payload = {
        "schema_version": 1,
        "runner": "agentlab",
        "mode": "phase4",
        "event": event,
        "timestamp": _utc_now(),
        "monotonic_s": round(time.monotonic(), 6),
        "step": step,
        "phase": runtime.get("current_phase"),
        "url": _string_or_none(obs.get("url")),
        "title": _active_title_from_obs(obs),
        "action": str(action) if action is not None else None,
        "screenshot": f"screenshots/step_{step}.png",
        "network_event_count": len(network.events),
        **network_summary,
        "reward": getattr(step_info, "reward", None),
        "raw_reward": getattr(step_info, "raw_reward", None),
        "terminated": getattr(step_info, "terminated", None),
        "truncated": getattr(step_info, "truncated", None),
        "last_action_error": obs.get("last_action_error") if isinstance(obs, dict) else None,
    }
    _append_jsonl(output_dir / _TIMELINE_ARTIFACT, payload)
    _append_agentlab_event(
        output_dir,
        f"step.{event}",
        runtime=runtime,
        step=step,
        phase=runtime.get("current_phase"),
        url=payload["url"],
        action=payload["action"],
        network_event_count=payload["network_event_count"],
        network_delta_count=payload.get("network_delta_count"),
    )


def _network_delta_summary(
    network: NetworkTraceRecorder,
    network_mark: int | None,
) -> dict[str, Any]:
    summarize = getattr(network, "summarize_since", None)
    if callable(summarize):
        return summarize(network_mark)
    events = getattr(network, "events", [])
    count = len(events) if isinstance(events, list) else 0
    start = network_mark if isinstance(network_mark, int) else 0
    start = max(0, min(start, count))
    return {
        "network_event_start": start,
        "network_event_end": count,
        "network_delta_count": max(0, count - start),
        "network_delta_failed_count": None,
        "network_delta_methods": [],
        "network_delta_statuses": [],
        "network_delta_latest_url": None,
        "network_delta_latest_method": None,
        "network_delta_latest_status": None,
        "network_delta_latest_resource_type": None,
    }


def _append_agentlab_event(output_dir: Path, event: str, **fields: Any) -> None:
    payload = {
        "schema_version": 1,
        "runner": "agentlab",
        "mode": "phase4",
        "event": event,
        "timestamp": _utc_now(),
        "monotonic_s": round(time.monotonic(), 6),
        **fields,
    }
    _append_jsonl(output_dir / _EVENTS_ARTIFACT, _redact_sidecar_payload(payload))


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _active_title_from_obs(obs: dict[str, Any]) -> str | None:
    titles = obs.get("open_pages_titles")
    active = obs.get("active_page_index")
    try:
        index = int(active[0]) if hasattr(active, "__getitem__") else int(active)
    except Exception:
        index = 0
    if isinstance(titles, (list, tuple)) and 0 <= index < len(titles):
        return _string_or_none(titles[index])
    return None


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@contextmanager
def _step_deadline(request: dict[str, Any], label: str):
    timeout_s = _optional_positive_float(request.get("step_timeout"))
    if timeout_s is None:
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)
    started_at = time.monotonic()

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"AgentLab {label} exceeded step timeout {timeout_s:g}s")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        previous_remaining = float(previous_timer[0] or 0)
        if previous_remaining > 0:
            elapsed = time.monotonic() - started_at
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.001, previous_remaining - elapsed),
                float(previous_timer[1] or 0),
            )


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


def _optional_positive_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"request missing required string field {key!r}")
    return value


def _recycle_cdp_browser(cdp_url: str) -> dict[str, Any]:
    try:
        from worldsim.phase_4.pvpo_beginframe import coordinator_for_pvpo_endpoint
        from worldsim.phase_4.pvpo_browser_lifecycle import recycle_pvpo_browser_after_task

        async def _recycle() -> dict[str, Any]:
            payload = await recycle_pvpo_browser_after_task(None, cdp_url)
            if payload.get("recycle_status") == "recycled":
                coordinator_for_pvpo_endpoint(cdp_url).reset_after_recycle()
            return payload

        return _run_async_in_thread(_recycle())
    except Exception as exc:
        return {
            "recycle_enabled": bool(cdp_url),
            "recycle_status": "failed",
            "recycle_method": "shared_lifecycle",
            "recycle_failure": f"{type(exc).__name__}: {exc}",
        }


def _run_async_in_thread(coro: Any) -> Any:
    result: concurrent.futures.Future[Any] = concurrent.futures.Future()

    def _worker() -> None:
        try:
            result.set_result(asyncio.run(coro))
        except Exception as exc:
            result.set_exception(exc)

    thread = threading.Thread(target=_worker, name="worldsim-agentlab-async", daemon=True)
    thread.start()
    thread.join()
    return result.result()
