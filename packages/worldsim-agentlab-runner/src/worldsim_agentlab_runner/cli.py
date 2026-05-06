from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WorldSim AgentLab sidecar runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run one AgentLab request JSON")
    run_parser.add_argument("request", type=Path)
    phase4_parser = subparsers.add_parser(
        "phase4-run",
        help="Run one WorldSim Phase 4 AgentLab request JSON",
    )
    phase4_parser.add_argument("request", type=Path)
    args = parser.parse_args(argv)

    if args.command == "run":
        result = run_request_path(args.request)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    if args.command == "phase4-run":
        from worldsim_agentlab_runner.phase4_loop import run_phase4_request_path

        result = run_phase4_request_path(args.request)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    parser.error(f"unknown command {args.command!r}")
    return 2


def run_request_path(path: Path) -> dict[str, Any]:
    request = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise ValueError("request JSON must be an object")
    return run_request(request)


def run_request(request: dict[str, Any]) -> dict[str, Any]:
    from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
    from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
    from agentlab.experiments.loop import EnvArgs, ExpArgs

    from worldsim_agentlab_runner.model_args import model_args_from_request

    output_dir = Path(_required_str(request, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _normalized_model_name(
        _required_str(request, "model"),
        provider=request.get("provider"),
    )
    chat_model_args = model_args_from_request(request)
    task_name = _required_str(request, "browsergym_task_name")
    max_steps = int(request.get("max_steps") or 30)
    task_seed = request.get("task_seed")
    if task_seed in ("", None):
        task_seed = None
    else:
        task_seed = int(task_seed)

    env_args = EnvArgs(
        task_name=task_name,
        task_seed=task_seed,
        max_steps=max_steps,
        headless=bool(request.get("headless", True)),
        storage_state=request.get("storage_state") or None,
    )
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

    start = time.monotonic()
    with _patched_env(_string_env(request.get("env_overrides"))):
        benchmark_config = _apply_benchmark_config(agent_args, request)
        exp_args = ExpArgs(agent_args=agent_args, env_args=env_args)
        exp_args.prepare(exp_root=output_dir.parent)
        exp_args.exp_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "exp_args.pkl").open("wb") as handle:
            pickle.dump(exp_args, handle)
        exp_args.run()

    elapsed = time.monotonic() - start
    summary = _load_summary(output_dir)
    err_msg = summary.get("err_msg")
    reward = float(summary.get("cum_reward") or 0.0)
    steps = int(summary.get("n_steps") or 0)
    passed = reward > 0 and err_msg is None
    terminated = bool(summary.get("terminated", False))
    truncated = bool(summary.get("truncated", False))
    status = "error" if err_msg else ("success" if passed else "failure")
    return {
        "schema_version": 1,
        "task_id": request.get("task_id"),
        "status": status,
        "passed": passed,
        "reward": reward,
        "steps": steps,
        "is_done": terminated or truncated,
        "terminated": terminated,
        "truncated": truncated,
        "elapsed": elapsed,
        "error": err_msg,
        "summary_info": summary,
        "artifacts": _artifact_manifest(output_dir),
        "versions": _versions(),
        "model": _model_result(request, model, chat_model_args),
        "benchmark_config": benchmark_config,
    }


def _apply_benchmark_config(agent_args: Any, request: dict[str, Any]) -> dict[str, Any]:
    """Apply the same benchmark action/tab settings AgentLab Study applies.

    AgentLab's standard study path calls ``agent_args.set_benchmark`` before
    creating experiments. The sidecar bypasses ``Study`` so it must reproduce
    that semantic hook explicitly, otherwise BrowserGym can receive the wrong
    action grammar.
    """

    benchmark_name = str(request.get("benchmark_name") or "").strip()
    if not benchmark_name:
        return {"status": "missing_benchmark_name"}

    try:
        import bgym
    except ImportError as exc:
        raise RuntimeError("bgym is required to apply AgentLab benchmark config") from exc

    factory = getattr(bgym, "DEFAULT_BENCHMARKS", {}).get(benchmark_name)
    if not callable(factory):
        raise ValueError(f"AgentLab benchmark {benchmark_name!r} is not available in bgym")

    benchmark = factory()
    agent_args.set_benchmark(benchmark, demo_mode=bool(request.get("demo_mode", False)))
    action_args = getattr(getattr(agent_args, "flags", None), "action", None)
    action_set = getattr(action_args, "action_set", None)
    obs_flags = getattr(getattr(agent_args, "flags", None), "obs", None)
    return {
        "status": "applied",
        "benchmark_name": getattr(benchmark, "name", benchmark_name),
        "is_multi_tab": bool(getattr(benchmark, "is_multi_tab", False)),
        "obs_use_tabs": bool(getattr(obs_flags, "use_tabs", False)),
        "action_set": repr(action_set),
        "action_set_type": type(action_set).__name__ if action_set is not None else None,
    }


def _model_result(
    request: dict[str, Any],
    normalized_model: str,
    chat_model_args: Any,
) -> dict[str, Any]:
    return {
        "requested_model": request.get("requested_model", request.get("model")),
        "requested_provider": request.get("requested_provider"),
        "normalized_model": normalized_model,
        "provider": request.get("provider"),
        "service_tier": request.get("service_tier"),
        "profile": request.get("model_profile") or {},
        "transport": getattr(chat_model_args, "transport", None),
        "transport_model": getattr(chat_model_args, "model_name", None),
        "required_env_var": getattr(chat_model_args, "required_env_var", None),
        "temperature": getattr(chat_model_args, "temperature", None),
        "max_new_tokens": getattr(chat_model_args, "max_new_tokens", None),
        "vision_support": getattr(chat_model_args, "vision_support", None),
        "extra_body": getattr(chat_model_args, "extra_body", None),
    }


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"request missing required string field {key!r}")
    return value


def _normalized_model_name(model: str, *, provider: object = None) -> str:
    """Return a LiteLLM model name for direct or parent-built sidecar requests."""

    model = model.strip()
    provider_name = str(provider or "").strip().lower()
    if "/" in model or provider_name in ("", "auto"):
        return model
    prefixes = {
        "openai": "openai",
        "anthropic": "anthropic",
        "google": "gemini",
        "openrouter": "openrouter",
    }
    prefix = prefixes.get(provider_name)
    return f"{prefix}/{model}" if prefix else model


def _default_temperature(model: str) -> float | None:
    """Pick AgentLab-compatible sampling defaults for model families."""

    if model.startswith("openai/gpt-5") or model.startswith("gpt-5"):
        # GPT-5-family chat routes reject most explicit temperature values.
        return None
    return 0


def _string_env(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): str(item)
        for key, item in value.items()
        if isinstance(key, str) and key and item not in (None, "")
    }


@contextmanager
def _patched_env(overrides: dict[str, str]):
    old_values = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _load_summary(output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "summary_info.json"
    if not summary_path.exists():
        raise RuntimeError(f"AgentLab did not write {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"AgentLab summary at {summary_path} is not a JSON object")
    return payload


def _artifact_manifest(output_dir: Path) -> dict[str, Any]:
    files = {
        "summary_info": output_dir / "summary_info.json",
        "experiment_log": output_dir / "experiment.log",
        "exp_args": output_dir / "exp_args.pkl",
        "agentlab_native_exp_args": output_dir / "agentlab_native_exp_args.pkl",
        "model_calls": output_dir / "worldsim_model_calls.jsonl",
        "package_versions": output_dir / "package_versions.txt",
        "goal_object": output_dir / "goal_object.pkl.gz",
        "history": output_dir / "history.json",
        "final_response": output_dir / "final_response.json",
        "network_trace": output_dir / "network_trace.json",
        "network_har": output_dir / "network.har",
        "navigation_trace": output_dir / "navigation_trace.json",
        "browser_runtime": output_dir / "browser_runtime.json",
        "needham_trace": output_dir / "needham_trace.json",
        "needham_xml": output_dir / "needham_trace.xml",
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


def _versions() -> dict[str, str | None]:
    packages = [
        "agentlab",
        "browsergym",
        "browsergym-core",
        "browsergym-experiments",
        "browsergym-webarena",
        "browsergym-webarena-verified",
        "openai",
        "litellm",
    ]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _default_obs_flags() -> Any:
    from agentlab.agents import dynamic_prompting as dp

    return dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=True,
        use_diff=False,
        use_screenshot=True,
        use_som=False,
        extract_visible_tag=True,
        extract_clickable_tag=True,
        extract_coords="False",
        filter_visible_elements_only=False,
    )


def _default_action_flags() -> Any:
    from agentlab.agents import dynamic_prompting as dp
    from bgym import HighLevelActionSetArgs

    return dp.ActionFlags(
        # AgentLab's Study path overwrites this via set_benchmark(). Keep this
        # fallback only for malformed direct requests that fail before then.
        action_set=HighLevelActionSetArgs(subsets=["bid"], multiaction=False),
        long_description=False,
        individual_examples=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
