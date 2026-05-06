"""CLI helpers for AgentLab/BrowserGym comparison runs."""

from __future__ import annotations

import asyncio
import json
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

from worldsim.agent_config import (
    bind_task_to_instance,
    instances_for_site,
    prepare_tasks_for_execution,
)
from worldsim.agent_models import supported_agentlab_model_profiles
from worldsim.config import BenchmarkConfig, BenchmarkInstance, load_benchmark_config
from worldsim.placeholders import normalize_site_name
from worldsim.runners.agentlab import AgentLabAgentWrapper, make_task_runner


def run(args: Namespace) -> int:
    """Run one AgentLab/BrowserGym comparison task from CLI args."""

    try:
        result = asyncio.run(_run_async(args))
    except Exception as exc:
        print(f"AgentLab comparison run failed: {exc}", file=sys.stderr)
        return 1

    if getattr(args, "json", False):
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            "AgentLab comparison "
            f"{result.get('status', 'unknown')}: task_id={result.get('task_id', 'unknown')} "
            f"passed={bool(result.get('passed', False))} "
            f"steps={int(result.get('steps', 0))} "
            f"message={result.get('message', '')}"
        )
        print(f"Artifacts: {result.get('trajectory_dir')}")
    return 1 if result.get("status") == "error" else 0


def models(args: Namespace) -> int:
    """Print the named AgentLab model profiles WorldSim knows how to route."""

    profiles = supported_agentlab_model_profiles()
    if getattr(args, "json", False):
        print(json.dumps([profile.to_sidecar_dict() for profile in profiles], sort_keys=True))
        return 0

    for profile in profiles:
        aliases = ", ".join(profile.aliases[:3])
        temp = "omitted" if profile.temperature is None else str(profile.temperature)
        print(
            f"{profile.key:12} {profile.display_name:22} "
            f"{profile.provider:10} {profile.model:32} "
            f"temp={temp:7} max_new={profile.max_new_tokens} "
            f"vision={str(profile.vision_support).lower()} aliases={aliases}"
        )
    return 0


async def _run_async(args: Namespace) -> dict[str, Any]:
    config = load_benchmark_config(args.instances)
    task = _task_from_args(args, config)
    instance = _select_instance(config, task, args)
    prepared_task = _prepare_single_task(task, config)
    bound_task = bind_task_to_instance(prepared_task, instance, config.instances)
    task_dir = _task_dir(args, bound_task)
    runner = make_task_runner(
        attack_mode=args.attack_mode,
        benchmark_prefix=args.benchmark_prefix,
        max_steps=args.max_steps,
    )
    agent = AgentLabAgentWrapper(
        model=args.agent_model,
        provider=args.agent_provider,
        service_tier=getattr(args, "agent_service_tier", None),
    )
    return await runner(bound_task, agent, instance, task_dir)


def _task_from_args(args: Namespace, config: BenchmarkConfig) -> dict[str, Any]:
    if args.task_json is not None:
        task = _load_task_json(args.task_json)
    else:
        if not args.browsergym_task_name:
            raise ValueError("--task-json or --browsergym-task-name is required")
        task_id = args.task_id or args.browsergym_task_name.replace("/", "_").replace(".", "_")
        task = {
            "id": task_id,
            "benchmark_name": args.benchmark_name or config.benchmark_name,
            "agentlab_task_name": args.browsergym_task_name,
            "data_seed": {"mechanism": "none"},
        }

    task = json.loads(json.dumps(task))
    task.setdefault("benchmark_name", args.benchmark_name or config.benchmark_name)
    if args.browsergym_task_name:
        task["agentlab_task_name"] = args.browsergym_task_name
    if args.task_id:
        task["id"] = args.task_id
    if args.site:
        task["site"] = normalize_site_name(args.site)
        task["sites"] = [normalize_site_name(args.site)]
    elif not task.get("site") and not task.get("sites") and len(config.instances) == 1:
        task["site"] = normalize_site_name(config.instances[0].site_name)
        task["sites"] = [normalize_site_name(config.instances[0].site_name)]
    return task


def _load_task_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError("--task-json list inputs must contain exactly one task")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError("--task-json must contain one JSON object")
    return payload


def _select_instance(
    config: BenchmarkConfig,
    task: dict[str, Any],
    args: Namespace,
) -> BenchmarkInstance:
    site = normalize_site_name(args.site or task.get("site", ""))
    if not site:
        sites = task.get("sites")
        if isinstance(sites, list) and sites:
            site = normalize_site_name(str(sites[0]))
    candidates = instances_for_site(config.instances, site) if site else list(config.instances)
    if args.replica_name:
        candidates = [
            instance for instance in candidates if instance.replica_name == args.replica_name
        ]
    if args.replica_index is not None:
        candidates = [
            instance for instance in candidates if instance.replica_index == args.replica_index
        ]
    if not candidates:
        raise ValueError(f"no configured instance matches site={site!r}")
    if not site and len(candidates) > 1:
        raise ValueError("multiple instances are configured; pass --site")
    return candidates[0]


def _prepare_single_task(task: dict[str, Any], config: BenchmarkConfig) -> dict[str, Any]:
    prepared, errors = prepare_tasks_for_execution(
        [task],
        config.instances,
        config_url_placeholders=config.url_placeholders,
    )
    if errors:
        messages = "; ".join(str(error.get("message", error)) for error in errors)
        raise ValueError(messages)
    if len(prepared) != 1:
        raise ValueError("expected exactly one prepared task")
    return prepared[0]


def _task_dir(args: Namespace, task: dict[str, Any]) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    task_id = str(task.get("id", task.get("task_id", "unknown")))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs") / "agentlab_comparison" / timestamp / task_id
