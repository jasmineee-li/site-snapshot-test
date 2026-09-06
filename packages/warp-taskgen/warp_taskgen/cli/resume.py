"""Resume flow for the WARP Taskgen CLI."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from warp_taskgen.agent_runtime import RUNNER_BROWSER_USE
from warp_taskgen.cli import dispatch as _dispatch
from warp_taskgen.cli.argument_defaults import DEFAULT_SANDBOX_MODEL
from warp_taskgen.cli.resume_plan import dispatch_resume_plan

# Ordered pipeline steps. Each entry maps step name -> (phase_id, sub) where
# sub is the sub-step for Phase 0, or None for later phases.
_PHASE_ORDER: list[str] = [
    "phase_0a",
    "phase_0b",
    "phase_0c",
    "phase_0d",
    "phase_1",
    "phase_2",
    "phase_3",
    "phase_4",
]


def _next_step(step: str) -> str | None:
    """Return the step after ``step``, or None if ``step`` is the last."""
    try:
        idx = _PHASE_ORDER.index(step)
    except ValueError:
        return None
    if idx + 1 < len(_PHASE_ORDER):
        return _PHASE_ORDER[idx + 1]
    return None


def _dispatch_resume(args: argparse.Namespace) -> int:
    """Read last checkpoint and dispatch to the appropriate phase."""
    if getattr(args, "command", None) == "derive-and-resume":
        from warp_taskgen.cli.derived_run import dispatch_derived_resume

        return dispatch_derived_resume(args)

    from warp_taskgen.state import load_state

    state = load_state()
    if state is None:
        print("No pipeline state found; run a phase first.", file=sys.stderr)
        return 1
    if getattr(args, "plan", False):
        return dispatch_resume_plan(args, state)
    from warp_taskgen.cli.run_identity import resume_state_inputs

    try:
        state = resume_state_inputs(state)
    except ValueError as exc:
        print(f"Resume rejected by Run Definition: {exc}", file=sys.stderr)
        return 2

    last_step = state.get("step", "")
    status = state.get("status", "")
    logs_dir = state.get("logs_dir")

    if (
        logs_dir
        and not os.environ.get("WARP_TASKGEN_STATE_DIR")
        and not os.environ.get("WORLDSIM_STATE_DIR")
    ):
        os.environ["WORLDSIM_STATE_DIR"] = str(logs_dir)

    pipeline_finished = False
    if status in {"complete", "partial_complete"}:
        target = _next_step(last_step)
        if target is None:
            pipeline_finished = True
            target = last_step
        else:
            qualifier = "partial and " if status == "partial_complete" else ""
            print(f"Last checkpoint: {last_step} {qualifier}complete. Resuming from {target}.")
    elif status == "running":
        target = last_step
        print(f"Last checkpoint: {last_step} was running (likely crashed). Re-running {target}.")
    elif status == "failed":
        target = last_step
        reason = state.get("reason")
        suffix = f" ({reason})" if reason else ""
        print(f"Last checkpoint: {last_step} failed{suffix}. Re-running {target}.")
    elif status in {"paused", "interrupted"}:
        target = last_step
        print(f"Last checkpoint: {last_step} was {status}. Re-running {target}.")
    else:
        print(f"Last checkpoint: {last_step} has unknown status {status!r}.", file=sys.stderr)
        return 1

    if state.get("process_pool"):
        if pipeline_finished:
            print(f"Last checkpoint: {last_step} complete. Pipeline finished — nothing to resume.")
            return 0
        try:
            from warp_taskgen.phase_4.process_pool_control import process_pool_resume_command

            command = process_pool_resume_command(state)
        except ValueError as exc:
            print(f"Process-pool resume rejected: {exc}", file=sys.stderr)
            return 2
        if status != "paused":
            print(
                "Process-pool crash recovery remains fail-closed; inspect or repair its partial "
                "artifacts instead of dispatching normal Phase 4.",
                file=sys.stderr,
            )
            return 2
        print(
            f"Process-pool roots resume through the isolated supervisor wrapper:\n  {command}",
            file=sys.stderr,
        )
        return 2

    # Build a synthetic argparse.Namespace that _dispatch_phase understands.
    # CLI flags override state metadata; state metadata fills gaps.
    benchmark = getattr(args, "benchmark", None)
    config = getattr(args, "config", None)
    instances = getattr(args, "instances", None)
    agent_model = getattr(args, "agent_model", None)
    agent_runner = getattr(args, "runner", None)
    sandbox_model = getattr(args, "sandbox_model", None)
    agent_provider = getattr(args, "agent_provider", None)
    agent_service_tier = getattr(args, "agent_service_tier", None)
    agent_llm_timeout = getattr(args, "agent_llm_timeout", None)
    agent_step_timeout = getattr(args, "agent_step_timeout", None)
    agent_task_timeout = getattr(args, "agent_task_timeout", None)
    phase_4_max_workers = getattr(args, "phase_4_max_workers", None)
    phase_4_variant_budget = getattr(args, "phase_4_variant_budget", None)
    phase_4_variant_system = getattr(args, "phase_4_variant_system", None)
    phase_4_eval_awareness_max_iterations = getattr(
        args, "phase_4_eval_awareness_max_iterations", None
    )
    skip_intermediate_asr = getattr(args, "skip_intermediate_asr", None)
    intermediate_asr_max_steps_per_task = getattr(args, "intermediate_asr_max_steps_per_task", None)
    generate_novel = getattr(args, "generate_novel", None)
    novel_tasks_per_site = getattr(args, "novel_tasks_per_site", None)
    task_card_plan = getattr(args, "task_card_plan", None)
    task_capability_profile = getattr(args, "task_capability_profile", None)
    phase_1_action_counts = getattr(args, "phase_1_action_counts", None)
    max_tasks_per_site = getattr(args, "max_tasks_per_site", None)
    task_origin = getattr(args, "task_origin", None)
    sites = getattr(args, "sites", None)
    phase_2b_texts_per_plan = getattr(args, "phase_2b_texts_per_plan", None)
    phase_2_text_fill_concurrency = getattr(args, "phase_2_text_fill_concurrency", None)
    phase_2_text_model = getattr(args, "phase_2_text_model", None)
    phase_2a_action_policy = getattr(args, "phase_2a_action_policy", None)

    skip_feasibility = getattr(args, "skip_feasibility", None)
    feasibility_only = getattr(args, "feasibility_only", None)
    feasibility_instances = getattr(args, "feasibility_instances", None)
    feasibility_concurrency = getattr(args, "feasibility_concurrency", None)
    feasibility_retry_count = getattr(args, "feasibility_retry_count", None)
    feasibility_ttl_hours = getattr(args, "feasibility_ttl_hours", None)
    force_reverify = getattr(args, "force_reverify", None)
    no_l3_l4 = getattr(args, "no_l3_l4", None)

    # Fall back to paths stored in state metadata
    if benchmark is None and "benchmark_path" in state:
        benchmark = Path(state["benchmark_path"])
    if config is None and "manifest_path" in state:
        config = Path(state["manifest_path"])
    if instances is None and "instances_path" in state:
        instances = Path(state["instances_path"])
    if agent_model is None:
        agent_model = state.get("agent_model")
    if agent_runner is None:
        agent_runner = state.get("agent_runner", RUNNER_BROWSER_USE)
    if sandbox_model is None:
        sandbox_model = state.get("sandbox_model", DEFAULT_SANDBOX_MODEL)
    if agent_provider is None:
        agent_provider = state.get("agent_provider")
    if agent_service_tier is None:
        agent_service_tier = state.get("agent_service_tier")
    if agent_llm_timeout is None:
        agent_llm_timeout = state.get("agent_llm_timeout")
    if agent_step_timeout is None:
        agent_step_timeout = state.get("agent_step_timeout")
    if agent_task_timeout is None:
        agent_task_timeout = state.get("agent_task_timeout")
    if phase_4_max_workers is None:
        phase_4_max_workers = state.get("phase_4_max_workers")
    if phase_4_variant_budget is None:
        phase_4_variant_budget = state.get("phase_4_variant_budget")
    if phase_4_variant_system is None:
        phase_4_variant_system = state.get("phase_4_variant_system")
    if (
        target == "phase_4"
        and phase_4_variant_system is None
        and "phase_4_variant_system" not in state
        and state.get("step") == "phase_4"
    ):
        # Pre-iterator Phase 4 states did not persist this field. Preserve the
        # behavior those runs started with instead of silently resuming into the
        # new default.
        phase_4_variant_system = "strategy-variation"
    if phase_4_eval_awareness_max_iterations is None:
        phase_4_eval_awareness_max_iterations = state.get("phase_4_eval_awareness_max_iterations")
    if skip_intermediate_asr is None:
        skip_intermediate_asr = state.get("skip_intermediate_asr", False)
    if intermediate_asr_max_steps_per_task is None:
        intermediate_asr_max_steps_per_task = state.get("intermediate_asr_max_steps_per_task")
    if generate_novel is None:
        generate_novel = state.get("generate_novel", False)
    if novel_tasks_per_site is None:
        novel_tasks_per_site = state.get("novel_tasks_per_site")
    if task_card_plan is None and "task_card_plan_path" in state:
        raw_task_card_plan = state.get("task_card_plan_path")
        if raw_task_card_plan:
            task_card_plan = Path(raw_task_card_plan)
    if task_capability_profile is None:
        task_capability_profile = state.get("task_capability_profile")
    if phase_1_action_counts is None:
        phase_1_action_counts = state.get("phase_1_action_counts", state.get("action_counts"))
    if task_origin is None:
        task_origin = state.get("task_origin")
    if sites is None:
        sites = state.get("sites")
    if phase_2b_texts_per_plan is None:
        phase_2b_texts_per_plan = state.get("phase_2b_texts_per_plan")
    if phase_2_text_fill_concurrency is None:
        phase_2_text_fill_concurrency = state.get("phase_2_text_fill_concurrency")
    if phase_2_text_model is None:
        phase_2_text_model = state.get("phase_2_text_model")
    if phase_2a_action_policy is None:
        phase_2a_action_policy = state.get("phase_2a_action_policy")
    if skip_feasibility is None:
        skip_feasibility = state.get("skip_feasibility", False)
    if feasibility_only is None:
        feasibility_only = state.get("feasibility_only", False)
    if feasibility_instances is None:
        feasibility_instances = state.get("feasibility_instances")
    if feasibility_concurrency is None:
        feasibility_concurrency = state.get("feasibility_concurrency", 10)
    if feasibility_retry_count is None:
        feasibility_retry_count = state.get("feasibility_retry_count", 1)
    if feasibility_ttl_hours is None:
        feasibility_ttl_hours = state.get("feasibility_ttl_hours")
    if force_reverify is None:
        force_reverify = state.get("force_reverify", False)
    if no_l3_l4 is None:
        resolution_sig = state.get("phase_2a_resolution_signature")
        if isinstance(resolution_sig, dict):
            no_l3_l4 = bool(resolution_sig.get("no_l3_l4", False))
        else:
            no_l3_l4 = False

    # Map target step to phase ID for _dispatch_phase (e.g. "phase_0a" -> "0a")
    phase_id = target.replace("phase_", "")

    allow_unknown_auth = getattr(args, "allow_unknown_auth", None)
    if allow_unknown_auth is None:
        allow_unknown_auth = state.get("allow_unknown_auth", False)
    skip_host_bound_storage_state_auth = getattr(args, "skip_host_bound_storage_state_auth", None)
    if skip_host_bound_storage_state_auth is None:
        skip_host_bound_storage_state_auth = state.get("skip_host_bound_storage_state_auth", False)

    synthetic = argparse.Namespace(
        command="phase",
        phase=phase_id,
        resume=True,
        benchmark=benchmark,
        config=config,
        instances=instances,
        agent_model=agent_model,
        runner=agent_runner,
        sandbox_model=sandbox_model,
        agent_provider=agent_provider,
        agent_service_tier=agent_service_tier,
        agent_llm_timeout=agent_llm_timeout,
        agent_step_timeout=agent_step_timeout,
        agent_task_timeout=agent_task_timeout,
        phase_4_max_workers=phase_4_max_workers,
        phase_4_variant_budget=phase_4_variant_budget,
        phase_4_variant_system=phase_4_variant_system,
        phase_4_eval_awareness_max_iterations=phase_4_eval_awareness_max_iterations,
        skip_intermediate_asr=skip_intermediate_asr,
        intermediate_asr_max_steps_per_task=intermediate_asr_max_steps_per_task,
        generate_novel=generate_novel,
        novel_tasks_per_site=novel_tasks_per_site,
        task_card_plan=task_card_plan,
        task_capability_profile=task_capability_profile,
        phase_1_action_counts=phase_1_action_counts,
        max_tasks_per_site=max_tasks_per_site,
        task_origin=task_origin,
        sites=sites,
        phase_2b_texts_per_plan=phase_2b_texts_per_plan,
        phase_2_text_fill_concurrency=phase_2_text_fill_concurrency,
        phase_2_text_model=phase_2_text_model,
        phase_2a_action_policy=phase_2a_action_policy,
        allow_unknown_auth=allow_unknown_auth,
        skip_host_bound_storage_state_auth=skip_host_bound_storage_state_auth,
        skip_feasibility=skip_feasibility,
        feasibility_only=feasibility_only,
        feasibility_instances=feasibility_instances,
        feasibility_concurrency=feasibility_concurrency,
        feasibility_retry_count=feasibility_retry_count,
        feasibility_ttl_hours=feasibility_ttl_hours,
        force_reverify=force_reverify,
        no_l3_l4=no_l3_l4,
    )

    from warp_taskgen.cli.run_identity import resolve_cli_run_transition

    try:
        transition = resolve_cli_run_transition(
            args,
            existing_state=state,
        )
    except ValueError as exc:
        print(f"Resume rejected by Run Definition: {exc}", file=sys.stderr)
        return 2
    if transition.kind == "derived_required":
        fields = ", ".join(transition.drift_fields) or "unknown inputs"
        print(
            "Resume requires an isolated Derived Run and explicit "
            "`warp-taskgen derive-and-resume` intent "
            f"({transition.reason_code}; changed: {fields}).",
            file=sys.stderr,
        )
        return 2
    if status in {"complete", "partial_complete", "failed", "paused", "interrupted"}:
        from warp_taskgen.run_control import clear_pause_request
        from warp_taskgen.state import get_state_dir

        clear_pause_request(Path(str(state.get("logs_dir") or get_state_dir())))
    if pipeline_finished:
        print(f"Last checkpoint: {last_step} complete. Pipeline finished — nothing to resume.")
        return 0
    synthetic._run_transition = transition

    try:
        _dispatch._install_verification_proxy_from_args(synthetic)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return _dispatch._dispatch_phase(synthetic)


__all__ = ["_PHASE_ORDER", "_dispatch_resume", "_next_step"]
