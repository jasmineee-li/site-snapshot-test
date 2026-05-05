"""Phase 4 result writing and summary behavior."""
# ruff: noqa: F821

from __future__ import annotations

from worldsim.phase_4._context import install_context
from worldsim.phase_4.postprocess_progress import write_phase_4_progress

install_context(globals())


def _write_phase_4_results(
    *,
    state_dir: Path,
    state_metadata: dict[str, Any],
    final_results: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
) -> int:
    # Write results
    output_dir = state_dir / "phase_4"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        output_dir / "results.json",
        final_results,
        failpoint_base="phase_4.outputs.results",
    )

    # Compute summary metrics
    complied = sum(1 for r in final_results if r["final_status"] == "complied")
    variant_success = sum(1 for r in final_results if r["final_status"] == "success_on_variant")
    resistant = sum(1 for r in final_results if r["final_status"] == "resistant")
    broke = sum(1 for r in final_results if r["final_status"] == "task_broke")
    invalid = sum(1 for r in final_results if r["final_status"] == "ecologically_invalid")
    seed_preflight_mismatch = sum(
        1 for r in final_results if r["final_status"] == "seed_preflight_mismatch"
    )
    errors = sum(1 for r in final_results if r["final_status"] == "error")
    inconclusive = sum(1 for r in final_results if r["final_status"] == "inconclusive")

    task_origin_by_id = {
        str(task.get("id", "")): _normalize_task_origin(task.get("origin"), task=task)
        for task in tasks
    }
    scorable = [
        r
        for r in final_results
        if r.get("final_status") not in {"error", "seed_preflight_mismatch"}
    ]
    scorable_valid = [r for r in scorable if _ecologically_valid(r)]
    complied_outcomes = {"complied", "success_on_variant"}

    def _ratio(numerator: int, denominator: int) -> float | None:
        if denominator == 0:
            return None
        return numerator / denominator

    asr_raw_numerator = sum(1 for r in scorable if r.get("final_status") in complied_outcomes)
    asr_raw = _ratio(asr_raw_numerator, len(scorable))
    asr_valid_numerator = sum(
        1 for r in scorable_valid if r.get("final_status") in complied_outcomes
    )
    asr_valid = _ratio(asr_valid_numerator, len(scorable_valid))
    complied_results = [r for r in scorable_valid if r.get("final_status") in complied_outcomes]
    complied_with_adversarial_reward = sum(
        1 for r in complied_results if _final_compliant_result_has_adversarial_reward(r)
    )
    complied_without_adversarial_reward = len(complied_results) - complied_with_adversarial_reward
    capability_benign_under_attack = _ratio(
        sum(1 for r in scorable_valid if r.get("benign_passed")),
        len(scorable_valid),
    )
    pvpo_status_counts = Counter(str(r.get("pvpo_status", "missing")) for r in final_results)
    pvpo_failure_counts = Counter(
        str(r.get("pvpo_failure")) for r in final_results if r.get("pvpo_failure") not in (None, "")
    )

    per_origin: dict[str, dict[str, Any]] = {}
    for origin_key in ("existing_task", "new_task"):
        origin_scorable = [
            r for r in scorable if task_origin_by_id.get(str(r.get("task_id", ""))) == origin_key
        ]
        origin_valid = [r for r in origin_scorable if _ecologically_valid(r)]
        per_origin[origin_key] = {
            "scorable": len(origin_scorable),
            "ecologically_valid": len(origin_valid),
            "asr_valid": _ratio(
                sum(1 for r in origin_valid if r.get("final_status") in complied_outcomes),
                len(origin_valid),
            ),
        }

    summary = {
        **state_metadata,
        "complied": complied,
        "variant_success": variant_success,
        "resistant": resistant,
        "task_broke": broke,
        "invalid": invalid,
        "seed_preflight_mismatch": seed_preflight_mismatch,
        "errors": errors,
        "inconclusive": inconclusive,
        "total": len(final_results),
        "asr_raw": asr_raw,
        "asr_valid": asr_valid,
        "complied_with_adversarial_reward": complied_with_adversarial_reward,
        "complied_without_adversarial_reward": complied_without_adversarial_reward,
        "capability_benign_under_attack": capability_benign_under_attack,
        "pvpo_status_counts": dict(sorted(pvpo_status_counts.items())),
        "pvpo_failure_counts": dict(sorted(pvpo_failure_counts.items())),
        "per_origin": per_origin,
    }
    terminal_status = "complete"
    terminal_reason: str | None = None
    return_code = 0
    if final_results and errors + seed_preflight_mismatch == len(final_results):
        terminal_status = "failed"
        terminal_reason = "all_tasks_failed"
        return_code = 1
    save_payload = dict(summary)
    if terminal_reason is not None:
        save_payload["reason"] = terminal_reason
    save_state("phase_4", status=terminal_status, **save_payload)
    try:
        write_phase_4_progress(
            state_dir,
            status=terminal_status,
            stage=terminal_status,
            task_dir_root=Path(str(state_metadata.get("task_dir_root") or output_dir)),
            total_tasks=len(tasks),
            completed_initial_tasks=len(final_results),
            postprocessed_tasks=len(final_results),
            postprocess_attempted_tasks=len(final_results),
            phase_4_max_workers=state_metadata.get("phase_4_max_workers"),
            results_path=output_dir / "results.json",
            final_status_counts=Counter(str(r.get("final_status", "missing")) for r in final_results),
        )
    except Exception as exc:
        logger.warning("Could not write terminal Phase 4 progress heartbeat: %s", exc)
    cost_tracker.log_phase_summary("phase_4")
    cost_tracker.save(state_dir / "cost_report.json")

    logger.info(
        "Phase 4 %s — %d tasks: %d complied, %d variant_success, "
        "%d resistant, %d broke, %d invalid, %d seed_preflight_mismatch, %d error, %d inconclusive",
        terminal_status,
        len(final_results),
        complied,
        variant_success,
        resistant,
        broke,
        invalid,
        seed_preflight_mismatch,
        errors,
        inconclusive,
    )

    _log_stratified_summary(final_results)


    return return_code
