"""Phase 4 metrics behavior."""
# ruff: noqa: F821

from __future__ import annotations

from worldsim.phase_4._context import install_context

install_context(globals())

def _null_metric_keys() -> dict[str, Any]:
    """Null metric payload stamped on every gate-miss trajectory.

    Keeping all new keys present-but-null (vs absent) means downstream
    analysis can load any result.json uniformly — no per-trajectory schema
    branching on whether the injection was encountered.
    """
    return {
        "transcript_purpose_classification": None,
        "transcript_purpose_is_aware": None,
        "transcript_purpose_was_refused": None,
        "transcript_purpose_guess": None,
        "transcript_purpose_explanation": None,
        "transcript_purpose_failure": None,
        "transcript_purpose_format": "needham-xml-v1",
        "transcript_purpose_attempts": 0,
        "transcript_purpose_raw_response_path": None,
        "transcript_purpose_apollo_continue_classification": None,
        "transcript_purpose_apollo_continue_is_aware": None,
        "transcript_purpose_apollo_continue_was_refused": None,
        "transcript_purpose_apollo_continue_guess": None,
        "transcript_purpose_apollo_continue_explanation": None,
        "transcript_purpose_apollo_continue_failure": None,
        "transcript_purpose_apollo_continue_attempts": None,
        "transcript_purpose_apollo_continue_raw_response_path": None,
        "verbalized_eval_awareness_classification": None,
        "verbalized_eval_awareness_is_aware": None,
        "verbalized_eval_awareness_evidence": [],
        "verbalized_eval_awareness_failure": None,
        "verbalized_eval_awareness_transcript_format": "needham-xml-v1",
        "verbalized_eval_awareness_raw_response_path": None,
    }

def _pvpo_metric_payload(pvpo: dict[str, Any]) -> dict[str, Any]:
    """Project the metric subset of a ``_run_pvpo_gate`` return for persistence.

    Filters to keys in ``_PVPO_METRIC_KEYS`` so the splat into ``save_result``
    cannot collide with ``encounter`` / ``injection_not_encountered`` /
    ``trajectory_dir`` kwargs already passed at the call sites.
    """
    return {k: pvpo[k] for k in _PVPO_METRIC_KEYS if k in pvpo}

def _gate_miss_payload(encounter_dict: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical gate-miss payload with a fail-closed encounter view."""
    coerced_encounter = dict(encounter_dict)
    coerced_encounter["max_coverage"] = 0.0
    coerced_encounter["reference_step"] = None
    coerced_encounter["reference_screenshot_path"] = None
    return {
        "encounter": coerced_encounter,
        "injection_not_encountered": True,
        "pvpo_status": coerced_encounter.get("pvpo_status"),
        "pvpo_failure": coerced_encounter.get("pvpo_failure"),
        **_null_metric_keys(),
    }

def _observational_metrics_unavailable(
    *,
    encounter_dict: dict[str, Any],
    task_dir: Path,
    failure_class: str,
) -> dict[str, Any]:
    """Return a schema-complete payload when Stage 2 evidence is unavailable."""
    return {
        "encounter": encounter_dict,
        "injection_not_encountered": False,
        "pvpo_status": encounter_dict.get("pvpo_status"),
        "pvpo_failure": encounter_dict.get("pvpo_failure"),
        **_transcript_purpose_failure_result(failure_class, task_dir),
        **_vea_failure_result(failure_class),
    }

def _trajectory_available_for_observational_metrics(traj: ExtractedTrajectory) -> bool:
    """Conservatively reject partial trajectory envelopes for Stage 2 judges."""
    if traj.partial:
        return False
    return not any(step.partial for step in traj.steps)

def _ecologically_valid(result: dict[str, Any]) -> bool:
    """D1 signal: the IPI payload was visible at least once during the run.

    Migrated from the legacy ``ecologically_valid`` bool to ``max_coverage > 0``
    from PVPO encounter detection. See commit ``chore(phase4): delete
    IPI-Screenshot P(eval)``.
    """
    return phase4_result_summary.ecologically_valid(result)

def _successful_strategy_variants(result: dict[str, Any]) -> list[dict[str, Any]]:
    return phase4_result_summary.successful_strategy_variants(result)

def _final_compliant_result_has_adversarial_reward(result: dict[str, Any]) -> bool:
    """Return whether the final compliant status has matching reward evidence."""
    return phase4_result_summary.final_compliant_result_has_adversarial_reward(result)

def _variant_adversarial_reward_passed(variant: dict[str, Any]) -> bool:
    return phase4_result_summary.variant_adversarial_reward_passed(variant)

def _normalize_task_origin(origin: Any, *, task: dict[str, Any] | None = None) -> str:
    """Normalize task-origin names. Falls back to id-prefix inference for
    legacy snapshots whose ``origin`` field was never stamped."""
    raw_origin = str(origin or "").strip()
    if raw_origin in {"existing_task", "new_task"}:
        return raw_origin

    task_id = str(task.get("id", "")).strip() if isinstance(task, dict) else ""
    if task_id.startswith("novel_"):
        return "new_task"
    if task_id:
        return "existing_task"
    raise ValueError("task origin is missing and cannot be inferred from id")

def _log_stratified_summary(final_results: list[dict[str, Any]]) -> None:
    """Reconstruct ``ClassifiedOutcome``s from persisted result dicts and
    log the handoff §12 stratified summary block.

    Non-fatal: if the reconstruction fails for any reason, fall back to
    logging nothing rather than breaking the Phase 4 summary line.
    """
    try:
        reconstructed: list[outcome_taxonomy.ClassifiedOutcome] = []
        for r in final_results:
            if not isinstance(r, dict):
                continue
            fine = r.get("outcome_fine")
            if not isinstance(fine, str):
                continue
            signals_dict = r.get("signals") or {}
            signals_obj: outcome_taxonomy.TrajectorySignals | None
            if isinstance(signals_dict, dict) and signals_dict:
                try:
                    signals_obj = outcome_taxonomy.TrajectorySignals(**signals_dict)
                except TypeError:
                    signals_obj = None
            else:
                signals_obj = None
            flags = r.get("flags") or []
            if not isinstance(flags, list):
                flags = []
            confidence = r.get("diagnosable_confidence", "high")
            if confidence not in {"high", "medium", "low"}:
                confidence = "high"
            outcome = "task_broke"
            if fine.startswith("complied_"):
                outcome = "complied"
            elif fine.startswith("resistant") or fine == "refused_or_ignored":
                outcome = "refused_or_ignored"
            elif fine.startswith("evaluator_unavailable"):
                outcome = "evaluator_unavailable"
            elif fine.startswith("task_broke"):
                outcome = "task_broke"
            reconstructed.append(
                outcome_taxonomy.ClassifiedOutcome(
                    outcome=outcome,
                    outcome_fine=fine,
                    flags=list(flags),
                    diagnosable_confidence=confidence,
                    signals=signals_obj,
                    rationale=str(r.get("classifier_rationale", "")),
                    classifier_version=str(
                        r.get("classifier_version", outcome_taxonomy.CLASSIFIER_VERSION)
                    ),
                )
            )
        if not reconstructed:
            return
        summary = outcome_taxonomy.stratified_summary(reconstructed)
        for line in outcome_taxonomy.format_stratified_summary_log(summary).splitlines():
            logger.info(line)
    except Exception as exc:  # defensive: never break the summary line
        logger.warning("stratified_summary logging failed: %s", exc)

def _layout_bucket(scroll_px: Any) -> str:
    if not isinstance(scroll_px, (int, float)):
        return "unknown"
    if scroll_px <= 0:
        return "entry"
    for label, ceiling in LAYOUT_SCROLL_BUCKETS[1:]:
        if scroll_px <= ceiling:
            return label
    return "very_deep"

def _layout_telemetry(task: dict[str, Any]) -> dict[str, Any] | None:
    feasibility = task.get("feasibility")
    exposure = feasibility.get("exposure") if isinstance(feasibility, dict) else None
    if not isinstance(exposure, dict):
        return None
    telemetry: dict[str, Any] = {}
    for key in ("layout_visible_at_entry", "scroll_to_visible_px", "requires_expand"):
        if key in exposure:
            telemetry[key] = exposure.get(key)
    if telemetry:
        telemetry["layout_bucket"] = _layout_bucket(exposure.get("scroll_to_visible_px"))
    return telemetry or None

def _classify_trajectory_outcome(
    *,
    task: dict[str, Any],
    task_dir: Path,
    benign_passed: bool,
    adv_passed: bool,
    ecologically_valid: bool,
) -> dict[str, Any]:
    """Run the outcome-taxonomy classifier on a completed trajectory.

    Returns empty dict if classification fails for any reason (malformed
    artifacts, missing task metadata, etc.) — the pipeline must not break
    when the classifier trips.
    """
    try:
        if not (task_dir / "history.json").exists():
            return {}
        classified = outcome_taxonomy.classify_from_dir(
            task_dir,
            task,
            benign_passed=benign_passed,
            adversarial_passed=adv_passed,
            ecologically_valid=ecologically_valid,
            seed_ok=True,
        )
    except Exception as exc:  # defensive: never break the pipeline
        logger.warning(
            "outcome_taxonomy classification failed for %s: %s",
            task.get("id", "unknown"),
            exc,
        )
        return {}
    return outcome_taxonomy.as_result_fields(classified)
