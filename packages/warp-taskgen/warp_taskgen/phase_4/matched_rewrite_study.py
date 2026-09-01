"""One fixed, study-only TP-guided versus ordinary rewrite comparison."""

from __future__ import annotations

import copy
from typing import cast

from warp_taskgen.phase_4.matched_rewrite_contracts import (
    AdmittedBaseline,
    Arm,
    AttemptOutcome,
    AttemptProvider,
    BrowserOutcome,
    DiagnosisOutcome,
    Guidance,
    JsonObject,
    MatchedAttemptRequest,
    MatchedRewriteStudyConfig,
    OrdinaryGuidance,
    PairAccounting,
    ProposalOutcome,
    Stage,
    TPGuidance,
    Usage,
)
from warp_taskgen.phase_4.matched_rewrite_identity import (
    STUDY_ID,
    STUDY_SCHEMA_VERSION,
    IncompatibleMatchedRewriteResume,
    checkpoint_payload,
    validate_baseline_binding,
    validate_checkpoint,
)

ARMS: tuple[Arm, Arm] = ("tp_guided", "ordinary")
_STAGES = {
    "tp_diagnosis",
    "ordinary_critique",
    "proposal",
    "repair",
    "browser",
}


def _failure_outcome(stage: Stage, reason: str) -> AttemptOutcome:
    usage = Usage.unavailable(f"{stage}_usage_unavailable")
    if stage in {"tp_diagnosis", "ordinary_critique"}:
        return DiagnosisOutcome(status="failed", guidance=None, usage=usage, failure=reason)
    if stage in {"proposal", "repair"}:
        return ProposalOutcome(status="failed", candidate=None, usage=usage, failure=reason)
    return BrowserOutcome(status="failed", result=None, usage=usage, failure=reason)


async def _invoke(
    provider: AttemptProvider | None, request: MatchedAttemptRequest
) -> AttemptOutcome:
    if provider is None:
        return _failure_outcome(request.stage, "attempt_provider_not_configured")
    try:
        outcome = await provider.run(request)
    except Exception as exc:
        return _failure_outcome(request.stage, f"{type(exc).__name__}: {exc}")
    expected = {
        "tp_diagnosis": DiagnosisOutcome,
        "ordinary_critique": DiagnosisOutcome,
        "proposal": ProposalOutcome,
        "repair": ProposalOutcome,
        "browser": BrowserOutcome,
    }[request.stage]
    if not isinstance(outcome, expected):
        return _failure_outcome(request.stage, "attempt_provider_returned_wrong_stage_type")
    return outcome


def _request(
    baseline: AdmittedBaseline,
    config: MatchedRewriteStudyConfig,
    *,
    arm: Arm,
    stage: Stage,
    guidance: Guidance | None = None,
    repair_attempt: int = 0,
    variant_task: JsonObject | None = None,
) -> MatchedAttemptRequest:
    if stage not in _STAGES:
        raise ValueError(f"unsupported matched rewrite stage {stage!r}")
    return MatchedAttemptRequest(
        binding=baseline.binding,
        condition=config.condition,
        schedule=config.schedule,
        arm=arm,
        stage=stage,
        pair_index=0,
        evidence=baseline.neutral_evidence(),
        guidance=guidance,
        repair_attempt=repair_attempt,
        baseline_task=baseline.task_copy(),
        baseline_result=baseline.result_copy(),
        variant_task=copy.deepcopy(variant_task),
        artifact_namespace=f"{STUDY_ID}/{arm}/pair-0",
    )


def _failure_text(value: str | None, fallback: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def _qa(task: JsonObject, candidate: JsonObject) -> tuple[JsonObject, JsonObject, str | None]:
    """Call canonical Phase 4 finalization and contract QA owners."""

    from warp_taskgen.phase_4.variant_contract_qa import build_variant_contract_qa
    from warp_taskgen.phase_4.variant_eval import _merge_variant_task

    finalized = cast(JsonObject, _merge_variant_task(task, candidate))
    report = cast(
        JsonObject,
        build_variant_contract_qa(task, candidate, finalized_candidate=finalized) or {},
    )
    failures = report.get("failure_classes")
    if finalized == task or not candidate.get("adversarial_data_seed"):
        return finalized, report, "unchanged_seed"
    if isinstance(failures, list) and failures:
        return finalized, report, "contract_qa_failed"
    return finalized, report, None


def _matched_inputs(baseline: AdmittedBaseline, config: MatchedRewriteStudyConfig) -> JsonObject:
    evidence = baseline.neutral_evidence().to_dict()
    evidence.update(
        {
            "condition": config.condition,
            "schedule": config.schedule,
            "repair_attempts": config.repair_attempts,
            "baseline_identity": baseline.identity,
            "run_definition_digest": baseline.run_definition.definition_digest,
        }
    )
    return evidence


async def _run_arm(
    baseline: AdmittedBaseline,
    config: MatchedRewriteStudyConfig,
    arm: Arm,
    provider: AttemptProvider | None,
) -> dict[str, object]:
    accounting = PairAccounting()
    row: dict[str, object] = {
        "pair_index": 0,
        "arm": arm,
        "schedule": config.schedule,
        "status": "scheduled",
        "matched_inputs": _matched_inputs(baseline, config),
        "guidance": None,
        "proposal": None,
        "repair_attempts": [],
        "qa": None,
        "result": None,
    }
    diagnosis_stage: Stage = "tp_diagnosis" if arm == "tp_guided" else "ordinary_critique"
    diagnosis = await _invoke(
        provider,
        _request(baseline, config, arm=arm, stage=diagnosis_stage),
    )
    accounting.record(diagnosis_stage, diagnosis)
    if not isinstance(diagnosis, DiagnosisOutcome) or diagnosis.status != "ok":
        row["status"] = "diagnosis_failed"
        row["failure"] = _failure_text(
            diagnosis.failure if isinstance(diagnosis, DiagnosisOutcome) else None,
            "diagnosis_failed",
        )
        row["accounting"] = accounting.to_dict()
        return row
    guidance = diagnosis.guidance
    if (arm == "tp_guided" and not isinstance(guidance, TPGuidance)) or (
        arm == "ordinary" and not isinstance(guidance, OrdinaryGuidance)
    ):
        row["status"] = "diagnosis_failed"
        row["failure"] = "diagnosis_guidance_type_mismatch"
        row["accounting"] = accounting.to_dict()
        return row
    row["guidance"] = guidance.to_dict()

    proposal = await _invoke(
        provider,
        _request(baseline, config, arm=arm, stage="proposal", guidance=guidance),
    )
    accounting.record("proposal", proposal)
    if not isinstance(proposal, ProposalOutcome):
        row["status"] = "generation_failed"
        row["failure"] = "proposal_outcome_type_mismatch"
        row["accounting"] = accounting.to_dict()
        return row
    row["proposal"] = {"status": proposal.status, "candidate": copy.deepcopy(proposal.candidate)}
    if proposal.status == "inapplicable":
        row["status"] = "inapplicable"
        row["failure"] = _failure_text(proposal.failure, "rewrite_inapplicable")
        row["accounting"] = accounting.to_dict()
        return row
    if proposal.status != "ok" or proposal.candidate is None:
        row["status"] = "generation_failed"
        row["failure"] = _failure_text(proposal.failure, "proposal_generation_failed")
        row["accounting"] = accounting.to_dict()
        return row

    task = baseline.task_copy()
    candidate = copy.deepcopy(proposal.candidate)
    finalized, qa, failure = _qa(task, candidate)
    proposal_record = cast(dict[str, object], row["proposal"])
    proposal_record["finalized"] = copy.deepcopy(finalized)
    row["qa"] = copy.deepcopy(qa)
    repairs = cast(list[object], row["repair_attempts"])
    repair_failed = False
    for repair_attempt in range(1, config.repair_attempts + 1) if failure else ():
        repair = await _invoke(
            provider,
            _request(
                baseline,
                config,
                arm=arm,
                stage="repair",
                guidance=guidance,
                repair_attempt=repair_attempt,
                variant_task=finalized,
            ),
        )
        accounting.record("repair", repair)
        if not isinstance(repair, ProposalOutcome):
            repairs.append(
                {
                    "repair_attempt": repair_attempt,
                    "status": "failed",
                    "failure": "repair_outcome_type_mismatch",
                }
            )
            failure = "repair_outcome_type_mismatch"
            repair_failed = True
            break
        repair_record: dict[str, object] = {
            "repair_attempt": repair_attempt,
            "status": repair.status,
            "candidate": copy.deepcopy(repair.candidate),
        }
        if repair.status != "ok" or repair.candidate is None:
            repair_record["failure"] = _failure_text(repair.failure, "repair_failed")
            repairs.append(repair_record)
            failure = "repair_failed"
            repair_failed = True
            break
        finalized, qa, failure = _qa(task, repair.candidate)
        repair_record["qa"] = copy.deepcopy(qa)
        repairs.append(repair_record)
        if failure is None:
            break
    if failure is not None:
        row["status"] = "repair_failed" if repair_failed else "qa_failed"
        row["failure"] = failure
        row["qa"] = copy.deepcopy(qa)
        row["accounting"] = accounting.to_dict()
        return row

    row["finalized_task"] = copy.deepcopy(finalized)
    browser = await _invoke(
        provider,
        _request(
            baseline,
            config,
            arm=arm,
            stage="browser",
            guidance=guidance,
            variant_task=finalized,
        ),
    )
    if not isinstance(browser, BrowserOutcome):
        row["status"] = "browser_attempt_failed"
        row["failure"] = "browser_outcome_type_mismatch"
        accounting.record("browser", browser)
        row["accounting"] = accounting.to_dict()
        return row
    accounting.record("browser", browser, browser_counted=browser.status != "no_rerun")
    if browser.status == "no_rerun":
        row["status"] = "no_rerun"
        row["failure"] = _failure_text(browser.failure, "browser_not_run")
    elif browser.status != "ok" or browser.result is None:
        row["status"] = "browser_attempt_failed"
        row["failure"] = _failure_text(browser.failure, "browser_attempt_failed")
    else:
        row["status"] = "evaluated"
        row["result"] = copy.deepcopy(browser.result)
    row["accounting"] = accounting.to_dict()
    return row


def _eligible(baseline: AdmittedBaseline) -> str | None:
    from warp_taskgen.phase_4.result_summary import ecologically_valid

    if not baseline.admitted:
        return "baseline_not_admitted"
    if not ecologically_valid(baseline.result):
        return "baseline_not_pvpo_valid"
    if baseline.tp_classification == "Real":
        return "baseline_tp_real"
    if not baseline.mutable_payload:
        return "baseline_payload_not_host_declared_mutable"
    return None


def _select(baseline: AdmittedBaseline, rows: dict[Arm, dict[str, object]]) -> dict[str, object]:
    """Use the existing eval-awareness iterator selector for both arms."""

    from warp_taskgen.phase_4.eval_awareness_iterator import _best_iterator_result

    selected: dict[str, object] = {}
    for arm in ARMS:
        result = rows[arm].get("result")
        records = [{"iteration": 1, "result": result}] if isinstance(result, dict) else []
        picked, iteration, reason = _best_iterator_result(
            baseline.result_copy(),
            cast(list[dict[str, object]], records),
        )
        selected[arm] = {
            "selected_iteration": iteration,
            "selection_reason": reason,
            "result": copy.deepcopy(picked),
        }
    return {
        "endpoint": "secondary_selected_result",
        "selector": "eval-awareness-iterator",
        "arms": selected,
    }


def _aggregate(rows: dict[Arm, dict[str, object]]) -> dict[str, object]:
    total = PairAccounting()
    for row in rows.values():
        accounting = row.get("accounting")
        if not isinstance(accounting, dict):
            continue
        total.diagnosis_calls += int(accounting.get("diagnosis_calls", 0))
        total.proposal_calls += int(accounting.get("proposal_calls", 0))
        total.repair_calls += int(accounting.get("repair_calls", 0))
        total.browser_attempts += int(accounting.get("browser_attempts", 0))
        if accounting.get("usage_status") == "unavailable":
            total.input_tokens = total.output_tokens = total.total_tokens = total.cost_usd = None
            reasons = accounting.get("usage_unavailable_reasons")
            if isinstance(reasons, list) and total.usage_unavailable_reasons is not None:
                total.usage_unavailable_reasons.extend(str(reason) for reason in reasons)
        elif total.input_tokens is not None:
            total.input_tokens += int(accounting.get("input_tokens", 0))
            total.output_tokens = cast(int, total.output_tokens) + int(
                accounting.get("output_tokens", 0)
            )
            total.total_tokens = cast(int, total.total_tokens) + int(
                accounting.get("total_tokens", 0)
            )
            total.cost_usd = cast(float, total.cost_usd) + float(accounting.get("cost_usd", 0.0))
    return total.to_dict()


async def run_matched_rewrite_study(
    baseline: AdmittedBaseline,
    *,
    attempt_provider: AttemptProvider | None = None,
    config: MatchedRewriteStudyConfig | None = None,
    checkpoint: JsonObject | None = None,
) -> dict[str, object]:
    """Run exactly one paired opportunity without changing the default iterator."""

    if not isinstance(baseline, AdmittedBaseline):
        raise TypeError("matched rewrite study requires an AdmittedBaseline")
    settings = config or MatchedRewriteStudyConfig()
    validate_baseline_binding(baseline, settings)
    if checkpoint is not None:
        validate_checkpoint(checkpoint, baseline=baseline, config=settings)
        return {
            "status": "resume_accepted",
            "study_id": STUDY_ID,
            "schema_version": STUDY_SCHEMA_VERSION,
        }

    result: dict[str, object] = {
        "study_id": STUDY_ID,
        "schema_version": STUDY_SCHEMA_VERSION,
        "condition": settings.condition,
        "schedule": settings.schedule,
        "baseline": baseline.to_dict(),
        "status": "scheduled",
        "primary": {"endpoint": "primary_fixed_index_scheduled_attempt", "pairs": []},
    }
    ineligible = _eligible(baseline)
    if ineligible is not None:
        result["status"] = "ineligible"
        result["ineligibility_reason"] = ineligible
        result["primary"] = {
            "endpoint": "primary_fixed_index_scheduled_attempt",
            "pairs": [],
            "accounting": PairAccounting().to_dict(),
            "denominators": {"scheduled_pairs": 0, "scheduled_arms": 0},
        }
        result["secondary"] = {"status": "ineligible", "arms": {}}
    else:
        if attempt_provider is not None:
            attempt_provider.bind(baseline.binding)
        rows = {arm: await _run_arm(baseline, settings, arm, attempt_provider) for arm in ARMS}
        result["status"] = "completed"
        evaluated = sum(row.get("status") == "evaluated" for row in rows.values())
        result["primary"] = {
            "endpoint": "primary_fixed_index_scheduled_attempt",
            "pairs": [
                {
                    "pair_index": 0,
                    "schedule": settings.schedule,
                    "fixed_inputs": _matched_inputs(baseline, settings),
                    "arms": rows,
                }
            ],
            "accounting": _aggregate(rows),
            "denominators": {
                "scheduled_pairs": 1,
                "scheduled_arms": 2,
                "evaluated_arms": evaluated,
            },
        }
        try:
            result["secondary"] = _select(baseline, rows)
        except ModuleNotFoundError as exc:
            result["secondary"] = {
                "status": "unavailable",
                "selector": "eval-awareness-iterator",
                "reason": f"selector_dependencies_unavailable: {exc}",
            }
    result["checkpoint"] = checkpoint_payload(baseline, settings)
    return result


def admit_matched_rewrite_baseline(baseline: AdmittedBaseline) -> AdmittedBaseline:
    """Validate the typed baseline once before a paired run."""

    if not isinstance(baseline, AdmittedBaseline):
        raise TypeError("matched rewrite baseline must be an AdmittedBaseline")
    if not baseline.admitted:
        raise ValueError("matched rewrite baseline must be admitted")
    return baseline


__all__ = [
    "AdmittedBaseline",
    "AttemptProvider",
    "IncompatibleMatchedRewriteResume",
    "MatchedAttemptRequest",
    "MatchedRewriteStudyConfig",
    "admit_matched_rewrite_baseline",
    "run_matched_rewrite_study",
]
