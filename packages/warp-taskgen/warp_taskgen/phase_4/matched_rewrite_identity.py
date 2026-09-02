"""Run Definition binding and checkpoint validation for the matched study."""

from __future__ import annotations

from typing import Final

from warp_taskgen.phase_4.matched_rewrite_contracts import (
    AdmittedBaseline,
    JsonObject,
    MatchedRewriteStudyConfig,
)
from warp_taskgen.run_definition import define_run

STUDY_ID: Final = "matched_tp_guided_vs_ordinary_rewrite"
STUDY_SCHEMA_VERSION: Final = 2
BASELINE_TASK_FIELD: Final = "phase_4_matched_rewrite_study_baseline_task"
BASELINE_RESULT_FIELD: Final = "phase_4_matched_rewrite_study_baseline_result"
BASELINE_SELECTED_PAYLOAD_FIELD: Final = "phase_4_matched_rewrite_study_selected_payload"
BASELINE_WITNESS_FIELD: Final = "phase_4_matched_rewrite_study_witness"
BASELINE_CONSTRAINTS_FIELD: Final = "phase_4_matched_rewrite_study_constraints"
CALL_POLICY_FIELD: Final = "phase_4_matched_rewrite_study_call_policy"
BUDGET_FIELD: Final = "phase_4_matched_rewrite_study_budget"
_CONDITION_FIELD: Final = "phase_4_matched_rewrite_study_condition"
_SCHEDULE_FIELD: Final = "phase_4_matched_rewrite_study_schedule"
_CONTEXT_FIELDS: Final = {
    "agent_model": "agent_model",
    "agent_provider": "agent_provider",
    "agent_runner": "agent_runner",
    "sandbox_model": "sandbox_model",
    "agent_service_tier": "agent_service_tier",
    "runtime_composition": "runtime_composition",
}


class IncompatibleMatchedRewriteResume(ValueError):
    """A checkpoint fails the fixed study schema or Run Definition binding."""

    def __init__(self, message: str, *, expected: object, observed: object = None) -> None:
        super().__init__(message)
        self.expected = expected
        self.observed = observed


def validate_baseline_binding(
    baseline: AdmittedBaseline,
    config: MatchedRewriteStudyConfig,
) -> None:
    """Require the full baseline contracts and execution context in Run Definition."""

    expected_inputs = {
        BASELINE_TASK_FIELD: baseline.task,
        BASELINE_RESULT_FIELD: baseline.result,
        BASELINE_SELECTED_PAYLOAD_FIELD: baseline.selected_payload,
        BASELINE_WITNESS_FIELD: baseline.witness,
        BASELINE_CONSTRAINTS_FIELD: baseline.constraints,
        _CONDITION_FIELD: config.condition,
        _SCHEDULE_FIELD: config.schedule,
        CALL_POLICY_FIELD: config.resolve_call_policy(baseline.model_context.sandbox_model).to_dict(),
    }
    if config.budget is not None:
        expected_inputs[BUDGET_FIELD] = config.budget.to_dict()
    expected_inputs.update(
        {
            field: value
            for field, value in baseline.model_context.to_projection().items()
            if value is not None
        }
    )
    if baseline.run_definition.legacy:
        normalized = define_run(expected_inputs).input_projection()
    else:
        normalized = define_run(
            {
                **expected_inputs,
                "run_definition_schema_version": baseline.run_definition.schema_version,
                "run_id": baseline.run_definition.run_id,
                "source_run_id": baseline.run_definition.source_run_id,
            }
        ).input_projection()
    projection = baseline.run_definition.input_projection()
    for field, value in normalized.items():
        if field not in projection or projection[field] != value:
            raise ValueError(f"admitted baseline Run Definition is missing or mismatched {field!r}")


def checkpoint_payload(
    baseline: AdmittedBaseline,
    config: MatchedRewriteStudyConfig,
) -> JsonObject:
    """Build the strict compatibility-only checkpoint metadata."""

    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "condition": config.condition,
        "schedule": config.schedule,
        "baseline_identity": baseline.identity,
        "run_definition": baseline.run_definition.to_dict(),
        "call_policy": config.resolve_call_policy(baseline.model_context.sandbox_model).to_dict(),
        "budget": config.budget.to_dict() if config.budget is not None else None,
        "status": "complete",
        "primary": {
            "status": "complete",
            "scheduled_pairs": 1,
            "scheduled_arms": 2,
        },
    }


def validate_checkpoint(
    checkpoint: JsonObject,
    *,
    baseline: AdmittedBaseline,
    config: MatchedRewriteStudyConfig,
) -> None:
    """Validate schema, study, condition, schedule, baseline, and primary shape."""

    expected = checkpoint_payload(baseline, config)
    if not isinstance(checkpoint, dict):
        raise IncompatibleMatchedRewriteResume(
            "matched rewrite checkpoint must be a JSON object",
            expected=expected,
            observed=checkpoint,
        )
    if set(checkpoint) != set(expected):
        raise IncompatibleMatchedRewriteResume(
            "matched rewrite checkpoint has an unsupported schema shape",
            expected=sorted(expected),
            observed=sorted(checkpoint),
        )
    for field in (
        "schema_version",
        "study_id",
        "condition",
        "schedule",
        "baseline_identity",
        "run_definition",
        "call_policy",
        "budget",
        "status",
    ):
        if checkpoint[field] != expected[field]:
            raise IncompatibleMatchedRewriteResume(
                f"matched rewrite checkpoint {field!r} is incompatible",
                expected=expected[field],
                observed=checkpoint[field],
            )
    primary = checkpoint["primary"]
    if primary != expected["primary"]:
        raise IncompatibleMatchedRewriteResume(
            "matched rewrite checkpoint primary endpoint is incomplete or incompatible",
            expected=expected["primary"],
            observed=primary,
        )


__all__ = [
    "BASELINE_CONSTRAINTS_FIELD",
    "BASELINE_RESULT_FIELD",
    "BASELINE_SELECTED_PAYLOAD_FIELD",
    "BASELINE_TASK_FIELD",
    "BASELINE_WITNESS_FIELD",
    "BUDGET_FIELD",
    "CALL_POLICY_FIELD",
    "STUDY_ID",
    "STUDY_SCHEMA_VERSION",
    "IncompatibleMatchedRewriteResume",
    "checkpoint_payload",
    "validate_baseline_binding",
    "validate_checkpoint",
]
