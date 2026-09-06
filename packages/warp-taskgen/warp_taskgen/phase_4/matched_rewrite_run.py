"""Materialize and persist one opt-in matched study from retained Phase 4 artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from warp_taskgen.atomic_io import write_json_atomic
from warp_taskgen.phase_4.matched_rewrite_contracts import (
    AdmittedBaseline,
    AttemptProvider,
    JsonObject,
    MatchedRewriteStudyConfig,
    ModelProviderContext,
    Phase4Runtime,
)
from warp_taskgen.phase_4.matched_rewrite_identity import (
    BASELINE_CONSTRAINTS_FIELD,
    BASELINE_RESULT_FIELD,
    BASELINE_SELECTED_PAYLOAD_FIELD,
    BASELINE_TASK_FIELD,
    BASELINE_WITNESS_FIELD,
    BUDGET_FIELD,
    CALL_POLICY_FIELD,
    STUDY_ID,
    assignment_inputs,
    assignment_projection,
    study_schema_version,
    validate_checkpoint,
)
from warp_taskgen.phase_4.matched_rewrite_study import (
    matched_rewrite_ineligibility,
    run_matched_rewrite_study,
)
from warp_taskgen.phase_4.payload_witnesses import payload_witnesses_for_task
from warp_taskgen.phase_4.prompt_contracts import rewrite_constraints
from warp_taskgen.run_definition import define_run
from warp_taskgen.run_definition_contracts import RunDefinition
from warp_taskgen.run_transition import resolve_run_request

_RESULT_NAME = "result.json"


@dataclass(frozen=True, slots=True)
class MatchedRewriteRunRequest:
    """Everything needed to bind one retained baseline to one study artifact."""

    source_run_dir: Path
    task_id: str
    study_run_id: str
    output_dir: Path
    config: MatchedRewriteStudyConfig

    def __post_init__(self) -> None:
        for name in ("source_run_dir", "output_dir"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                raise ValueError(f"matched rewrite {name} must be a pathlib.Path")
        for name in ("task_id", "study_run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"matched rewrite {name} must be non-empty")
            object.__setattr__(self, name, value.strip())
        if not isinstance(self.config, MatchedRewriteStudyConfig):
            raise ValueError("matched rewrite config must be MatchedRewriteStudyConfig")
        policy = self.config.call_policy
        if policy is None or policy.provider == "unconfigured" or policy.runner == "unconfigured":
            raise ValueError("matched rewrite requires a configured call policy")
        if self.config.budget is None:
            raise ValueError("matched rewrite requires an explicit bounded budget")

    @property
    def result_path(self) -> Path:
        return self.output_dir / _RESULT_NAME


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"required matched rewrite artifact is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"matched rewrite artifact is unreadable: {path}") from exc


def _source_definition(source_run_dir: Path) -> tuple[RunDefinition, JsonObject]:
    state_path = source_run_dir / "pipeline_state.json"
    state = _load_json(state_path)
    if not isinstance(state, dict):
        raise ValueError("source pipeline_state.json must contain an object")
    definition = define_run(state)
    if definition.legacy or definition.run_id is None:
        raise ValueError("matched rewrite requires an identified non-legacy source run")
    for field in ("run_id", "source_run_id", "definition_digest"):
        if field in state and state[field] != getattr(definition, field):
            raise ValueError(
                f"source pipeline_state.json {field} conflicts with its Run Definition"
            )
    try:
        transition = resolve_run_request({}, existing_state=state)
    except ValueError as exc:
        raise ValueError(
            "source pipeline_state.json effective inputs conflict with its Run Definition"
        ) from exc
    if transition.kind != "exact" or transition.definition != definition:
        raise ValueError("source pipeline_state.json does not have one exact Run Definition")
    if state.get("step") != "phase_4" or state.get("status") != "complete":
        raise ValueError("matched rewrite source run must have a complete Phase 4 checkpoint")
    logs_dir = state.get("logs_dir")
    if not isinstance(logs_dir, str) or not logs_dir.strip():
        raise ValueError("source pipeline_state.json logs_dir must be a non-empty string")
    if Path(logs_dir).resolve(strict=False) != source_run_dir.resolve(strict=False):
        raise ValueError("source pipeline_state.json belongs to a different run directory")
    return definition, cast(JsonObject, state)


def _unique_by_id(path: Path, *, id_field: str, wanted: str) -> JsonObject:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"matched rewrite artifact must contain a JSON array: {path}")
    matches = [item for item in payload if isinstance(item, dict) and item.get(id_field) == wanted]
    if len(matches) != 1:
        raise ValueError(
            f"matched rewrite requires exactly one {id_field}={wanted!r} in {path}; "
            f"found {len(matches)}"
        )
    return cast(JsonObject, matches[0])


def _required_context(projection: dict[str, object]) -> ModelProviderContext:
    required = ("agent_model", "agent_provider", "agent_runner", "sandbox_model")
    missing = [
        field
        for field in required
        if not isinstance(projection.get(field), str) or not str(projection[field]).strip()
    ]
    if missing:
        raise ValueError(
            "source Run Definition is missing matched rewrite execution context: "
            + ", ".join(missing)
        )
    return ModelProviderContext(
        agent_model=cast(str, projection["agent_model"]),
        agent_provider=cast(str, projection["agent_provider"]),
        agent_runner=cast(str, projection["agent_runner"]),
        sandbox_model=cast(str, projection["sandbox_model"]),
        agent_service_tier=cast(str | None, projection.get("agent_service_tier")),
        runtime_composition=cast(str | None, projection.get("runtime_composition")),
    )


def _selected_payload(task: JsonObject) -> JsonObject:
    payloads = task.get("payload_texts")
    index = task.get("selected_payload_index")
    if not isinstance(payloads, list) or not payloads:
        raise ValueError("matched rewrite task has no payload_texts")
    if type(index) is not int or not 0 <= index < len(payloads):
        raise ValueError("matched rewrite task has an invalid selected_payload_index")
    chosen = payloads[index]
    if not isinstance(chosen, dict):
        raise ValueError("matched rewrite selected payload must be an object")
    rendered = chosen.get("rendered_payload")
    if not isinstance(rendered, str) or not rendered.strip():
        raise ValueError("matched rewrite task has no host-selected mutable payload")
    return cast(JsonObject, chosen)


def _host_declares_mutable_payload(task: JsonObject) -> bool:
    from warp_taskgen.phase_4.payload_rendering import build_payload_renderer_contract

    contract = build_payload_renderer_contract(cast(dict[str, Any], task))
    if not isinstance(contract, dict):
        return False
    slots = contract.get("editable_slots")
    return isinstance(slots, list) and any(
        isinstance(slot, dict)
        and slot.get("required") is True
        and isinstance(slot.get("name"), str)
        and bool(str(slot["name"]).strip())
        for slot in slots
    )


def _study_definition(
    *,
    source: RunDefinition,
    request: MatchedRewriteRunRequest,
    task: JsonObject,
    result: JsonObject,
    payload: JsonObject,
    witness: list[dict[str, str]],
    constraints: JsonObject,
    model_context: ModelProviderContext,
) -> RunDefinition:
    inputs = {
        **source.input_projection(),
        **assignment_inputs(request.config),
        "run_definition_schema_version": source.schema_version,
        "run_id": request.study_run_id,
        "source_run_id": source.run_id,
        BASELINE_TASK_FIELD: task,
        BASELINE_RESULT_FIELD: result,
        BASELINE_SELECTED_PAYLOAD_FIELD: payload,
        BASELINE_WITNESS_FIELD: witness,
        BASELINE_CONSTRAINTS_FIELD: constraints,
        "phase_4_matched_rewrite_study_condition": request.config.condition,
        "phase_4_matched_rewrite_study_schedule": request.config.schedule,
        CALL_POLICY_FIELD: request.config.resolve_call_policy(
            model_context.sandbox_model
        ).to_dict(),
        BUDGET_FIELD: request.config.budget.to_dict(),
    }
    return define_run(inputs)


def materialize_retained_baseline(request: MatchedRewriteRunRequest) -> AdmittedBaseline:
    """Bind one exact task/result pair from a completed, identified Phase 4 run."""

    source, _state = _source_definition(request.source_run_dir)
    task = _unique_by_id(
        request.source_run_dir / "phase_2" / "adversarial_tasks.json",
        id_field="id",
        wanted=request.task_id,
    )
    result = _unique_by_id(
        request.source_run_dir / "phase_4" / "results.json",
        id_field="task_id",
        wanted=request.task_id,
    )
    if result.get("task_id") != task.get("id"):
        raise ValueError("matched rewrite task_id differs between task and result")
    if result.get("run_id") is not None and result.get("run_id") != source.run_id:
        raise ValueError("matched rewrite result run_id does not match its source Run Definition")
    if (
        result.get("definition_digest") is not None
        and result.get("definition_digest") != source.definition_digest
    ):
        raise ValueError(
            "matched rewrite result definition_digest does not match its source Run Definition"
        )
    chosen = _selected_payload(task)
    witnesses = [
        witness.as_dict() for witness in payload_witnesses_for_task(cast(dict[str, Any], task))
    ]
    if not witnesses:
        raise ValueError("matched rewrite task has no attack-specific payload witness")
    constraints = cast(JsonObject, rewrite_constraints(task))
    model_context = _required_context(source.input_projection())
    definition = _study_definition(
        source=source,
        request=request,
        task=task,
        result=result,
        payload=cast(JsonObject, chosen),
        witness=witnesses,
        constraints=constraints,
        model_context=model_context,
    )
    return AdmittedBaseline(
        task=task,
        result=result,
        selected_payload=cast(JsonObject, chosen),
        witness=cast(list[object], witnesses),
        constraints=constraints,
        run_definition=definition,
        model_context=model_context,
        admitted=True,
        mutable_payload=_host_declares_mutable_payload(task),
        tp_classification=cast(str | None, result.get("transcript_purpose_classification")),
    )


def _result_identity(
    baseline: AdmittedBaseline, config: MatchedRewriteStudyConfig
) -> dict[str, object]:
    """The same effective inputs bind admission and completed artifact reuse."""

    return {
        "study_id": STUDY_ID,
        "schema_version": study_schema_version(config),
        **assignment_projection(config),
        "condition": config.condition,
        "schedule": config.schedule,
        "call_policy": config.resolve_call_policy(baseline.model_context.sandbox_model).to_dict(),
        "budget": config.budget.to_dict(),
        "baseline": baseline.to_dict(),
    }


def _validate_retained_result(
    result: dict[str, object],
    *,
    baseline: AdmittedBaseline,
    config: MatchedRewriteStudyConfig,
) -> None:
    expected = _result_identity(baseline, config)
    for field, value in expected.items():
        if result.get(field) != value:
            raise ValueError(f"matched rewrite result {field!r} is incompatible")
    status = result.get("status")
    if status not in {"completed", "ineligible"}:
        raise ValueError(
            "matched rewrite result status is not complete; automatic redispatch is refused"
        )
    checkpoint = result.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("matched rewrite result is missing its completed checkpoint")
    validate_checkpoint(cast(JsonObject, checkpoint), baseline=baseline, config=config)

    primary = result.get("primary")
    secondary = result.get("secondary")
    if not isinstance(primary, dict) or not isinstance(secondary, dict):
        raise ValueError("matched rewrite result is missing primary or secondary endpoints")
    if primary.get("endpoint") != "primary_fixed_index_scheduled_attempt":
        raise ValueError("matched rewrite primary endpoint is incompatible")
    pairs = primary.get("pairs")
    denominators = primary.get("denominators")
    if status == "ineligible":
        reason = matched_rewrite_ineligibility(baseline)
        if reason is None or result.get("ineligibility_reason") != reason:
            raise ValueError("matched rewrite ineligibility reason is incompatible")
        if pairs != [] or denominators != {"scheduled_pairs": 0, "scheduled_arms": 0}:
            raise ValueError("matched rewrite ineligible primary endpoint is malformed")
        if secondary.get("status") != "ineligible":
            raise ValueError("matched rewrite ineligible secondary endpoint is malformed")
        return

    if not isinstance(pairs, list) or len(pairs) != 1 or not isinstance(pairs[0], dict):
        raise ValueError("matched rewrite completed result must contain exactly one pair")
    if not isinstance(denominators, dict):
        raise ValueError("matched rewrite completed result is missing denominators")
    if denominators.get("scheduled_pairs") != 1 or denominators.get("scheduled_arms") != 2:
        raise ValueError("matched rewrite completed denominators are incompatible")
    pair = pairs[0]
    if config.arm_order is not None and pair.get("arm_order") != list(config.arm_order):
        raise ValueError("matched rewrite completed pair arm_order is incompatible")
    arms = pair.get("arms")
    if pair.get("pair_index") != 0 or pair.get("schedule") != config.schedule:
        raise ValueError("matched rewrite completed pair identity is incompatible")
    if not isinstance(arms, dict) or set(arms) != {"tp_guided", "ordinary"}:
        raise ValueError("matched rewrite completed pair must contain both arms")
    fixed_inputs = pair.get("fixed_inputs")
    for arm in ("tp_guided", "ordinary"):
        row = arms[arm]
        if not isinstance(row, dict):
            raise ValueError(f"matched rewrite {arm} arm must be an object")
        if (
            row.get("pair_index") != 0
            or row.get("arm") != arm
            or row.get("schedule") != config.schedule
            or not isinstance(row.get("status"), str)
            or not isinstance(row.get("accounting"), dict)
            or row.get("matched_inputs") != fixed_inputs
        ):
            raise ValueError(f"matched rewrite {arm} arm is incompatible")


async def run_retained_matched_rewrite(
    request: MatchedRewriteRunRequest,
    *,
    attempt_provider: AttemptProvider | None = None,
    runtime: Phase4Runtime | None = None,
) -> dict[str, object]:
    """Run once, or validate and reuse the exact completed study artifact."""

    if attempt_provider is not None and runtime is not None:
        raise ValueError("pass either an attempt provider or a preflighted runtime, not both")
    baseline = materialize_retained_baseline(request)
    if request.result_path.exists():
        existing = _load_json(request.result_path)
        if not isinstance(existing, dict):
            raise ValueError("matched rewrite result.json must contain an object")
        _validate_retained_result(
            cast(dict[str, object], existing),
            baseline=baseline,
            config=request.config,
        )
        return cast(dict[str, object], existing)

    if request.config.arm_order is None:
        raise ValueError("new matched rewrite runs require explicit arm_order and assignment_seed")
    ineligibility = matched_rewrite_ineligibility(baseline)
    if attempt_provider is None:
        if runtime is None and ineligibility is None:
            raise ValueError("a new matched rewrite run requires a preflighted runtime")
        if runtime is not None and ineligibility is None:
            _validate_runtime_identity(runtime, baseline, request.config)
            attempt_provider = _provider_for_runtime(runtime)
    # Persist the assignment before any provider dispatch. Interrupted studies
    # retain this incomplete artifact and fail closed on reopen; only completed
    # checkpoints are reusable. There is no inferred mid-arm resume schedule.
    if ineligibility is None:
        scheduled_arms = {
            arm: {
                "arm": arm,
                "pair_index": 0,
                "schedule": request.config.schedule,
                "status": "scheduled",
            }
            for arm in request.config.arm_order
        }
        write_json_atomic(
            request.result_path,
            {
                **_result_identity(baseline, request.config),
                "status": "scheduled",
                "primary": {
                    "endpoint": "primary_fixed_index_scheduled_attempt",
                    "pairs": [
                        {
                            "pair_index": 0,
                            "schedule": request.config.schedule,
                            "arm_order": list(request.config.arm_order),
                            "arms": scheduled_arms,
                        }
                    ],
                    "denominators": {"scheduled_pairs": 1, "scheduled_arms": 2},
                },
            },
            failpoint_base="phase_4.matched_rewrite.assignment",
        )
    result = await run_matched_rewrite_study(
        baseline,
        attempt_provider=attempt_provider,
        config=request.config,
    )
    write_json_atomic(
        request.result_path,
        result,
        failpoint_base="phase_4.matched_rewrite.result",
    )
    return result


def _provider_for_runtime(runtime: Phase4Runtime) -> AttemptProvider:
    """Adapt explicit preflighted handles without importing model SDKs on readback."""

    from warp_taskgen.phase_4.matched_rewrite_provider import ExistingPhase4AttemptAdapter

    return ExistingPhase4AttemptAdapter(runtime)


def _validate_runtime_identity(
    runtime: Phase4Runtime,
    baseline: AdmittedBaseline,
    config: MatchedRewriteStudyConfig,
) -> None:
    """Bind live browser handles to the retained run before adapting them."""

    if not isinstance(runtime, Phase4Runtime):
        raise ValueError("matched rewrite runtime must be Phase4Runtime")
    expected = baseline.model_context
    observed = {
        "sandbox_model": runtime.sandbox_model,
        "agent_model": runtime.browser_model,
        "agent_provider": runtime.browser_provider,
        "agent_runner": runtime.browser_runner,
    }
    expected_identity = {
        "sandbox_model": expected.sandbox_model,
        "agent_model": expected.agent_model,
        "agent_provider": expected.agent_provider,
        "agent_runner": expected.agent_runner,
    }
    for field, value in expected_identity.items():
        if observed[field] != value:
            raise ValueError(
                f"matched rewrite runtime {field} does not match the retained Run Definition"
            )
    observed_composition = getattr(runtime.runtime_composition, "name", None)
    if observed_composition != expected.runtime_composition:
        raise ValueError(
            "matched rewrite runtime runtime_composition does not match the retained Run Definition"
        )
    call_policy = config.resolve_call_policy(expected.sandbox_model)
    if runtime.host_provider != call_policy.provider:
        raise ValueError(
            "matched rewrite runtime host_provider does not match the study call policy"
        )
    execution = runtime.agent_execution
    if not isinstance(execution, dict):
        raise ValueError("matched rewrite runtime requires the Phase 4 agent execution identity")
    for field in ("agent_model", "agent_provider", "agent_runner", "agent_service_tier"):
        expected_value = getattr(expected, field)
        if execution.get(field) != expected_value:
            raise ValueError(
                f"matched rewrite runtime agent_execution.{field} does not match "
                "the retained Run Definition"
            )
    if not callable(runtime.agent_factory):
        raise ValueError("matched rewrite runtime requires a callable agent factory")
    if not runtime.all_instances or runtime.primary_instance not in runtime.all_instances:
        raise ValueError("matched rewrite runtime primary instance must belong to all_instances")
    primary_site = getattr(runtime.primary_instance, "site_name", None)
    task_site = baseline.task.get("site")
    if not isinstance(primary_site, str) or primary_site != task_site:
        raise ValueError("matched rewrite runtime primary instance does not match the task site")


__all__ = [
    "MatchedRewriteRunRequest",
    "materialize_retained_baseline",
    "run_retained_matched_rewrite",
]
