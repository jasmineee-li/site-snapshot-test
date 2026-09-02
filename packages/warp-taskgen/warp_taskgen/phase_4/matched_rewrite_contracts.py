"""Typed contracts for the fixed matched-rewrite study."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

from warp_taskgen.phase_4.matched_rewrite_accounting import (
    MatchedCallPolicy,
    MatchedStudyBudget,
    PairAccounting,
    Usage,
)
from warp_taskgen.phase_4.prompt_contracts import rewrite_constraints, trajectory_summary
from warp_taskgen.phase_4.prompt_payloads import sanitize_task_for_model_prompt
from warp_taskgen.run_definition_contracts import RunDefinition

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]

Condition = Literal["tp_guided_vs_ordinary"]
Schedule = Literal["one_opportunity"]
Arm = Literal["tp_guided", "ordinary"]
Stage = Literal["tp_diagnosis", "ordinary_critique", "proposal", "repair", "browser"]
Confidence = Literal["low", "medium", "high"]

STUDY_CONDITION: Condition = "tp_guided_vs_ordinary"
STUDY_SCHEDULE: Schedule = "one_opportunity"
STUDY_REPAIR_ATTEMPTS = 1


def _copy(value: JsonValue) -> JsonValue:
    return cast(JsonValue, copy.deepcopy(value))


def _validate_json(value: object, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain finite numbers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} keys must be non-empty strings")
            _validate_json(item, path=f"{path}.{key}")
        return
    raise ValueError(f"{path} must contain JSON-shaped values")


@dataclass(frozen=True, slots=True)
class ModelProviderContext:
    """Execution identity that must match the admitted Run Definition."""

    agent_model: str
    agent_provider: str
    agent_runner: str
    sandbox_model: str
    agent_service_tier: str | None = None
    runtime_composition: str | None = None

    def __post_init__(self) -> None:
        for name in ("agent_model", "agent_provider", "agent_runner", "sandbox_model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"model/provider context {name} must be non-empty")
            object.__setattr__(self, name, value.strip())
        for name in ("agent_service_tier", "runtime_composition"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"model/provider context {name} must be non-empty or null")
            if isinstance(value, str):
                object.__setattr__(self, name, value.strip())

    def to_projection(self) -> JsonObject:
        return {
            "agent_model": self.agent_model,
            "agent_provider": self.agent_provider,
            "agent_runner": self.agent_runner,
            "sandbox_model": self.sandbox_model,
            "agent_service_tier": self.agent_service_tier,
            "runtime_composition": self.runtime_composition,
        }


@dataclass(frozen=True, slots=True)
class MatchedRewriteStudyConfig:
    """Only the supported condition and one-opportunity schedule are admitted."""

    condition: Condition = STUDY_CONDITION
    schedule: Schedule = STUDY_SCHEDULE
    call_policy: MatchedCallPolicy | None = None
    budget: MatchedStudyBudget | None = None

    def __post_init__(self) -> None:
        if self.condition != STUDY_CONDITION:
            raise ValueError(f"unsupported matched rewrite condition: {self.condition!r}")
        if self.schedule != STUDY_SCHEDULE:
            raise ValueError(f"unsupported matched rewrite schedule: {self.schedule!r}")
        if self.call_policy is not None and not isinstance(self.call_policy, MatchedCallPolicy):
            raise ValueError("matched rewrite call_policy must be MatchedCallPolicy or null")
        if self.budget is not None and not isinstance(self.budget, MatchedStudyBudget):
            raise ValueError("matched rewrite budget must be MatchedStudyBudget or null")

    @property
    def repair_attempts(self) -> int:
        return STUDY_REPAIR_ATTEMPTS

    def resolve_call_policy(self, model: str) -> MatchedCallPolicy:
        """Resolve and verify the policy against the baseline model identity."""

        policy = self.call_policy or MatchedCallPolicy.for_model(model)
        if policy.model != model.strip():
            raise ValueError(
                "matched rewrite call policy model must match the baseline sandbox model"
            )
        return policy


@dataclass(frozen=True, slots=True)
class BaselineBinding:
    """Non-secret binding an attempt provider must verify on every request."""

    identity: str
    definition_digest: str

    def __post_init__(self) -> None:
        if not self.identity.strip() or len(self.definition_digest) != 64:
            raise ValueError("baseline binding identity and definition digest are required")


@dataclass(frozen=True, slots=True)
class AdmittedBaseline:
    """Deep-copied, execution-complete baseline admitted to the one opportunity."""

    task: JsonObject
    result: JsonObject
    selected_payload: JsonObject
    witness: JsonValue
    constraints: JsonObject
    run_definition: RunDefinition
    model_context: ModelProviderContext
    admitted: bool = True
    mutable_payload: bool = True
    tp_classification: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task, dict) or not isinstance(self.result, dict):
            raise ValueError("admitted baseline task and result must be JSON objects")
        if not isinstance(self.selected_payload, dict) or not isinstance(self.constraints, dict):
            raise ValueError("admitted baseline payload and constraints must be JSON objects")
        if not isinstance(self.run_definition, RunDefinition):
            raise ValueError("admitted baseline requires a Run Definition")
        if not isinstance(self.model_context, ModelProviderContext):
            raise ValueError("admitted baseline requires model/provider context")
        if type(self.admitted) is not bool or type(self.mutable_payload) is not bool:
            raise ValueError("admitted baseline admission flags must be boolean")
        if self.tp_classification is not None and not isinstance(self.tp_classification, str):
            raise ValueError("baseline TP classification must be a string or null")
        for name in ("task", "result", "selected_payload", "witness", "constraints"):
            _validate_json(getattr(self, name), path=f"baseline.{name}")
            object.__setattr__(self, name, _copy(cast(JsonValue, getattr(self, name))))

    @property
    def binding(self) -> BaselineBinding:
        run_id = self.run_definition.run_id or "legacy"
        return BaselineBinding(
            identity=f"{run_id}:{self.run_definition.definition_digest}",
            definition_digest=self.run_definition.definition_digest,
        )

    @property
    def identity(self) -> str:
        return self.binding.identity

    def task_copy(self) -> JsonObject:
        return cast(JsonObject, _copy(self.task))

    def result_copy(self) -> JsonObject:
        return cast(JsonObject, _copy(self.result))

    def neutral_evidence(self) -> NeutralEvidence:
        return NeutralEvidence(
            task=cast(JsonObject, sanitize_task_for_model_prompt(self.task_copy())),
            selected_payload=cast(JsonObject, _copy(self.selected_payload)),
            witness=_copy(self.witness),
            trajectory_summary=cast(JsonObject, trajectory_summary(self.result_copy())),
            constraints=cast(JsonObject, _copy(self.constraints)),
        )

    def to_dict(self) -> JsonObject:
        return {
            "identity": self.identity,
            "task": cast(JsonObject, sanitize_task_for_model_prompt(self.task_copy())),
            "result_summary": cast(JsonObject, trajectory_summary(self.result_copy())),
            "selected_payload": _copy(self.selected_payload),
            "witness": _copy(self.witness),
            "constraints": _copy(self.constraints),
            "admitted": self.admitted,
            "mutable_payload": self.mutable_payload,
            "tp_classification": self.tp_classification,
            "run_definition": cast(JsonObject, self.run_definition.to_dict()),
            "model_context": self.model_context.to_projection(),
        }


@dataclass(frozen=True, slots=True)
class NeutralEvidence:
    """Equal model-facing baseline evidence shared by both arms."""

    task: JsonObject
    selected_payload: JsonObject
    witness: JsonValue
    trajectory_summary: JsonObject
    constraints: JsonObject

    def __post_init__(self) -> None:
        for name in ("task", "selected_payload", "trajectory_summary", "constraints"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"neutral evidence {name} must be a JSON object")
        for name in ("task", "selected_payload", "witness", "trajectory_summary", "constraints"):
            _validate_json(getattr(self, name), path=f"neutral evidence.{name}")
            object.__setattr__(self, name, _copy(cast(JsonValue, getattr(self, name))))

    def to_dict(self) -> JsonObject:
        return {
            "task": _copy(self.task),
            "selected_payload": _copy(self.selected_payload),
            "witness": _copy(self.witness),
            "trajectory_summary": _copy(self.trajectory_summary),
            "constraints": _copy(self.constraints),
        }


@dataclass(frozen=True, slots=True)
class TPGuidance:
    trigger_source: str
    mutable_payload_cues: tuple[str, ...] = ()
    protected_payload_cues: tuple[str, ...] = ()
    capability_eval_cues: tuple[str, ...] = ()
    trajectory_cues: tuple[str, ...] = ()
    real_world_reframe: str = ""
    do_not_change: tuple[str, ...] = ()
    rewrite_guidance: str = ""
    confidence: Confidence = "medium"
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.trigger_source.strip():
            raise ValueError("TP guidance trigger_source must be non-empty")
        if self.confidence not in {"low", "medium", "high"}:
            raise ValueError("TP guidance confidence is unsupported")

    def to_dict(self) -> JsonObject:
        return {
            "trigger_source": self.trigger_source,
            "mutable_payload_cues": list(self.mutable_payload_cues),
            "protected_payload_cues": list(self.protected_payload_cues),
            "capability_eval_cues": list(self.capability_eval_cues),
            "trajectory_cues": list(self.trajectory_cues),
            "real_world_reframe": self.real_world_reframe,
            "do_not_change": list(self.do_not_change),
            "rewrite_guidance": self.rewrite_guidance,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class OrdinaryGuidance:
    critique: str
    guidance: str = ""
    rewrite_guidance: str = ""
    focus: str = ""
    confidence: Confidence = "medium"
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.critique.strip():
            raise ValueError("ordinary guidance critique must be non-empty")
        if self.confidence not in {"low", "medium", "high"}:
            raise ValueError("ordinary guidance confidence is unsupported")

    def to_dict(self) -> JsonObject:
        return {
            "critique": self.critique,
            "guidance": self.guidance,
            "rewrite_guidance": self.rewrite_guidance,
            "focus": self.focus,
            "confidence": self.confidence,
            "reason": self.reason,
        }


Guidance = TPGuidance | OrdinaryGuidance


@dataclass(frozen=True, slots=True)
class MatchedAttemptRequest:
    """One typed system-boundary request; raw execution data is never serialized to prompts."""

    binding: BaselineBinding
    condition: Condition
    schedule: Schedule
    arm: Arm
    stage: Stage
    pair_index: int
    evidence: NeutralEvidence
    guidance: Guidance | None
    repair_attempt: int
    baseline_task: JsonObject
    baseline_result: JsonObject
    variant_task: JsonObject | None = None
    artifact_namespace: str = ""
    call_policy: MatchedCallPolicy | None = None
    budget: MatchedStudyBudget | None = None

    def __post_init__(self) -> None:
        if self.pair_index != 0:
            raise ValueError("the matched rewrite study has exactly one pair")
        if self.stage == "tp_diagnosis" and self.arm != "tp_guided":
            raise ValueError("TP diagnosis belongs only to the TP-guided arm")
        if self.stage == "ordinary_critique" and self.arm != "ordinary":
            raise ValueError("ordinary critique belongs only to the ordinary arm")
        if self.stage in {"proposal", "repair"} and self.guidance is None:
            raise ValueError("proposal and repair requests require arm guidance")
        if self.repair_attempt < 0:
            raise ValueError("repair attempt must be non-negative")
        if self.call_policy is not None and not isinstance(self.call_policy, MatchedCallPolicy):
            raise ValueError("attempt call_policy must be MatchedCallPolicy or null")
        if self.budget is not None and not isinstance(self.budget, MatchedStudyBudget):
            raise ValueError("attempt budget must be MatchedStudyBudget or null")
        for name in ("baseline_task", "baseline_result"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"attempt {name} must be a JSON object")
            _validate_json(getattr(self, name), path=f"attempt.{name}")
        if self.variant_task is not None:
            _validate_json(self.variant_task, path="attempt.variant_task")

    def to_dict(self) -> JsonObject:
        return {
            "condition": self.condition,
            "schedule": self.schedule,
            "arm": self.arm,
            "stage": self.stage,
            "pair_index": self.pair_index,
            "binding": {
                "identity": self.binding.identity,
                "definition_digest": self.binding.definition_digest,
            },
            "evidence": self.evidence.to_dict(),
            "guidance": self.guidance.to_dict() if self.guidance is not None else None,
            "repair_attempt": self.repair_attempt,
            "call_policy": self.call_policy.to_dict() if self.call_policy is not None else None,
            "budget": self.budget.to_dict() if self.budget is not None else None,
        }


@dataclass(frozen=True, slots=True)
class DiagnosisOutcome:
    status: Literal["ok", "failed"]
    guidance: Guidance | None
    usage: Usage
    failure: str | None = None
    diagnostics: JsonObject | None = None

    def __post_init__(self) -> None:
        if self.status not in {"ok", "failed"}:
            raise ValueError("diagnosis outcome status is unsupported")
        if not isinstance(self.usage, Usage):
            raise ValueError("diagnosis outcome requires Usage")
        if self.status == "ok" and self.guidance is None:
            raise ValueError("successful diagnosis outcome requires guidance")
        if self.status == "failed" and self.guidance is not None:
            raise ValueError("failed diagnosis outcome cannot include guidance")
        if self.diagnostics is not None:
            _validate_json(self.diagnostics, path="diagnosis.diagnostics")
            object.__setattr__(self, "diagnostics", _copy(self.diagnostics))


@dataclass(frozen=True, slots=True)
class ProposalOutcome:
    status: Literal["ok", "inapplicable", "failed"]
    candidate: JsonObject | None
    usage: Usage
    failure: str | None = None
    diagnostics: JsonObject | None = None

    def __post_init__(self) -> None:
        if self.status not in {"ok", "inapplicable", "failed"}:
            raise ValueError("proposal outcome status is unsupported")
        if not isinstance(self.usage, Usage):
            raise ValueError("proposal outcome requires Usage")
        if self.status == "ok" and self.candidate is None:
            raise ValueError("successful proposal outcome requires a candidate")
        if self.status != "ok" and self.candidate is not None:
            raise ValueError("non-successful proposal outcome cannot include a candidate")
        if self.diagnostics is not None:
            _validate_json(self.diagnostics, path="proposal.diagnostics")
            object.__setattr__(self, "diagnostics", _copy(self.diagnostics))


@dataclass(frozen=True, slots=True)
class BrowserOutcome:
    status: Literal["ok", "no_rerun", "failed"]
    result: JsonObject | None
    usage: Usage
    failure: str | None = None
    diagnostics: JsonObject | None = None

    def __post_init__(self) -> None:
        if self.status not in {"ok", "no_rerun", "failed"}:
            raise ValueError("browser outcome status is unsupported")
        if not isinstance(self.usage, Usage):
            raise ValueError("browser outcome requires Usage")
        if self.status == "ok" and self.result is None:
            raise ValueError("successful browser outcome requires a result")
        if self.status != "ok" and self.result is not None:
            raise ValueError("non-successful browser outcome cannot include a result")
        if self.diagnostics is not None:
            _validate_json(self.diagnostics, path="browser.diagnostics")
            object.__setattr__(self, "diagnostics", _copy(self.diagnostics))


AttemptOutcome = DiagnosisOutcome | ProposalOutcome | BrowserOutcome


class AttemptProvider(Protocol):
    def bind(self, binding: BaselineBinding) -> None: ...

    async def run(self, request: MatchedAttemptRequest) -> AttemptOutcome: ...


@dataclass(frozen=True, slots=True)
class Phase4Runtime:
    """Explicit runtime handles for the optional existing Phase 4 adapter."""

    primary_instance: object
    all_instances: tuple[object, ...]
    agent_factory: Callable[[], object]
    task_dir_root: Path
    sandbox_model: str = "claude-sonnet-4-6"
    benchmark_root: Path | None = None
    site_profile: JsonObject | None = None
    agent_execution: JsonObject | None = None
    browser_worker_semaphore: object | None = None
    runtime_composition: object | None = None
    host_client: object | None = None


__all__ = [
    "STUDY_CONDITION",
    "STUDY_REPAIR_ATTEMPTS",
    "STUDY_SCHEDULE",
    "AdmittedBaseline",
    "Arm",
    "AttemptOutcome",
    "AttemptProvider",
    "BaselineBinding",
    "BrowserOutcome",
    "Condition",
    "DiagnosisOutcome",
    "Guidance",
    "JsonObject",
    "JsonValue",
    "MatchedAttemptRequest",
    "MatchedCallPolicy",
    "MatchedRewriteStudyConfig",
    "MatchedStudyBudget",
    "ModelProviderContext",
    "NeutralEvidence",
    "OrdinaryGuidance",
    "PairAccounting",
    "Phase4Runtime",
    "ProposalOutcome",
    "Schedule",
    "Stage",
    "TPGuidance",
    "Usage",
    "rewrite_constraints",
]
