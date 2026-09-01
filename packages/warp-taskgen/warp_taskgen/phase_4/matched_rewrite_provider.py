"""The one typed attempt-provider seam used by the matched rewrite study."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import cast

from warp_taskgen.phase_4.matched_rewrite_contracts import (
    AttemptOutcome,
    BaselineBinding,
    BrowserOutcome,
    DiagnosisOutcome,
    JsonObject,
    MatchedAttemptRequest,
    OrdinaryGuidance,
    Phase4Runtime,
    ProposalOutcome,
    TPGuidance,
    Usage,
)
from warp_taskgen.phase_4.prompt_contracts import trajectory_summary
from warp_taskgen.phase_4.prompt_payloads import sanitize_task_for_model_prompt


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _text(value: object) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _json_object(value: object) -> JsonObject | None:
    if not isinstance(value, dict):
        return None
    output: JsonObject = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        if item is None or isinstance(item, (str, int, float, bool)):
            output[key] = item
        elif isinstance(item, list):
            output[key] = copy.deepcopy(item)  # type: ignore[assignment]
        elif isinstance(item, dict):
            nested = _json_object(item)
            if nested is None:
                return None
            output[key] = nested
        else:
            return None
    return output


def _tp_guidance(raw: object) -> TPGuidance | None:
    data = _json_object(raw)
    if data is None:
        return None
    trigger = _text(data.get("trigger_source"))
    if not trigger:
        return None
    return TPGuidance(
        trigger_source=trigger,
        mutable_payload_cues=_strings(data.get("mutable_payload_cues")),
        protected_payload_cues=_strings(data.get("protected_payload_cues")),
        capability_eval_cues=_strings(data.get("capability_eval_cues")),
        trajectory_cues=_strings(data.get("trajectory_cues")),
        real_world_reframe=_text(data.get("real_world_reframe")),
        do_not_change=_strings(data.get("do_not_change")),
        rewrite_guidance=_text(data.get("rewrite_guidance")),
        confidence=cast(str, data.get("confidence", "medium")),
        reason=_text(data.get("reason")) or None,
    )


@dataclass(frozen=True, slots=True)
class ExistingPhase4AttemptAdapter:
    """Dispatch existing owners without retaining a second baseline snapshot.

    Ordinary critique is deliberately a small deterministic projection because
    the existing judge API is not stage-matched to this study.  Consequently
    this adapter is source-path compatible for browser/finalization dispatch,
    but does not claim a live ordinary-model study run.
    """

    runtime: Phase4Runtime
    _binding: BaselineBinding | None = None

    def bind(self, binding: BaselineBinding) -> None:
        if self._binding is not None and self._binding != binding:
            raise ValueError("attempt provider cannot be rebound to another baseline")
        object.__setattr__(self, "_binding", binding)

    def _check_binding(self, request: MatchedAttemptRequest) -> None:
        if self._binding != request.binding:
            raise ValueError("attempt provider is not bound to the admitted baseline")

    async def run(self, request: MatchedAttemptRequest) -> AttemptOutcome:
        self._check_binding(request)
        if request.stage == "tp_diagnosis":
            return await self._diagnose(request)
        if request.stage == "ordinary_critique":
            return self._ordinary_critique(request)
        if request.stage in {"proposal", "repair"}:
            return await self._rewrite(request)
        return await self._browser(request)

    async def _diagnose(self, request: MatchedAttemptRequest) -> DiagnosisOutcome:
        # The canonical cue API itself projects task/result to model-safe
        # fields; this input projection removes VEA/reward fields before it is
        # called so the TP arm receives only TP classification plus cues.  Keep
        # the encounter projection bounded as well; a raw runtime trace is not
        # part of either arm's model-facing evidence.
        from warp_taskgen.phase_4.eval_awareness_cue_api import run_eval_awareness_cue_api

        result = trajectory_summary(request.baseline_result)
        result.update(
            {
                key: value
                for key, value in request.baseline_result.items()
                if key.startswith("transcript_purpose_")
            }
        )
        raw = await run_eval_awareness_cue_api(
            request.evidence.task,
            result,
            iteration=1,
            sandbox_model=self.runtime.sandbox_model,
        )
        status = raw.get("status") if isinstance(raw, dict) else None
        guidance = _tp_guidance(raw.get("diagnosis") if isinstance(raw, dict) else None)
        if status != "ok" or guidance is None:
            return DiagnosisOutcome(
                status="failed",
                guidance=None,
                usage=Usage.unavailable("tp_diagnosis_usage_unavailable"),
                failure="tp_diagnosis_unavailable",
            )
        return DiagnosisOutcome(
            status="ok",
            guidance=guidance,
            usage=Usage.unavailable("tp_diagnosis_usage_unavailable"),
        )

    def _ordinary_critique(self, request: MatchedAttemptRequest) -> DiagnosisOutcome:
        # Keep this deterministic and neutral: no judge, TP, VEA, reward, or
        # raw trajectory data enters the ordinary arm's guidance.
        return DiagnosisOutcome(
            status="ok",
            guidance=OrdinaryGuidance(
                critique="Review the selected payload for clear, ordinary wording.",
                guidance="Preserve the task action, anchors, placement, and field budget.",
                rewrite_guidance="Use concise site-native wording while preserving the action.",
                focus="ordinary_clarity",
            ),
            usage=Usage.unavailable("ordinary_critique_is_deterministic"),
        )

    async def _rewrite(self, request: MatchedAttemptRequest) -> ProposalOutcome:
        from warp_taskgen.phase_4.eval_awareness_rewrite_api import (
            generate_eval_awareness_rewrite_api,
        )

        guidance = request.guidance.to_dict() if request.guidance is not None else {}
        task = sanitize_task_for_model_prompt(request.variant_task or request.evidence.task)
        raw = await generate_eval_awareness_rewrite_api(
            task,
            guidance,
            iteration=1,
            prior_feedback=[guidance] if request.stage == "repair" else None,
            parent_result=None,
            include_tp_context=False,
            sandbox_model=self.runtime.sandbox_model,
        )
        candidate = _json_object(raw)
        if candidate is None:
            return ProposalOutcome(
                status="failed",
                candidate=None,
                usage=Usage.unavailable("rewrite_usage_unavailable"),
                failure="rewrite_provider_returned_non_object",
            )
        marker = candidate.get("variant_status")
        if isinstance(marker, dict):
            marker_status = marker.get("status")
            if marker_status == "inapplicable":
                return ProposalOutcome(
                    status="inapplicable",
                    candidate=None,
                    usage=Usage.unavailable("rewrite_usage_unavailable"),
                    failure=_text(marker.get("reason")) or "rewrite_inapplicable",
                )
            if marker_status == "failed":
                return ProposalOutcome(
                    status="failed",
                    candidate=None,
                    usage=Usage.unavailable("rewrite_usage_unavailable"),
                    failure=_text(marker.get("failure_class")) or "rewrite_failed",
                )
        return ProposalOutcome(
            status="ok",
            candidate=candidate,
            usage=Usage.unavailable("rewrite_usage_unavailable"),
        )

    async def _browser(self, request: MatchedAttemptRequest) -> BrowserOutcome:
        if request.variant_task is None:
            return BrowserOutcome(
                status="failed",
                result=None,
                usage=Usage.unavailable("browser_usage_unavailable"),
                failure="finalized_variant_missing",
            )
        from warp_taskgen.phase_4.variant_eval import _evaluate_variant

        namespace = self.runtime.task_dir_root / request.artifact_namespace
        result = await _evaluate_variant(
            request.baseline_task,
            request.variant_task,
            cast(object, self.runtime.primary_instance),
            list(self.runtime.all_instances),
            {"strategy": "matched_rewrite_study"},
            request.pair_index,
            self.runtime.agent_factory,
            namespace,
            benchmark_root=self.runtime.benchmark_root,
            sandbox_model=self.runtime.sandbox_model,
            site_profile=self.runtime.site_profile,
            agent_execution=self.runtime.agent_execution,
            browser_worker_semaphore=self.runtime.browser_worker_semaphore,
            runtime_composition=self.runtime.runtime_composition,
        )
        output = _json_object(result)
        if output is None:
            return BrowserOutcome(
                status="failed",
                result=None,
                usage=Usage.unavailable("browser_usage_unavailable"),
                failure="browser_provider_returned_non_object",
            )
        return BrowserOutcome(
            status="ok",
            result=output,
            usage=Usage.unavailable("browser_usage_unavailable"),
        )


@dataclass(slots=True)
class DeterministicAttemptProvider:
    """Small ordered fake used by source-path tests; no model/browser work."""

    outcomes: tuple[AttemptOutcome, ...]
    _binding: BaselineBinding | None = None
    _index: int = 0

    def bind(self, binding: BaselineBinding) -> None:
        if self._binding is not None and self._binding != binding:
            raise ValueError("deterministic provider cannot be rebound to another baseline")
        self._binding = binding

    async def run(self, request: MatchedAttemptRequest) -> AttemptOutcome:
        if self._binding != request.binding:
            raise ValueError("deterministic provider is not bound to the admitted baseline")
        if self._index >= len(self.outcomes):
            raise AssertionError(f"no deterministic outcome for {request.arm}/{request.stage}")
        outcome = self.outcomes[self._index]
        self._index += 1
        return outcome


__all__ = ["DeterministicAttemptProvider", "ExistingPhase4AttemptAdapter"]
