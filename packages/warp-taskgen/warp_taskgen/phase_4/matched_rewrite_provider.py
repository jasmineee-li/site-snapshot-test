"""The one typed attempt-provider seam used by the matched rewrite study."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import cast

from warp_taskgen.phase_4.anthropic_client import (
    classify_api_exception,
    get_client,
)
from warp_taskgen.phase_4.matched_rewrite_contracts import (
    AttemptOutcome,
    BaselineBinding,
    BrowserOutcome,
    DiagnosisOutcome,
    JsonObject,
    MatchedAttemptRequest,
    MatchedCallPolicy,
    Phase4Runtime,
    ProposalOutcome,
    TPGuidance,
    Usage,
)
from warp_taskgen.phase_4.matched_rewrite_ordinary_api import (
    browser_usage,
    run_ordinary_critique,
    usage_from_diagnostics,
)
from warp_taskgen.phase_4.prompt_contracts import trajectory_summary
from warp_taskgen.phase_4.prompt_payloads import (
    sanitize_task_for_model_prompt,
)


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
    """Dispatch existing Phase 4 owners for one matched study opportunity.

    The ordinary diagnosis is a neutral, stage-matched host Messages call. The
    deterministic provider below remains available only for source tests; this
    adapter never substitutes it on the live study path.
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

    def _client(self) -> object:
        """Use an injected test client or the shared host Messages client."""

        return self.runtime.host_client or get_client()

    async def run(self, request: MatchedAttemptRequest) -> AttemptOutcome:
        self._check_binding(request)
        if request.stage == "tp_diagnosis":
            return await self._diagnose(request)
        if request.stage == "ordinary_critique":
            return await self._ordinary_critique(request)
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
        policy = self._call_policy(request)
        try:
            raw = await run_eval_awareness_cue_api(
                request.evidence.task,
                result,
                iteration=1,
                sandbox_model=policy.model,
                client=self._client(),  # type: ignore[arg-type]
                max_tokens=policy.max_tokens,
                semantic_retries=policy.semantic_retries,
                transport_retries=policy.transport_retries,
                temperature=policy.temperature,
            )
        except Exception as exc:
            return DiagnosisOutcome(
                status="failed",
                guidance=None,
                usage=Usage.unavailable(f"tp_diagnosis_api_error:{classify_api_exception(exc)}"),
                failure="tp_diagnosis_unavailable",
            )
        status = raw.get("status") if isinstance(raw, dict) else None
        guidance = _tp_guidance(
            raw.get("diagnosis") if isinstance(raw, dict) and raw.get("diagnosis") else raw
        )
        usage = usage_from_diagnostics(
            raw.get("api_diagnostics") if isinstance(raw, dict) else None,
            model=policy.model,
            fallback_reason="tp_diagnosis_usage_unavailable",
        )
        if status != "ok" or guidance is None:
            return DiagnosisOutcome(
                status="failed",
                guidance=None,
                usage=usage,
                failure="tp_diagnosis_unavailable",
            )
        return DiagnosisOutcome(
            status="ok",
            guidance=guidance,
            usage=usage,
        )

    def _call_policy(self, request: MatchedAttemptRequest) -> MatchedCallPolicy:
        return request.call_policy or MatchedCallPolicy.for_model(self.runtime.sandbox_model)

    async def _ordinary_critique(self, request: MatchedAttemptRequest) -> DiagnosisOutcome:
        """Run a neutral ordinary model critique at the TP diagnosis boundary."""

        try:
            client = self._client()
        except Exception as exc:
            return DiagnosisOutcome(
                status="failed",
                guidance=None,
                usage=Usage.unavailable(
                    f"ordinary_critique_api_error:{classify_api_exception(exc)}"
                ),
                failure="ordinary_critique_unavailable",
            )
        return await run_ordinary_critique(
            request,
            policy=self._call_policy(request),
            client=client,
        )

    async def _rewrite(self, request: MatchedAttemptRequest) -> ProposalOutcome:
        from warp_taskgen.phase_4.eval_awareness_rewrite_api import (
            generate_eval_awareness_rewrite_api,
        )

        policy = self._call_policy(request)
        guidance = request.guidance.to_dict() if request.guidance is not None else {}
        task = sanitize_task_for_model_prompt(request.variant_task or request.evidence.task)
        try:
            raw = await generate_eval_awareness_rewrite_api(
                task,
                guidance,
                iteration=1,
                prior_feedback=[guidance] if request.stage == "repair" else None,
                parent_result=None,
                include_tp_context=False,
                sandbox_model=policy.model,
                client=self._client(),  # type: ignore[arg-type]
                include_api_diagnostics=True,
                max_tokens=policy.max_tokens,
                semantic_retries=policy.semantic_retries,
                transport_retries=policy.transport_retries,
                temperature=policy.temperature,
            )
        except Exception as exc:
            return ProposalOutcome(
                status="failed",
                candidate=None,
                usage=Usage.unavailable(f"rewrite_api_error:{classify_api_exception(exc)}"),
                failure="rewrite_provider_failed",
            )
        candidate = _json_object(raw)
        if candidate is None:
            return ProposalOutcome(
                status="failed",
                candidate=None,
                usage=Usage.unavailable("rewrite_usage_unavailable"),
                failure="rewrite_provider_returned_non_object",
            )
        diagnostics = candidate.pop("matched_rewrite_api_diagnostics", None)
        marker = candidate.get("variant_status")
        if diagnostics is None and isinstance(marker, dict):
            diagnostics = marker.get("api_diagnostics")
        usage = usage_from_diagnostics(
            diagnostics,
            model=policy.model,
            fallback_reason="rewrite_usage_unavailable",
        )
        if isinstance(marker, dict):
            marker_status = marker.get("status")
            if marker_status == "inapplicable":
                return ProposalOutcome(
                    status="inapplicable",
                    candidate=None,
                    usage=usage,
                    failure=_text(marker.get("reason")) or "rewrite_inapplicable",
                )
            if marker_status == "failed":
                return ProposalOutcome(
                    status="failed",
                    candidate=None,
                    usage=usage,
                    failure=_text(marker.get("failure_class")) or "rewrite_failed",
                )
        return ProposalOutcome(
            status="ok",
            candidate=candidate,
            usage=usage,
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
            sandbox_model=self._call_policy(request).model,
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
            usage=browser_usage(output),
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
