"""Behavioral tests for the diagnostic Site composition seam."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

import pytest

from warp_taskgen.adversarial_actions.capability_adapters import (
    CapabilityTaskAdapter,
    capability_adapters_for_profile,
)
from warp_taskgen.editors._registry import EditorMethodSpec
from warp_taskgen.rewards.evidence import EvidencePolicy
from warp_taskgen.rewards.final_state_catalog import (
    FinalStateEvaluationRequest,
    FinalStateEvaluatorCatalog,
)
from warp_taskgen.seeding import EditorSeedResult, SeedSiteRegistration
from warp_taskgen.site_composition import (
    ActiveSitePolicy,
    CapabilityReference,
    OperationalEvidence,
    SiteBenchmarkBinding,
    SiteDefinition,
    SiteDoctorRequest,
    compile_site_definitions,
    default_site_definitions,
)
from warp_taskgen.site_composition_contracts import CapabilityFinding, SiteDoctorReport
from warp_taskgen.sites import (
    CanonicalRoute,
    ReadbackDecision,
    ReadbackObservation,
    ReadSurfacePlanFailure,
    ReadSurfaceVerificationPlan,
    ResolvedTarget,
    SiteCatalog,
    SiteRouteContractFacts,
    SurfaceResolution,
    TargetingContext,
)
from warp_taskgen.sites.read_surface import build_read_surface_plan


class FakeForumSite:
    site = "proof_forum"
    supported_benchmarks = frozenset({"webarena_verified"})

    def validate(self) -> None:
        return None

    def validate_task(self, task: Mapping[str, object]) -> tuple[str, str] | None:
        return None

    def routes(self, context: TargetingContext) -> tuple[CanonicalRoute, ...]:
        return (
            CanonicalRoute(
                id="proof_forum.thread_reply",
                site=self.site,
                kind="thread",
                allowed_start_url_patterns=(r"^/threads/[0-9]+$",),
                anchor_examples=({"thread_id": "17"},),
            ),
        )

    def match(
        self, url: str, task: Mapping[str, object], context: TargetingContext
    ) -> tuple[str, dict[str, object]] | None:
        prefix = f"{context.site_origin()}/threads/"
        if not url.startswith(prefix):
            return None
        thread_id = url.removeprefix(prefix).split("/", 1)[0]
        return ("thread", {"thread_id": thread_id}) if thread_id.isdigit() else None

    def reconstruct(
        self, kind: str, anchors: Mapping[str, object], context: TargetingContext
    ) -> str | None:
        thread_id = str(anchors.get("thread_id") or "")
        origin = context.site_origin()
        if kind != "thread" or not thread_id.isdigit() or origin is None:
            return None
        return f"{origin}/threads/{thread_id}"

    def is_listing(self, kind: str) -> bool:
        return False

    def listing_start_url(
        self, kind: str, resolved_url: str, fallback_url: str | None
    ) -> str | None:
        return resolved_url

    def canonicalize_surface_id(self, *, benchmark: str, raw_surface_id: str | None) -> str | None:
        if benchmark != "webarena_verified":
            return None
        aliases = {
            "thread_reply": "comment.body",
            "comment_body": "comment.body",
            "comment.body": "comment.body",
        }
        return aliases.get(str(raw_surface_id or "").strip())

    def resolve_profile_surface(
        self,
        *,
        benchmark: str,
        profile: Mapping[str, object],
        target_surface_id: str,
        kind: str | None = None,
        method: str | None = None,
        editor_surface_id: str | None = None,
    ) -> SurfaceResolution | None:
        canonical = self.canonicalize_surface_id(
            benchmark=benchmark, raw_surface_id=target_surface_id
        )
        surfaces = profile.get("injection_surface")
        if canonical != "comment.body" or not isinstance(surfaces, list):
            return None
        surface = next(
            (
                item
                for item in surfaces
                if isinstance(item, Mapping)
                and self.canonicalize_surface_id(
                    benchmark=benchmark, raw_surface_id=str(item.get("id") or "")
                )
                == canonical
            ),
            None,
        )
        if surface is None:
            return None
        return SurfaceResolution(
            benchmark=benchmark,
            site=self.site,
            canonical_surface_id=canonical,
            profile_surface_id=str(surface.get("id")),
            profile_surface=surface,
            evidence="fake_profile_alias",
            editor_surface_id=editor_surface_id,
        )

    def route_contract_facts(
        self, *, benchmark: str, profile: Mapping[str, object], kind: str
    ) -> SiteRouteContractFacts:
        route = self.routes(TargetingContext(benchmark=benchmark, site=self.site))[0]
        return SiteRouteContractFacts(
            allowed_start_url_patterns=route.allowed_start_url_patterns,
            anchor_examples=route.anchor_examples,
        )

    def build_read_surface_plan(
        self,
        *,
        seed_result: EditorSeedResult,
        signature: str,
        origin: str,
    ) -> object:
        return build_read_surface_plan(
            site=self.site,
            seed_result=seed_result,
            signature=signature,
            origin=origin,
            identity_keys=("thread_id", "reply_id", "actor"),
        )

    def interpret_readback(self, observation: ReadbackObservation) -> ReadbackDecision:
        payload = observation.payload
        expected = observation.identity_tokens
        if not isinstance(payload, Mapping):
            return ReadbackDecision(False, "malformed_payload")
        exact = all(str(payload.get(key) or "") == str(expected.get(key) or "") for key in expected)
        visible = payload.get("independent_reader_visible") is True
        signature = str(observation.signature or "")
        body = str(payload.get("body") or "")
        return ReadbackDecision(
            exact and visible and bool(signature) and signature in body,
            "exact_reply_visible" if exact and visible and signature in body else "reply_mismatch",
            matched_signature=signature if signature in body else None,
            rendered_text=body,
        )


class FakeForumEditor:
    def __init__(self) -> None:
        self.cleaned = False

    def create_comment(self, *, thread_id: str, actor: str, body: str) -> EditorSeedResult:
        return EditorSeedResult(
            write_tokens=MappingProxyType(
                {"thread_id": thread_id, "reply_id": "42", "actor": actor}
            ),
            created_resources=(),
            read_surface_urls=(f"/threads/{thread_id}#reply-42",),
            read_surface_provenance_source="fake_regular_user_editor",
            editor_method="create_comment",
        )

    def cleanup(self) -> None:
        self.cleaned = True


class FakeFeasibilityPolicy:
    benchmark = "webarena_verified"
    site = "proof_forum"

    def auth_self_test_path(self) -> str | None:
        return None

    def requires_authenticated_preflight(self) -> bool:
        return False

    def probe_targets(self, task: dict[str, object], instance_site_url: str) -> list[object]:
        return [{"url": f"{instance_site_url}/threads/17", "thread_id": "17"}]

    def classify_probe(self, **_: object) -> object:
        return "verified"

    def decide_source_data(self, **_: object) -> object:
        return "admissible"

    def counts_toward_run_bailout(self, classification: object) -> bool:
        return False

    def should_bailout_source_data_run(self, **_: object) -> bool:
        return False

    def restore_drop_on_run_bailout(self, issue: dict[str, object]) -> bool:
        return False


class FakeEvaluator:
    benchmark = "webarena_verified"
    site = "proof_forum"

    def evaluate(self, request: FinalStateEvaluationRequest) -> tuple[bool, str]:
        state = request.state_probe or {}
        source_event = any(
            event.get("method") == request.network_expected.get("method")
            and str(event.get("url") or "").endswith(
                str(request.network_expected.get("path") or "")
            )
            for event in request.network_trace
        )
        exact = (
            source_event
            and state.get("thread_id") == "17"
            and state.get("reply_id") == "42"
            and state.get("actor") == "researcher"
            and request.witness in str(state.get("body") or "")
        )
        return exact, "exact fake forum final state" if exact else "fake forum mismatch"


def _editor_spec() -> EditorMethodSpec:
    return EditorMethodSpec(
        benchmark="webarena_verified",
        site="proof_forum",
        method="create_comment",
        kinds=frozenset({"thread"}),
        http=("POST", "/threads/{thread_id}/replies"),
        bindings={},
        surface_id_per_kind={"thread": "comment.body"},
        required_editor_args=(),
    )


def _action_adapter() -> CapabilityTaskAdapter:
    class FakeForumActionAdapter(CapabilityTaskAdapter):
        def validate(self) -> None:
            if (
                self.site != "proof_forum"
                or self.action_kind != "submit_comment"
                or self.route_ids != ("proof_forum.comment_body.thread.create_comment",)
            ):
                raise ValueError("invalid fake forum action adapter")

    return FakeForumActionAdapter(
        id="proof_forum_reply",
        benchmark_family="webarena_verified",
        site="proof_forum",
        action_kind="submit_comment",
        route_ids=("proof_forum.comment_body.thread.create_comment",),
        archetype_id="proof_forum_reply",
        benign_task_family_id="submission_discussion_followup",
    )


def _complete_definition() -> SiteDefinition:
    site = FakeForumSite()
    binding = SiteBenchmarkBinding(
        benchmark="webarena_verified",
        targeting=CapabilityReference("supported", site, ("test.fake_forum",)),
        profile=CapabilityReference("supported", site, ("test.profile",)),
        editor_specs=CapabilityReference("supported", (_editor_spec(),), ("test.editor",)),
        seed=CapabilityReference(
            "supported",
            SeedSiteRegistration("webarena_verified", "proof_forum", lambda *_: FakeForumEditor()),
            ("test.seed",),
        ),
        feasibility=CapabilityReference("supported", FakeFeasibilityPolicy(), ("test.policy",)),
        read_surface=CapabilityReference("supported", site, ("test.read_surface",)),
        readback=CapabilityReference("supported", site, ("test.readback",)),
        final_state=CapabilityReference("supported", FakeEvaluator(), ("test.evaluator",)),
        action_cards=CapabilityReference("supported", (_action_adapter(),), ("test.action_card",)),
    )
    return SiteDefinition("proof_forum", (binding,), ("test.definition",))


def _request(
    *, site: str = "proof_forum", benchmark: str = "webarena_verified"
) -> SiteDoctorRequest:
    return SiteDoctorRequest(site=site, benchmark=benchmark, use_case="ugc_reply")


def test_complete_fake_forum_static_chain_is_complete_but_not_authorized() -> None:
    report = compile_site_definitions(
        (_complete_definition(),),
        _request(),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )

    assert report.static_status == "complete"
    assert report.status == "blocked"
    assert {finding.capability for finding in report.findings} >= {
        "registration",
        "static_readiness",
        "active_policy",
        "configured_host_feasibility",
        "admission",
        "execution",
        "scoring",
    }
    assert report.finding("active_policy").state == "missing"
    assert report.finding("benchmark_capability").state == "not_applicable"


def test_fake_forum_conformance_crosses_existing_owner_seams() -> None:
    """The diagnostic's complete fake chain is behaviorally realizable."""

    definition = _complete_definition()
    binding = definition.bindings[0]
    site = binding.targeting.owner
    assert isinstance(site, FakeForumSite)
    assert binding.profile.owner is site
    bound = SiteCatalog((site,)).bind(
        benchmark="webarena_verified",
        site="proof_forum",
        origin="https://forum.test",
    )
    resolved = bound.resolve(
        {
            "sites": ["proof_forum"],
            "start_urls": ["https://forum.test/threads/17"],
            "reward_function": {
                "eval": [
                    {
                        "evaluator": "NetworkEventEvaluator",
                        "expected": {"url": "https://forum.test/threads/17"},
                    }
                ]
            },
        }
    )
    assert isinstance(resolved, ResolvedTarget)
    assert (resolved.kind, resolved.anchors["thread_id"]) == ("thread", "17")
    profile_resolution = site.resolve_profile_surface(
        benchmark="webarena_verified",
        profile={"injection_surface": [{"id": "thread_reply"}]},
        target_surface_id="thread_reply",
        kind="thread",
        method="create_comment",
        editor_surface_id="comment.body",
    )
    assert profile_resolution is not None
    assert profile_resolution.canonical_surface_id == "comment.body"

    editor_specs = binding.editor_specs.owner
    assert editor_specs and editor_specs[0].method == "create_comment"
    policy = binding.feasibility.owner
    assert policy is not None
    assert policy.probe_targets({}, "https://forum.test") == [
        {"url": "https://forum.test/threads/17", "thread_id": "17"}
    ]
    assert policy.classify_probe() == "verified"
    assert policy.decide_source_data() == "admissible"

    registration = binding.seed.owner
    assert isinstance(registration, SeedSiteRegistration)
    editor = registration.create({}, object())
    seed_result = editor.create_comment(
        thread_id="17",
        actor="researcher",
        body="unique reply signature",
    )
    plan = bound.read_surface_plan(seed_result=seed_result, signature="unique reply signature")
    assert isinstance(plan, ReadSurfaceVerificationPlan)
    assert plan.urls == ("https://forum.test/threads/17#reply-42",)
    assert dict(plan.identity_tokens) == {
        "thread_id": "17",
        "reply_id": "42",
        "actor": "researcher",
    }

    decision = bound.interpret_readback(
        ReadbackObservation(
            "resource_signature",
            plan.identity_tokens,
            {
                "thread_id": "17",
                "reply_id": "42",
                "actor": "researcher",
                "body": "unique reply signature",
                "independent_reader_visible": True,
            },
            signature=plan.signature,
        )
    )
    assert decision == ReadbackDecision(
        True,
        "exact_reply_visible",
        matched_signature="unique reply signature",
        rendered_text="unique reply signature",
    )
    foreign = EditorSeedResult(
        write_tokens=seed_result.write_tokens,
        created_resources=(),
        read_surface_urls=("https://foreign.invalid/threads/17#reply-42",),
        read_surface_provenance_source="fake_regular_user_editor",
        editor_method="create_comment",
    )
    foreign_plan = bound.read_surface_plan(
        seed_result=foreign,
        signature="unique reply signature",
    )
    assert isinstance(foreign_plan, ReadSurfacePlanFailure)
    assert foreign_plan.reason == "foreign_read_surface"
    wrong_actor = bound.interpret_readback(
        ReadbackObservation(
            "resource_signature",
            plan.identity_tokens,
            {
                "thread_id": "17",
                "reply_id": "42",
                "actor": "someone_else",
                "body": "unique reply signature",
                "independent_reader_visible": True,
            },
            signature=plan.signature,
        )
    )
    assert (wrong_actor.verified, wrong_actor.reason) == (False, "reply_mismatch")

    request = FinalStateEvaluationRequest(
        benchmark="webarena_verified",
        site="proof_forum",
        action_kind="submit_reply",
        witness="unique reply signature",
        network_expected={"method": "POST", "path": "/threads/17/replies"},
        state_probe={
            "thread_id": "17",
            "reply_id": "42",
            "actor": "researcher",
            "body": "unique reply signature",
        },
        evidence_policy=EvidencePolicy(
            required=frozenset({"source_event", "state_readback"}),
            allowed_source=frozenset({"network_event"}),
        ),
        network_trace=({"method": "POST", "url": "https://forum.test/threads/17/replies"},),
        instance={},
        initial_events=(),
        initial_message="",
    )
    evaluator = binding.final_state.owner
    assert evaluator is not None
    evaluator_catalog = FinalStateEvaluatorCatalog.from_evaluators((evaluator,))
    assert evaluator_catalog.evaluate(request) == (True, "exact fake forum final state")
    missing_source = FinalStateEvaluationRequest(
        benchmark=request.benchmark,
        site=request.site,
        action_kind=request.action_kind,
        witness=request.witness,
        network_expected=request.network_expected,
        state_probe=request.state_probe,
        evidence_policy=request.evidence_policy,
        network_trace=(),
        instance=request.instance,
        initial_events=request.initial_events,
        initial_message=request.initial_message,
    )
    assert evaluator_catalog.evaluate(missing_source)[0] is False
    action_cards = binding.action_cards.owner
    assert action_cards and action_cards[0].route_ids == (
        "proof_forum.comment_body.thread.create_comment",
    )
    action_cards[0].validate()
    editor.cleanup()
    editor.cleanup()
    assert editor.cleaned is True


def test_missing_seed_is_actionable_without_collapsing_other_edges() -> None:
    definition = _complete_definition()
    binding = SiteBenchmarkBinding(
        benchmark=definition.bindings[0].benchmark,
        targeting=definition.bindings[0].targeting,
        profile=definition.bindings[0].profile,
        editor_specs=definition.bindings[0].editor_specs,
        seed=CapabilityReference("missing", None, ("test.seed",)),
        feasibility=definition.bindings[0].feasibility,
        read_surface=definition.bindings[0].read_surface,
        readback=definition.bindings[0].readback,
        final_state=definition.bindings[0].final_state,
        action_cards=definition.bindings[0].action_cards,
    )
    report = compile_site_definitions(
        (SiteDefinition("proof_forum", (binding,), ("test.definition",)),),
        _request(),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )

    assert report.static_status == "incomplete"
    assert report.finding("seed").state == "missing"
    assert report.finding("targeting").state == "supported"
    assert report.finding("final_state").state == "supported"


def test_cross_site_readback_owner_fails_closed() -> None:
    definition = _complete_definition()

    class OtherSite(FakeForumSite):
        site = "other_forum"

    binding = definition.bindings[0]
    conflicted = SiteBenchmarkBinding(
        benchmark=binding.benchmark,
        targeting=binding.targeting,
        profile=binding.profile,
        editor_specs=binding.editor_specs,
        seed=binding.seed,
        feasibility=binding.feasibility,
        read_surface=binding.read_surface,
        readback=CapabilityReference("supported", OtherSite(), ("test.other_site",)),
        final_state=binding.final_state,
        action_cards=binding.action_cards,
    )
    report = compile_site_definitions(
        (SiteDefinition("proof_forum", (conflicted,), ("test.definition",)),),
        _request(),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )

    assert report.static_status == "incomplete"
    assert report.finding("readback").state == "unsupported"
    assert "identity" in report.finding("readback").detail


def test_cross_site_profile_owner_fails_closed() -> None:
    definition = _complete_definition()

    class OtherSite(FakeForumSite):
        site = "other_forum"

    binding = definition.bindings[0]
    conflicted = SiteBenchmarkBinding(
        benchmark=binding.benchmark,
        targeting=binding.targeting,
        profile=CapabilityReference("supported", OtherSite(), ("test.other_profile",)),
        editor_specs=binding.editor_specs,
        seed=binding.seed,
        feasibility=binding.feasibility,
        read_surface=binding.read_surface,
        readback=binding.readback,
        final_state=binding.final_state,
        action_cards=binding.action_cards,
    )
    report = compile_site_definitions(
        (SiteDefinition("proof_forum", (conflicted,), ("test.definition",)),),
        _request(),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )

    assert report.static_status == "incomplete"
    assert report.finding("profile").state == "unsupported"


def test_unknown_or_removed_site_fails_closed() -> None:
    report = compile_site_definitions(
        (_complete_definition(),),
        _request(site="removed_forum"),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )

    assert report.static_status == "invalid"
    assert report.finding("registration").state == "unsupported"


def test_duplicate_site_definition_is_invalid_instead_of_order_selected() -> None:
    definition = _complete_definition()
    report = compile_site_definitions(
        (definition, definition),
        _request(),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )

    assert report.static_status == "invalid"
    assert report.finding("registration").state == "unsupported"
    assert "duplicate" in report.finding("registration").detail
    assert report.finding("registration").provenance


def test_default_projection_reports_missing_editor_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from warp_taskgen.editors import EDITOR_REGISTRY

    monkeypatch.delitem(EDITOR_REGISTRY, ("webarena_verified", "gitlab"))

    definitions = default_site_definitions()
    report = compile_site_definitions(
        definitions,
        SiteDoctorRequest(
            site="gitlab",
            benchmark="webarena_verified",
            use_case="phase_2_feasibility",
        ),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )

    assert report.static_status == "incomplete"
    assert report.finding("seed").state == "missing"


def test_default_projection_reports_malformed_editor_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from warp_taskgen.editors import EDITOR_REGISTRY

    monkeypatch.setitem(EDITOR_REGISTRY, ("webarena_verified", "gitlab"), object())

    definitions = default_site_definitions()
    report = compile_site_definitions(
        definitions,
        SiteDoctorRequest(
            site="gitlab",
            benchmark="webarena_verified",
            use_case="phase_2_feasibility",
        ),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )

    assert report.static_status == "incomplete"
    assert report.finding("seed").state == "missing"


def test_targeting_owner_exception_becomes_structured_failure() -> None:
    class BrokenRoutes(FakeForumSite):
        def routes(self, context: TargetingContext) -> tuple[CanonicalRoute, ...]:
            raise RuntimeError("must not escape the doctor")

    definition = _complete_definition()
    binding = definition.bindings[0]
    broken = SiteBenchmarkBinding(
        benchmark=binding.benchmark,
        targeting=CapabilityReference("supported", BrokenRoutes(), ("test.broken",)),
        profile=binding.profile,
        editor_specs=binding.editor_specs,
        seed=binding.seed,
        feasibility=binding.feasibility,
        read_surface=binding.read_surface,
        readback=binding.readback,
        final_state=binding.final_state,
        action_cards=binding.action_cards,
    )

    report = compile_site_definitions(
        (SiteDefinition("proof_forum", (broken,), ("test.definition",)),),
        _request(),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )

    assert report.static_status == "incomplete"
    assert report.finding("targeting").state == "unsupported"
    assert "RuntimeError" in report.finding("targeting").detail


def test_duplicate_editor_specs_fail_closed() -> None:
    definition = _complete_definition()
    binding = definition.bindings[0]
    duplicate = SiteBenchmarkBinding(
        benchmark=binding.benchmark,
        targeting=binding.targeting,
        profile=binding.profile,
        editor_specs=CapabilityReference(
            "supported", (_editor_spec(), _editor_spec()), ("test.editor",)
        ),
        seed=binding.seed,
        feasibility=binding.feasibility,
        read_surface=binding.read_surface,
        readback=binding.readback,
        final_state=binding.final_state,
        action_cards=binding.action_cards,
    )

    report = compile_site_definitions(
        (SiteDefinition("proof_forum", (duplicate,), ("test.definition",)),),
        _request(),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )

    assert report.static_status == "incomplete"
    assert report.finding("editor_specs").state == "unsupported"
    assert "duplicate" in report.finding("editor_specs").detail


@pytest.mark.parametrize(
    ("field", "value"),
    (("method", None), ("kinds", None), ("surface_id_per_kind", None)),
)
def test_malformed_editor_spec_fails_closed_without_digest_crash(field: str, value: object) -> None:
    definition = _complete_definition()
    binding = definition.bindings[0]
    malformed = replace(_editor_spec(), **{field: value})
    changed = SiteBenchmarkBinding(
        benchmark=binding.benchmark,
        targeting=binding.targeting,
        profile=binding.profile,
        editor_specs=CapabilityReference("supported", (malformed,), ("test.editor",)),
        seed=binding.seed,
        feasibility=binding.feasibility,
        read_surface=binding.read_surface,
        readback=binding.readback,
        final_state=binding.final_state,
        action_cards=binding.action_cards,
    )

    report = compile_site_definitions(
        (SiteDefinition("proof_forum", (changed,), ("test.definition",)),),
        _request(),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )

    assert report.static_status == "incomplete"
    assert report.finding("editor_specs").state == "unsupported"
    assert report.finding("action_cards").state == "unsupported"
    assert report.definition_digest


def test_action_card_profile_exception_fails_closed() -> None:
    class BadProfile(FakeForumSite):
        def canonicalize_surface_id(
            self, *, benchmark: str, raw_surface_id: str | None
        ) -> str | None:
            raise RuntimeError("must not escape the doctor")

    definition = _complete_definition()
    binding = definition.bindings[0]
    changed = SiteBenchmarkBinding(
        benchmark=binding.benchmark,
        targeting=binding.targeting,
        profile=CapabilityReference("supported", BadProfile(), ("test.profile",)),
        editor_specs=binding.editor_specs,
        seed=binding.seed,
        feasibility=binding.feasibility,
        read_surface=binding.read_surface,
        readback=binding.readback,
        final_state=binding.final_state,
        action_cards=binding.action_cards,
    )

    report = compile_site_definitions(
        (SiteDefinition("proof_forum", (changed,), ("test.definition",)),),
        _request(),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )

    assert report.static_status == "incomplete"
    assert report.finding("action_cards").state == "unsupported"
    assert "RuntimeError" in report.finding("action_cards").detail


def test_duplicate_action_cards_fail_closed() -> None:
    definition = next(item for item in default_site_definitions() if item.site == "reddit")
    binding = definition.bindings[0]
    cards = binding.action_cards.owner
    assert cards
    duplicate = SiteBenchmarkBinding(
        benchmark=binding.benchmark,
        targeting=binding.targeting,
        profile=binding.profile,
        editor_specs=binding.editor_specs,
        seed=binding.seed,
        feasibility=binding.feasibility,
        read_surface=binding.read_surface,
        readback=binding.readback,
        final_state=binding.final_state,
        action_cards=CapabilityReference("supported", (cards[0], cards[0]), ("test.action_cards",)),
    )

    report = compile_site_definitions(
        (SiteDefinition("reddit", (duplicate,), ("test.definition",)),),
        SiteDoctorRequest(
            site="reddit",
            benchmark="webarena_verified",
            use_case="phase_1_generation",
        ),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )

    assert report.static_status == "incomplete"
    assert report.finding("action_cards").state == "unsupported"
    assert "duplicate" in report.finding("action_cards").detail


def test_action_card_with_unknown_route_fails_closed() -> None:
    definition = next(item for item in default_site_definitions() if item.site == "gitlab")
    binding = definition.bindings[0]
    cards = binding.action_cards.owner
    assert cards
    invalid_cards = (replace(cards[0], route_ids=("gitlab.missing.route.unknown",)),)
    changed = SiteBenchmarkBinding(
        benchmark=binding.benchmark,
        targeting=binding.targeting,
        profile=binding.profile,
        editor_specs=binding.editor_specs,
        seed=binding.seed,
        feasibility=binding.feasibility,
        read_surface=binding.read_surface,
        readback=binding.readback,
        final_state=binding.final_state,
        action_cards=CapabilityReference("supported", invalid_cards, ("test.action_cards",)),
    )

    report = compile_site_definitions(
        (SiteDefinition("gitlab", (changed,), ("test.definition",)),),
        SiteDoctorRequest(
            site="gitlab",
            benchmark="webarena_verified",
            use_case="phase_1_generation",
        ),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )

    assert report.static_status == "incomplete"
    assert report.finding("action_cards").state == "unsupported"
    assert "closure" in report.finding("action_cards").detail


def test_action_target_contract_requires_matching_target_editor() -> None:
    definition = next(item for item in default_site_definitions() if item.site == "gitlab")
    binding = definition.bindings[0]
    card = next(
        item
        for item in capability_adapters_for_profile(
            "tier2_gitlab_public_comment_pilot",
            benchmark_family="webarena_verified",
            sites=("gitlab",),
        )
        if item.id == "gitlab_issue_description_public_followup_comment"
    )
    assert card.action_target_contract is not None

    def report_for(candidate: CapabilityTaskAdapter) -> SiteDoctorReport:
        changed = SiteBenchmarkBinding(
            benchmark=binding.benchmark,
            targeting=binding.targeting,
            profile=binding.profile,
            editor_specs=binding.editor_specs,
            seed=binding.seed,
            feasibility=binding.feasibility,
            read_surface=binding.read_surface,
            readback=binding.readback,
            final_state=binding.final_state,
            action_cards=CapabilityReference("supported", (candidate,), ("test.action_cards",)),
        )
        return compile_site_definitions(
            (SiteDefinition("gitlab", (changed,), ("test.definition",)),),
            SiteDoctorRequest(
                site="gitlab",
                benchmark="webarena_verified",
                use_case="phase_1_generation",
            ),
            active_policy=ActiveSitePolicy(),
            operational_evidence=OperationalEvidence(),
        )

    valid = report_for(card)
    wrong_target = replace(card.action_target_contract, target_editor_method="create_group")
    invalid = report_for(replace(card, action_target_contract=wrong_target))

    assert valid.finding("action_cards").state == "supported"
    assert invalid.static_status == "incomplete"
    assert invalid.finding("action_cards").state == "unsupported"


def test_malformed_action_card_fails_closed_without_digest_crash() -> None:
    definition = _complete_definition()
    binding = definition.bindings[0]
    malformed = replace(_action_adapter(), route_ids=None)  # type: ignore[arg-type]
    changed = SiteBenchmarkBinding(
        benchmark=binding.benchmark,
        targeting=binding.targeting,
        profile=binding.profile,
        editor_specs=binding.editor_specs,
        seed=binding.seed,
        feasibility=binding.feasibility,
        read_surface=binding.read_surface,
        readback=binding.readback,
        final_state=binding.final_state,
        action_cards=CapabilityReference("supported", (malformed,), ("test.action_cards",)),
    )

    report = compile_site_definitions(
        (SiteDefinition("proof_forum", (changed,), ("test.definition",)),),
        _request(),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )

    assert report.static_status == "incomplete"
    assert report.finding("action_cards").state == "unsupported"
    assert report.definition_digest


def test_definition_digest_ignores_editor_snapshot_order() -> None:
    definition = _complete_definition()
    binding = definition.bindings[0]
    second = EditorMethodSpec(
        benchmark="webarena_verified",
        site="proof_forum",
        method="update_reply",
        kinds=frozenset({"thread"}),
        http=("POST", "/threads/{thread_id}/replies/{reply_id}"),
        bindings={},
        surface_id_per_kind={"thread": "comment.body"},
        required_editor_args=(),
    )

    def with_specs(specs: tuple[EditorMethodSpec, ...]) -> SiteDefinition:
        changed = SiteBenchmarkBinding(
            benchmark=binding.benchmark,
            targeting=binding.targeting,
            profile=binding.profile,
            editor_specs=CapabilityReference("supported", specs, ("test.editor",)),
            seed=binding.seed,
            feasibility=binding.feasibility,
            read_surface=binding.read_surface,
            readback=binding.readback,
            final_state=binding.final_state,
            action_cards=binding.action_cards,
        )
        return SiteDefinition("proof_forum", (changed,), ("test.definition",))

    first = compile_site_definitions(
        (with_specs((_editor_spec(), second)),),
        _request(),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )
    reversed_order = compile_site_definitions(
        (with_specs((second, _editor_spec())),),
        _request(),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )

    assert first.definition_digest == reversed_order.definition_digest


def test_definition_digest_tracks_editor_and_action_contract_semantics() -> None:
    definition = _complete_definition()
    binding = definition.bindings[0]

    def changed_definition(
        *,
        editor_specs: tuple[EditorMethodSpec, ...] | None = None,
        action_cards: tuple[CapabilityTaskAdapter, ...] | None = None,
    ) -> SiteDefinition:
        changed = SiteBenchmarkBinding(
            benchmark=binding.benchmark,
            targeting=binding.targeting,
            profile=binding.profile,
            editor_specs=CapabilityReference(
                "supported", editor_specs or binding.editor_specs.owner, ("test.editor",)
            ),
            seed=binding.seed,
            feasibility=binding.feasibility,
            read_surface=binding.read_surface,
            readback=binding.readback,
            final_state=binding.final_state,
            action_cards=CapabilityReference(
                "supported", action_cards or binding.action_cards.owner, ("test.action_card",)
            ),
        )
        return SiteDefinition("proof_forum", (changed,), ("test.definition",))

    baseline = compile_site_definitions(
        (definition,),
        _request(),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )
    editor_changed = compile_site_definitions(
        (
            changed_definition(
                editor_specs=(replace(_editor_spec(), required_editor_args=("body",)),)
            ),
        ),
        _request(),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )
    action_changed = compile_site_definitions(
        (
            changed_definition(
                action_cards=(replace(_action_adapter(), archetype_id="changed_archetype"),)
            ),
        ),
        _request(),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )

    assert editor_changed.definition_digest != baseline.definition_digest
    assert action_changed.definition_digest != baseline.definition_digest


def test_definition_digest_ignores_action_route_set_order() -> None:
    definition = next(item for item in default_site_definitions() if item.site == "reddit")
    binding = definition.bindings[0]
    cards = binding.action_cards.owner
    assert cards and len(cards) >= 2
    route_ids = (cards[0].route_ids[0], cards[1].route_ids[0])

    def report_for(candidate: CapabilityTaskAdapter) -> SiteDoctorReport:
        changed = SiteBenchmarkBinding(
            benchmark=binding.benchmark,
            targeting=binding.targeting,
            profile=binding.profile,
            editor_specs=binding.editor_specs,
            seed=binding.seed,
            feasibility=binding.feasibility,
            read_surface=binding.read_surface,
            readback=binding.readback,
            final_state=binding.final_state,
            action_cards=CapabilityReference("supported", (candidate,), ("test.action_cards",)),
        )
        return compile_site_definitions(
            (SiteDefinition("reddit", (changed,), ("test.definition",)),),
            SiteDoctorRequest(
                site="reddit",
                benchmark="webarena_verified",
                use_case="phase_1_generation",
            ),
            active_policy=ActiveSitePolicy(),
            operational_evidence=OperationalEvidence(),
        )

    first = report_for(replace(cards[0], route_ids=route_ids))
    reversed_order = report_for(replace(cards[0], route_ids=tuple(reversed(route_ids))))

    assert first.static_status == reversed_order.static_status == "complete"
    assert first.definition_digest == reversed_order.definition_digest


def test_report_freezes_findings_supplied_as_a_list() -> None:
    finding = CapabilityFinding(
        capability="registration",
        state="supported",
        outcome="pass",
        code="registration.supported",
        detail="typed definition",
    )
    mutable = [finding]
    report = SiteDoctorReport(
        site="proof_forum",
        benchmark="webarena_verified",
        use_case="ugc_reply",
        static_status="complete",
        status="blocked",
        definition_digest="abc",
        findings=mutable,  # type: ignore[arg-type]
    )
    mutable.clear()

    assert report.findings == (finding,)


def test_comparison_only_benchmark_is_rejected() -> None:
    report = compile_site_definitions(
        (_complete_definition(),),
        SiteDoctorRequest(site="proof_forum", benchmark="wasp", use_case="ugc_reply"),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )

    assert report.static_status == "invalid"
    assert any("comparison" in finding.detail for finding in report.findings)


def test_report_json_and_digest_are_stable_and_secret_free() -> None:
    first = compile_site_definitions(
        (_complete_definition(),),
        _request(),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )
    second = compile_site_definitions(
        (_complete_definition(),),
        _request(),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )

    assert first.to_json() == second.to_json()
    assert first.digest == second.digest
    payload = json.loads(first.to_json())
    assert payload["contract_kit_version"] == "warp-site-composition-v1"
    assert payload["definition_digest"] == first.definition_digest
    assert "https://" not in first.to_json()
    assert "cookie" not in first.to_json().lower()
    assert "header" not in first.to_json().lower()


def test_capability_reference_rejects_owner_for_non_supported_state() -> None:
    with pytest.raises(ValueError, match="owner"):
        CapabilityReference("missing", object(), ())


def test_definition_rejects_url_provenance_before_it_can_reach_a_report() -> None:
    with pytest.raises(ValueError, match="provenance"):
        CapabilityReference("supported", FakeForumSite(), ("https://private.invalid",))


@pytest.mark.parametrize(
    "provenance",
    (
        "Authorization:Bearer:SECRET",
        "cookie=SECRET",
        "raw_payload",
    ),
)
def test_definition_rejects_sensitive_provenance(provenance: str) -> None:
    with pytest.raises(ValueError, match="provenance"):
        CapabilityReference("supported", FakeForumSite(), (provenance,))


def test_finding_rejects_sensitive_detail() -> None:
    with pytest.raises(ValueError, match="detail"):
        CapabilityFinding(
            capability="registration",
            state="unsupported",
            outcome="failure",
            code="registration.unsupported",
            detail="Authorization: Bearer SECRET",
        )


def test_typed_report_contract_rejects_contradictory_states() -> None:
    with pytest.raises(ValueError, match="contradict"):
        CapabilityFinding(
            capability="registration",
            state="supported",
            outcome="failure",
            code="registration.supported",
            detail="typed definition",
        )
    finding = CapabilityFinding(
        capability="registration",
        state="supported",
        outcome="pass",
        code="registration.supported",
        detail="typed definition",
    )
    with pytest.raises(ValueError, match="contradict"):
        SiteDoctorReport(
            site="proof_forum",
            benchmark="webarena_verified",
            use_case="ugc_reply",
            static_status="invalid",
            status="blocked",
            definition_digest="",
            findings=(finding,),
        )
