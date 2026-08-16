"""Behavioral tests for the immutable static Site Composition seam."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from warp_taskgen.site_composition import (
    SITE_OWNER_ROLE_ORDER,
    SiteBenchmarkComposition,
    SiteComposition,
    SiteCompositionCheckRequest,
    SiteCompositionUseCase,
    SiteCompositionUseCaseCatalog,
    SiteOwnerDeclaration,
    check_site_composition,
    default_site_compositions,
    site_composition_digest,
)
from warp_taskgen.site_composition_contracts import (
    SiteCompositionCheckReport,
    SiteCompositionFinding,
)


def _owner(
    owner_id: str,
    *,
    state: str = "supported",
    provenance: tuple[str, ...] = (),
) -> SiteOwnerDeclaration:
    return SiteOwnerDeclaration(
        state=state,
        owner_id=owner_id if state == "supported" else None,
        contract_version="v1",
        provenance=provenance or (f"test.{owner_id}",),
    )


def _composition(
    *,
    site: str = "synthetic_discussion_forum",
    benchmark: str = "webarena_verified",
    carriers: tuple[str, ...] = ("comment.body",),
    action_kinds: tuple[str, ...] = ("submit_comment",),
) -> SiteComposition:
    projection = SiteBenchmarkComposition(
        benchmark=benchmark,
        site_targeting=_owner("test.site_targeting"),
        profile=_owner("test.profile"),
        editor_specification=_owner("test.editor_specification"),
        regular_participant_writer=_owner("test.regular_participant_writer"),
        feasibility=_owner("test.feasibility"),
        read_surface=_owner("test.read_surface"),
        readback=_owner("test.readback"),
        final_state_evaluation=_owner("test.final_state_evaluation"),
        action_cards=_owner("test.action_cards"),
        supported_carriers=carriers,
        supported_action_kinds=action_kinds,
        provenance=(f"test.{site}",),
    )
    return SiteComposition(
        site=site,
        benchmark_compositions=(projection,),
        provenance=(f"test.{site}",),
    )


def _request(
    *,
    site: str = "synthetic_discussion_forum",
    benchmark: str = "webarena_verified",
    use_case: str = "public_reply",
    carrier: str | None = "comment.body",
    action_kind: str | None = "submit_comment",
) -> SiteCompositionCheckRequest:
    return SiteCompositionCheckRequest(
        site=site,
        benchmark=benchmark,
        use_case=use_case,
        carrier=carrier,
        action_kind=action_kind,
    )


def test_site_owner_declaration_is_immutable_data_only() -> None:
    declaration = _owner("test.site_targeting")

    assert declaration.state == "supported"
    assert declaration.owner_id == "test.site_targeting"
    assert declaration.contract_version == "v1"
    with pytest.raises((AttributeError, TypeError)):
        declaration.owner_id = "changed"  # type: ignore[misc]
    assert not hasattr(declaration, "routes")


@pytest.mark.parametrize("state", ("missing", "unsupported", "not_applicable"))
def test_non_supported_declarations_forbid_owner_ids(state: str) -> None:
    with pytest.raises(ValueError, match="forbid"):
        SiteOwnerDeclaration(state=state, owner_id="test.owner")


def test_supported_declaration_requires_owner_id() -> None:
    with pytest.raises(ValueError, match="require owner_id"):
        SiteOwnerDeclaration(state="supported")


def test_not_applicable_is_compiler_derived_and_forbidden_in_declarations() -> None:
    with pytest.raises(ValueError, match="compiler-derived"):
        SiteOwnerDeclaration(state="not_applicable")  # type: ignore[arg-type]


def test_site_owner_ids_are_canonicalized_for_digest_identity() -> None:
    first = SiteOwnerDeclaration("supported", "Test.Owner", "V1", ("test.owner",))
    second = SiteOwnerDeclaration("supported", "test.owner", "v1", ("test.owner",))

    assert first == second


def test_public_reply_is_host_owned_static_diagnostic_use_case() -> None:
    use_case = SiteCompositionUseCaseCatalog.default().resolve("public_reply")

    assert use_case is not None
    assert use_case.scope == "static_diagnostic"
    assert use_case.requires_carrier is True
    assert use_case.requires_action_kind is True
    assert "final_state_evaluation" not in use_case.required_owner_roles
    assert use_case.required_owner_roles == (
        "site_targeting",
        "profile",
        "editor_specification",
        "regular_participant_writer",
        "feasibility",
        "read_surface",
        "readback",
        "action_cards",
    )


def test_site_owner_role_order_is_canonical() -> None:
    assert SITE_OWNER_ROLE_ORDER == (
        "site_targeting",
        "profile",
        "editor_specification",
        "regular_participant_writer",
        "feasibility",
        "read_surface",
        "readback",
        "final_state_evaluation",
        "action_cards",
    )


def test_complete_public_reply_is_static_complete_and_final_state_is_compiler_na() -> None:
    composition = _composition(
        site="classifieds",
        benchmark="visualwebarena",
        carriers=("listing_reply.body",),
        action_kinds=("answer_opposite_binary_label",),
    )
    report = check_site_composition(
        (composition,),
        _request(
            site="classifieds",
            benchmark="visualwebarena",
            carrier="listing_reply.body",
            action_kind="answer_opposite_binary_label",
        ),
    )

    assert report.static_status == "complete"
    assert report.finding("final_state_evaluation").state == "not_applicable"
    assert report.finding("final_state_evaluation").outcome == "pass"
    assert report.site_composition_digest is not None
    assert report.site_composition_digest.startswith("sha256:")
    assert len(report.site_composition_digest) == len("sha256:") + 64


def test_public_reply_requires_exact_carrier_and_action_kind() -> None:
    composition = _composition(
        site="classifieds",
        benchmark="visualwebarena",
        carriers=("listing_reply.body",),
        action_kinds=("answer_opposite_binary_label",),
    )
    missing = check_site_composition(
        (composition,),
        _request(
            site="classifieds",
            benchmark="visualwebarena",
            carrier=None,
            action_kind=None,
        ),
    )
    wrong = check_site_composition(
        (composition,),
        _request(
            site="classifieds",
            benchmark="visualwebarena",
            carrier="listing.title",
            action_kind="unknown_action",
        ),
    )

    assert missing.static_status == "incomplete"
    assert missing.finding("profile").state == "missing"
    assert missing.finding("action_cards").state == "missing"
    assert wrong.static_status == "incomplete"
    assert wrong.finding("editor_specification").state == "missing"
    assert wrong.finding("action_cards").state == "missing"


def test_not_required_roles_are_compiler_derived_not_declaration_claims() -> None:
    composition = _composition(
        site="classifieds",
        benchmark="visualwebarena",
        carriers=("listing_reply.body",),
        action_kinds=("answer_opposite_binary_label",),
    )
    report = check_site_composition(
        (composition,),
        _request(
            site="classifieds",
            benchmark="visualwebarena",
            carrier="listing_reply.body",
            action_kind="answer_opposite_binary_label",
        ),
    )

    assert report.finding("final_state_evaluation").state == "not_applicable"


def test_phase_2_feasibility_requires_feasibility_and_readback() -> None:
    composition = _composition()
    projection = composition.benchmark_compositions[0]
    changed = replace(
        projection,
        feasibility=_owner("test.feasibility", state="missing"),
        readback=_owner("test.readback", state="unsupported"),
    )
    composition = replace(composition, benchmark_compositions=(changed,))

    report = check_site_composition(
        (composition,),
        _request(use_case="phase_2_feasibility", carrier=None, action_kind=None),
    )

    assert report.static_status == "incomplete"
    assert report.finding("feasibility").state == "missing"
    assert report.finding("readback").state == "unsupported"
    assert report.finding("action_cards").state == "not_applicable"


def test_static_check_never_dispatches_behavior_methods() -> None:
    class ExplodingDeclaration:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError(f"behavior was dispatched: {name}")

    # Declarations accept semantic owner IDs only, so an executable behavior
    # object cannot cross the static seam at all.
    with pytest.raises((TypeError, ValueError)):
        SiteOwnerDeclaration("supported", ExplodingDeclaration())  # type: ignore[arg-type]
    report = check_site_composition(
        (_composition(),),
        _request(),
    )
    assert report.static_status == "complete"


def test_missing_owner_is_actionable_without_collapsing_other_edges() -> None:
    composition = _composition()
    projection = composition.benchmark_compositions[0]
    changed = replace(
        projection,
        regular_participant_writer=_owner("test.regular_participant_writer", state="missing"),
    )
    report = check_site_composition(
        (replace(composition, benchmark_compositions=(changed,)),),
        _request(),
    )

    assert report.static_status == "incomplete"
    assert report.finding("regular_participant_writer").state == "missing"
    assert report.finding("site_targeting").state == "supported"


def test_unknown_or_removed_site_fails_closed() -> None:
    report = check_site_composition(
        default_site_compositions(),
        _request(
            site="removed_forum", use_case="phase_2_feasibility", carrier=None, action_kind=None
        ),
    )

    assert report.static_status == "invalid"
    assert report.site_composition_digest is None
    assert report.finding("registration").state == "unsupported"


def test_duplicate_site_composition_is_invalid() -> None:
    composition = _composition()
    report = check_site_composition(
        (composition, composition),
        _request(),
    )

    assert report.static_status == "invalid"
    assert report.site_composition_digest is None
    assert "duplicate" in report.finding("registration").detail


def test_missing_benchmark_projection_is_invalid() -> None:
    report = check_site_composition(
        (_composition(benchmark="visualwebarena"),),
        _request(benchmark="webarena_verified"),
    )

    assert report.static_status == "invalid"
    assert report.site_composition_digest is None


def test_comparison_only_benchmark_is_invalid() -> None:
    report = check_site_composition(
        (_composition(benchmark="wasp"),),
        _request(benchmark="wasp"),
    )

    assert report.static_status == "invalid"
    assert report.site_composition_digest is None
    assert "comparison" in report.finding("registration").detail


def test_digest_is_stable_for_site_and_benchmark_order() -> None:
    compositions = default_site_compositions()
    request = SiteCompositionCheckRequest(
        site="gitlab",
        benchmark="webarena_verified",
        use_case="phase_2_feasibility",
    )

    first = check_site_composition(compositions, request)
    second = check_site_composition(tuple(reversed(compositions)), request)

    assert first.to_json() == second.to_json()
    assert first.site_composition_digest == second.site_composition_digest
    assert first.site_composition_digest == site_composition_digest(
        next(item for item in compositions if item.site == "gitlab")
    )


def test_digest_changes_for_semantic_declaration_changes() -> None:
    baseline = _composition()
    changed = replace(
        baseline,
        benchmark_compositions=(
            replace(
                baseline.benchmark_compositions[0],
                action_cards=_owner("test.changed_action_cards"),
            ),
        ),
    )

    assert site_composition_digest(baseline) != site_composition_digest(changed)


def test_digest_payload_contains_contract_and_catalog_requirements() -> None:
    composition = _composition()
    report = check_site_composition((composition,), _request())
    payload = json.loads(report.to_json())

    assert payload["schema"] == "warp-site-composition-check-v1"
    assert payload["scope"] == "static_site_composition_only"
    assert payload["site_composition_digest"].startswith("sha256:")
    assert payload["active_policy_checked"] is False
    assert payload["live_evidence_checked"] is False


def test_report_is_static_only_and_findings_are_frozen() -> None:
    finding = SiteCompositionFinding(
        capability="registration",
        state="supported",
        outcome="pass",
        code="registration.supported",
        detail="typed composition",
    )
    closure = SiteCompositionFinding(
        capability="static_closure",
        state="supported",
        outcome="pass",
        code="static_closure.supported",
        detail="complete static closure",
    )
    report = SiteCompositionCheckReport(
        site="synthetic_discussion_forum",
        benchmark="webarena_verified",
        use_case="public_reply",
        static_status="complete",
        site_composition_digest="sha256:" + "0" * 64,
        findings=[finding, closure],  # type: ignore[arg-type]
    )

    assert report.findings == (finding, closure)
    assert not hasattr(report, "status")
    assert report.to_dict()["findings"] == [finding.to_dict(), closure.to_dict()]


def test_report_rejects_contradictory_static_status_and_findings() -> None:
    finding = SiteCompositionFinding(
        capability="registration",
        state="supported",
        outcome="pass",
        code="registration.supported",
        detail="typed composition",
    )
    with pytest.raises(ValueError, match="registration and static_closure"):
        SiteCompositionCheckReport(
            site="synthetic_discussion_forum",
            benchmark="webarena_verified",
            use_case="public_reply",
            static_status="complete",
            site_composition_digest="sha256:" + "0" * 64,
            findings=(finding,),
        )
    non_registration = replace(finding, state="not_applicable")
    closure = SiteCompositionFinding(
        capability="static_closure",
        state="supported",
        outcome="pass",
        code="static_closure.supported",
        detail="complete static closure",
    )
    with pytest.raises(ValueError, match="passing registration"):
        SiteCompositionCheckReport(
            site="synthetic_discussion_forum",
            benchmark="webarena_verified",
            use_case="public_reply",
            static_status="complete",
            site_composition_digest="sha256:" + "0" * 64,
            findings=(non_registration, closure),
        )


def test_invalid_report_rejects_digest() -> None:
    finding = SiteCompositionFinding(
        capability="registration",
        state="unsupported",
        outcome="failure",
        code="registration.unsupported",
        detail="invalid composition",
    )
    with pytest.raises(ValueError, match="cannot carry"):
        SiteCompositionCheckReport(
            site="invalid",
            benchmark="invalid",
            use_case="invalid",
            static_status="invalid",
            site_composition_digest="sha256:" + "0" * 64,
            findings=(finding,),
        )


def test_sensitive_provenance_and_detail_are_rejected() -> None:
    with pytest.raises(ValueError, match="provenance"):
        SiteOwnerDeclaration("supported", "test.owner", "v1", ("https://private.invalid",))
    with pytest.raises(ValueError, match="detail"):
        SiteCompositionFinding(
            capability="registration",
            state="unsupported",
            outcome="failure",
            code="registration.unsupported",
            detail="Authorization: Bearer SECRET",
        )
    with pytest.raises(ValueError, match="identities"):
        SiteCompositionCheckReport(
            site="https://private.invalid",
            benchmark="webarena_verified",
            use_case="public_reply",
            static_status="invalid",
            site_composition_digest=None,
            findings=(),
        )


def test_use_case_catalog_rejects_unknown_scope_and_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="scope"):
        SiteCompositionUseCase(
            id="bad_scope",
            scope="live_host",  # type: ignore[arg-type]
            required_owner_roles=(),
        )
    entry = SiteCompositionUseCase(
        id="duplicate",
        scope="static_diagnostic",
        required_owner_roles=(),
    )
    with pytest.raises(ValueError, match="unique"):
        SiteCompositionUseCaseCatalog((entry, entry))
