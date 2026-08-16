from __future__ import annotations

from warp_taskgen.site_composition import (
    ActiveSitePolicy,
    OperationalEvidence,
    SiteDoctorRequest,
    compile_site_definitions,
    default_site_definitions,
)


def _request(*, carrier: str = "listing_reply.body") -> SiteDoctorRequest:
    return SiteDoctorRequest(
        site="classifieds",
        benchmark="visualwebarena",
        use_case="ugc_reply",
        carrier=carrier,
        action_kind="answer_opposite_binary_label",
    )


def _compile(*, active: bool = False, carrier: str = "listing_reply.body"):
    policy = (
        ActiveSitePolicy(frozenset({("visualwebarena", "classifieds", "ugc_reply")}))
        if active
        else ActiveSitePolicy()
    )
    return compile_site_definitions(
        default_site_definitions(),
        _request(carrier=carrier),
        active_policy=policy,
        operational_evidence=OperationalEvidence(),
    )


def test_classifieds_definition_closes_without_hidden_activation() -> None:
    report = _compile()

    assert report.site == "classifieds"
    assert report.benchmark == "visualwebarena"
    assert report.static_status == "complete"
    assert report.status == "blocked"
    assert report.finding("active_policy").state == "missing"
    assert report.finding("configured_host_feasibility").state == "missing"
    assert report.finding("final_state").state == "not_applicable"
    assert report.finding("action_cards").state == "supported"
    assert report.definition_digest


def test_named_policy_does_not_invent_configured_host_evidence() -> None:
    report = _compile(active=True)

    assert report.static_status == "complete"
    assert report.status == "blocked"
    assert report.finding("active_policy").state == "supported"
    assert report.finding("configured_host_feasibility").state == "missing"


def test_classifieds_title_is_not_a_supported_carrier() -> None:
    report = _compile(carrier="listing.title")

    assert report.static_status == "incomplete"
    assert report.finding("profile").state == "missing"
    assert report.finding("editor_specs").state == "missing"


def test_removing_classifieds_definition_fails_closed_without_affecting_defaults() -> None:
    definitions = default_site_definitions()
    without_classifieds = tuple(item for item in definitions if item.site != "classifieds")

    removed = compile_site_definitions(
        without_classifieds,
        _request(),
        active_policy=ActiveSitePolicy(),
        operational_evidence=OperationalEvidence(),
    )

    assert removed.static_status == "invalid"
    assert removed.status == "invalid"
    assert "classifieds" not in removed.to_json()
    assert {item.site for item in without_classifieds} == {"gitlab", "reddit"}
