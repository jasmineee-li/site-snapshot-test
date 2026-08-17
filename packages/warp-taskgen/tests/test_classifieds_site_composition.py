"""Static Site Composition checks for the Classifieds diagnostic."""

from __future__ import annotations

from warp_taskgen.site_composition import (
    SiteCompositionCheckRequest,
    check_site_composition,
    default_site_compositions,
)


def _request(
    *,
    carrier: str | None = "listing_reply.body",
    action_kind: str | None = "answer_opposite_binary_label",
) -> SiteCompositionCheckRequest:
    return SiteCompositionCheckRequest(
        site="classifieds",
        benchmark="visualwebarena",
        use_case="public_reply",
        carrier=carrier,
        action_kind=action_kind,
    )


def _compile(
    *,
    carrier: str | None = "listing_reply.body",
    action_kind: str | None = "answer_opposite_binary_label",
):
    return check_site_composition(
        default_site_compositions(),
        _request(carrier=carrier, action_kind=action_kind),
    )


def test_classifieds_public_reply_is_static_complete_without_hidden_activation() -> None:
    report = _compile()

    assert report.site == "classifieds"
    assert report.benchmark == "visualwebarena"
    assert report.static_status == "complete"
    assert report.finding("final_state_evaluation").state == "not_applicable"
    assert report.finding("action_cards").state == "supported"
    assert report.site_composition_digest is not None


def test_classifieds_requires_exact_carrier() -> None:
    report = _compile(carrier="listing.title")

    assert report.static_status == "incomplete"
    assert report.finding("profile").state == "missing"
    assert report.finding("editor_specification").state == "missing"
    assert report.finding("action_cards").state == "supported"


def test_classifieds_requires_exact_action_kind() -> None:
    report = _compile(action_kind="unknown_action")

    assert report.static_status == "incomplete"
    assert report.finding("profile").state == "supported"
    assert report.finding("editor_specification").state == "supported"
    assert report.finding("action_cards").state == "missing"


def test_classifieds_missing_request_metadata_is_incomplete() -> None:
    report = _compile(carrier=None, action_kind=None)

    assert report.static_status == "incomplete"
    assert report.finding("profile").state == "missing"
    assert report.finding("action_cards").state == "missing"


def test_removing_classifieds_definition_fails_closed_without_affecting_defaults() -> None:
    definitions = tuple(item for item in default_site_compositions() if item.site != "classifieds")

    removed = check_site_composition(definitions, _request())

    assert removed.static_status == "invalid"
    assert removed.site_composition_digest is None
    assert "classifieds" not in removed.to_json()
    assert {item.site for item in definitions} == {"gitlab", "reddit"}
