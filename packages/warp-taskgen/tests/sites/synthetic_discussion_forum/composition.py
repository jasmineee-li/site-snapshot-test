"""Data-only Site Composition for the test-only discussion forum."""

from __future__ import annotations

from warp_taskgen.site_composition import (
    SiteBenchmarkComposition,
    SiteComposition,
    SiteCompositionCheckRequest,
    SiteOwnerDeclaration,
    check_site_composition,
)
from warp_taskgen.site_composition_contracts import SiteCompositionCheckReport

BENCHMARK = "webarena_verified"
SITE = "synthetic_discussion_forum"
USE_CASE = "public_reply"
CARRIER = "comment.body"
ACTION_KIND = "submit_comment"


def _owner(owner_role: str) -> SiteOwnerDeclaration:
    return SiteOwnerDeclaration(
        state="supported",
        owner_id=f"tests.synthetic_discussion_forum.{owner_role}",
        contract_version="v1",
        provenance=(f"tests.sites.synthetic_discussion_forum.{owner_role}",),
    )


def site_composition() -> SiteComposition:
    """Describe this Site's static owner identities without importing behavior."""

    benchmark_composition = SiteBenchmarkComposition(
        benchmark=BENCHMARK,
        site_targeting=_owner("site_targeting"),
        profile=_owner("profile"),
        editor_specification=_owner("editor_specification"),
        regular_participant_writer=_owner("regular_participant_writer"),
        feasibility=_owner("feasibility"),
        read_surface=_owner("read_surface"),
        readback=_owner("readback"),
        final_state_evaluation=_owner("final_state_evaluation"),
        action_cards=_owner("action_cards"),
        supported_carriers=(CARRIER,),
        supported_action_kinds=(ACTION_KIND,),
        provenance=("tests.sites.synthetic_discussion_forum.composition",),
    )
    return SiteComposition(
        site=SITE,
        benchmark_compositions=(benchmark_composition,),
        provenance=("tests.sites.synthetic_discussion_forum.composition",),
    )


def composition_check_request() -> SiteCompositionCheckRequest:
    """Return the one exact public-reply request exercised by this fake Site."""

    return SiteCompositionCheckRequest(
        site=SITE,
        benchmark=BENCHMARK,
        use_case=USE_CASE,
        carrier=CARRIER,
        action_kind=ACTION_KIND,
    )


def static_composition_report() -> SiteCompositionCheckReport:
    """Check the feature-local static declaration only."""

    return check_site_composition((site_composition(),), composition_check_request())


__all__ = [
    "ACTION_KIND",
    "BENCHMARK",
    "CARRIER",
    "SITE",
    "USE_CASE",
    "composition_check_request",
    "site_composition",
    "static_composition_report",
]
