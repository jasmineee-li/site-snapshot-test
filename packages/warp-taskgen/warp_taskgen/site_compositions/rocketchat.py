"""Pure static Rocket.Chat Site Composition declarations.

This projection describes only the source-level response workflow.  It does
not claim deployed authentication, painted visibility, cleanup, reset, or a
final-state/action-card capability.
"""

from __future__ import annotations

from warp_taskgen.site_composition import check_site_composition
from warp_taskgen.site_composition_contracts import (
    SiteBenchmarkComposition,
    SiteComposition,
    SiteCompositionCheckReport,
    SiteCompositionCheckRequest,
    SiteCompositionUseCaseCatalog,
    SiteOwnerDeclaration,
)


def _owner(
    owner_id: str,
    provenance: tuple[str, ...],
    *,
    state: str = "unsupported",
) -> SiteOwnerDeclaration:
    return SiteOwnerDeclaration(
        state=state,
        owner_id=owner_id if state == "supported" else None,
        contract_version="v1",
        provenance=provenance,
    )


def rocket_chat_site_composition() -> SiteComposition:
    projection = SiteBenchmarkComposition(
        benchmark="theagentcompany",
        # The pure route grammar is the only concrete owner in this source
        # slice.  Authentication, editor, writer, readback, and evaluator
        # behavior remain explicitly unsupported until their owners exist.
        site_targeting=_owner(
            "warp.rocketchat.site_targeting",
            ("sites.rocketchat",),
            state="supported",
        ),
        profile=_owner("warp.rocketchat.profile", ("sites.rocketchat",)),
        editor_specification=_owner(
            "warp.rocketchat.editor_specification",
            ("phase_1.rocket_chat_decisions",),
        ),
        regular_participant_writer=_owner(
            "warp.rocketchat.regular_participant_writer",
            ("phase_1.rocket_chat_decisions",),
        ),
        feasibility=_owner(
            "warp.rocketchat.feasibility",
            ("phase_1.rocket_chat_decisions",),
        ),
        read_surface=_owner(
            "warp.rocketchat.read_surface",
            ("phase_1.rocket_chat_decisions",),
        ),
        readback=_owner("warp.rocketchat.readback", ("phase_1.rocket_chat_decisions",)),
        final_state_evaluation=SiteOwnerDeclaration(
            state="unsupported",
            contract_version="v1",
            provenance=("phase_1.rocket_chat_evaluator",),
        ),
        action_cards=SiteOwnerDeclaration(
            state="unsupported",
            contract_version="v1",
            provenance=("unsupported.rocketchat.action_cards",),
        ),
        # No carrier is declared: a Rocket.Chat surface identity has not yet
        # been resolved through the Site-owned mapping.
        supported_carriers=(),
        supported_action_kinds=(),
        provenance=("phase_1.rocket_chat_decisions", "sites.rocketchat"),
    )
    return SiteComposition(
        site="rocketchat",
        benchmark_compositions=(projection,),
        provenance=("phase_1.rocket_chat_decisions", "sites.rocketchat"),
    )


def rocket_chat_static_composition_request() -> SiteCompositionCheckRequest:
    """Use the canonical host-owned Phase 1 generation check."""

    return SiteCompositionCheckRequest(
        site="rocketchat",
        benchmark="theagentcompany",
        use_case="phase_1_generation",
    )


def rocket_chat_static_composition_report() -> SiteCompositionCheckReport:
    """Report immutable declarations; deployment readiness remains blocked."""

    return check_site_composition(
        (rocket_chat_site_composition(),),
        rocket_chat_static_composition_request(),
        use_case_catalog=SiteCompositionUseCaseCatalog.default(),
    )


__all__ = [
    "rocket_chat_site_composition",
    "rocket_chat_static_composition_report",
    "rocket_chat_static_composition_request",
]
