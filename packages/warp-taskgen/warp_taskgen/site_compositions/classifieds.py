"""Pure Classifieds Site Composition declarations."""

from __future__ import annotations

from warp_taskgen.site_composition_contracts import (
    SiteBenchmarkComposition,
    SiteComposition,
    SiteOwnerDeclaration,
)


def _owner(owner_id: str, provenance: tuple[str, ...]) -> SiteOwnerDeclaration:
    return SiteOwnerDeclaration(
        state="supported",
        owner_id=owner_id,
        contract_version="v1",
        provenance=provenance,
    )


def classifieds_site_composition() -> SiteComposition:
    projection = SiteBenchmarkComposition(
        benchmark="visualwebarena",
        site_targeting=_owner("warp.classifieds.site_targeting", ("sites.classifieds",)),
        profile=_owner("warp.classifieds.profile", ("sites.classifieds_profile",)),
        editor_specification=_owner(
            "warp.classifieds.editor_specification", ("sites.classifieds_editor",)
        ),
        regular_participant_writer=_owner(
            "warp.classifieds.regular_participant_writer",
            ("seeding.site_contracts", "sites.classifieds_editor"),
        ),
        feasibility=_owner(
            "warp.classifieds.feasibility", ("phase_2.phase_2c.classifieds_policy",)
        ),
        read_surface=_owner("warp.classifieds.read_surface", ("sites.classifieds_read_surface",)),
        readback=_owner("warp.classifieds.readback", ("sites.classifieds_readback",)),
        final_state_evaluation=SiteOwnerDeclaration(
            state="unsupported",
            contract_version="v1",
            provenance=("static.public_reply",),
        ),
        action_cards=_owner(
            "warp.classifieds.action_cards",
            ("adversarial_actions.classifieds_capability",),
        ),
        supported_carriers=("listing_reply.body",),
        supported_action_kinds=("answer_opposite_binary_label",),
        provenance=("sites.classifieds",),
    )
    return SiteComposition(
        site="classifieds",
        benchmark_compositions=(projection,),
        provenance=("sites.classifieds",),
    )


__all__ = ["classifieds_site_composition"]
