"""Pure Reddit/Postmill Site Composition declarations."""

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


def reddit_site_composition() -> SiteComposition:
    projection = SiteBenchmarkComposition(
        benchmark="webarena_verified",
        site_targeting=_owner("warp.reddit.site_targeting", ("sites.reddit",)),
        profile=_owner("warp.reddit.profile", ("sites.reddit_profile",)),
        editor_specification=_owner("warp.reddit.editor_specification", ("editors._registry",)),
        regular_participant_writer=_owner(
            "warp.reddit.regular_participant_writer", ("seeding.site_contracts",)
        ),
        feasibility=_owner("warp.reddit.feasibility", ("phase_2.phase_2c.webarena_policy",)),
        read_surface=_owner("warp.reddit.read_surface", ("sites.reddit",)),
        readback=_owner("warp.reddit.readback", ("sites.reddit",)),
        final_state_evaluation=_owner(
            "warp.reddit.final_state_evaluation", ("rewards.final_state_catalog",)
        ),
        action_cards=_owner(
            "warp.reddit.action_cards", ("adversarial_actions.capability_adapters",)
        ),
        supported_carriers=(
            "comment_body_thread",
            "submission.title",
            "submission_body_detail",
        ),
        supported_action_kinds=("answer_opposite_binary_label",),
        provenance=("sites.reddit",),
    )
    return SiteComposition(
        site="reddit",
        benchmark_compositions=(projection,),
        provenance=("sites.reddit",),
    )


__all__ = ["reddit_site_composition"]
