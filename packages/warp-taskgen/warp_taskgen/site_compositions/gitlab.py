"""Pure GitLab Site Composition declarations.

Only immutable owner IDs and static carrier/action facts live here; runtime
owners remain in their existing feature modules.
"""

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


def gitlab_site_composition() -> SiteComposition:
    projection = SiteBenchmarkComposition(
        benchmark="webarena_verified",
        site_targeting=_owner("warp.gitlab.site_targeting", ("sites.gitlab",)),
        profile=_owner("warp.gitlab.profile", ("sites.gitlab_profile",)),
        editor_specification=_owner("warp.gitlab.editor_specification", ("editors._registry",)),
        regular_participant_writer=_owner(
            "warp.gitlab.regular_participant_writer", ("seeding.site_contracts",)
        ),
        feasibility=_owner("warp.gitlab.feasibility", ("phase_2.phase_2c.webarena_policy",)),
        read_surface=_owner("warp.gitlab.read_surface", ("sites.gitlab",)),
        readback=_owner("warp.gitlab.readback", ("sites.gitlab",)),
        final_state_evaluation=_owner(
            "warp.gitlab.final_state_evaluation", ("rewards.final_state_catalog",)
        ),
        action_cards=_owner(
            "warp.gitlab.action_cards", ("adversarial_actions.capability_adapters",)
        ),
        supported_carriers=(
            "group_description_page",
            "issue.description",
            "issue.title",
            "label_description_page",
            "milestone_description_detail",
            "note_on_issue",
            "note_on_mr",
            "snippet_content_view",
            "snippet_title_list",
            "user_profile_bio",
            "user_status_message_profile",
        ),
        supported_action_kinds=("answer_opposite_binary_label",),
        provenance=("sites.gitlab",),
    )
    return SiteComposition(
        site="gitlab",
        benchmark_compositions=(projection,),
        provenance=("sites.gitlab",),
    )


__all__ = ["gitlab_site_composition"]
