"""Exposure contract signature behavior."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from warp_taskgen.phase_2.exposure_contract.constants import (
    PREFERRED_PAYLOAD_ARGS,
    PREFERRED_TOKEN_ORDER,
)


def exposure_contract_signature() -> dict[str, Any]:
    """Fingerprint knobs whose change invalidates persisted Phase 2 plans."""
    return {
        "version": 18,
        "modes": [
            "direct_detail",
            "inline_listing",
            "inline_listing_created_child",
            "bounded_transitive_existing",
            "bounded_transitive_created_child",
            "ineligible",
        ],
        "eligibility_policy": "seed_capability_and_phase4_exposure",
        "phase4_exposure_schema_version": 1,
        "payload_arg_preference": list(PREFERRED_PAYLOAD_ARGS),
        "token_preference": list(PREFERRED_TOKEN_ORDER),
        "surface_visibility_preference": ["always_shown", "conditional"],
        "core_surface_policy": "path_a_canonical_core_ugc",
        "created_child_target_source": "seed_metadata.created_resource.url",
        "appended_comment_exposure_policy": "gitlab_exact_region_reddit_seed_specific_visibility_or_runtime_hook",
        "visible_listing_title_preference": "prefer_payload_in_created_child_title_rows_when_rich_route_unproven",
        "surface_route_metadata": "entry_seed_transition_capacity_v2",
        "surface_candidate_policy": "enumerate_then_select_best_eligible_route",
        "title_surface_policy": "requires_task_salient_title_content_or_row_action",
        "ordered_append_guard": "created_children_and_appended_comments_pre_call_delay",
    }


def signature_hash() -> str:
    payload = json.dumps(exposure_contract_signature(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


__all__ = ["exposure_contract_signature", "signature_hash"]
