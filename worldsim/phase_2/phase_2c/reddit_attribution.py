"""Phase 2c Reddit attribution patch exports."""

from __future__ import annotations

from worldsim.phase_2.phase_2c._impl import (
    _attach_reddit_comment_attribution_contract,
    _iter_final_state_reward_configs,
    _patch_reddit_submit_comment_state_probes,
    _reddit_seed_comment_ids_from_seed_metadata,
    _task_has_reddit_submit_comment_reward,
)

__all__ = [
    "_attach_reddit_comment_attribution_contract",
    "_iter_final_state_reward_configs",
    "_patch_reddit_submit_comment_state_probes",
    "_reddit_seed_comment_ids_from_seed_metadata",
    "_task_has_reddit_submit_comment_reward",
]
