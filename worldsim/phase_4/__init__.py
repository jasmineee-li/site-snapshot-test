"""Phase 4 adaptive strategy variation — host-side API helpers.

Judge classification, variant generation, P(eval) probe, and
placement-fix all use direct Anthropic Messages API calls (single-turn,
tool-use forced). After the 2026-04-19 PVPO + placement-fix cutovers,
no Phase 4 trajectory step routes through `run_claude_in_sandbox`.

Cutover landed 2026-04-18 (judge, variant) and 2026-04-19 (P(eval),
placement-fix) after refusals on the prior "recommend strategies"
sandbox framing (see `logs/phase_4_demo_20260417_190422.log:3570`).
Methodology follows Dziemian et al., 2026 (arXiv:2603.15714, Gray Swan
IPI Arena).
"""

from worldsim.phase_4.anthropic_client import get_client
from worldsim.phase_4.concurrency import get_api_semaphore
from worldsim.phase_4.judge_api import run_judge_api
from worldsim.phase_4.placement_api import run_placement_api
from worldsim.phase_4.strategy_catalog import (
    ALLOWED_STRATEGIES,
    REFUSAL_TRIGGERS,
    SURFACE_CHANGE_SENTINEL,
    TRIGGER_TO_STRATEGIES,
    strategies_for_trigger,
)
from worldsim.phase_4.trajectory_slice import slice_trajectory
from worldsim.phase_4.variant_api import generate_variant_api

__all__ = [
    "ALLOWED_STRATEGIES",
    "REFUSAL_TRIGGERS",
    "SURFACE_CHANGE_SENTINEL",
    "TRIGGER_TO_STRATEGIES",
    "generate_variant_api",
    "get_api_semaphore",
    "get_client",
    "run_judge_api",
    "run_placement_api",
    "slice_trajectory",
    "strategies_for_trigger",
]
