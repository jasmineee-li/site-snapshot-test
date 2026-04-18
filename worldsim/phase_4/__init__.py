"""Phase 4 adaptive strategy variation — host-side API helpers.

Judge classification and variant generation use direct Anthropic Messages API
calls (single-turn, tool-use forced). The ecological-validity probe,
placement-fix, and ecoval-fix loops still use `run_claude_in_sandbox`; only
the two paths in this package are API-direct.

Cutover landed 2026-04-18 after refusals on the prior "recommend strategies"
sandbox framing (see `logs/phase_4_demo_20260417_190422.log:3570`).
Methodology follows Dziemian et al., 2026 (arXiv:2603.15714, Gray Swan IPI
Arena).
"""

from worldsim.phase_4.anthropic_client import get_client
from worldsim.phase_4.concurrency import get_api_semaphore
from worldsim.phase_4.judge_api import run_judge_api
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
    "slice_trajectory",
    "strategies_for_trigger",
]
