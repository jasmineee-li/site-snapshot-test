"""Live smoke: does the judge API path actually classify a real refusal?

Skipped unless credentials are present in the environment. Uses the actual
refused trajectory from logs/phase_4/20260417_190423/adv_shopping_admin_002
(the task whose judge refused on the old sandbox prompt).

Acceptance: returns a structured response with a valid refusal_trigger —
NOT a refusal. A single refusal here means the new prompt regressed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from warp_taskgen.phase_4 import run_judge_api
from warp_taskgen.phase_4.strategy_catalog import REFUSAL_TRIGGERS

REPO = Path(__file__).resolve().parents[2]
TRAJ_DIR = REPO / "logs/phase_4/20260417_190423/adv_shopping_admin_002__placement_2"
TASK_SRC = REPO / "logs/phase_4/20260417_190423/adv_shopping_admin_002/processed_result.json"


def _has_credentials() -> bool:
    for k in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"):
        if os.environ.get(k, "").strip():
            return True
    if (
        os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
        and os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    ):
        return True
    return False


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_credentials(),
    reason="requires CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN+BASE_URL",
)
@pytest.mark.skipif(
    not TRAJ_DIR.exists() or not TASK_SRC.exists(),
    reason="reference trajectory from 20260417_190423 run not present; run Phase 4 first",
)
@pytest.mark.asyncio
async def test_judge_classifies_real_refused_trajectory():
    processed = json.loads(TASK_SRC.read_text())
    task = processed["placement_fix"]["final_task"]

    result = await run_judge_api(task, TRAJ_DIR)

    # The new prompt must not refuse on this task.
    assert result["status"] in (
        "judge_ok_actionable",
        "judge_ok_unactionable",
    ), f"judge regressed: {result}"
    assert result["refusal_trigger"] in REFUSAL_TRIGGERS
    # The raw response is persisted.
    assert (TRAJ_DIR / "judge_raw_response.json").exists()
