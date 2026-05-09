"""Phase 4 setup preflight — runs via ``pytest -m preflight``.

The setup script ``scripts/setup_phase4_on_host.sh`` drives these tests as
the final gate after bootstrapping uv/playwright/docker/artifacts. A green
preflight proves: the GitLab Phase 0d artifact exists and the WebArena-Verified
evaluator venv resolves. Current page-surface-stable Phase 4 captures from the
runner-owned browser and does not require dedicated PVPO CDP endpoints or
browser containers.

Inputs:
- ``WORLDSIM_STATE_DIR`` — optional state directory containing Phase 0d artifacts

Each test fails loudly with a clear remediation if its precondition is
unmet. Exit code from pytest is the gate the bash orchestrator checks.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.preflight


# test_magento_base_urls_resolved removed 2026-04-21 with the WASP-aligned
# scoping decision (see docs/handoffs/wasp-aligned-scoping-decision.md).


def test_gitlab_storage_state_present() -> None:
    """Phase 0d gitlab storage_state must exist and contain cookies."""
    state_dir = Path(os.environ.get("WORLDSIM_STATE_DIR", "logs"))
    artifact = state_dir / "phase_0d" / "gitlab" / "storage_state.json"
    if not artifact.exists():
        pytest.fail(
            f"missing gitlab Phase 0d storage_state at {artifact}. "
            f"Rerun setup_phase4_on_host.sh step 5 (or scripts/login_gitlab_r5.py)."
        )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    cookies = payload.get("cookies")
    assert isinstance(cookies, list) and cookies, (
        f"gitlab storage_state at {artifact} has no cookies; re-mint via step 5"
    )


def test_eval_venv_reachable() -> None:
    """The WebArena Verified evaluator venv must resolve + import."""
    try:
        from worldsim.rewards.vendor_webarena import (
            _default_eval_python,
            webarena_eval_python_override,
        )
    except ImportError as exc:
        pytest.skip(f"worldsim.rewards unavailable: {exc}")
    python_exe = webarena_eval_python_override() or _default_eval_python()
    if not python_exe:
        pytest.fail(
            "no evaluator Python resolved. Run `cd packages/worldsim-webarena-verified "
            "&& uv sync --locked` (setup_phase4_on_host.sh step 1)."
        )
    completed = subprocess.run(
        [python_exe, "-c", "from webarena_verified import WebArenaVerified"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, (
        f"evaluator venv python {python_exe!r} cannot import webarena_verified: "
        f"rc={completed.returncode} stderr={completed.stderr[:300]}"
    )
