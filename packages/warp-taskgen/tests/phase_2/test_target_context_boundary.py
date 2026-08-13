"""Structural guard for the explicit target-resolution dependency boundary."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[2]
PHASE_2_ROOT = PACKAGE_ROOT / "worldsim" / "phase_2"


def _source(name: str) -> str:
    return (PHASE_2_ROOT / name).read_text()


def test_target_resolution_modules_do_not_use_linked_context() -> None:
    """Target modules own imports rather than inheriting runner globals."""
    for name in ("target_inputs.py", "target_stage.py", "reuse.py"):
        source = _source(name)
        assert "install_context" not in source
        assert "ruff: noqa: F821" not in source


def test_plan_validation_owns_reuse_exposure_guard() -> None:
    """Validation must not need reuse-to-runner global linking."""
    from worldsim.phase_2 import plan_validation

    reason = plan_validation._stale_reusable_exposure_contract_reason(
        {
            "site": "reddit",
            "target_surface_id": "comment.body",
            "seed_template": {"editor_calls": [{"site": "reddit", "method": "create_comment"}]},
            "exposure_contract": {"phase4_exposure": {"admissible": True}},
        }
    )

    assert reason == "reddit_create_comment_missing_exact_comment_region_gate"
