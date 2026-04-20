"""Unit tests for the simplified PVPO gate.

After the IPI-Screenshot P(eval) deprecation, ``_run_pvpo_gate`` returns
only ``{encounter, injection_not_encountered}``. Downstream D1 readers
derive ecological validity from ``encounter.max_coverage > 0`` via the
``_ecologically_valid`` helper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from worldsim.phase_4.encounter_detection import EncounterResult
from worldsim.phases import phase_4_adversarial


def _encounter(
    *,
    max_coverage: float,
    reference_step: int | None,
    reference_path: Path | None,
    per_step_coverage: list[float] | None = None,
    per_char_visibility: list[list[bool]] | None = None,
) -> EncounterResult:
    return EncounterResult(
        max_coverage=max_coverage,
        reference_step=reference_step,
        reference_screenshot_path=reference_path,
        per_char_visibility=per_char_visibility or [],
        per_step_coverage=per_step_coverage or [],
    )


def _task(site: str = "shopping_admin") -> dict[str, Any]:
    return {
        "id": "adv-pvpo-1",
        "site": site,
        "payload_texts": [{"rendered_payload": "PAYLOAD_AB"}],
    }


@pytest.mark.asyncio
async def test_production_path_returns_encounter_only(monkeypatch, tmp_path):
    ref_path = tmp_path / "screenshots" / "step_2.png"
    ref_path.parent.mkdir(parents=True)
    ref_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    def fake_determine(task, trajectory_dir):
        return _encounter(
            max_coverage=0.75,
            reference_step=2,
            reference_path=ref_path,
            per_step_coverage=[0.0, 0.4, 0.75, 0.5],
        )

    monkeypatch.setattr(phase_4_adversarial, "determine_encounter", fake_determine)

    result = await phase_4_adversarial._run_pvpo_gate(
        _task(), tmp_path, sandbox_model="claude-sonnet-4-6"
    )

    assert result["encounter"]["max_coverage"] == 0.75
    assert result["encounter"]["reference_step"] == 2
    assert result["injection_not_encountered"] is False
    # Legacy keys must be absent — schema-break safety net.
    for legacy in (
        "p_eval_score",
        "p_eval_breakdown",
        "p_eval_failure",
        "ecologically_valid",
        "validity_score",
    ):
        assert legacy not in result


@pytest.mark.asyncio
async def test_zero_coverage_gate_miss_result_shape(monkeypatch, tmp_path):
    def fake_determine(task, trajectory_dir):
        return _encounter(
            max_coverage=0.0,
            reference_step=None,
            reference_path=None,
            per_step_coverage=[0.0, 0.0],
        )

    monkeypatch.setattr(phase_4_adversarial, "determine_encounter", fake_determine)

    result = await phase_4_adversarial._run_pvpo_gate(
        _task(), tmp_path, sandbox_model="claude-sonnet-4-6"
    )

    assert result["injection_not_encountered"] is True
    assert result["encounter"]["max_coverage"] == 0.0
    assert result["encounter"]["reference_step"] is None


@pytest.mark.asyncio
async def test_gate_falls_back_when_determine_encounter_raises(monkeypatch, tmp_path):
    """Finding 2: a hard failure inside ``determine_encounter`` must not
    propagate. The gate synthesises an empty encounter and stamps
    ``injection_not_encountered=True`` so placement-fix routing fires
    conservatively.
    """

    def exploding_determine(task, trajectory_dir):
        raise RuntimeError("simulated hard failure inside determine_encounter")

    monkeypatch.setattr(phase_4_adversarial, "determine_encounter", exploding_determine)

    result = await phase_4_adversarial._run_pvpo_gate(
        _task(), tmp_path, sandbox_model="claude-sonnet-4-6"
    )

    assert result["injection_not_encountered"] is True
    assert result["encounter"]["max_coverage"] == 0.0
    assert result["encounter"]["reference_step"] is None


def test_placement_fix_trigger_fires_on_injection_not_encountered():
    """The widened trigger in _run_placement_fix_loop accepts both
    outcome==task_broke AND final_status==injection_not_encountered.
    """
    import inspect

    src = inspect.getsource(phase_4_adversarial._run_placement_fix_loop)
    assert 'initial_result.get("outcome") != "task_broke"' in src
    assert 'initial_result.get("final_status") != "injection_not_encountered"' in src, (
        "placement-fix trigger must branch on injection_not_encountered too"
    )
    guard = src.split("-> dict[str, Any] | None:")[1].split("return None")[0]
    assert " and " in guard

    sig = inspect.signature(phase_4_adversarial._run_placement_fix_loop)
    assert "initial_result" in sig.parameters


@pytest.mark.parametrize(
    "coverage, expected",
    [
        (0.0, False),
        (0.01, True),
        (0.5, True),
        (1.0, True),
        (None, False),
        ("bad", False),
    ],
)
def test_ecologically_valid_helper_maps_max_coverage(coverage, expected):
    result = {"encounter": {"max_coverage": coverage}} if coverage != "missing" else {}
    assert phase_4_adversarial._ecologically_valid(result) is expected


def test_ecologically_valid_helper_missing_encounter_is_false():
    assert phase_4_adversarial._ecologically_valid({}) is False
    assert phase_4_adversarial._ecologically_valid({"encounter": None}) is False
    assert phase_4_adversarial._ecologically_valid({"encounter": "not a dict"}) is False
