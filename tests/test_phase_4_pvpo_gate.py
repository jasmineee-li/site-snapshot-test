"""Unit tests for the production PVPO gate path.

The 19 legacy ``probe_ecological_validity`` fixtures in
``test_phase_4_adversarial.py`` exercise the no-artifacts fallback branch
inside ``_run_pvpo_gate``. They do NOT cover the production path where
``determine_encounter`` returns ``max_coverage > 0`` and ``run_p_eval_api``
produces a real probe result.

This module fills that gap by monkey-patching the two collaborators
directly (``determine_encounter``, ``run_p_eval_api``) so each code path
through ``_run_pvpo_gate`` is independently testable without needing a
running Docker container or Anthropic credentials.
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
async def test_production_path_runs_p_eval_and_populates_fields(monkeypatch, tmp_path):
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

    probes = [
        {"name": "content_realism", "score": 0.1, "justification": "ok"},
        {"name": "ui_realism", "score": 0.2, "justification": "ok"},
        {"name": "data_coherence", "score": 0.3, "justification": "ok"},
        {"name": "artifact_presence", "score": 0.4, "justification": "ok"},
        {"name": "overall_plausibility", "score": 0.5, "justification": "ok"},
    ]

    captured_args: dict[str, Any] = {}

    async def fake_p_eval(ref, site_context, trajectory_dir, *, sandbox_model):
        captured_args["ref"] = ref
        captured_args["site_context"] = site_context
        captured_args["trajectory_dir"] = trajectory_dir
        captured_args["sandbox_model"] = sandbox_model
        return {
            "p_eval_score": sum(p["score"] for p in probes) / len(probes),
            "p_eval_breakdown": probes,
            "failure_class": None,
        }

    monkeypatch.setattr(phase_4_adversarial, "determine_encounter", fake_determine)
    monkeypatch.setattr(phase_4_adversarial, "run_p_eval_api", fake_p_eval)

    result = await phase_4_adversarial._run_pvpo_gate(
        _task(), tmp_path, sandbox_model="claude-sonnet-4-6"
    )

    assert result["encounter"]["max_coverage"] == 0.75
    assert result["encounter"]["reference_step"] == 2
    assert result["p_eval_score"] == pytest.approx(0.3)
    assert result["p_eval_breakdown"] == probes
    assert result["p_eval_failure"] is None
    assert result["ecologically_valid"] is True
    assert result["validity_score"] == pytest.approx(0.3)
    assert result["injection_not_encountered"] is False

    assert captured_args["ref"] == ref_path
    assert captured_args["site_context"] == "shopping_admin"
    assert captured_args["trajectory_dir"] == tmp_path
    assert captured_args["sandbox_model"] == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_zero_coverage_sets_injection_not_encountered_and_skips_p_eval(monkeypatch, tmp_path):
    def fake_determine(task, trajectory_dir):
        return _encounter(
            max_coverage=0.0,
            reference_step=None,
            reference_path=None,
            per_step_coverage=[0.0, 0.0],
        )

    async def unexpected_p_eval(*args, **kwargs):
        raise AssertionError("run_p_eval_api should not be called when max_coverage == 0")

    async def legacy_probe(task_dir, task, *, sandbox_model):
        # Legacy fallback path: default (unpatched) wrapper returns 1.0,
        # which signals "no encounter" and must preserve injection_not_encountered.
        return 1.0

    monkeypatch.setattr(phase_4_adversarial, "determine_encounter", fake_determine)
    monkeypatch.setattr(phase_4_adversarial, "run_p_eval_api", unexpected_p_eval)
    monkeypatch.setattr(phase_4_adversarial, "probe_ecological_validity", legacy_probe)

    result = await phase_4_adversarial._run_pvpo_gate(
        _task(), tmp_path, sandbox_model="claude-sonnet-4-6"
    )

    assert result["injection_not_encountered"] is True
    assert result["p_eval_score"] is None
    assert result["p_eval_breakdown"] is None
    assert result["p_eval_failure"] is None
    assert result["ecologically_valid"] is True
    assert result["validity_score"] is None


@pytest.mark.asyncio
async def test_legacy_fallback_path_threads_score_from_probe_ecological_validity(
    monkeypatch, tmp_path
):
    """When no PVPO artifacts exist, a monkey-patched probe_ecological_validity
    score flows into the result via the legacy fallback branch."""

    def fake_determine(task, trajectory_dir):
        return _encounter(
            max_coverage=0.0,
            reference_step=None,
            reference_path=None,
        )

    async def legacy_probe(task_dir, task, *, sandbox_model):
        return 0.42

    monkeypatch.setattr(phase_4_adversarial, "determine_encounter", fake_determine)
    monkeypatch.setattr(phase_4_adversarial, "probe_ecological_validity", legacy_probe)

    result = await phase_4_adversarial._run_pvpo_gate(
        _task(), tmp_path, sandbox_model="claude-sonnet-4-6"
    )

    assert result["p_eval_score"] == pytest.approx(0.42)
    assert result["validity_score"] == pytest.approx(0.42)
    assert result["injection_not_encountered"] is False
    assert result["ecologically_valid"] is True


@pytest.mark.asyncio
async def test_p_eval_failure_populates_failure_class_and_nulls_score(monkeypatch, tmp_path):
    ref_path = tmp_path / "screenshots" / "step_0.png"
    ref_path.parent.mkdir(parents=True)
    ref_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    def fake_determine(task, trajectory_dir):
        return _encounter(
            max_coverage=1.0,
            reference_step=0,
            reference_path=ref_path,
        )

    async def failing_p_eval(ref, site_context, trajectory_dir, *, sandbox_model):
        return {
            "p_eval_score": None,
            "p_eval_breakdown": None,
            "failure_class": "schema_violation",
        }

    monkeypatch.setattr(phase_4_adversarial, "determine_encounter", fake_determine)
    monkeypatch.setattr(phase_4_adversarial, "run_p_eval_api", failing_p_eval)

    result = await phase_4_adversarial._run_pvpo_gate(
        _task(), tmp_path, sandbox_model="claude-sonnet-4-6"
    )

    assert result["p_eval_score"] is None
    assert result["p_eval_breakdown"] is None
    assert result["p_eval_failure"] == "schema_violation"
    assert result["validity_score"] is None
    assert result["ecologically_valid"] is True
    assert result["injection_not_encountered"] is False


def test_placement_fix_trigger_fires_on_injection_not_encountered():
    """The widened trigger in _run_placement_fix_loop must accept both
    outcome==task_broke AND final_status==injection_not_encountered.

    Source-inspected because the function has many collaborators that are
    unwieldy to stub from a unit test; the trigger is a two-line guard we
    can pin by reading it.
    """
    import inspect

    src = inspect.getsource(phase_4_adversarial._run_placement_fix_loop)
    assert 'initial_result.get("outcome") != "task_broke"' in src
    assert 'initial_result.get("final_status") != "injection_not_encountered"' in src, (
        "placement-fix trigger must branch on injection_not_encountered too"
    )
    # The two conditions are joined with ``and`` (both must fail to skip).
    guard = src.split("-> dict[str, Any] | None:")[1].split("return None")[0]
    assert " and " in guard

    sig = inspect.signature(phase_4_adversarial._run_placement_fix_loop)
    assert "initial_result" in sig.parameters


@pytest.mark.asyncio
async def test_run_p_eval_wrapper_delegates_to_pvpo_gate(monkeypatch, tmp_path):
    called_with: dict[str, Any] = {}

    async def fake_gate(task, task_dir, *, sandbox_model):
        called_with["task"] = task
        called_with["task_dir"] = task_dir
        called_with["sandbox_model"] = sandbox_model
        return {
            "encounter": {"max_coverage": 0.5},
            "p_eval_score": 0.1,
            "p_eval_breakdown": [],
            "p_eval_failure": None,
            "ecologically_valid": True,
            "validity_score": 0.1,
            "injection_not_encountered": False,
        }

    monkeypatch.setattr(phase_4_adversarial, "_run_pvpo_gate", fake_gate)

    result = await phase_4_adversarial.run_p_eval(
        tmp_path, {"id": "x", "site": "reddit"}, sandbox_model="claude-sonnet-4-6"
    )

    assert result["p_eval_score"] == 0.1
    assert called_with["task"] == {"id": "x", "site": "reddit"}
    assert called_with["task_dir"] == tmp_path
    assert called_with["sandbox_model"] == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_probe_ecological_validity_default_returns_no_encounter_sentinel(tmp_path):
    """With no monkey-patch, the deprecated wrapper returns 1.0 (sentinel)."""
    score = await phase_4_adversarial.probe_ecological_validity(
        tmp_path, {"id": "x", "site": "reddit"}
    )
    assert score == 1.0


@pytest.mark.asyncio
async def test_ecological_validity_fix_loop_stub_is_noop(tmp_path):
    """Deleted fix loop survives as a no-op stub so existing monkey-patches keep
    resolving. Calling it with arbitrary args must return None, never raise."""
    result = await phase_4_adversarial._run_ecological_validity_fix_loop(
        task={"id": "x"},
        initial_result={},
        instance=None,
        all_instances=[],
        agent_factory=None,
        profile_path=tmp_path,
        task_dir_root=tmp_path,
    )
    assert result is None
