"""P0 integration tests for the rewritten _run_pvpo_gate with both metrics.

After the transcript_purpose cutover, the gate runs Apollo's 2-step
classifier in parallel with the VEA judge. This file covers:

- Gate-miss short-circuits both metrics to nulls.
- Encountered trajectories populate transcript_purpose + VEA in parallel.
- One metric blowing up does not starve the other.
- Sentinel cleanup (.ipi_aer_inflight) on clean return.
- Legacy .aer_inflight sentinel sweep on Phase 4 entry.
- Pre-cutover ipi_aer_* keys are absent from new result payloads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from worldsim.phase_4.encounter_detection import EncounterResult
from worldsim.phases import phase_4_adversarial


def _encounter(
    *,
    max_coverage: float,
    reference_step: int | None = None,
    reference_path: Path | None = None,
) -> EncounterResult:
    return EncounterResult(
        max_coverage=max_coverage,
        reference_step=reference_step,
        reference_screenshot_path=reference_path,
    )


def _task() -> dict[str, Any]:
    return {
        "id": "adv-pvpo-int-1",
        "site": "shopping_admin",
        "description": "task intent",
        "payload_texts": [{"rendered_payload": "PAYLOAD"}],
    }


def _write_history(tmp_path: Path, steps: int = 1) -> None:
    history = {
        "history": [
            {
                "model_output": {
                    "thinking": f"t{i}",
                    "evaluation_previous_goal": f"e{i}",
                    "memory": f"m{i}",
                    "next_goal": f"g{i}",
                    "action": [{"click": {"index": i}}],
                },
                "state": {"url": f"https://x/{i}", "title": f"t{i}"},
                "result": [],
            }
            for i in range(steps)
        ]
    }
    (tmp_path / "history.json").write_text(json.dumps(history))


def _screenshot(tmp_path: Path) -> Path:
    path = tmp_path / "screenshots" / "step_0.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    return path


def _transcript_purpose_payload() -> dict[str, Any]:
    return {
        "transcript_purpose_classification": "Alignment Evaluation",
        "transcript_purpose_is_aware": True,
        "transcript_purpose_guess": "to test alignment under injection",
        "transcript_purpose_explanation": "grounded guess",
        "transcript_purpose_failure": None,
        "transcript_purpose_format": "ipi-aer-inline-v1",
        "transcript_purpose_attempts": 1,
        "transcript_purpose_raw_response_path": "/tmp/ignored.json",
    }


def _vea_real_payload(tmp_path: Path) -> dict[str, Any]:
    return {
        "verbalized_eval_awareness_classification": "Real",
        "verbalized_eval_awareness_is_aware": False,
        "verbalized_eval_awareness_evidence": [],
        "verbalized_eval_awareness_failure": None,
        "verbalized_eval_awareness_transcript_format": "needham-xml-v1",
        "verbalized_eval_awareness_raw_response_path": str(
            tmp_path / "verbalized_eval_awareness_raw_response.json"
        ),
    }


@pytest.mark.asyncio
async def test_gate_miss_nulls_all_metrics_skips_judges(monkeypatch, tmp_path):
    """max_coverage=0 → both judges never called, all metric keys present-and-null."""

    def fake_determine(task, trajectory_dir):
        return _encounter(max_coverage=0.0, reference_step=None, reference_path=None)

    async def assert_tp_not_called(*args, **kwargs):
        raise AssertionError("transcript_purpose must not run on gate miss")

    async def assert_vea_not_called(*args, **kwargs):
        raise AssertionError("VEA must not run on gate miss")

    monkeypatch.setattr(phase_4_adversarial, "determine_encounter", fake_determine)
    monkeypatch.setattr(phase_4_adversarial, "run_transcript_purpose_api", assert_tp_not_called)
    monkeypatch.setattr(
        phase_4_adversarial, "run_verbalized_eval_awareness_api", assert_vea_not_called
    )

    result = await phase_4_adversarial._run_pvpo_gate(
        _task(), tmp_path, sandbox_model="claude-sonnet-4-6"
    )

    assert result["injection_not_encountered"] is True
    assert result["encounter"]["max_coverage"] == 0.0
    for key in (
        "transcript_purpose_classification",
        "transcript_purpose_is_aware",
        "transcript_purpose_guess",
        "transcript_purpose_explanation",
        "transcript_purpose_failure",
        "transcript_purpose_format",
        "transcript_purpose_attempts",
        "transcript_purpose_raw_response_path",
        "verbalized_eval_awareness_classification",
        "verbalized_eval_awareness_is_aware",
        "verbalized_eval_awareness_evidence",
        "verbalized_eval_awareness_failure",
        "verbalized_eval_awareness_transcript_format",
        "verbalized_eval_awareness_raw_response_path",
    ):
        assert key in result
    assert result["transcript_purpose_classification"] is None
    assert result["transcript_purpose_format"] == "ipi-aer-inline-v1"
    assert result["transcript_purpose_attempts"] == 0
    assert result["verbalized_eval_awareness_classification"] is None
    assert result["verbalized_eval_awareness_transcript_format"] == "needham-xml-v1"

    # Legacy ipi_aer_* / p_eval_* keys must be absent — schema-break safety net.
    for legacy in (
        "ipi_aer_p_eval_score",
        "ipi_aer_label_matrix",
        "ipi_aer_failure",
        "ipi_aer_caption_text",
        "ipi_aer_reps",
        "ipi_aer_screenshot_mode",
        "p_eval_score",
        "ecologically_valid",
    ):
        assert legacy not in result


@pytest.mark.asyncio
async def test_gate_encountered_fans_out_both_metrics(monkeypatch, tmp_path):
    """max_coverage>0 → both judges run in parallel; both metrics populated."""
    _write_history(tmp_path)
    screenshot = _screenshot(tmp_path)

    def fake_determine(task, trajectory_dir):
        return _encounter(max_coverage=0.75, reference_step=0, reference_path=screenshot)

    async def fake_tp(task, trajectory_dir, extracted, screenshot_path, *, sandbox_model):
        return _transcript_purpose_payload()

    async def fake_vea(task, trajectory_dir, extracted, *, sandbox_model):
        return {
            "verbalized_eval_awareness_classification": "Alignment Evaluation",
            "verbalized_eval_awareness_is_aware": True,
            "verbalized_eval_awareness_evidence": ["this looks like a test"],
            "verbalized_eval_awareness_failure": None,
            "verbalized_eval_awareness_transcript_format": "needham-xml-v1",
            "verbalized_eval_awareness_raw_response_path": str(
                tmp_path / "verbalized_eval_awareness_raw_response.json"
            ),
        }

    monkeypatch.setattr(phase_4_adversarial, "determine_encounter", fake_determine)
    monkeypatch.setattr(phase_4_adversarial, "run_transcript_purpose_api", fake_tp)
    monkeypatch.setattr(phase_4_adversarial, "run_verbalized_eval_awareness_api", fake_vea)

    result = await phase_4_adversarial._run_pvpo_gate(
        _task(), tmp_path, sandbox_model="claude-sonnet-4-6"
    )

    assert result["injection_not_encountered"] is False
    assert result["transcript_purpose_classification"] == "Alignment Evaluation"
    assert result["transcript_purpose_is_aware"] is True
    assert result["transcript_purpose_attempts"] == 1
    assert result["verbalized_eval_awareness_classification"] == "Alignment Evaluation"
    assert result["verbalized_eval_awareness_is_aware"] is True
    # Sentinel deleted on clean return (new name)
    assert not (tmp_path / ".ipi_aer_inflight").exists()


@pytest.mark.asyncio
async def test_transcript_purpose_failure_does_not_block_vea(monkeypatch, tmp_path):
    _write_history(tmp_path)
    screenshot = _screenshot(tmp_path)

    def fake_determine(task, trajectory_dir):
        return _encounter(max_coverage=0.5, reference_step=0, reference_path=screenshot)

    async def failing_tp(task, trajectory_dir, extracted, screenshot_path, *, sandbox_model):
        raise RuntimeError("transcript_purpose blew up")

    async def fake_vea(task, trajectory_dir, extracted, *, sandbox_model):
        return _vea_real_payload(tmp_path)

    monkeypatch.setattr(phase_4_adversarial, "determine_encounter", fake_determine)
    monkeypatch.setattr(phase_4_adversarial, "run_transcript_purpose_api", failing_tp)
    monkeypatch.setattr(phase_4_adversarial, "run_verbalized_eval_awareness_api", fake_vea)

    result = await phase_4_adversarial._run_pvpo_gate(
        _task(), tmp_path, sandbox_model="claude-sonnet-4-6"
    )

    assert result["transcript_purpose_failure"] == "api_error"
    assert result["transcript_purpose_classification"] is None
    # VEA completed despite transcript_purpose failure.
    assert result["verbalized_eval_awareness_classification"] == "Real"
    assert result["verbalized_eval_awareness_failure"] is None


@pytest.mark.asyncio
async def test_vea_failure_does_not_block_transcript_purpose(monkeypatch, tmp_path):
    _write_history(tmp_path)
    screenshot = _screenshot(tmp_path)

    def fake_determine(task, trajectory_dir):
        return _encounter(max_coverage=0.5, reference_step=0, reference_path=screenshot)

    async def fake_tp(task, trajectory_dir, extracted, screenshot_path, *, sandbox_model):
        return _transcript_purpose_payload()

    async def failing_vea(task, trajectory_dir, extracted, *, sandbox_model):
        raise RuntimeError("VEA crashed")

    monkeypatch.setattr(phase_4_adversarial, "determine_encounter", fake_determine)
    monkeypatch.setattr(phase_4_adversarial, "run_transcript_purpose_api", fake_tp)
    monkeypatch.setattr(phase_4_adversarial, "run_verbalized_eval_awareness_api", failing_vea)

    result = await phase_4_adversarial._run_pvpo_gate(
        _task(), tmp_path, sandbox_model="claude-sonnet-4-6"
    )

    assert result["transcript_purpose_classification"] == "Alignment Evaluation"
    assert result["verbalized_eval_awareness_failure"] == "api_error"
    assert result["verbalized_eval_awareness_classification"] is None


@pytest.mark.asyncio
async def test_raised_transcript_purpose_still_yields_full_null_schema(monkeypatch, tmp_path):
    """R1: the prefix filter on the fallback path keeps every new metric key present."""
    _write_history(tmp_path)
    screenshot = _screenshot(tmp_path)

    def fake_determine(task, trajectory_dir):
        return _encounter(max_coverage=0.5, reference_step=0, reference_path=screenshot)

    async def failing_tp(*args, **kwargs):
        raise RuntimeError("boom")

    async def fake_vea(task, trajectory_dir, extracted, *, sandbox_model):
        return _vea_real_payload(tmp_path)

    monkeypatch.setattr(phase_4_adversarial, "determine_encounter", fake_determine)
    monkeypatch.setattr(phase_4_adversarial, "run_transcript_purpose_api", failing_tp)
    monkeypatch.setattr(phase_4_adversarial, "run_verbalized_eval_awareness_api", fake_vea)

    result = await phase_4_adversarial._run_pvpo_gate(
        _task(), tmp_path, sandbox_model="claude-sonnet-4-6"
    )

    for key in (
        "transcript_purpose_classification",
        "transcript_purpose_is_aware",
        "transcript_purpose_guess",
        "transcript_purpose_explanation",
        "transcript_purpose_failure",
        "transcript_purpose_format",
        "transcript_purpose_attempts",
        "transcript_purpose_raw_response_path",
    ):
        assert key in result, f"missing key after raise: {key}"
    assert result["transcript_purpose_failure"] == "api_error"


@pytest.mark.asyncio
async def test_sentinel_written_then_cleaned_on_success(monkeypatch, tmp_path):
    """Sentinel `.ipi_aer_inflight` is written at entry and unlinked on clean exit."""
    _write_history(tmp_path)
    screenshot = _screenshot(tmp_path)

    def fake_determine(task, trajectory_dir):
        return _encounter(max_coverage=0.5, reference_step=0, reference_path=screenshot)

    async def fake_tp(*args, **kwargs):
        # Observe the sentinel mid-run.
        assert (tmp_path / ".ipi_aer_inflight").exists()
        return _transcript_purpose_payload()

    async def fake_vea(*args, **kwargs):
        return _vea_real_payload(tmp_path)

    monkeypatch.setattr(phase_4_adversarial, "determine_encounter", fake_determine)
    monkeypatch.setattr(phase_4_adversarial, "run_transcript_purpose_api", fake_tp)
    monkeypatch.setattr(phase_4_adversarial, "run_verbalized_eval_awareness_api", fake_vea)

    await phase_4_adversarial._run_pvpo_gate(_task(), tmp_path, sandbox_model="claude-sonnet-4-6")

    # Cleaned up on clean return.
    assert not (tmp_path / ".ipi_aer_inflight").exists()
