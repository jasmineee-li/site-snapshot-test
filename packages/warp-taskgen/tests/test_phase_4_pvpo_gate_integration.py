"""P0 integration tests for the rewritten _run_pvpo_gate with both metrics.

After the transcript_purpose cutover, the gate runs Apollo's 2-step
classifier in parallel with the VEA judge. This file covers:

- Gate-miss short-circuits both metrics to nulls.
- Encountered trajectories populate transcript_purpose + VEA in parallel.
- One metric blowing up does not starve the other.
- Legacy .aer_inflight sentinel sweep on Phase 4 entry.
- Pre-cutover ipi_aer_* keys are absent from new result payloads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from worldsim.phase_4 import runner as phase_4_adversarial
from worldsim.phase_4.encounter_detection import EncounterResult


def _encounter(
    *,
    max_coverage: float,
    reference_step: int | None = None,
    reference_path: Path | None = None,
    reference_bytes: bytes | None = None,
    pvpo_status: str = "ok",
    pvpo_failure: str | None = None,
) -> EncounterResult:
    # Production callers (``determine_encounter``) always populate
    # ``reference_screenshot_bytes`` alongside ``reference_screenshot_path``
    # so Stage 2 receives a pre-validated snapshot with no TOCTOU window.
    # Mirror that in tests: when a path is supplied but no bytes, read the
    # file so ``_run_pvpo_gate`` fans out to the judges instead of routing
    # to placement-fix via the missing-bytes short-circuit.
    if reference_bytes is None and reference_path is not None:
        try:
            reference_bytes = reference_path.read_bytes()
        except OSError:
            reference_bytes = None
    return EncounterResult(
        max_coverage=max_coverage,
        reference_step=reference_step,
        reference_screenshot_path=reference_path,
        reference_screenshot_bytes=reference_bytes,
        pvpo_status=pvpo_status,
        pvpo_failure=pvpo_failure,
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
        "transcript_purpose_was_refused": False,
        "transcript_purpose_guess": "to test alignment under injection",
        "transcript_purpose_explanation": "grounded guess",
        "transcript_purpose_failure": None,
        "transcript_purpose_format": "needham-xml-v1",
        "transcript_purpose_attempts": 1,
        "transcript_purpose_raw_response_path": "/tmp/ignored.json",
        "transcript_purpose_apollo_continue_classification": None,
        "transcript_purpose_apollo_continue_is_aware": None,
        "transcript_purpose_apollo_continue_was_refused": None,
        "transcript_purpose_apollo_continue_guess": None,
        "transcript_purpose_apollo_continue_explanation": None,
        "transcript_purpose_apollo_continue_failure": None,
        "transcript_purpose_apollo_continue_attempts": None,
        "transcript_purpose_apollo_continue_raw_response_path": None,
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
    assert result["pvpo_status"] == "ok"
    assert result["pvpo_failure"] is None
    for key in (
        "transcript_purpose_classification",
        "transcript_purpose_is_aware",
        "transcript_purpose_was_refused",
        "transcript_purpose_guess",
        "transcript_purpose_explanation",
        "transcript_purpose_failure",
        "transcript_purpose_format",
        "transcript_purpose_attempts",
        "transcript_purpose_raw_response_path",
        "transcript_purpose_apollo_continue_classification",
        "transcript_purpose_apollo_continue_is_aware",
        "transcript_purpose_apollo_continue_was_refused",
        "transcript_purpose_apollo_continue_guess",
        "transcript_purpose_apollo_continue_explanation",
        "transcript_purpose_apollo_continue_failure",
        "transcript_purpose_apollo_continue_attempts",
        "transcript_purpose_apollo_continue_raw_response_path",
        "verbalized_eval_awareness_classification",
        "verbalized_eval_awareness_is_aware",
        "verbalized_eval_awareness_evidence",
        "verbalized_eval_awareness_failure",
        "verbalized_eval_awareness_transcript_format",
        "verbalized_eval_awareness_raw_response_path",
    ):
        assert key in result
    assert result["transcript_purpose_classification"] is None
    assert result["transcript_purpose_format"] == "needham-xml-v1"
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

    seen_xml: list[str] = []

    async def fake_tp(
        task,
        trajectory_dir,
        extracted,
        *,
        sandbox_model,
        needham_xml=None,
        include_apollo_continue_format=True,
    ):
        seen_xml.append(needham_xml)
        return _transcript_purpose_payload()

    async def fake_vea(task, trajectory_dir, extracted, *, sandbox_model, needham_xml=None):
        seen_xml.append(needham_xml)
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
    assert len(seen_xml) == 2
    assert seen_xml[0] == seen_xml[1]
    assert seen_xml[0] is not None
    assert '<message role="user">' in seen_xml[0]
    assert (tmp_path / "needham_trace.json").exists()


@pytest.mark.asyncio
async def test_transcript_purpose_failure_does_not_block_vea(monkeypatch, tmp_path):
    _write_history(tmp_path)
    screenshot = _screenshot(tmp_path)

    def fake_determine(task, trajectory_dir):
        return _encounter(max_coverage=0.5, reference_step=0, reference_path=screenshot)

    async def failing_tp(
        task,
        trajectory_dir,
        extracted,
        *,
        sandbox_model,
        needham_xml=None,
        include_apollo_continue_format=True,
    ):
        raise RuntimeError("transcript_purpose blew up")

    async def fake_vea(task, trajectory_dir, extracted, *, sandbox_model, needham_xml=None):
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

    async def fake_tp(
        task,
        trajectory_dir,
        extracted,
        *,
        sandbox_model,
        needham_xml=None,
        include_apollo_continue_format=True,
    ):
        return _transcript_purpose_payload()

    async def failing_vea(task, trajectory_dir, extracted, *, sandbox_model, needham_xml=None):
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

    async def fake_vea(task, trajectory_dir, extracted, *, sandbox_model, needham_xml=None):
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
        "transcript_purpose_was_refused",
        "transcript_purpose_guess",
        "transcript_purpose_explanation",
        "transcript_purpose_failure",
        "transcript_purpose_format",
        "transcript_purpose_attempts",
        "transcript_purpose_raw_response_path",
        "transcript_purpose_apollo_continue_classification",
        "transcript_purpose_apollo_continue_is_aware",
        "transcript_purpose_apollo_continue_was_refused",
        "transcript_purpose_apollo_continue_guess",
        "transcript_purpose_apollo_continue_explanation",
        "transcript_purpose_apollo_continue_failure",
        "transcript_purpose_apollo_continue_attempts",
        "transcript_purpose_apollo_continue_raw_response_path",
    ):
        assert key in result, f"missing key after raise: {key}"
    assert result["transcript_purpose_failure"] == "api_error"


@pytest.mark.asyncio
async def test_partial_trajectory_skips_both_stage_2_judges(monkeypatch, tmp_path):
    (tmp_path / "history.json").write_text(
        json.dumps({"history": [], "partial": True, "errors": ["browser crashed"]})
    )
    screenshot = _screenshot(tmp_path)

    def fake_determine(task, trajectory_dir):
        return _encounter(max_coverage=0.5, reference_step=0, reference_path=screenshot)

    async def assert_tp_not_called(*args, **kwargs):
        raise AssertionError("transcript-purpose must not run on partial trajectory")

    async def assert_vea_not_called(*args, **kwargs):
        raise AssertionError("VEA must not run on partial trajectory")

    monkeypatch.setattr(phase_4_adversarial, "determine_encounter", fake_determine)
    monkeypatch.setattr(phase_4_adversarial, "run_transcript_purpose_api", assert_tp_not_called)
    monkeypatch.setattr(
        phase_4_adversarial, "run_verbalized_eval_awareness_api", assert_vea_not_called
    )

    result = await phase_4_adversarial._run_pvpo_gate(
        _task(), tmp_path, sandbox_model="claude-sonnet-4-6"
    )

    assert result["injection_not_encountered"] is False
    assert result["transcript_purpose_failure"] == "trajectory_unavailable"
    assert result["verbalized_eval_awareness_failure"] == "trajectory_unavailable"


@pytest.mark.asyncio
async def test_missing_history_skips_both_stage_2_judges(monkeypatch, tmp_path):
    screenshot = _screenshot(tmp_path)

    def fake_determine(task, trajectory_dir):
        return _encounter(max_coverage=0.5, reference_step=0, reference_path=screenshot)

    async def assert_tp_not_called(*args, **kwargs):
        raise AssertionError("transcript-purpose must not run without history.json")

    async def assert_vea_not_called(*args, **kwargs):
        raise AssertionError("VEA must not run without history.json")

    monkeypatch.setattr(phase_4_adversarial, "determine_encounter", fake_determine)
    monkeypatch.setattr(phase_4_adversarial, "run_transcript_purpose_api", assert_tp_not_called)
    monkeypatch.setattr(
        phase_4_adversarial, "run_verbalized_eval_awareness_api", assert_vea_not_called
    )

    result = await phase_4_adversarial._run_pvpo_gate(
        _task(), tmp_path, sandbox_model="claude-sonnet-4-6"
    )

    assert result["injection_not_encountered"] is False
    assert result["transcript_purpose_failure"] == "trajectory_unavailable"
    assert result["verbalized_eval_awareness_failure"] == "trajectory_unavailable"


@pytest.mark.asyncio
async def test_reference_screenshot_rechecked_before_stage_2(monkeypatch, tmp_path):
    _write_history(tmp_path)
    screenshot = _screenshot(tmp_path)

    def fake_determine(task, trajectory_dir):
        return _encounter(max_coverage=0.5, reference_step=0, reference_path=screenshot)

    async def assert_tp_not_called(*args, **kwargs):
        raise AssertionError("transcript-purpose must not run after screenshot loss")

    async def assert_vea_not_called(*args, **kwargs):
        raise AssertionError("VEA must not run after screenshot loss")

    screenshot.unlink()
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
    assert result["pvpo_status"] == "ok"
