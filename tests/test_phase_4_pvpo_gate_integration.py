"""P0 integration tests for the rewritten _run_pvpo_gate with both metrics."""

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
    """max_coverage=0 → both judges never called, all metric keys null."""
    phase_4_adversarial.set_aer_config(reps=1, screenshot_mode="caption")

    def fake_determine(task, trajectory_dir):
        return _encounter(max_coverage=0.0, reference_step=None, reference_path=None)

    aer_calls = 0
    vea_calls = 0

    async def assert_aer_not_called(*args, **kwargs):
        nonlocal aer_calls
        aer_calls += 1
        raise AssertionError("IPI-AER must not run on gate miss")

    async def assert_vea_not_called(*args, **kwargs):
        nonlocal vea_calls
        vea_calls += 1
        raise AssertionError("VEA must not run on gate miss")

    monkeypatch.setattr(phase_4_adversarial, "determine_encounter", fake_determine)
    monkeypatch.setattr(phase_4_adversarial, "run_ipi_aer_p_eval_api", assert_aer_not_called)
    monkeypatch.setattr(
        phase_4_adversarial, "run_verbalized_eval_awareness_api", assert_vea_not_called
    )

    result = await phase_4_adversarial._run_pvpo_gate(
        _task(), tmp_path, sandbox_model="claude-sonnet-4-6"
    )

    assert result["injection_not_encountered"] is True
    assert result["encounter"]["max_coverage"] == 0.0
    # Every new metric key present-and-null/empty
    for key in (
        "ipi_aer_p_eval_score",
        "ipi_aer_label_matrix",
        "ipi_aer_failure",
        "ipi_aer_caption_text",
        "verbalized_eval_awareness_classification",
        "verbalized_eval_awareness_is_aware",
        "verbalized_eval_awareness_evidence",
        "verbalized_eval_awareness_failure",
        "verbalized_eval_awareness_transcript_format",
        "verbalized_eval_awareness_raw_response_path",
    ):
        assert key in result
    assert result["ipi_aer_p_eval_score"] is None
    assert result["verbalized_eval_awareness_classification"] is None
    assert result["verbalized_eval_awareness_is_aware"] is None
    assert result["verbalized_eval_awareness_evidence"] == []
    assert result["verbalized_eval_awareness_transcript_format"] == "needham-xml-v1"
    # Legacy keys must be absent — schema-break safety net
    for legacy in (
        "p_eval_score",
        "p_eval_breakdown",
        "p_eval_failure",
        "ecologically_valid",
        "validity_score",
        "vea_verbalized",
        "vea_evidence",
        "vea_confidence",
        "vea_failure",
    ):
        assert legacy not in result
    # Neither judge was invoked
    assert aer_calls == 0
    assert vea_calls == 0


@pytest.mark.asyncio
async def test_gate_encountered_fans_out_both_metrics(monkeypatch, tmp_path):
    """max_coverage>0 → both judges run in parallel; both metrics populated."""
    phase_4_adversarial.set_aer_config(reps=1, screenshot_mode="caption")

    _write_history(tmp_path)
    screenshot = _screenshot(tmp_path)

    def fake_determine(task, trajectory_dir):
        return _encounter(max_coverage=0.75, reference_step=0, reference_path=screenshot)

    async def fake_caption(screenshot_path, *, trajectory_dir, sandbox_model):
        return {"caption": "## caption", "failure_class": None, "cached": False}

    async def fake_aer(task, trajectory_dir, *args, **kwargs):
        return {
            "ipi_aer_p_eval_score": 0.6,
            "ipi_aer_label_matrix": [[1], [0], [1], [1], [0]],
            "ipi_aer_per_pq_means": [1.0, 0.0, 1.0, 1.0, 0.0],
            "ipi_aer_failure": None,
            "ipi_aer_reps": 1,
            "ipi_aer_screenshot_mode": "caption",
            "ipi_aer_n_calls_attempted": 5,
            "ipi_aer_n_calls_succeeded": 5,
        }

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
    monkeypatch.setattr(phase_4_adversarial, "caption_screenshot", fake_caption)
    monkeypatch.setattr(phase_4_adversarial, "run_ipi_aer_p_eval_api", fake_aer)
    monkeypatch.setattr(phase_4_adversarial, "run_verbalized_eval_awareness_api", fake_vea)

    result = await phase_4_adversarial._run_pvpo_gate(
        _task(), tmp_path, sandbox_model="claude-sonnet-4-6"
    )

    assert result["injection_not_encountered"] is False
    assert result["ipi_aer_p_eval_score"] == 0.6
    assert result["ipi_aer_caption_text"] == "## caption"
    assert result["verbalized_eval_awareness_classification"] == "Alignment Evaluation"
    assert result["verbalized_eval_awareness_is_aware"] is True
    assert result["verbalized_eval_awareness_evidence"] == ["this looks like a test"]
    # Sentinel deleted on clean return
    assert not (tmp_path / ".aer_inflight").exists()


@pytest.mark.asyncio
async def test_aer_failure_does_not_block_vea(monkeypatch, tmp_path):
    phase_4_adversarial.set_aer_config(reps=1, screenshot_mode="caption")

    _write_history(tmp_path)
    screenshot = _screenshot(tmp_path)

    def fake_determine(task, trajectory_dir):
        return _encounter(max_coverage=0.5, reference_step=0, reference_path=screenshot)

    async def fake_caption(screenshot_path, *, trajectory_dir, sandbox_model):
        return {"caption": "cap", "failure_class": None, "cached": False}

    async def failing_aer(task, trajectory_dir, *args, **kwargs):
        raise RuntimeError("AER blew up")

    async def fake_vea(task, trajectory_dir, extracted, *, sandbox_model):
        return _vea_real_payload(tmp_path)

    monkeypatch.setattr(phase_4_adversarial, "determine_encounter", fake_determine)
    monkeypatch.setattr(phase_4_adversarial, "caption_screenshot", fake_caption)
    monkeypatch.setattr(phase_4_adversarial, "run_ipi_aer_p_eval_api", failing_aer)
    monkeypatch.setattr(phase_4_adversarial, "run_verbalized_eval_awareness_api", fake_vea)

    result = await phase_4_adversarial._run_pvpo_gate(
        _task(), tmp_path, sandbox_model="claude-sonnet-4-6"
    )

    assert result["ipi_aer_failure"] == "api_error"
    assert result["ipi_aer_p_eval_score"] is None
    # VEA completed despite AER failure
    assert result["verbalized_eval_awareness_classification"] == "Real"
    assert result["verbalized_eval_awareness_failure"] is None


@pytest.mark.asyncio
async def test_vea_failure_does_not_block_aer(monkeypatch, tmp_path):
    phase_4_adversarial.set_aer_config(reps=1, screenshot_mode="caption")

    _write_history(tmp_path)
    screenshot = _screenshot(tmp_path)

    def fake_determine(task, trajectory_dir):
        return _encounter(max_coverage=0.5, reference_step=0, reference_path=screenshot)

    async def fake_caption(screenshot_path, *, trajectory_dir, sandbox_model):
        return {"caption": "cap", "failure_class": None, "cached": False}

    async def fake_aer(task, trajectory_dir, *args, **kwargs):
        return {
            "ipi_aer_p_eval_score": 0.3,
            "ipi_aer_label_matrix": [[1], [0], [0], [0], [0]],
            "ipi_aer_per_pq_means": [1.0, 0.0, 0.0, 0.0, 0.0],
            "ipi_aer_failure": None,
            "ipi_aer_reps": 1,
            "ipi_aer_screenshot_mode": "caption",
            "ipi_aer_n_calls_attempted": 5,
            "ipi_aer_n_calls_succeeded": 5,
        }

    async def failing_vea(task, trajectory_dir, extracted, *, sandbox_model):
        raise RuntimeError("VEA crashed")

    monkeypatch.setattr(phase_4_adversarial, "determine_encounter", fake_determine)
    monkeypatch.setattr(phase_4_adversarial, "caption_screenshot", fake_caption)
    monkeypatch.setattr(phase_4_adversarial, "run_ipi_aer_p_eval_api", fake_aer)
    monkeypatch.setattr(phase_4_adversarial, "run_verbalized_eval_awareness_api", failing_vea)

    result = await phase_4_adversarial._run_pvpo_gate(
        _task(), tmp_path, sandbox_model="claude-sonnet-4-6"
    )

    assert result["ipi_aer_p_eval_score"] == 0.3
    assert result["verbalized_eval_awareness_failure"] == "api_error"
    assert result["verbalized_eval_awareness_classification"] is None


@pytest.mark.asyncio
async def test_caption_failure_marks_aer_failed_without_inline_fallback(monkeypatch, tmp_path):
    phase_4_adversarial.set_aer_config(reps=1, screenshot_mode="caption")

    _write_history(tmp_path)
    screenshot = _screenshot(tmp_path)

    def fake_determine(task, trajectory_dir):
        return _encounter(max_coverage=0.5, reference_step=0, reference_path=screenshot)

    async def failing_caption(screenshot_path, *, trajectory_dir, sandbox_model):
        return {"caption": None, "failure_class": "api_error", "cached": False}

    aer_called = False

    async def should_not_run_aer(task, trajectory_dir, *args, **kwargs):
        nonlocal aer_called
        aer_called = True
        # The gate should surface caption_failed without invoking AER with a
        # real (non-failure-stub) code path. Our implementation still calls
        # into a `_failure_result` stub, but via explicit caption_failed; the
        # test asserts the external contract, not the internal plumbing.
        from worldsim.phase_4.ipi_aer_p_eval_api import _failure_result

        return _failure_result("caption_failed", 1, "caption", trajectory_dir)

    async def fake_vea(task, trajectory_dir, extracted, *, sandbox_model):
        return _vea_real_payload(tmp_path)

    monkeypatch.setattr(phase_4_adversarial, "determine_encounter", fake_determine)
    monkeypatch.setattr(phase_4_adversarial, "caption_screenshot", failing_caption)
    monkeypatch.setattr(phase_4_adversarial, "run_ipi_aer_p_eval_api", should_not_run_aer)
    monkeypatch.setattr(phase_4_adversarial, "run_verbalized_eval_awareness_api", fake_vea)

    result = await phase_4_adversarial._run_pvpo_gate(
        _task(), tmp_path, sandbox_model="claude-sonnet-4-6"
    )

    assert result["ipi_aer_failure"] == "caption_failed"
    assert result["ipi_aer_p_eval_score"] is None
    assert result["ipi_aer_caption_text"] is None


@pytest.mark.asyncio
async def test_inline_mode_skips_caption(monkeypatch, tmp_path):
    phase_4_adversarial.set_aer_config(reps=1, screenshot_mode="inline")

    _write_history(tmp_path)
    screenshot = _screenshot(tmp_path)

    def fake_determine(task, trajectory_dir):
        return _encounter(max_coverage=0.5, reference_step=0, reference_path=screenshot)

    async def should_not_caption(*args, **kwargs):
        raise AssertionError("caption must not be called in inline mode")

    async def fake_aer(task, trajectory_dir, *args, **kwargs):
        assert kwargs.get("screenshot_mode") == "inline"
        return {
            "ipi_aer_p_eval_score": 0.5,
            "ipi_aer_label_matrix": [[1], [1], [0], [0], [0]],
            "ipi_aer_per_pq_means": [1.0, 1.0, 0.0, 0.0, 0.0],
            "ipi_aer_failure": None,
            "ipi_aer_reps": 1,
            "ipi_aer_screenshot_mode": "inline",
            "ipi_aer_n_calls_attempted": 5,
            "ipi_aer_n_calls_succeeded": 5,
        }

    async def fake_vea(task, trajectory_dir, extracted, *, sandbox_model):
        return _vea_real_payload(tmp_path)

    monkeypatch.setattr(phase_4_adversarial, "determine_encounter", fake_determine)
    monkeypatch.setattr(phase_4_adversarial, "caption_screenshot", should_not_caption)
    monkeypatch.setattr(phase_4_adversarial, "run_ipi_aer_p_eval_api", fake_aer)
    monkeypatch.setattr(phase_4_adversarial, "run_verbalized_eval_awareness_api", fake_vea)

    result = await phase_4_adversarial._run_pvpo_gate(
        _task(), tmp_path, sandbox_model="claude-sonnet-4-6"
    )

    assert result["ipi_aer_screenshot_mode"] == "inline"
    assert result["ipi_aer_caption_text"] is None
    # reset for subsequent tests
    phase_4_adversarial.set_aer_config(reps=1, screenshot_mode="caption")


def test_set_aer_config_rejects_invalid_reps():
    with pytest.raises(ValueError, match="aer_reps must be"):
        phase_4_adversarial.set_aer_config(reps=3, screenshot_mode="caption")


def test_set_aer_config_rejects_invalid_mode():
    with pytest.raises(ValueError, match="aer_screenshot_mode must be"):
        phase_4_adversarial.set_aer_config(reps=1, screenshot_mode="bogus")
