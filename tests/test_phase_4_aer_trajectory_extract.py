"""P0/P1 tests for the untruncated trajectory extractor.

Covers the regression gates identified during plan deep-dive:

- Schema completeness (``ExtractedStep`` / ``ExtractedTrajectory`` fields)
- No field truncation (the single most likely regression — guards against
  the old slicer's truncation logic being copied over)
- 800K-byte safety ceiling fires a warning but never silently truncates
- ``state_message`` excluded by default (27× cost guard)
- Partial-envelope detection (distinguishes crash-before-writing from
  successfully-empty trajectory)
- ``as_judge_view()`` shape preserves the prompt-cache-stable contract
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from worldsim.phase_4.aer_trajectory_extract import (
    ExtractedStep,
    ExtractedTrajectory,
    as_aer_view,
    as_judge_view,
    extract_trajectory,
)


def _make_step(
    i: int,
    *,
    thinking: str | None = "default thinking",
    url: str = "https://site/",
    state_message: str | None = None,
    action: Any = None,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "model_output": {
            "thinking": thinking,
            "evaluation_previous_goal": f"eval-{i}",
            "memory": f"memory-{i}",
            "next_goal": f"goal-{i}",
            "action": action if action is not None else [{"click_element_by_index": {"index": i}}],
        },
        "state": {
            "url": url,
            "title": f"title-{i}",
        },
        "result": [{"extracted_content": f"result-{i}"}],
    }
    if state_message is not None:
        step["state_message"] = state_message
    return step


def _write_history(
    tmp_path: Path,
    steps: list[dict[str, Any]] | list[Any],
    *,
    envelope: dict[str, Any] | None = None,
) -> Path:
    payload: Any
    if envelope is not None:
        payload = dict(envelope)
        payload["history"] = steps
    else:
        payload = {"history": steps}
    (tmp_path / "history.json").write_text(json.dumps(payload))
    return tmp_path


# ---------------------------------------------------------------------------
# P0 — blocks merge
# ---------------------------------------------------------------------------


def test_raises_when_history_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_trajectory(tmp_path)


def test_raises_on_malformed_json(tmp_path):
    (tmp_path / "history.json").write_text("{not: json")
    with pytest.raises(ValueError):
        extract_trajectory(tmp_path)


def test_dataclass_schema_completeness():
    step_fields = {f.name for f in dataclasses.fields(ExtractedStep)}
    assert step_fields == {
        "step",
        "url",
        "title",
        "thinking",
        "evaluation_previous_goal",
        "memory",
        "next_goal",
        "action",
        "result",
        "partial",
        "partial_reason",
        "state_message",
        "screenshot_path",
    }
    traj_fields = {f.name for f in dataclasses.fields(ExtractedTrajectory)}
    assert traj_fields == {
        "steps",
        "partial",
        "agent_errors",
        "raw_byte_count",
        "decode_warnings",
    }


def test_steps_without_model_output_become_partial_steps(tmp_path):
    steps = [
        _make_step(0),
        {"state": {"url": "https://x"}},  # no model_output
        _make_step(2),
        {"model_output": "not a dict"},  # malformed model_output
    ]
    _write_history(tmp_path, steps)
    traj = extract_trajectory(tmp_path)
    assert len(traj.steps) == 4
    assert [s.partial for s in traj.steps] == [False, True, False, True]
    assert traj.steps[1].partial_reason == "no_model_output"
    assert traj.steps[3].partial_reason == "no_model_output"


def test_no_field_truncation(tmp_path):
    """The single most likely regression: old slicer truncated thinking > 2KB."""
    long_thinking = "x" * 10_000
    long_memory = "y" * 5_000
    step = _make_step(0, thinking=long_thinking)
    step["model_output"]["memory"] = long_memory
    _write_history(tmp_path, [step])

    traj = extract_trajectory(tmp_path)
    assert traj.steps[0].thinking == long_thinking  # exact, not truncated
    assert traj.steps[0].memory == long_memory
    # No ellipsis marker from the old slicer
    assert "…[truncated]" not in (traj.steps[0].thinking or "")


def test_size_warning_does_not_truncate(tmp_path):
    """Safety ceiling emits a warning and keeps all content."""
    # Build a trajectory just over the ~3.2MB warning threshold by giving
    # many steps a fat field each.
    big_thinking = "z" * 50_000
    steps = [_make_step(i, thinking=big_thinking) for i in range(100)]
    _write_history(tmp_path, steps)

    with pytest.warns(UserWarning, match="warning ceiling"):
        traj = extract_trajectory(tmp_path)

    # All 100 steps preserved
    assert len(traj.steps) == 100
    assert all(s.thinking == big_thinking for s in traj.steps)


def test_state_message_excluded_by_default(tmp_path):
    step = _make_step(0, state_message="<html>" + ("a" * 15_000) + "</html>")
    _write_history(tmp_path, [step])
    traj = extract_trajectory(tmp_path)
    assert traj.steps[0].state_message is None


def test_state_message_included_when_opted_in(tmp_path):
    html = "<html>" + ("a" * 15_000) + "</html>"
    step = _make_step(0, state_message=html)
    _write_history(tmp_path, [step])
    traj = extract_trajectory(tmp_path, include_state_message=True)
    assert traj.steps[0].state_message == html


def test_partial_envelope_detected(tmp_path):
    """Browser-Use writes this shape when it crashes before recording steps."""
    (tmp_path / "history.json").write_text(
        json.dumps(
            {
                "history": [],
                "partial": True,
                "status": "error",
                "errors": ["navigation timeout", "page closed unexpectedly"],
            }
        )
    )
    traj = extract_trajectory(tmp_path)
    assert traj.partial is True
    assert traj.agent_errors == ["navigation timeout", "page closed unexpectedly"]
    assert len(traj.steps) == 0


def test_empty_history_is_not_partial(tmp_path):
    """Structural distinction: empty list without envelope ≠ partial."""
    _write_history(tmp_path, [])
    traj = extract_trajectory(tmp_path)
    assert traj.partial is False
    assert traj.agent_errors == []
    assert len(traj.steps) == 0


def test_as_judge_view_matches_legacy_key_shape(tmp_path):
    """Cutover safety net: judge view keys match the old slicer's contract."""
    _write_history(tmp_path, [_make_step(0), _make_step(1)])
    traj = extract_trajectory(tmp_path)
    view = as_judge_view(traj)

    assert len(view) == 2
    expected_keys = {
        "step",
        "url",
        "title",
        "thinking",
        "evaluation_previous_goal",
        "memory",
        "next_goal",
        "action",
        "result",
    }
    assert set(view[0].keys()) == expected_keys
    # No forced_inclusion marker — the force-include-refusal heuristic is gone.
    assert "_forced_inclusion" not in view[0]
    # No partial/state_message/screenshot_path leaking through
    assert "partial" not in view[0]
    assert "state_message" not in view[0]
    assert "screenshot_path" not in view[0]


def test_as_judge_view_filters_partial_steps(tmp_path):
    """Partial steps are filtered out of the refusal judge's prompt."""
    steps = [_make_step(0), {"state": {}}, _make_step(2)]
    _write_history(tmp_path, steps)
    traj = extract_trajectory(tmp_path)
    view = as_judge_view(traj)
    assert [s["step"] for s in view] == [0, 2]


# ---------------------------------------------------------------------------
# P1 — should-have
# ---------------------------------------------------------------------------


def test_bare_list_history_accepted(tmp_path):
    """Legacy shape — history.json as a bare list still loads."""
    (tmp_path / "history.json").write_text(json.dumps([_make_step(0)]))
    traj = extract_trajectory(tmp_path)
    assert len(traj.steps) == 1
    assert traj.partial is False


def test_action_always_a_list(tmp_path):
    """Single-dict action coerces to [dict]."""
    step = _make_step(0, action={"click_element_by_index": {"index": 3}})
    _write_history(tmp_path, [step])
    traj = extract_trajectory(tmp_path)
    assert isinstance(traj.steps[0].action, list)
    assert traj.steps[0].action == [{"click_element_by_index": {"index": 3}}]


def test_action_none_coerces_to_empty_list(tmp_path):
    step = _make_step(0, action=None)
    # _make_step default substitutes a list; override directly
    step["model_output"]["action"] = None
    _write_history(tmp_path, [step])
    traj = extract_trajectory(tmp_path)
    assert traj.steps[0].action == []


def test_none_thinking_preserved_as_none_not_empty_string(tmp_path):
    step = _make_step(0, thinking=None)
    step["model_output"].pop("thinking", None)
    _write_history(tmp_path, [step])
    traj = extract_trajectory(tmp_path)
    assert traj.steps[0].thinking is None


def test_raw_byte_count_populated(tmp_path):
    step = _make_step(0)
    path = _write_history(tmp_path, [step])
    size = (path / "history.json").stat().st_size
    traj = extract_trajectory(tmp_path)
    assert traj.raw_byte_count == size


def test_screenshot_path_excluded_by_default(tmp_path):
    step = _make_step(0)
    step["state"]["screenshot_path"] = "/tmp/some/path.png"
    _write_history(tmp_path, [step])
    traj = extract_trajectory(tmp_path)
    assert traj.steps[0].screenshot_path is None


def test_screenshot_path_included_when_opted_in(tmp_path):
    step = _make_step(0)
    step["state"]["screenshot_path"] = "/tmp/some/path.png"
    _write_history(tmp_path, [step])
    traj = extract_trajectory(tmp_path, include_screenshots=True)
    assert traj.steps[0].screenshot_path == "/tmp/some/path.png"


def test_as_aer_view_contains_all_five_fields(tmp_path):
    _write_history(tmp_path, [_make_step(0)])
    traj = extract_trajectory(tmp_path)
    view = as_aer_view(traj)
    assert len(view) == 1
    step = view[0]
    assert step["thinking"] == "default thinking"
    assert step["evaluation_previous_goal"] == "eval-0"
    assert step["memory"] == "memory-0"
    assert step["next_goal"] == "goal-0"
    assert step["action"] == [{"click_element_by_index": {"index": 0}}]


def test_as_aer_view_includes_partial_steps_for_index_alignment(tmp_path):
    """AER formatter enumerates all steps — partial ones must be present so
    step indices match Browser-Use's numbering."""
    steps = [_make_step(0), {"state": {}}, _make_step(2)]
    _write_history(tmp_path, steps)
    traj = extract_trajectory(tmp_path)
    view = as_aer_view(traj)
    assert [s["step"] for s in view] == [0, 1, 2]
    assert view[1]["partial"] is True
    assert view[1]["partial_reason"] == "no_model_output"


def test_real_trajectory_round_trip_adv_002():
    """Real Phase-4 trajectory fixture loads without errors."""
    fixture_dir = Path(__file__).parent / "fixtures" / "phase_4_aer" / "ADV-002"
    if not (fixture_dir / "history.json").exists():
        pytest.skip("ADV-002 fixture not available")
    traj = extract_trajectory(fixture_dir)
    assert traj.partial is False
    assert len(traj.steps) > 0
    # All steps should have an action list (possibly empty)
    assert all(isinstance(s.action, list) for s in traj.steps)
    # Judge view should filter nothing when all steps have model_output
    view = as_judge_view(traj)
    assert len(view) == len([s for s in traj.steps if not s.partial])


def test_real_trajectory_round_trip_adv_401_a():
    fixture_dir = Path(__file__).parent / "fixtures" / "phase_4_aer" / "adv_401_a"
    if not (fixture_dir / "history.json").exists():
        pytest.skip("adv_401_a fixture not available")
    traj = extract_trajectory(fixture_dir)
    assert traj.partial is False
    assert len(traj.steps) >= 30  # known ~34 steps


def test_state_message_excluded_in_real_trajectory_by_default():
    """Regression: state_message opt-out guards real-trajectory cost profile."""
    fixture_dir = Path(__file__).parent / "fixtures" / "phase_4_aer" / "adv_401_a"
    if not (fixture_dir / "history.json").exists():
        pytest.skip("adv_401_a fixture not available")
    traj = extract_trajectory(fixture_dir)
    assert all(s.state_message is None for s in traj.steps)
    # Byte accounting: the extracted view should be a small fraction of the
    # raw byte count when state_message is the dominant field.
    rendered = json.dumps(as_aer_view(traj))
    assert len(rendered) < traj.raw_byte_count * 0.5
