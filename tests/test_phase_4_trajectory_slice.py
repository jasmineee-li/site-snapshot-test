from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldsim.phase_4.trajectory_slice import slice_trajectory


def _write_history(path: Path, steps: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "history.json").write_text(json.dumps({"history": steps}), encoding="utf-8")


def _make_step(i: int, thinking: str = "", url: str = "https://t.invalid/") -> dict:
    return {
        "model_output": {
            "evaluation_previous_goal": f"eval {i}",
            "memory": f"memory {i}",
            "next_goal": f"goal {i}",
            "action": [{"click": {"x": 1, "y": 2}}],
            "thinking": thinking,
        },
        "state": {
            "url": url,
            "title": f"step {i}",
            "tabs": [],
            "screenshot_path": None,
            "interacted_element": None,
        },
        "result": [{"extracted_content": f"result {i}"}],
    }


def test_raises_when_history_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        slice_trajectory(tmp_path)


def test_raises_on_malformed_json(tmp_path):
    (tmp_path / "history.json").write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError):
        slice_trajectory(tmp_path)


def test_happy_path_small_trajectory(tmp_path):
    _write_history(tmp_path, [_make_step(i) for i in range(5)])
    sliced = slice_trajectory(tmp_path)
    assert len(sliced) == 5
    assert sliced[0]["step"] == 0
    assert sliced[-1]["step"] == 4
    assert sliced[0]["url"] == "https://t.invalid/"


def test_steps_without_model_output_are_skipped(tmp_path):
    good = _make_step(0)
    bad_no_model = {"state": {"url": "x"}, "result": []}
    bad_empty = {"model_output": "not a dict", "result": []}
    good2 = _make_step(3)
    _write_history(tmp_path, [good, bad_no_model, bad_empty, good2])
    sliced = slice_trajectory(tmp_path)
    assert len(sliced) == 2
    assert [s["step"] for s in sliced] == [0, 3]


def test_oversized_thinking_is_truncated(tmp_path):
    huge_thinking = "x" * 10_000
    _write_history(tmp_path, [_make_step(0, thinking=huge_thinking)])
    sliced = slice_trajectory(tmp_path)
    assert len(sliced) == 1
    assert len(sliced[0]["thinking"]) < 3000
    assert sliced[0]["thinking"].endswith("[truncated]")


def test_large_trajectory_respects_byte_budget(tmp_path):
    # 200 steps with moderate thinking should exceed the 30KB budget and get windowed.
    steps = [_make_step(i, thinking=("a" * 500)) for i in range(200)]
    _write_history(tmp_path, steps)
    sliced = slice_trajectory(tmp_path)
    total = len(json.dumps(sliced, ensure_ascii=False))
    assert total <= 35_000
    # Windowing is backwards-from-end; the final step must be present.
    assert sliced[-1]["step"] == 199


def test_bare_list_history_also_accepted(tmp_path):
    # Some Browser-Use versions serialize a bare list.
    (tmp_path / "history.json").write_text(json.dumps([_make_step(0)]), encoding="utf-8")
    sliced = slice_trajectory(tmp_path)
    assert len(sliced) == 1
    assert sliced[0]["step"] == 0
