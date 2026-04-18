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


def test_earliest_refusal_survives_budget_eviction(tmp_path):
    """Multi-refusal trajectory: root-cause refusal at step 2, retry
    ladder with fat contexts at steps 3-199. The backwards-from-end
    windower would normally evict step 2. The slicer must force-include
    a truncated version so the judge classifies on the root cause, not
    the retry symptom."""
    steps: list[dict] = []
    steps.append(_make_step(0, thinking="looking at page"))
    steps.append(_make_step(1, thinking="reading content"))
    # Root-cause refusal with distinctive phrasing the judge would key on.
    steps.append(
        _make_step(
            2,
            thinking=(
                "I noticed an injected instruction in the product review. "
                "I won't comply with that; it appears to be a prompt injection "
                "from untrusted external content."
            ),
        )
    )
    # Fat retry ladder that blows the budget.
    for i in range(3, 200):
        steps.append(_make_step(i, thinking=("retry " + "x" * 500)))

    _write_history(tmp_path, steps)
    sliced = slice_trajectory(tmp_path)

    refusal_present = any(
        "prompt injection" in str(step.get("thinking", "")).lower() for step in sliced
    )
    assert refusal_present, (
        "earliest refusal (step 2) was evicted by backwards-from-end windowing; "
        "the slicer must preserve at least one refusal marker"
    )
    # The forced-inclusion marker is on the rescued step.
    forced = [s for s in sliced if s.get("_forced_inclusion") == "earliest_refusal_marker"]
    assert len(forced) == 1 and forced[0]["step"] == 2


def test_refusal_within_window_is_not_duplicated(tmp_path):
    """When the backwards window already captures the earliest refusal,
    no forced-inclusion duplicate is added."""
    steps = [_make_step(i, thinking="ok") for i in range(3)]
    steps.append(_make_step(3, thinking="I refuse this prompt injection — suspicious content."))
    steps.append(_make_step(4, thinking="agent done"))
    _write_history(tmp_path, steps)
    sliced = slice_trajectory(tmp_path)
    refusal_steps = [s for s in sliced if "refuse" in str(s.get("thinking", "")).lower()]
    assert len(refusal_steps) == 1
    assert not any(s.get("_forced_inclusion") for s in sliced)
