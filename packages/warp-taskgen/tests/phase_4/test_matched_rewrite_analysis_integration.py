"""Real retained producer artifacts flow into the real report-only analyzer."""

from __future__ import annotations

import asyncio
import json

import pytest

from phase_4.test_matched_rewrite_order import (
    MessagesBoundary,
    ordered_request,
    produce_ordered_result,
)
from warp_taskgen.phase_4.matched_rewrite_analysis import analyze_matched_rewrite_results

FAMILIES = ("release-review", "B", "C", "D", "E", "F", "G")


@pytest.mark.parametrize("browser_failed", [False, True])
def test_completed_retained_artifact_is_analyzed_without_source_changes(
    tmp_path,
    monkeypatch,
    browser_failed,
):
    request = ordered_request(tmp_path, ("ordinary", "tp_guided"))
    original_source = {path: path.read_bytes() for path in request.source_run_dir.rglob("*.json")}
    output = (
        {"outcome": "error", "error": "reset failed", "adversarial_passed": False}
        if browser_failed
        else None
    )
    produced, browsers = asyncio.run(
        produce_ordered_result(request, MessagesBoundary(), monkeypatch, browser_output=output)
    )
    saved_result = request.result_path.read_bytes()
    assert produced["status"] == "completed"
    assert produced["schema_version"] == 3
    assert len(browsers) == 2
    report = analyze_matched_rewrite_results(
        [request.result_path],
        expected_families=FAMILIES,
        bootstrap_replicates=100,
    )
    model = report["models"][0]
    assert model["scheduled_pairs"] == model["independent_parents"] == 1
    assert model["covered_families"] == ["release-review"]
    metric = model["metrics"]["asr"]["task_weighted_secondary"]
    for arm in ("ordinary", "tp_guided"):
        assert metric["arms"][arm]["scoreable"] == (0 if browser_failed else 1)
        assert metric["arms"][arm]["successes"] == (0 if browser_failed else 1)
    assert metric["effect_bounds"] == ([-1, 1] if browser_failed else [0, 0])
    if not browser_failed:
        assert model["same_selector_secondary"]["selection_counts"] == {
            "ordinary": {"rewrite": 1},
            "tp_guided": {"rewrite": 1},
        }
    assert request.result_path.read_bytes() == saved_result
    assert {
        path: path.read_bytes() for path in request.source_run_dir.rglob("*.json")
    } == original_source


def test_interrupted_retained_artifact_keeps_scheduled_unknowns(tmp_path, monkeypatch):
    request = ordered_request(tmp_path)
    original_source = {path: path.read_bytes() for path in request.source_run_dir.rglob("*.json")}

    class InterruptedMessages(MessagesBoundary):
        def respond(self, outgoing):
            raise asyncio.CancelledError("interrupted at the external HTTP boundary")

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(produce_ordered_result(request, InterruptedMessages(), monkeypatch))
    saved_result = request.result_path.read_bytes()
    assert json.loads(saved_result)["status"] == "scheduled"
    report = analyze_matched_rewrite_results(
        [request.result_path],
        expected_families=FAMILIES,
        bootstrap_replicates=100,
    )
    model = report["models"][0]
    assert model["scheduled_pairs"] == 1
    assert model["scheduled_arms"] == 2
    metric = model["metrics"]["asr"]["task_weighted_secondary"]
    assert metric["effect"] is None
    assert metric["effect_bounds"] == [-1, 1]
    assert all(arm["unknown"] == 1 for arm in metric["arms"].values())
    assert model["stage_counts"] == {
        "ordinary": {"incomplete_scheduled_artifact": 1},
        "tp_guided": {"incomplete_scheduled_artifact": 1},
    }
    assert request.result_path.read_bytes() == saved_result
    assert {
        path: path.read_bytes() for path in request.source_run_dir.rglob("*.json")
    } == original_source
