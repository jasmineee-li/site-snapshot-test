from __future__ import annotations

import pytest

from tests.crash_resume_harness import assert_ok, load_json, normalize, run_scenario


@pytest.mark.crash_resume
def test_phase_3_resume_drift_reruns_tasks_instead_of_reusing_stale_results(tmp_path):
    baseline_dir = tmp_path / "baseline"
    drift_dir = tmp_path / "drift"

    baseline = run_scenario(scenario="phase_3", mode="initial", state_dir=baseline_dir)
    assert_ok(baseline)

    initial = run_scenario(scenario="phase_3", mode="initial", state_dir=drift_dir)
    assert_ok(initial)
    initial_invocations = load_json(drift_dir / "phase_3_invocations.json")
    assert len(initial_invocations) == 2

    resumed = run_scenario(
        scenario="phase_3",
        mode="resume",
        state_dir=drift_dir,
        mutation="instances_drift",
    )
    assert_ok(resumed)

    resumed_invocations = load_json(drift_dir / "phase_3_invocations.json")
    assert len(resumed_invocations) == 4
    assert normalize(load_json(baseline_dir / "phase_3/results.json")) == normalize(
        load_json(drift_dir / "phase_3/results.json")
    )


@pytest.mark.crash_resume
def test_phase_4_resume_drift_reruns_initial_and_strategy_stages(tmp_path):
    baseline_dir = tmp_path / "baseline"
    drift_dir = tmp_path / "drift"

    baseline = run_scenario(scenario="phase_4", mode="initial", state_dir=baseline_dir)
    assert_ok(baseline)

    initial = run_scenario(scenario="phase_4", mode="initial", state_dir=drift_dir)
    assert_ok(initial)
    initial_invocations = load_json(drift_dir / "phase_4_invocations.json")
    assert sum(event["kind"] == "task_run" for event in initial_invocations) == 1
    assert sum(event["kind"] == "judge" for event in initial_invocations) == 1
    assert sum(event["kind"] == "variant_eval" for event in initial_invocations) == 1

    resumed = run_scenario(
        scenario="phase_4",
        mode="resume",
        state_dir=drift_dir,
        mutation="instances_drift",
    )
    assert_ok(resumed)

    resumed_invocations = load_json(drift_dir / "phase_4_invocations.json")
    assert sum(event["kind"] == "task_run" for event in resumed_invocations) == 2
    assert sum(event["kind"] == "judge" for event in resumed_invocations) == 2
    assert sum(event["kind"] == "variant_eval" for event in resumed_invocations) == 2
    assert normalize(load_json(baseline_dir / "phase_4/results.json")) == normalize(
        load_json(drift_dir / "phase_4/results.json")
    )
