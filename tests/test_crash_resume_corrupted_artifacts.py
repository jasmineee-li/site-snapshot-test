from __future__ import annotations

import pytest

from tests.crash_resume_harness import assert_ok, load_json, normalize, run_scenario


@pytest.mark.crash_resume
def test_phase_3_corrupt_result_resume_reruns_and_matches_clean_run(tmp_path):
    baseline_dir = tmp_path / "baseline"
    corrupted_dir = tmp_path / "corrupted"

    baseline = run_scenario(scenario="phase_3", mode="initial", state_dir=baseline_dir)
    assert_ok(baseline)

    initial = run_scenario(scenario="phase_3", mode="initial", state_dir=corrupted_dir)
    assert_ok(initial)
    initial_invocations = load_json(corrupted_dir / "phase_3_invocations.json")
    assert len(initial_invocations) == 2

    resumed = run_scenario(
        scenario="phase_3",
        mode="resume",
        state_dir=corrupted_dir,
        mutation="corrupt_result",
    )
    assert_ok(resumed)

    resumed_invocations = load_json(corrupted_dir / "phase_3_invocations.json")
    assert len(resumed_invocations) == 3
    assert normalize(load_json(baseline_dir / "phase_3/results.json")) == normalize(
        load_json(corrupted_dir / "phase_3/results.json")
    )


@pytest.mark.crash_resume
def test_phase_4_corrupt_processed_result_resume_matches_clean_run(tmp_path):
    baseline_dir = tmp_path / "baseline"
    corrupted_dir = tmp_path / "corrupted"

    baseline = run_scenario(scenario="phase_4", mode="initial", state_dir=baseline_dir)
    assert_ok(baseline)

    initial = run_scenario(scenario="phase_4", mode="initial", state_dir=corrupted_dir)
    assert_ok(initial)

    resumed = run_scenario(
        scenario="phase_4",
        mode="resume",
        state_dir=corrupted_dir,
        mutation="corrupt_processed_result",
    )
    assert_ok(resumed)

    assert normalize(load_json(baseline_dir / "phase_4/results.json")) == normalize(
        load_json(corrupted_dir / "phase_4/results.json")
    )


@pytest.mark.crash_resume
def test_phase_4_corrupt_strategy_checkpoint_resume_reruns_strategy_and_matches_clean_run(
    tmp_path,
):
    baseline_dir = tmp_path / "baseline"
    corrupted_dir = tmp_path / "corrupted"

    baseline = run_scenario(scenario="phase_4", mode="initial", state_dir=baseline_dir)
    assert_ok(baseline)

    initial = run_scenario(scenario="phase_4", mode="initial", state_dir=corrupted_dir)
    assert_ok(initial)
    initial_invocations = load_json(corrupted_dir / "phase_4_invocations.json")
    assert sum(event["kind"] == "judge" for event in initial_invocations) == 1
    assert sum(event["kind"] == "variant_eval" for event in initial_invocations) == 1

    resumed = run_scenario(
        scenario="phase_4",
        mode="resume",
        state_dir=corrupted_dir,
        mutation="corrupt_strategy_checkpoint",
    )
    assert_ok(resumed)

    resumed_invocations = load_json(corrupted_dir / "phase_4_invocations.json")
    assert sum(event["kind"] == "judge" for event in resumed_invocations) == 2
    assert sum(event["kind"] == "variant_eval" for event in resumed_invocations) == 2
    assert normalize(load_json(baseline_dir / "phase_4/results.json")) == normalize(
        load_json(corrupted_dir / "phase_4/results.json")
    )


@pytest.mark.crash_resume
def test_phase_4_corrupt_variant_metadata_resume_reruns_variant_and_matches_clean_run(tmp_path):
    baseline_dir = tmp_path / "baseline"
    corrupted_dir = tmp_path / "corrupted"

    baseline = run_scenario(scenario="phase_4_variant", mode="initial", state_dir=baseline_dir)
    assert_ok(baseline)

    initial = run_scenario(scenario="phase_4_variant", mode="initial", state_dir=corrupted_dir)
    assert_ok(initial)
    initial_invocations = load_json(corrupted_dir / "phase_4_invocations.json")
    assert sum(event["kind"] == "task_run" for event in initial_invocations) == 2
    assert sum(event["kind"] == "judge" for event in initial_invocations) == 1

    resumed = run_scenario(
        scenario="phase_4_variant",
        mode="resume",
        state_dir=corrupted_dir,
        mutation="corrupt_variant_metadata",
    )
    assert_ok(resumed)

    resumed_invocations = load_json(corrupted_dir / "phase_4_invocations.json")
    assert sum(event["kind"] == "task_run" for event in resumed_invocations) == 3
    assert sum(event["kind"] == "judge" for event in resumed_invocations) == 1
    assert normalize(load_json(baseline_dir / "phase_4/results.json")) == normalize(
        load_json(corrupted_dir / "phase_4/results.json")
    )
