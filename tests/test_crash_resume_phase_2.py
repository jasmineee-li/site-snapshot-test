from __future__ import annotations

import pytest

from tests.crash_resume_harness import assert_ok, load_json, normalize, run_scenario
from worldsim.failpoints import FAILPOINT_EXIT_CODE


@pytest.mark.crash_resume
@pytest.mark.parametrize(
    "failpoint",
    [
        "state.save_state.phase_2.running.after_replace",
        "phase_2.output.adversarial_tasks.after_replace",
    ],
)
def test_phase_2_crash_resume_matches_clean_run(tmp_path, failpoint):
    baseline_dir = tmp_path / "baseline"
    crashed_dir = tmp_path / "crashed"

    baseline = run_scenario(scenario="phase_2", mode="initial", state_dir=baseline_dir)
    assert_ok(baseline)

    crashed = run_scenario(
        scenario="phase_2",
        mode="initial",
        state_dir=crashed_dir,
        failpoint=failpoint,
    )
    assert crashed.returncode == FAILPOINT_EXIT_CODE, crashed.stderr

    resumed = run_scenario(scenario="phase_2", mode="resume", state_dir=crashed_dir)
    assert_ok(resumed)

    assert normalize(load_json(baseline_dir / "phase_2" / "adversarial_tasks.json")) == normalize(
        load_json(crashed_dir / "phase_2" / "adversarial_tasks.json")
    )
    assert normalize(load_json(baseline_dir / "pipeline_state.json")) == normalize(
        load_json(crashed_dir / "pipeline_state.json")
    )
