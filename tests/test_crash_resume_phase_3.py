from __future__ import annotations

import pytest

from tests.crash_resume_harness import assert_ok, load_json, normalize, run_scenario
from worldsim.failpoints import FAILPOINT_EXIT_CODE


@pytest.mark.crash_resume
@pytest.mark.parametrize(
    "failpoint",
    [
        "phase_3.outputs.contracts.before_replace",
        "phase_3.outputs.contracts.after_replace",
    ],
)
def test_phase_3_crash_resume_matches_clean_run(tmp_path, failpoint):
    baseline_dir = tmp_path / "baseline"
    crashed_dir = tmp_path / "crashed"

    baseline = run_scenario(scenario="phase_3", mode="initial", state_dir=baseline_dir)
    assert_ok(baseline)

    crashed = run_scenario(
        scenario="phase_3",
        mode="initial",
        state_dir=crashed_dir,
        failpoint=failpoint,
    )
    assert crashed.returncode == FAILPOINT_EXIT_CODE, crashed.stderr

    resumed = run_scenario(scenario="phase_3", mode="resume", state_dir=crashed_dir)
    assert_ok(resumed)

    for relative_path in (
        "phase_3/contracts.json",
        "pipeline_state.json",
    ):
        assert normalize(load_json(baseline_dir / relative_path)) == normalize(
            load_json(crashed_dir / relative_path)
        )
