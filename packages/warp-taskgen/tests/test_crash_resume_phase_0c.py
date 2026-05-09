from __future__ import annotations

import pytest

from tests.crash_resume_harness import assert_ok, load_json, normalize, run_scenario
from worldsim.failpoints import FAILPOINT_EXIT_CODE


@pytest.mark.crash_resume
def test_phase_0c_partial_artifact_crash_reruns_and_publishes_complete_outputs(tmp_path):
    baseline_dir = tmp_path / "baseline"
    crashed_dir = tmp_path / "crashed"

    baseline = run_scenario(scenario="phase_0c", mode="initial", state_dir=baseline_dir)
    assert_ok(baseline)

    crashed = run_scenario(
        scenario="phase_0c",
        mode="initial",
        state_dir=crashed_dir,
        failpoint="phase_0c.outputs.after_replace",
    )
    assert crashed.returncode == FAILPOINT_EXIT_CODE, crashed.stderr

    rerun = run_scenario(scenario="phase_0c", mode="initial", state_dir=crashed_dir)
    assert_ok(rerun)

    for relative_path in (
        "phase_0c/BENCHMARK_PROFILE_shopping.json",
        "phase_0c/AGENT_CONTEXT_shopping.json",
        "phase_0c/PROFILE_METADATA_shopping.json",
    ):
        assert normalize(load_json(baseline_dir / relative_path)) == normalize(
            load_json(crashed_dir / relative_path)
        )
