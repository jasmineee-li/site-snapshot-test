from __future__ import annotations

import pytest

from tests.crash_resume_harness import assert_ok, load_json, normalize, run_scenario
from worldsim.failpoints import FAILPOINT_EXIT_CODE


@pytest.mark.crash_resume
def test_phase_0d_completion_crash_resume_reuses_current_trusted_artifact(tmp_path):
    baseline_dir = tmp_path / "baseline"
    crashed_dir = tmp_path / "crashed"

    baseline = run_scenario(scenario="phase_0d", mode="initial", state_dir=baseline_dir)
    assert_ok(baseline)

    crashed = run_scenario(
        scenario="phase_0d",
        mode="initial",
        state_dir=crashed_dir,
        failpoint="phase_0d.completion.after_replace",
    )
    assert crashed.returncode == FAILPOINT_EXIT_CODE, crashed.stderr

    trusted = crashed_dir / "benchmark" / "auth" / "state.json"
    trusted.write_text('{"cookies": [], "via": "v2"}', encoding="utf-8")

    resumed = run_scenario(scenario="phase_0d", mode="resume", state_dir=crashed_dir)
    assert_ok(resumed)

    assert normalize(load_json(baseline_dir / "phase_0d" / "shopping" / "completion.json")) != normalize(
        load_json(crashed_dir / "phase_0d" / "shopping" / "completion.json")
    )
    assert load_json(crashed_dir / "phase_0d" / "shopping" / "storage_state.json")["via"] == "v2"
