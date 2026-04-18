from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_MODULE = "tests.crash_resume_scenarios"
_DROP_KEYS = {
    "adversarial_tasks_path",
    "benchmark_path",
    "benchmark_root",
    "instances_path",
    "logs_dir",
    "task_dir_root",
    "timestamp",
    "trajectory_dir",
    "contracts_path",
    "feasibility_completed_at",
    "feasibility_report_path",
    "feasibility_infeasible_path",
    "skipped_at",
    "verified_at",
    "first_failed_at",
}


def run_scenario(
    *,
    scenario: str,
    mode: str,
    state_dir: Path,
    failpoint: str | None = None,
    mutation: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["WORLDSIM_STATE_DIR"] = str(state_dir)
    # Synthetic scenarios don't build tasks with a Phase 2c feasibility
    # stanza. Disable strict admission in the child so Phase 4 still runs.
    env["WORLDSIM_STRICT_FEASIBILITY"] = "false"
    # Synthetic scenarios mock run_judge/generate_variant at the module level
    # and don't need real Anthropic credentials. Bypass the Phase 4 preflight
    # so subprocess launches succeed in CI environments with no auth.
    env["WORLDSIM_PHASE_4_SKIP_PREFLIGHT"] = "1"
    if failpoint is None:
        env.pop("WORLDSIM_FAILPOINTS", None)
    else:
        env["WORLDSIM_FAILPOINTS"] = failpoint
    return subprocess.run(
        [
            sys.executable,
            "-m",
            SCENARIO_MODULE,
            "--scenario",
            scenario,
            "--mode",
            mode,
            "--state-dir",
            str(state_dir),
            *([] if mutation is None else ["--mutation", mutation]),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        items = {key: normalize(val) for key, val in value.items() if key not in _DROP_KEYS}
        return {key: items[key] for key in sorted(items)}
    if isinstance(value, list):
        normalized = [normalize(item) for item in value]
        if all(isinstance(item, dict) for item in normalized):
            for key in ("id", "task_id", "strategy"):
                if all(key in item for item in normalized):
                    return sorted(normalized, key=lambda item: str(item[key]))
        return normalized
    return value


def assert_ok(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        f"command failed with {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
