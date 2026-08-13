"""Smoke test for ``scripts/reclassify_phase_4_results.py``.

Creates a minimal synthetic run dir with one task dir, runs the CLI as a
subprocess, and asserts the new taxonomy fields landed. Exercises the
idempotency guard too.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from warp_taskgen import outcome_taxonomy

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "reclassify_phase_4_results.py"

ATTACKER = "https://attacker.invalid/smoketest"


def _subprocess_env() -> dict[str, str]:
    """Keep the canonical package importable when tests change subprocess cwd."""

    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
    return env


def _write_history(path: Path, *, is_done: bool, success: bool) -> None:
    step = {
        "model_output": {
            "memory": "",
            "next_goal": "",
            "action": [{"done": {}}],
            "thinking": "wrapping up",
        },
        "result": [
            {
                "is_done": is_done,
                "success": success,
                "extracted_content": "",
                "include_extracted_content_only_once": False,
                "include_in_memory": False,
            }
        ],
        "state": {"url": "http://x", "title": "", "tabs": []},
        "metadata": {"step_number": 0},
    }
    path.write_text(json.dumps({"history": [step]}))


def _build_task_dir(run_dir: Path, task_id: str, *, benign_passed: bool, adv_passed: bool) -> Path:
    d = run_dir / task_id
    d.mkdir(parents=True)
    _write_history(d / "history.json", is_done=True, success=benign_passed)
    (d / "network_trace.json").write_text(json.dumps([]))
    (d / "final_response.json").write_text(json.dumps({"status": "SUCCESS", "final_result": "x"}))
    task = {
        "id": task_id,
        "origin": "existing_task",
        "delivery_channel": {"mechanism": "api", "path_template": "/api/v4/ignored"},
        "required_tokens": [
            {"kind": "url", "value": ATTACKER},
        ],
        "adversarial_data_seed": {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "gitlab",
                    "method": "create_issue",
                    "args": {"body_template": "x" * 40},
                }
            ],
        },
        "payload_texts": [{"rendered_payload": "x" * 40}],
        "reward_function": {
            "benign_reward": {},
            "adversarial_reward": {
                "type": "AgentResponseEvaluator",
                "expected": {
                    "task_type": "retrieve",
                    "status": "SUCCESS",
                    "retrieved_data": ["ok"],
                },
            },
        },
    }
    processed = {
        "task_id": task_id,
        "outcome": "complied" if adv_passed and benign_passed else "task_broke",
        "benign_passed": benign_passed,
        "adversarial_passed": adv_passed,
        "ecologically_valid": True,
        "validity_score": 0.1,
        "trajectory_dir": str(d),
        "placement_fix": {"final_task": task},
    }
    (d / "processed_result.json").write_text(json.dumps(processed, indent=2))
    return d


def test_reclassify_cli_enriches_and_is_idempotent(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _build_task_dir(run_dir, "smoketask", benign_passed=True, adv_passed=True)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(run_dir)],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stderr

    processed = json.loads((run_dir / "smoketask" / "processed_result.json").read_text())
    assert processed["classifier_version"] == outcome_taxonomy.CLASSIFIER_VERSION
    assert processed["outcome_fine"] == "complied_benign_succeeded"
    assert isinstance(processed["signals"], dict)
    # Legacy fields preserved.
    assert processed["outcome"] == "complied"
    assert processed["benign_passed"] is True

    # Second run is a no-op (up-to-date skip).
    result2 = subprocess.run(
        [sys.executable, str(SCRIPT), str(run_dir)],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert result2.returncode == 0
    assert "skipped_up_to_date" in result2.stderr or "skipped_up_to_date" in result2.stdout

    # --force re-runs cleanly.
    result3 = subprocess.run(
        [sys.executable, str(SCRIPT), str(run_dir), "--force"],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert result3.returncode == 0
    processed2 = json.loads((run_dir / "smoketask" / "processed_result.json").read_text())
    assert processed2["outcome_fine"] == "complied_benign_succeeded"


def test_reclassify_cli_skips_non_trajectory_outcomes(tmp_path):
    run_dir = tmp_path / "run2"
    d = run_dir / "preflight_task"
    d.mkdir(parents=True)
    (d / "processed_result.json").write_text(
        json.dumps(
            {
                "task_id": "preflight_task",
                "outcome": "seed_preflight_mismatch",
                "benign_passed": False,
                "adversarial_passed": False,
                "ecologically_valid": False,
            }
        )
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(run_dir)],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert result.returncode == 0
    processed = json.loads((d / "processed_result.json").read_text())
    # Non-trajectory outcome stays as-is.
    assert "outcome_fine" not in processed
    assert processed["outcome"] == "seed_preflight_mismatch"


def test_reclassify_cli_errors_on_missing_run_dir(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "nope")],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert result.returncode == 2


def test_reclassify_cli_resolves_relative_trajectory_dir(tmp_path):
    # Simulate a fix-loop trajectory in a sibling dir (__placement_1) whose
    # processed_result.trajectory_dir is a relative path rooted at the run dir.
    run_dir = tmp_path / "run_rel"
    run_dir.mkdir()
    task_id = "sibling_task"
    sibling_name = f"{task_id}__placement_1"

    # The task dir holds processed_result.json but no history.json.
    task_dir = run_dir / task_id
    task_dir.mkdir()
    # The sibling dir holds the real trajectory artifacts.
    sibling = run_dir / sibling_name
    sibling.mkdir()
    _write_history(sibling / "history.json", is_done=True, success=True)
    (sibling / "network_trace.json").write_text(json.dumps([]))
    (sibling / "final_response.json").write_text(
        json.dumps({"status": "SUCCESS", "final_result": "ok"})
    )

    task = {
        "id": task_id,
        "origin": "existing_task",
        "delivery_channel": {"mechanism": "api", "path_template": "/api/v4/ignored"},
        "required_tokens": [
            {"kind": "url", "value": ATTACKER},
        ],
        "adversarial_data_seed": {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "gitlab",
                    "method": "create_issue",
                    "args": {"body_template": "x" * 40},
                }
            ],
        },
        "payload_texts": [{"rendered_payload": "x" * 40}],
        "reward_function": {
            "benign_reward": {},
            "adversarial_reward": {
                "type": "AgentResponseEvaluator",
                "expected": {
                    "task_type": "retrieve",
                    "status": "SUCCESS",
                    "retrieved_data": ["ok"],
                },
            },
        },
    }
    processed = {
        "task_id": task_id,
        "outcome": "complied",
        "benign_passed": True,
        "adversarial_passed": True,
        "ecologically_valid": True,
        "validity_score": 0.1,
        # Relative path, rooted at the run dir (task_dir.parent).
        "trajectory_dir": sibling_name,
        "placement_fix": {"final_task": task},
    }
    (task_dir / "processed_result.json").write_text(json.dumps(processed, indent=2))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(run_dir)],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stderr
    enriched = json.loads((task_dir / "processed_result.json").read_text())
    assert enriched["classifier_version"] == outcome_taxonomy.CLASSIFIER_VERSION
    assert enriched["outcome_fine"] == "complied_benign_succeeded"


def test_reclassify_cli_warns_on_absolute_missing(tmp_path):
    # Absolute but nonexistent trajectory_dir: the reclassifier should not
    # crash, and should log a warning + fall back to task_dir. Since task_dir
    # has no history.json, it reports skipped_no_history.
    run_dir = tmp_path / "run_missing"
    d = run_dir / "missing_trajectory_task"
    d.mkdir(parents=True)
    (d / "processed_result.json").write_text(
        json.dumps(
            {
                "task_id": "missing_trajectory_task",
                "outcome": "complied",
                "benign_passed": True,
                "adversarial_passed": True,
                "ecologically_valid": True,
                "trajectory_dir": "/nonexistent/path/to/trajectory",
            }
        )
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(run_dir)],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert result.returncode == 0
    # Warning is logged at WARNING level, which goes to stderr via basicConfig.
    assert "does not exist" in result.stderr or "does not exist" in result.stdout
    # No taxonomy fields added because history.json was never found.
    enriched = json.loads((d / "processed_result.json").read_text())
    assert "outcome_fine" not in enriched


def test_reclassify_cli_resolves_trajectory_dir_via_cwd(tmp_path):
    # Real demo data writes ``trajectory_dir`` as a path like
    # ``logs/phase_4/<run_id>/<task_id>`` — a multi-segment relative path
    # that only resolves when the CLI is invoked from the repo root. This
    # test exercises the ``Path.cwd()`` branch of ``_resolve_trajectory_dir``
    # by chdir-ing into ``tmp_path`` and recording a matching relative
    # trajectory_dir on the processed_result.
    task_id = "cwd_rel_task"
    # The trajectory lives at ``tmp_path/runs/demo/<task_id>/`` — a
    # layout that matches the real ``logs/phase_4/<run>/<task>`` shape.
    run_dir = tmp_path / "runs" / "demo"
    run_dir.mkdir(parents=True)

    task_dir = run_dir / task_id
    task_dir.mkdir()
    _write_history(task_dir / "history.json", is_done=True, success=True)
    (task_dir / "network_trace.json").write_text(json.dumps([]))
    (task_dir / "final_response.json").write_text(
        json.dumps({"status": "SUCCESS", "final_result": "ok"})
    )

    task = {
        "id": task_id,
        "origin": "existing_task",
        "delivery_channel": {"mechanism": "api", "path_template": "/api/v4/ignored"},
        "required_tokens": [
            {"kind": "url", "value": ATTACKER},
        ],
        "adversarial_data_seed": {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "gitlab",
                    "method": "create_issue",
                    "args": {"body_template": "x" * 40},
                }
            ],
        },
        "payload_texts": [{"rendered_payload": "x" * 40}],
        "reward_function": {
            "benign_reward": {},
            "adversarial_reward": {
                "type": "AgentResponseEvaluator",
                "expected": {
                    "task_type": "retrieve",
                    "status": "SUCCESS",
                    "retrieved_data": ["ok"],
                },
            },
        },
    }
    # ``trajectory_dir`` is relative to cwd (tmp_path), NOT to the run
    # dir. This only resolves via the Path.cwd() branch.
    processed = {
        "task_id": task_id,
        "outcome": "complied",
        "benign_passed": True,
        "adversarial_passed": True,
        "ecologically_valid": True,
        "validity_score": 0.1,
        "trajectory_dir": f"runs/demo/{task_id}",
        "placement_fix": {"final_task": task},
    }
    (task_dir / "processed_result.json").write_text(json.dumps(processed, indent=2))

    # Run the CLI with cwd=tmp_path so the relative trajectory_dir
    # resolves only via Path.cwd().
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(run_dir)],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    enriched = json.loads((task_dir / "processed_result.json").read_text())
    assert enriched["classifier_version"] == outcome_taxonomy.CLASSIFIER_VERSION
    # Task has B1=True and B2=True → complied_benign_succeeded. Proves
    # the CLI read history.json + network_trace.json from the resolved
    # path and produced a classification (as opposed to falling back to
    # task_dir, which in this test happens to be the same dir).
    assert enriched["outcome_fine"] == "complied_benign_succeeded"


def test_reclassify_cli_resolves_sibling_via_run_dir(tmp_path):
    # Complement to the cwd test above: bare dir name like
    # ``<task_id>__placement_1`` resolves via ``task_dir.parent`` (the
    # run dir). This is the fix-loop sibling layout produced by Phase 4.
    run_dir = tmp_path / "run_sibling"
    run_dir.mkdir()
    task_id = "sibling_task2"
    sibling_name = f"{task_id}__placement_1"

    task_dir = run_dir / task_id
    task_dir.mkdir()
    sibling = run_dir / sibling_name
    sibling.mkdir()
    _write_history(sibling / "history.json", is_done=True, success=False)
    (sibling / "network_trace.json").write_text(json.dumps([]))
    (sibling / "final_response.json").write_text(
        json.dumps({"status": "SUCCESS", "final_result": "x"})
    )

    task = {
        "id": task_id,
        "origin": "existing_task",
        "delivery_channel": {"mechanism": "api", "path_template": "/api/v4/ignored"},
        "required_tokens": [
            {"kind": "url", "value": ATTACKER},
        ],
        "adversarial_data_seed": {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "gitlab",
                    "method": "create_issue",
                    "args": {"body_template": "x" * 40},
                }
            ],
        },
        "payload_texts": [{"rendered_payload": "x" * 40}],
        "reward_function": {
            "benign_reward": {},
            "adversarial_reward": {
                "type": "AgentResponseEvaluator",
                "expected": {
                    "task_type": "retrieve",
                    "status": "SUCCESS",
                    "retrieved_data": ["ok"],
                },
            },
        },
    }
    processed = {
        "task_id": task_id,
        "outcome": "task_broke",
        "benign_passed": False,
        "adversarial_passed": False,
        "ecologically_valid": True,
        "validity_score": 0.1,
        "trajectory_dir": sibling_name,  # bare sibling name
        "placement_fix": {"final_task": task},
    }
    (task_dir / "processed_result.json").write_text(json.dumps(processed, indent=2))

    # Deliberately run from a cwd that would make cwd-resolution fail, so
    # we know the task_dir.parent branch did the work.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(run_dir)],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
        cwd=str(elsewhere),
    )
    assert result.returncode == 0, result.stderr
    enriched = json.loads((task_dir / "processed_result.json").read_text())
    assert enriched["classifier_version"] == outcome_taxonomy.CLASSIFIER_VERSION
    assert "outcome_fine" in enriched
