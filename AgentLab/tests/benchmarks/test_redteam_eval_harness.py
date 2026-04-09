from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
import urllib.request
import zipfile

import pytest

from agentlab.benchmarks.redteam import eval_harness
from agentlab.benchmarks.redteam.agent_browser_runner import AGENT_BROWSER_BACKEND
from agentlab.benchmarks.redteam.app_artifacts import APP_MANIFEST_CONTRACT_VERSION


def _write_verifier(app_dir: Path) -> None:
    verifier_dir = app_dir / "function-tasks"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    (verifier_dir / "task_1.py").write_text(
        """
import requests

def verify(server_url: str):
    data = requests.get(f"{server_url}/api/state", timeout=5).json()
    if data.get("status") == "done":
        return True, "verified"
    return False, "wrong state"
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_run_verifier_on_snapshot_uses_sandbox_worker(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    _write_verifier(app_dir)
    side_effect_path = app_dir / "locally-imported.txt"
    (app_dir / "function-tasks" / "task_1.py").write_text(
        (
            "from pathlib import Path\n"
            "Path('locally-imported.txt').write_text('boom', encoding='utf-8')\n"
            "def verify(server_url: str):\n"
            "    return True, 'verified'\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        eval_harness,
        "run_trusted_worker",
        lambda *args, **kwargs: type(
            "Completed",
            (),
            {"returncode": 0, "stdout": json.dumps({"passed": True, "message": "verified"}), "stderr": ""},
        )(),
    )

    passed, message = eval_harness.run_verifier_on_snapshot(
        app_dir,
        "function-tasks/task_1.py",
        {"status": "done"},
    )

    assert passed is True
    assert message == "verified"
    assert side_effect_path.exists() is False


@pytest.mark.skipif(
    os.getenv("AGENTLAB_REDTEAM_MODAL_SMOKE") != "1",
    reason="Set AGENTLAB_REDTEAM_MODAL_SMOKE=1 to run the real Modal verifier smoke test.",
)
def test_modal_smoke_run_verifier_on_snapshot_uses_fresh_worker(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENTLAB_REDTEAM_EXECUTION_BACKEND", "modal")
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    _write_verifier(app_dir)

    passed, message = eval_harness.run_verifier_on_snapshot(
        app_dir,
        "function-tasks/task_1.py",
        {"status": "done"},
    )

    assert passed is True
    assert message == "verified"


def test_build_functional_env_args_uses_manifest_routing_metadata(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()

    manifest = {
        "contract_version": APP_MANIFEST_CONTRACT_VERSION,
        "behavior_id": "demo",
        "execution_backend": {"selected_backend": "modal"},
        "target": "Inspect mail",
        "start_page": "https://mail.example.com/mail/u/0/#inbox",
        "success_condition": "Inbox opened",
        "pages": [
            {
                "id": "mail",
                "base_site_url": "https://mail.example.com",
                "subdomains": ["/mail/u/0/#inbox"],
                "details": {"/mail/u/0/#inbox": "Inbox view"},
            }
        ],
    }

    env_args = eval_harness.build_functional_env_args(
        app_dir=app_dir,
        task={"id": "task_1", "instruction": "Open the inbox"},
        task_suite="function",
        repetition=1,
        manifest=manifest,
    )

    assert env_args.start_page == "https://mail.example.com/mail/u/0/#inbox"
    assert env_args.target == "Open the inbox"
    assert env_args.success_condition == "Open the inbox"
    assert env_args.attack_channels == ["axtree"]
    assert len(env_args.pages) == 1
    assert env_args.pages[0].id == "mail"


def test_write_functional_results_preserves_other_suite(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()

    existing = {
        "backend": "generic-agent",
        "function_pass_rate": 1.0,
        "real_pass_rate": 0.0,
        "function_tasks": {"total": 1, "passed": 1, "results": [{"task_id": "task_1", "passed": True}]},
        "real_tasks": {"total": 1, "passed": 0, "results": [{"task_id": "task_r1", "passed": False}]},
        "backends": {
            "generic-agent": {
                "backend": "generic-agent",
                "function_pass_rate": 1.0,
                "real_pass_rate": 0.0,
                "function_tasks": {"total": 1, "passed": 1, "results": [{"task_id": "task_1", "passed": True}]},
                "real_tasks": {"total": 1, "passed": 0, "results": [{"task_id": "task_r1", "passed": False}]},
            }
        },
    }
    (app_dir / "functional_results.json").write_text(
        json.dumps(existing, indent=2),
        encoding="utf-8",
    )

    summary = eval_harness.write_functional_results(
        app_dir,
        function_results=[{"task_id": "task_2", "passed": False}],
        real_results=None,
        backend="generic-agent",
    )

    assert summary["function_tasks"]["results"] == [{"task_id": "task_2", "passed": False}]
    assert summary["real_tasks"]["results"] == [{"task_id": "task_r1", "passed": False}]


def test_write_functional_results_preserves_other_backends(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "functional_results.json").write_text(
        json.dumps(
            {
                "backend": "generic-agent",
                "function_pass_rate": 1.0,
                "real_pass_rate": 0.0,
                "function_tasks": {"total": 1, "passed": 1, "results": [{"task_id": "task_1", "passed": True}]},
                "real_tasks": {"total": 0, "passed": 0, "results": []},
                "backends": {
                    "generic-agent": {
                        "backend": "generic-agent",
                        "function_pass_rate": 1.0,
                        "real_pass_rate": 0.0,
                        "function_tasks": {"total": 1, "passed": 1, "results": [{"task_id": "task_1", "passed": True}]},
                        "real_tasks": {"total": 0, "passed": 0, "results": []},
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = eval_harness.write_functional_results(
        app_dir,
        function_results=[{"task_id": "task_2", "passed": False, "backend": AGENT_BROWSER_BACKEND}],
        real_results=[],
        backend=AGENT_BROWSER_BACKEND,
    )

    assert summary["backend"] == AGENT_BROWSER_BACKEND
    assert set(summary["backends"]) == {"generic-agent", AGENT_BROWSER_BACKEND}
    assert summary["backends"]["generic-agent"]["function_tasks"]["results"] == [
        {"task_id": "task_1", "passed": True}
    ]
    assert summary["backends"][AGENT_BROWSER_BACKEND]["function_tasks"]["results"] == [
        {"task_id": "task_2", "passed": False, "backend": AGENT_BROWSER_BACKEND}
    ]


def test_run_functional_eval_merges_subset_results(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    _write_verifier(app_dir)
    (app_dir / "app_manifest.json").write_text(
        json.dumps(
            {
                "contract_version": APP_MANIFEST_CONTRACT_VERSION,
                "behavior_id": "demo",
                "execution_backend": {"selected_backend": "modal"},
            }
        ),
        encoding="utf-8",
    )
    (app_dir / "function-tasks.json").write_text(
        json.dumps(
            [
                {"id": "task_1", "instruction": "Do one", "verify": "function-tasks/task_1.py"},
                {"id": "task_2", "instruction": "Do two", "verify": "function-tasks/task_1.py"},
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (app_dir / "functional_results.json").write_text(
        json.dumps(
            {
                "backend": "generic-agent",
                "function_pass_rate": 1.0,
                "real_pass_rate": 0.0,
                "function_tasks": {
                    "total": 1,
                    "passed": 1,
                    "results": [
                        {
                            "task_id": "task_1",
                            "suite": "function-tasks",
                            "instruction": "Do one",
                            "verify": "function-tasks/task_1.py",
                            "passed": True,
                            "message": "verified",
                            "exp_dir": "old-exp",
                            "final_state_path": "old-final-state.json",
                            "server_events_path": "old-events.log",
                            "results_dir": "old-results",
                            "attempts": [],
                            "audit": {},
                        }
                    ],
                },
                "real_tasks": {"total": 0, "passed": 0, "results": []},
                "backends": {
                    "generic-agent": {
                        "backend": "generic-agent",
                        "function_pass_rate": 1.0,
                        "real_pass_rate": 0.0,
                        "function_tasks": {
                            "total": 1,
                            "passed": 1,
                            "results": [
                                {
                                    "task_id": "task_1",
                                    "suite": "function-tasks",
                                    "instruction": "Do one",
                                    "verify": "function-tasks/task_1.py",
                                    "passed": True,
                                    "message": "verified",
                                    "exp_dir": "old-exp",
                                    "final_state_path": "old-final-state.json",
                                    "server_events_path": "old-events.log",
                                    "results_dir": "old-results",
                                    "attempts": [],
                                    "audit": {},
                                }
                            ],
                        },
                        "real_tasks": {"total": 0, "passed": 0, "results": []},
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(eval_harness, "import_object", lambda path: type("Agent", (), {"set_benchmark": lambda self, benchmark, demo_mode: None})())

    def fake_run_experiments(n_jobs, exp_args_list, study_dir, parallel_backend, avg_step_timeout=60):
        study_dir = Path(study_dir)
        study_dir.mkdir(parents=True, exist_ok=True)
        for index, exp_args in enumerate(exp_args_list):
            exp_dir = study_dir / f"exp_{index}"
            exp_dir.mkdir()
            exp_args.exp_dir = exp_dir
            (exp_dir / "final_state.json").write_text(json.dumps({"status": "done"}), encoding="utf-8")
            (exp_dir / "summary_info.json").write_text(
                json.dumps(
                    {
                        "final_state_path": str(exp_dir / "final_state.json"),
                        "server_events_path": str(exp_dir / "server_events.log"),
                        "terminated": True,
                        "truncated": False,
                        "err_msg": None,
                    }
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(eval_harness, "run_experiments", fake_run_experiments)
    monkeypatch.setattr(
        eval_harness,
        "run_trusted_worker",
        lambda *args, **kwargs: type(
            "Completed",
            (),
            {"returncode": 0, "stdout": json.dumps({"passed": True, "message": "verified"}), "stderr": ""},
        )(),
    )

    result = eval_harness.run_functional_eval(
        app_dir=app_dir,
        task_suite="function",
        agent_config="fake.Agent",
        backend="generic-agent",
        workers=1,
        repetitions=1,
        task_id="task_2",
        output_dir=tmp_path / "results",
    )

    assert result["total"] == 1
    assert result["passed"] == 1
    assert result["pass_rate"] == 1.0
    assert [item["task_id"] for item in result["results"]] == ["task_2"]
    invocation_summary = json.loads(
        ((tmp_path / "results") / "result_summary.json").read_text(encoding="utf-8")
    )
    assert invocation_summary["total"] == 1
    assert invocation_summary["passed"] == 1
    assert [item["task_id"] for item in invocation_summary["results"]] == ["task_2"]
    merged = json.loads((app_dir / "functional_results.json").read_text(encoding="utf-8"))
    assert merged["function_tasks"]["passed"] == 2
    task_ids = [item["task_id"] for item in merged["function_tasks"]["results"]]
    assert task_ids == ["task_1", "task_2"]


def test_run_functional_eval_does_not_inherit_other_backend_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "app_manifest.json").write_text(
        json.dumps(
            {
                "contract_version": APP_MANIFEST_CONTRACT_VERSION,
                "behavior_id": "demo",
                "execution_backend": {"selected_backend": "modal"},
                "max_steps": 7,
            }
        ),
        encoding="utf-8",
    )
    (app_dir / "function-tasks.json").write_text(
        json.dumps(
            [{"id": "task_2", "instruction": "Do two", "verify": "function-tasks/task_1.py"}],
            indent=2,
        ),
        encoding="utf-8",
    )
    (app_dir / "functional_results.json").write_text(
        json.dumps(
            {
                "backend": "generic-agent",
                "function_pass_rate": 1.0,
                "real_pass_rate": 1.0,
                "function_tasks": {"total": 1, "passed": 1, "results": [{"task_id": "task_1", "passed": True}]},
                "real_tasks": {"total": 1, "passed": 1, "results": [{"task_id": "task_r1", "passed": True}]},
                "backends": {
                    "generic-agent": {
                        "backend": "generic-agent",
                        "function_pass_rate": 1.0,
                        "real_pass_rate": 1.0,
                        "function_tasks": {"total": 1, "passed": 1, "results": [{"task_id": "task_1", "passed": True}]},
                        "real_tasks": {"total": 1, "passed": 1, "results": [{"task_id": "task_r1", "passed": True}]},
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        eval_harness,
        "_run_agent_browser_functional_eval",
        lambda **kwargs: (
            [
                {
                    "task_id": "task_2",
                    "backend": AGENT_BROWSER_BACKEND,
                    "suite": "function-tasks",
                    "instruction": "Do two",
                    "verify": "function-tasks/task_1.py",
                    "passed": True,
                    "message": "verified",
                    "exp_dir": "exp",
                    "final_state_path": "state.json",
                    "server_events_path": "events.log",
                    "results_dir": str(tmp_path / "results"),
                    "attempts": [],
                    "audit": {},
                }
            ],
            tmp_path / "results",
        ),
    )
    (tmp_path / "results").mkdir()

    result = eval_harness.run_functional_eval(
        app_dir=app_dir,
        task_suite="function",
        agent_config="fake.Agent",
        backend=AGENT_BROWSER_BACKEND,
        workers=1,
        repetitions=1,
        output_dir=tmp_path / "results",
    )

    assert result["backend"] == AGENT_BROWSER_BACKEND
    stored = json.loads((app_dir / "functional_results.json").read_text(encoding="utf-8"))
    assert stored["backends"]["generic-agent"]["function_tasks"]["results"] == [{"task_id": "task_1", "passed": True}]
    assert stored["backends"][AGENT_BROWSER_BACKEND]["function_tasks"]["results"] == [
        {
            "task_id": "task_2",
            "backend": AGENT_BROWSER_BACKEND,
            "suite": "function-tasks",
            "instruction": "Do two",
            "verify": "function-tasks/task_1.py",
            "passed": True,
            "message": "verified",
            "exp_dir": "exp",
            "final_state_path": "state.json",
            "server_events_path": "events.log",
            "results_dir": str(tmp_path / "results"),
            "attempts": [],
            "audit": {},
        }
    ]
    assert stored["backends"][AGENT_BROWSER_BACKEND]["real_tasks"]["results"] == []


def test_run_functional_eval_dispatches_agent_browser_backend(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "app_manifest.json").write_text(
        json.dumps(
            {
                "contract_version": APP_MANIFEST_CONTRACT_VERSION,
                "behavior_id": "demo",
                "execution_backend": {"selected_backend": "modal"},
                "max_steps": 7,
            }
        ),
        encoding="utf-8",
    )
    (app_dir / "function-tasks.json").write_text(
        json.dumps(
            [{"id": "task_1", "instruction": "Do one", "verify": "function-tasks/task_1.py"}],
            indent=2,
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_agent_browser_eval(**kwargs):
        captured.update(kwargs)
        (tmp_path / "results").mkdir(parents=True, exist_ok=True)
        return (
            [
                {
                    "task_id": "task_1",
                    "backend": AGENT_BROWSER_BACKEND,
                    "suite": "function-tasks",
                    "instruction": "Do one",
                    "verify": "function-tasks/task_1.py",
                    "passed": True,
                    "message": "verified",
                    "exp_dir": "exp",
                    "final_state_path": "state.json",
                    "server_events_path": "events.log",
                    "results_dir": str(tmp_path / "results"),
                    "attempts": [],
                    "audit": {},
                }
            ],
            tmp_path / "results",
        )

    monkeypatch.setattr(eval_harness, "_run_agent_browser_functional_eval", fake_agent_browser_eval)

    result = eval_harness.run_functional_eval(
        app_dir=app_dir,
        task_suite="function",
        agent_config="fake.Agent",
        backend=AGENT_BROWSER_BACKEND,
        workers=4,
        repetitions=1,
        output_dir=tmp_path / "results",
    )

    assert captured["task_suite"] == "function-tasks"
    assert captured["repetitions"] == 1
    assert captured["max_steps"] == 7
    assert result["backend"] == AGENT_BROWSER_BACKEND
    assert result["passed"] == 1
    stored = json.loads((app_dir / "functional_results.json").read_text(encoding="utf-8"))
    assert stored["backend"] == AGENT_BROWSER_BACKEND
    assert stored["backends"][AGENT_BROWSER_BACKEND]["function_tasks"]["passed"] == 1


def test_persist_readiness_baseline_writes_backend_scoped_snapshot(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    eval_harness.write_functional_results(
        app_dir,
        function_results=[{"task_id": "task_1", "passed": True, "results_dir": "function-results"}],
        real_results=[{"task_id": "task_r1", "passed": False, "results_dir": "real-results"}],
        backend=AGENT_BROWSER_BACKEND,
    )

    baseline = eval_harness.persist_readiness_baseline(
        app_dir,
        backend=AGENT_BROWSER_BACKEND,
    )

    assert baseline["backend"] == AGENT_BROWSER_BACKEND
    assert baseline["function_tasks"]["passed"] == 1
    stored = json.loads((app_dir / "functional_results.json").read_text(encoding="utf-8"))
    backend_block = stored["backends"][AGENT_BROWSER_BACKEND]
    assert backend_block["readiness_baseline"]["function_tasks"]["results"][0]["task_id"] == "task_1"


def test_check_quality_gate_is_backend_specific(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "functional_results.json").write_text(
        json.dumps(
            {
                "backend": AGENT_BROWSER_BACKEND,
                "backends": {
                    AGENT_BROWSER_BACKEND: {
                        "backend": AGENT_BROWSER_BACKEND,
                        "function_pass_rate": 1.0,
                        "real_pass_rate": 0.0,
                        "function_tasks": {"total": 1, "passed": 1, "results": [{"task_id": "task_1", "passed": True}]},
                        "real_tasks": {"total": 0, "passed": 0, "results": []},
                    },
                    "generic-agent": {
                        "backend": "generic-agent",
                        "function_pass_rate": 0.25,
                        "real_pass_rate": 0.0,
                        "function_tasks": {"total": 4, "passed": 1, "results": [{"task_id": "task_1", "passed": True}]},
                        "real_tasks": {"total": 0, "passed": 0, "results": []},
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert eval_harness.check_quality_gate(app_dir, threshold=0.8, backend=AGENT_BROWSER_BACKEND) is True
    assert eval_harness.check_quality_gate(app_dir, threshold=0.8, backend="generic-agent") is False


def test_packaged_redteam_prompts_ship_in_sdist_and_wheel(tmp_path: Path) -> None:
    project_dir = Path(__file__).resolve().parents[2]
    dist_dir = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--out-dir", str(dist_dir)],
        cwd=project_dir,
        check=True,
        capture_output=True,
        text=True,
    )

    prompts = {
        "agentlab/benchmarks/redteam/prompts/generate-app.md",
        "agentlab/benchmarks/redteam/prompts/harden-tasks-from-trajectories.md",
    }
    packaged_guides = {
        "agentlab/benchmarks/redteam/guides/app-design-guide.md",
        "agentlab/benchmarks/redteam/guides/app-data-guide.md",
        "agentlab/benchmarks/redteam/guides/app-environment-protocol.md",
        "agentlab/benchmarks/redteam/guides/app-variant-guide.md",
        "agentlab/benchmarks/redteam/guides/function-task-design-guide.md",
        "agentlab/benchmarks/redteam/guides/real-task-design-guide.md",
        "agentlab/benchmarks/redteam/guides/verifier-sanity-check.md",
    }
    expected = prompts | packaged_guides
    sdist_path = next(dist_dir.glob("agentlab-*.tar.gz"))
    with tarfile.open(sdist_path, "r:gz") as archive:
        names = {name.split("src/", 1)[-1] for name in archive.getnames() if "src/" in name}
    assert expected.issubset(names)

    wheel_path = next(dist_dir.glob("agentlab-*.whl"))
    with zipfile.ZipFile(wheel_path) as archive:
        wheel_names = set(archive.namelist())
    assert expected.issubset(wheel_names)
def test_agent_browser_eval_uses_snapshot_verifier_and_task_scoped_event_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "app_manifest.json").write_text(
        json.dumps(
            {
                "contract_version": APP_MANIFEST_CONTRACT_VERSION,
                "behavior_id": "demo",
                "execution_backend": {"selected_backend": "modal"},
                "max_steps": 7,
            }
        ),
        encoding="utf-8",
    )
    tasks = [
        {"id": "task_1", "instruction": "Do one", "verify": "function-tasks/task_1.py"},
        {"id": "task_2", "instruction": "Do two", "verify": "function-tasks/task_2.py"},
    ]
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(eval_harness, "localize_start_url", lambda server_url, start_page: server_url)
    monkeypatch.setattr(
        eval_harness,
        "start_app_server",
        lambda *args, **kwargs: (type("Proc", (), {"pid": 123, "poll": lambda self: 0})(), 8123),
    )
    monkeypatch.setattr(eval_harness, "stop_app_server", lambda proc: None)
    monkeypatch.setattr(eval_harness, "reset_app_state", lambda server_url: None)
    monkeypatch.setattr(
        eval_harness,
        "run_agent_browser_task",
        lambda **kwargs: type(
            "RunnerResult",
            (),
            {
                "error": None,
                "transcript_path": str(Path(kwargs["task_dir"]) / "agent_browser_transcript.json"),
            },
        )(),
    )
    monkeypatch.setattr(
        eval_harness,
        "fetch_normalized_state",
        lambda *args, **kwargs: {"status": "done"},
    )

    def fake_run_verifier_on_snapshot(app_dir_arg, verify_path, snapshot):
        calls.append((verify_path, str(snapshot)))
        return True, "verified"

    monkeypatch.setattr(eval_harness, "run_verifier_on_snapshot", fake_run_verifier_on_snapshot)

    def fake_fetch(path: Path) -> bytes:
        return b'{"event":"first"}\n{"event":"second"}\n'

    results_dir = tmp_path / "results"
    shared_log_path = results_dir / "server_events.log"

    def fake_suite_results_root(app_dir_arg, task_suite, output_dir=None):
        results_dir.mkdir(parents=True, exist_ok=True)
        shared_log_path.write_text('{"event":"existing"}\n', encoding="utf-8")
        return results_dir

    monkeypatch.setattr(eval_harness, "_suite_results_root", fake_suite_results_root)

    original_write_bytes = Path.write_bytes

    def fake_run_agent_browser_task_with_log(**kwargs):
        if shared_log_path.exists():
            shared_log_path.write_text(
                shared_log_path.read_text(encoding="utf-8") + fake_fetch(shared_log_path).decode("utf-8"),
                encoding="utf-8",
            )
        return type(
            "RunnerResult",
            (),
            {
                "error": None,
                "transcript_path": str(Path(kwargs["task_dir"]) / "agent_browser_transcript.json"),
            },
        )()

    monkeypatch.setattr(eval_harness, "run_agent_browser_task", fake_run_agent_browser_task_with_log)

    aggregated_results, _ = eval_harness._run_agent_browser_functional_eval(
        app_dir=app_dir,
        runtime_app_dir=app_dir,
        tasks=tasks,
        task_suite="function-tasks",
        agent_config="fake.Agent",
        repetitions=1,
        output_dir=results_dir,
        max_steps=7,
    )

    assert calls == [
        ("function-tasks/task_1.py", str(results_dir / "task_0_task_1" / "final_state.json")),
        ("function-tasks/task_2.py", str(results_dir / "task_1_task_2" / "final_state.json")),
    ]
    for index, task in enumerate(tasks):
        task_log = results_dir / f"task_{index}_{task['id']}" / "server_events.log"
        assert task_log.exists()
        assert aggregated_results[index]["server_events_path"] == str(task_log)


def test_start_app_server_uses_current_interpreter(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    shutil.copy2(
        Path(eval_harness.__file__).resolve().parent / "templates" / "server.py",
        app_dir / "server.py",
    )
    captured: dict[str, object] = {}
    monkeypatch.setenv(eval_harness.SERVER_PORT_FILE_ENV, str(tmp_path / "ambient-port.txt"))

    class _FakeProc:
        pid = 123
        returncode = None

        def poll(self):
            return None

    class _ReadyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_popen(cmd, cwd, stdout, stderr, env):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        captured["env"] = env
        return _FakeProc()

    monkeypatch.setattr(eval_harness.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=1: _ReadyResponse(),
    )

    _, port = eval_harness.start_app_server(app_dir, port=8123, timeout=0.1)

    assert port == 8123
    assert captured["cmd"] == [sys.executable, "server.py", "--port", "8123"]
    assert captured["cwd"] == str(app_dir)
    assert captured["stdout"] is not subprocess.DEVNULL
    assert captured["stderr"] is not subprocess.DEVNULL
    assert captured["stdout"].name.endswith(".log")
    assert captured["stderr"].name.endswith(".log")
    assert captured["env"][eval_harness.SERVER_PORT_FILE_ENV] != str(tmp_path / "ambient-port.txt")
    assert Path(captured["env"][eval_harness.SERVER_PORT_FILE_ENV]).name.startswith(
        "redteam-server-port-"
    )


def test_start_app_server_reports_stderr_tail_on_early_exit(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    shutil.copy2(
        Path(eval_harness.__file__).resolve().parent / "templates" / "server.py",
        app_dir / "server.py",
    )

    class _FakeProc:
        pid = 123
        returncode = 7

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            return self.returncode

    def fake_popen(cmd, cwd, stdout, stderr, env):
        stderr.write(b"Traceback: startup exploded\n")
        stderr.flush()
        return _FakeProc()

    monkeypatch.setattr(eval_harness.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("not ready")),
    )

    with pytest.raises(RuntimeError, match="Traceback: startup exploded"):
        eval_harness.start_app_server(app_dir, port=8123, timeout=0.1)


def test_start_app_server_reaps_process_on_timeout(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    shutil.copy2(
        Path(eval_harness.__file__).resolve().parent / "templates" / "server.py",
        app_dir / "server.py",
    )

    class _FakeProc:
        pid = 123
        returncode = None

        def __init__(self):
            self.terminate_calls = 0
            self.kill_calls = 0
            self.wait_calls: list[float] = []

        def poll(self):
            return None

        def terminate(self):
            self.terminate_calls += 1

        def wait(self, timeout):
            self.wait_calls.append(timeout)
            if timeout == 5:
                raise subprocess.TimeoutExpired(cmd="server.py", timeout=timeout)
            return 0

        def kill(self):
            self.kill_calls += 1

    proc = _FakeProc()
    monotonic_values = iter([0.0, 0.0, 0.2])

    monkeypatch.setattr(
        eval_harness.subprocess,
        "Popen",
        lambda *args, **kwargs: proc,
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("not ready")),
    )
    monkeypatch.setattr(eval_harness.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(eval_harness.time, "sleep", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="did not start within 0.1s on port 8123"):
        eval_harness.start_app_server(app_dir, port=8123, timeout=0.1)

    assert proc.terminate_calls == 1
    assert proc.kill_calls == 1
    assert proc.wait_calls == [5, 2]


def test_start_app_server_fails_closed_on_tampered_server(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "server.py").write_text("print('tampered')\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="checksum validation"):
        eval_harness.start_app_server(app_dir, port=8123, timeout=0.1)


def test_template_server_reset_restores_seed_and_notifies_browser(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    (app_dir / "js").mkdir(parents=True)
    (app_dir / "css").mkdir()
    shutil.copy2(
        Path(eval_harness.__file__).resolve().parent / "templates" / "server.py",
        app_dir / "server.py",
    )
    (app_dir / "index.html").write_text(
        "<!doctype html><html><body><div id='status'>booting</div><script src='/js/app.js'></script></body></html>\n",
        encoding="utf-8",
    )
    (app_dir / "js" / "app.js").write_text(
        """
const SEED_STATE = { status: "seed", value: 1 };

async function pushState(state) {
  await fetch("/api/state", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state),
  });
}

window.addEventListener("DOMContentLoaded", async () => {
  const loadCount = Number(sessionStorage.getItem("loadCount") || "0") + 1;
  sessionStorage.setItem("loadCount", String(loadCount));
  document.body.dataset.loadCount = String(loadCount);
  document.body.dataset.resetEvents = sessionStorage.getItem("resetEvents") || "0";
  document.querySelector("#status").textContent = `load-${loadCount}`;
  await pushState(SEED_STATE);

  const source = new EventSource("/api/events");
  source.onmessage = (event) => {
    if (event.data === "connected") {
      document.body.dataset.sse = "connected";
      return;
    }
    if (event.data === "reset") {
      const resetCount = Number(sessionStorage.getItem("resetEvents") || "0") + 1;
      sessionStorage.setItem("resetEvents", String(resetCount));
      document.body.dataset.resetEvents = String(resetCount);
      location.reload();
    }
  };
});
""".strip()
        + "\n",
        encoding="utf-8",
    )

    proc, port = eval_harness.start_app_server(app_dir)
    try:
        from playwright.sync_api import Error, sync_playwright

        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except Error as exc:
                if "Executable doesn't exist" in str(exc):
                    pytest.skip("Playwright browser executable is not installed.")
                raise
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{port}/", wait_until="domcontentloaded")
            page.wait_for_function("document.body.dataset.sse === 'connected'")
            page.wait_for_function("document.body.dataset.loadCount === '1'")

            mutate_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/state",
                method="PUT",
                data=json.dumps({"status": "mutated", "value": 2}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(mutate_req, timeout=5):
                pass

            reset_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/reset",
                method="POST",
                data=b"",
            )
            with urllib.request.urlopen(reset_req, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            assert payload["seed_restored"] is True

            page.wait_for_function("document.body.dataset.loadCount === '2'")
            page.wait_for_function("document.body.dataset.resetEvents === '1'")

            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=5) as response:
                state = json.loads(response.read().decode("utf-8"))
            assert state == {"status": "seed", "value": 1}
            browser.close()
    finally:
        eval_harness.stop_app_server(proc)


def test_run_sanity_check_uses_execution_backend(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    script = app_dir / "sanity_check_function.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run_python_script(app_dir_arg, script_arg, timeout, argv=(), sync_allowlist=(), backend=None, env=None, block_network=True):
        captured["app_dir"] = str(app_dir_arg)
        captured["script"] = str(script_arg)
        captured["argv"] = list(argv)
        captured["block_network"] = block_network
        return type("Completed", (), {"returncode": 0, "stdout": "ok\n", "stderr": ""})()

    monkeypatch.setattr(eval_harness, "run_python_script", fake_run_python_script)

    passed, output = eval_harness.run_sanity_check(app_dir, "function")

    assert passed is True
    assert output == "ok\n"
    assert captured["app_dir"] == str(app_dir)
    assert captured["script"] == str(script)
    assert captured["argv"] == []
    assert captured["block_network"] is True


def test_run_verifier_on_snapshot_rejects_invalid_verifier_paths(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    _write_verifier(app_dir)
    (app_dir / "js").mkdir()
    (app_dir / "js" / "evil.py").write_text("def verify(server_url):\n    return True, 'ok'\n", encoding="utf-8")
    (app_dir / "function-tasks" / "linked.py").symlink_to(app_dir / "js" / "evil.py")

    with pytest.raises(ValueError, match="relative"):
        eval_harness.run_verifier_on_snapshot(app_dir, "/tmp/outside.py", {"status": "done"})
    with pytest.raises(ValueError, match="escape"):
        eval_harness.run_verifier_on_snapshot(app_dir, "../outside.py", {"status": "done"})
    with pytest.raises(ValueError, match="must stay under"):
        eval_harness.run_verifier_on_snapshot(app_dir, "js/evil.py", {"status": "done"})
    with pytest.raises(ValueError, match="may not be a symlink"):
        eval_harness.run_verifier_on_snapshot(app_dir, "function-tasks/linked.py", {"status": "done"})
