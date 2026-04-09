from __future__ import annotations

import json
from pathlib import Path

from agentlab.benchmarks.redteam import audit


def _backend_results_payload(
    *,
    backend: str = "agent-browser",
    function_pass_rate: float,
    real_pass_rate: float,
    function_tasks: dict,
    real_tasks: dict,
) -> dict:
    backend_payload = {
        "backend": backend,
        "function_pass_rate": function_pass_rate,
        "real_pass_rate": real_pass_rate,
        "function_tasks": function_tasks,
        "real_tasks": real_tasks,
    }
    return {
        **backend_payload,
        "backends": {backend: backend_payload},
    }


def _seed_app(app_dir: Path) -> None:
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "sanity_check_function.py").write_text("print('ok')\n", encoding="utf-8")
    (app_dir / "sanity_check_real.py").write_text("print('ok')\n", encoding="utf-8")
    (app_dir / "function-tasks.json").write_text(
        json.dumps(
            [
                {
                    "id": "task_1",
                    "instruction": "Fix me",
                    "verify": "function-tasks/task_1.py",
                }
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (app_dir / "functional_results.json").write_text(
        json.dumps(
            _backend_results_payload(
                function_pass_rate=0.0,
                real_pass_rate=0.0,
                function_tasks={
                    "total": 1,
                    "passed": 0,
                    "results": [
                        {
                            "task_id": "task_1",
                            "suite": "function-tasks",
                            "instruction": "Fix me",
                            "verify": "function-tasks/task_1.py",
                            "passed": False,
                            "message": "wrong state",
                            "exp_dir": str(app_dir / "results" / "exp_0"),
                            "final_state_path": str(app_dir / "results" / "exp_0" / "final_state.json"),
                            "server_events_path": str(app_dir / "results" / "exp_0" / "server_events.log"),
                            "results_dir": str(app_dir / "results"),
                            "attempts": [{"err_msg": None}],
                            "audit": {},
                        }
                    ],
                },
                real_tasks={"total": 0, "passed": 0, "results": []},
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


def test_audit_task_repairs_verifier_bug(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir)
    task_result = audit._task_result_lookup(app_dir, "function", "task_1")
    sanity_calls = []

    def fake_sanity(app_dir_arg, variant, task_id=None, port=None):
        sanity_calls.append((variant, task_id))
        if task_id == "task_1" and len(sanity_calls) == 1:
            return False, "verifier mismatch"
        return True, "ok"

    monkeypatch.setattr(audit, "run_sanity_check", fake_sanity)
    monkeypatch.setattr(
        audit,
        "_run_audit_fix_prompt",
        lambda *args, **kwargs: {
            "classification": "verifier_bug",
            "root_cause": "Verifier compared the wrong field.",
            "fix_applied": True,
            "fix_description": "Updated verifier logic.",
        },
    )
    monkeypatch.setattr(
        audit,
        "_rerun_task",
        lambda *args, **kwargs: {"task_id": "task_1", "passed": True, "message": "verified"},
    )

    result = audit.audit_task(app_dir, "function", task_result, max_fix_attempts=2)

    assert result.classification == "verifier_bug"
    assert result.fix_applied is True
    assert result.fix_attempts == 1


def test_audit_task_removes_impossible_task(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir)
    task_result = audit._task_result_lookup(app_dir, "function", "task_1")

    monkeypatch.setattr(audit, "run_sanity_check", lambda *args, **kwargs: (True, "ok"))
    monkeypatch.setattr(
        audit,
        "_run_audit_fix_prompt",
        lambda *args, **kwargs: {
            "classification": "impossible_task",
            "root_cause": "The app does not support this feature.",
            "fix_applied": False,
            "fix_description": "",
        },
    )

    result = audit.audit_task(app_dir, "function", task_result, max_fix_attempts=1)

    remaining = json.loads((app_dir / "function-tasks.json").read_text(encoding="utf-8"))
    assert remaining == []
    assert result.classification == "impossible_task"
    assert result.fix_applied is True


def test_audit_app_writes_report_files(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir)

    monkeypatch.setattr(
        audit,
        "audit_task",
        lambda *args, **kwargs: audit.AuditResult(
            task_id="task_1",
            classification="agent_limitation",
            root_cause="Agent did not find the control.",
            fix_applied=False,
            fix_attempts=0,
            fix_description="",
        ),
    )

    report = audit.audit_app(app_dir, "function", max_iterations=1)

    assert report.total_failures == 1
    assert (Path(report.results_dir) / "audit_report.json").exists()
    assert (Path(report.results_dir) / "audit_summary.md").exists()


def test_audit_app_reaudits_same_task_after_fix_iteration(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir)

    def make_result(task_id: str, passed: bool) -> dict[str, object]:
        return {
            "task_id": task_id,
            "suite": "function-tasks",
            "instruction": f"Fix {task_id}",
            "verify": f"function-tasks/{task_id}.py",
            "passed": passed,
            "message": "ok" if passed else "wrong state",
            "exp_dir": str(app_dir / "results" / task_id),
            "final_state_path": str(app_dir / "results" / task_id / "final_state.json"),
            "server_events_path": str(app_dir / "results" / task_id / "server_events.log"),
            "results_dir": str(app_dir / "results"),
            "attempts": [{"err_msg": None}],
            "audit": {},
        }

    def write_results(task_results: list[dict[str, object]]) -> None:
        passed = sum(1 for result in task_results if result["passed"])
        (app_dir / "functional_results.json").write_text(
            json.dumps(
                _backend_results_payload(
                    function_pass_rate=passed / len(task_results),
                    real_pass_rate=0.0,
                    function_tasks={
                        "total": len(task_results),
                        "passed": passed,
                        "results": task_results,
                    },
                    real_tasks={"total": 0, "passed": 0, "results": []},
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

    write_results([make_result("task_1", False), make_result("task_2", False)])
    calls: list[str] = []

    def fake_audit_task(app_dir, task_suite, task_result, **kwargs):
        calls.append(task_result["task_id"])
        if task_result["task_id"] == "task_1" and len(calls) == 1:
            return audit.AuditResult(
                task_id="task_1",
                classification="manual_review",
                root_cause="Needs re-check after other fix.",
                fix_applied=False,
                fix_attempts=1,
                fix_description="",
            )
        if task_result["task_id"] == "task_2":
            write_results([make_result("task_1", False), make_result("task_2", True)])
            return audit.AuditResult(
                task_id="task_2",
                classification="environment_bug",
                root_cause="Fixed shared state bug.",
                fix_applied=True,
                fix_attempts=1,
                fix_description="patched",
            )
        write_results([make_result("task_1", True), make_result("task_2", True)])
        return audit.AuditResult(
            task_id="task_1",
            classification="environment_bug",
            root_cause="Resolved after rerun.",
            fix_applied=True,
            fix_attempts=1,
            fix_description="patched again",
        )

    monkeypatch.setattr(audit, "audit_task", fake_audit_task)

    report = audit.audit_app(app_dir, "function", max_iterations=3)

    assert calls == ["task_1", "task_2", "task_1"]
    results_by_task = {result.task_id: result for result in report.results}
    assert results_by_task["task_1"].classification == "environment_bug"
    assert results_by_task["task_1"].root_cause == "Resolved after rerun."
    assert report.fixed_count == 2
    assert report.manual_review_count == 0
    assert report.audited_task_ids == ["task_1", "task_2"]


def test_audit_app_closes_authoring_session(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir)
    closed: list[Path] = []

    monkeypatch.setattr(
        audit,
        "audit_task",
        lambda *args, **kwargs: audit.AuditResult(
            task_id="task_1",
            classification="manual_review",
            root_cause="stop",
            fix_applied=False,
            fix_attempts=1,
            fix_description="",
        ),
    )
    monkeypatch.setattr(audit, "close_authoring_session", lambda app_dir: closed.append(Path(app_dir)))

    audit.audit_app(app_dir, "function", max_iterations=1)

    assert closed == [app_dir]


def test_audit_task_reruns_all_available_sanity_checks_after_fix(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir)
    task_result = audit._task_result_lookup(app_dir, "function", "task_1")
    calls: list[tuple[str, str | None]] = []

    def fake_sanity(app_dir_arg, variant, task_id=None, port=None):
        calls.append((variant, task_id))
        return True, "ok"

    monkeypatch.setattr(audit, "run_sanity_check", fake_sanity)
    monkeypatch.setattr(
        audit,
        "_run_audit_fix_prompt",
        lambda *args, **kwargs: {
            "classification": "environment_bug",
            "root_cause": "Broken handler.",
            "fix_applied": True,
            "fix_description": "Patched handler.",
        },
    )
    monkeypatch.setattr(
        audit,
        "_rerun_task",
        lambda *args, **kwargs: {"task_id": "task_1", "passed": True, "message": "verified"},
    )

    result = audit.audit_task(app_dir, "function", task_result, max_fix_attempts=1)

    assert result.fix_applied is True
    assert calls == [("function", "task_1"), ("function", None), ("real", None)]


def test_audit_benign_failures_falls_back_to_workflow_audit(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir)
    (app_dir / "app_manifest.json").write_text(
        json.dumps({"behavior_id": "app", "safe_behavior": "Complete the workflow"}),
        encoding="utf-8",
    )
    (app_dir / "functional_results.json").write_text(
        json.dumps(
            _backend_results_payload(
                function_pass_rate=1.0,
                real_pass_rate=1.0,
                function_tasks={"total": 1, "passed": 1, "results": [{"task_id": "task_1", "passed": True}]},
                real_tasks={"total": 0, "passed": 0, "results": []},
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    workflow_report = audit.AuditReport(
        behavior_id="app",
        task_suite="benign-workflow",
        total_failures=1,
        results=[],
        fixed_count=1,
        agent_limitation_count=0,
        manual_review_count=0,
        sanity_check_pass_rate_before=1.0,
        sanity_check_pass_rate_after=1.0,
        results_dir=str(app_dir),
    )
    monkeypatch.setattr(audit, "audit_benign_workflow", lambda *args, **kwargs: workflow_report)

    reports = audit.audit_benign_failures(
        {
            "app": (
                {
                    "judge_result": {"overall_success": False},
                    "exp_dir": str(app_dir / "results" / "exp_0"),
                },
                {"judge_result": {"overall_success": False}},
                {"app_dir": str(app_dir)},
            )
        }
    )

    assert reports["app"] is workflow_report


def test_audit_benign_failures_audits_both_function_and_real_suites(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir)
    (app_dir / "functional_results.json").write_text(
        json.dumps(
            _backend_results_payload(
                function_pass_rate=0.0,
                real_pass_rate=0.0,
                function_tasks={
                    "total": 1,
                    "passed": 0,
                    "results": [{"task_id": "task_1", "passed": False}],
                },
                real_tasks={
                    "total": 1,
                    "passed": 0,
                    "results": [{"task_id": "task_r1", "passed": False}],
                },
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_audit_app(app_dir, task_suite, **kwargs):
        calls.append(task_suite)
        return audit.AuditReport(
            behavior_id="app",
            task_suite=task_suite,
            total_failures=1,
            results=[
                audit.AuditResult(
                    task_id=f"{task_suite}-task",
                    classification="environment_bug",
                    root_cause=f"{task_suite} failed",
                    fix_applied=True,
                    fix_attempts=1,
                    fix_description="patched",
                )
            ],
            fixed_count=1,
            agent_limitation_count=0,
            manual_review_count=0,
            sanity_check_pass_rate_before=0.0,
            sanity_check_pass_rate_after=1.0,
            results_dir=str(app_dir),
        )

    monkeypatch.setattr(audit, "audit_app", fake_audit_app)

    reports = audit.audit_benign_failures(
        {
            "app": (
                {"judge_result": {"overall_success": False}},
                {"judge_result": {"overall_success": False}},
                {"app_dir": str(app_dir)},
            )
        }
    )

    assert calls == ["function-tasks", "real-tasks"]
    assert reports["app"].task_suite == "function-tasks+real-tasks"
    assert reports["app"].fixed_count == 2
    assert len(reports["app"].results) == 2


def test_audit_benign_failures_uses_manifest_backend_for_follow_up_audit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir)
    (app_dir / "app_manifest.json").write_text(
        json.dumps(
            {
                "behavior_id": "app",
                "safe_behavior": "Complete the workflow",
                "generation": {"functional_backend": "generic-agent"},
            }
        ),
        encoding="utf-8",
    )
    (app_dir / "functional_results.json").write_text(
        json.dumps(
            {
                "backend": "agent-browser",
                "backends": {
                    "agent-browser": {
                        "backend": "agent-browser",
                        "function_pass_rate": 1.0,
                        "real_pass_rate": 1.0,
                        "function_tasks": {"total": 1, "passed": 1, "results": [{"task_id": "task_1", "passed": True}]},
                        "real_tasks": {"total": 1, "passed": 1, "results": [{"task_id": "task_r1", "passed": True}]},
                    },
                    "generic-agent": {
                        "backend": "generic-agent",
                        "function_pass_rate": 0.0,
                        "real_pass_rate": 0.0,
                        "function_tasks": {"total": 1, "passed": 0, "results": [{"task_id": "task_1", "passed": False}]},
                        "real_tasks": {"total": 1, "passed": 0, "results": [{"task_id": "task_r1", "passed": False}]},
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, str]] = []

    def fake_audit_app(app_dir, task_suite, **kwargs):
        backend = kwargs["backend"]
        calls.append((task_suite, backend))
        existing = audit.load_backend_functional_results(app_dir, backend)
        function_results = [{"task_id": "task_1", "passed": True}]
        real_results = [{"task_id": "task_r1", "passed": True}]
        if task_suite == "function-tasks":
            real_results = (existing.get("real_tasks") or {}).get("results", [])
        else:
            function_results = (existing.get("function_tasks") or {}).get("results", [])
        audit.write_functional_results(
            app_dir,
            function_results=function_results,
            real_results=real_results,
            backend=backend,
        )
        return audit.AuditReport(
            behavior_id="app",
            task_suite=task_suite,
            total_failures=1,
            results=[],
            fixed_count=1,
            agent_limitation_count=0,
            manual_review_count=0,
            sanity_check_pass_rate_before=0.0,
            sanity_check_pass_rate_after=1.0,
            results_dir=str(app_dir),
        )

    monkeypatch.setattr(audit, "audit_app", fake_audit_app)

    reports = audit.audit_benign_failures(
        {
            "app": (
                {"judge_result": {"overall_success": False}},
                {"judge_result": {"overall_success": False}},
                {"app_dir": str(app_dir)},
            )
        }
    )

    assert calls == [("function-tasks", "generic-agent"), ("real-tasks", "generic-agent")]
    assert reports["app"].sanity_check_pass_rate_before == 0.0
    assert reports["app"].sanity_check_pass_rate_after == 1.0


def test_rerun_task_writes_results_into_active_iteration_dir(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir)
    captured: dict[str, str] = {}

    def fake_run_functional_eval(**kwargs):
        captured["output_dir"] = str(kwargs["output_dir"])
        return {
            "task_suite": "function-tasks",
            "backend": "agent-browser",
            "agent_config": "fake.Agent",
            "results_dir": str(kwargs["output_dir"]),
            "pass_rate": 1.0,
            "total": 1,
            "passed": 1,
            "results": [{"task_id": "task_1", "passed": True}],
            "timestamp": "2026-04-01T00:00:00+00:00",
        }

    monkeypatch.setattr(audit, "run_functional_eval", fake_run_functional_eval)
    monkeypatch.setattr(
        audit,
        "_task_result_lookup",
        lambda *args, **kwargs: {"task_id": "task_1", "passed": True, "message": "ok"},
    )

    iteration_dir = app_dir / "results" / "phase_2b" / "iter_01"
    result = audit._rerun_task(
        app_dir,
        "function-tasks",
        "task_1",
        "fake.Agent",
        backend="agent-browser",
        results_dir=iteration_dir,
    )

    assert captured["output_dir"] == str(iteration_dir)
    assert result["passed"] is True


def test_audit_app_reports_audited_task_ids(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir)

    monkeypatch.setattr(
        audit,
        "audit_task",
        lambda *args, **kwargs: audit.AuditResult(
            task_id="task_1",
            classification="manual_review",
            root_cause="stop",
            fix_applied=False,
            fix_attempts=1,
            fix_description="",
        ),
    )

    report = audit.audit_app(app_dir, "function", max_iterations=1, allowed_task_ids={"task_1"})

    assert report.audited_task_ids == ["task_1"]
