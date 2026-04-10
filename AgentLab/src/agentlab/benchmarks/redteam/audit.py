"""Auditing loop for functional correctness failures."""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentlab.benchmarks.redteam.execution import close_authoring_session
from agentlab.benchmarks.redteam.app_artifacts import load_behavior_contract
from agentlab.benchmarks.redteam.app_pipeline import (
    ensure_trusted_server_template,
    load_prompt_template,
)
from agentlab.benchmarks.redteam.claude_code import run_claude_code
from agentlab.benchmarks.redteam.eval_harness import (
    DEFAULT_AGENT_CONFIG,
    DEFAULT_FUNCTIONAL_BACKEND,
    load_backend_functional_results,
    load_functional_results,
    load_app_manifest,
    normalize_functional_backend,
    normalize_task_suite,
    run_functional_eval,
    run_sanity_check,
    suite_variant_name,
    write_functional_results,
)

logger = logging.getLogger(__name__)

FIXABLE_CLASSIFICATIONS = {"verifier_bug", "environment_bug", "ambiguous_instruction"}
TERMINAL_CLASSIFICATIONS = {"agent_limitation", "manual_review"}
TASK_PROBLEM_CLASSIFICATIONS = {"impossible_task"}
ALL_CLASSIFICATIONS = FIXABLE_CLASSIFICATIONS | TERMINAL_CLASSIFICATIONS | TASK_PROBLEM_CLASSIFICATIONS


@dataclass
class AuditResult:
    task_id: str
    classification: str
    root_cause: str
    fix_applied: bool
    fix_attempts: int
    fix_description: str


@dataclass
class AuditReport:
    behavior_id: str
    task_suite: str
    total_failures: int
    results: list[AuditResult]
    fixed_count: int
    agent_limitation_count: int
    manual_review_count: int
    sanity_check_pass_rate_before: float
    sanity_check_pass_rate_after: float
    results_dir: str = ""
    audited_task_ids: list[str] = field(default_factory=list)


def _resolved_app_id(app_dir: str | Path) -> str:
    manifest = load_app_manifest(app_dir)
    return str(manifest.get("app_id") or Path(app_dir).name)


def _resolved_behavior_instruction(
    app_dir: str | Path,
    *,
    behavior_id: str | None = None,
) -> str:
    app_dir = Path(app_dir)
    if behavior_id:
        behavior_contract = load_behavior_contract(app_dir, behavior_id)
        safe_behavior = str(behavior_contract.get("safe_behavior") or "").strip()
        if safe_behavior:
            return safe_behavior
    manifest = load_app_manifest(app_dir)
    return (
        str(manifest.get("app_description") or "").strip()
        or str(manifest.get("docs_path") or "").strip()
        or _resolved_app_id(app_dir)
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    for candidate in [cleaned]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        raise ValueError("No JSON object found in Claude Code output")
    return json.loads(match.group(0))


def _load_suite_results(
    app_dir: str | Path,
    task_suite: str,
    *,
    backend: str = DEFAULT_FUNCTIONAL_BACKEND,
) -> list[dict[str, Any]]:
    results = load_backend_functional_results(app_dir, backend)
    key = "function_tasks" if normalize_task_suite(task_suite) == "function-tasks" else "real_tasks"
    return (results.get(key) or {}).get("results", [])


def _suite_pass_rate(
    app_dir: str | Path,
    task_suite: str,
    *,
    backend: str = DEFAULT_FUNCTIONAL_BACKEND,
) -> float:
    results = _load_suite_results(app_dir, task_suite, backend=backend)
    if not results:
        return 0.0
    passed = sum(1 for result in results if result.get("passed"))
    return passed / len(results)


def _task_result_lookup(
    app_dir: str | Path,
    task_suite: str,
    task_id: str,
    *,
    backend: str = DEFAULT_FUNCTIONAL_BACKEND,
) -> dict[str, Any] | None:
    for result in _load_suite_results(app_dir, task_suite, backend=backend):
        if result.get("task_id") == task_id:
            return result
    return None


def _run_audit_fix_prompt(
    app_dir: str | Path,
    task_suite: str,
    task_result: dict[str, Any],
    diagnosis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    suite_variant = suite_variant_name(task_suite)
    prompt_name = (
        "repair-from-function-audit.md"
        if suite_variant == "function"
        else "repair-from-real-audit.md"
    )
    prompt = load_prompt_template(
        prompt_name,
        task_id=task_result["task_id"],
        instruction=task_result["instruction"],
        task_result_json=json.dumps(task_result, indent=2, ensure_ascii=False),
        diagnosis_json=json.dumps(diagnosis or {}, indent=2, ensure_ascii=False),
        app_dir=str(app_dir),
    )
    rc, stdout, stderr = run_claude_code(
        prompt=prompt,
        working_dir=app_dir,
        timeout=1800,
    )
    server_error = ensure_trusted_server_template(app_dir)
    if server_error:
        raise RuntimeError(server_error)
    if rc != 0:
        raise RuntimeError(
            f"Claude Code audit prompt failed (rc={rc}): {(stderr or stdout)[-500:]}"
        )
    return _extract_json_object(stdout or stderr or "{}")


def _run_benign_workflow_fix_prompt(
    app_dir: str | Path,
    workflow_result: dict[str, Any],
    diagnosis: dict[str, Any] | None = None,
    *,
    behavior_id: str | None = None,
) -> dict[str, Any]:
    prompt = load_prompt_template(
        "audit-benign-workflow.md",
        app_id=_resolved_app_id(app_dir),
        behavior_id=behavior_id or "",
        instruction=workflow_result["instruction"],
        workflow_result_json=json.dumps(workflow_result, indent=2, ensure_ascii=False),
        diagnosis_json=json.dumps(diagnosis or {}, indent=2, ensure_ascii=False),
        app_dir=str(app_dir),
    )
    rc, stdout, stderr = run_claude_code(
        prompt=prompt,
        working_dir=app_dir,
        timeout=1800,
    )
    server_error = ensure_trusted_server_template(app_dir)
    if server_error:
        raise RuntimeError(server_error)
    if rc != 0:
        raise RuntimeError(
            f"Claude Code benign workflow audit prompt failed (rc={rc}): {(stderr or stdout)[-500:]}"
        )
    return _extract_json_object(stdout or stderr or "{}")


def _available_sanity_variants(app_dir: str | Path) -> list[str]:
    app_dir = Path(app_dir)
    variants: list[str] = []
    seen_paths: set[Path] = set()
    for variant in ("function", "real"):
        script = app_dir / f"sanity_check_{variant}.py"
        if not script.exists():
            script = app_dir / "sanity_check.py"
        if script.exists() and script not in seen_paths:
            variants.append(variant)
            seen_paths.add(script)
    return variants


def _run_all_sanity_checks(app_dir: str | Path) -> tuple[bool, str]:
    outputs: list[str] = []
    variants = _available_sanity_variants(app_dir)
    if not variants:
        return False, "No sanity check scripts found."

    for variant in variants:
        passed, output = run_sanity_check(app_dir, variant)
        outputs.append(f"[{variant}] {'PASS' if passed else 'FAIL'}\n{output.strip()}")
        if not passed:
            return False, "\n\n".join(outputs)
    return True, "\n\n".join(outputs)


def _resolve_benign_audit_backend(app_dir: str | Path) -> str:
    app_dir = Path(app_dir)
    try:
        manifest = load_app_manifest(app_dir)
    except (OSError, json.JSONDecodeError):
        manifest = {}

    manifest_backend = ((manifest.get("generation") or {}).get("functional_backend"))
    if manifest_backend:
        return normalize_functional_backend(manifest_backend)

    results = load_functional_results(app_dir)
    stored_backend = results.get("backend")
    if stored_backend:
        return normalize_functional_backend(stored_backend)

    backends = results.get("backends")
    if isinstance(backends, dict) and len(backends) == 1:
        only_backend = next(iter(backends))
        return normalize_functional_backend(only_backend)

    return DEFAULT_FUNCTIONAL_BACKEND


def _combined_functional_pass_rate(
    app_dir: str | Path,
    *,
    backend: str = DEFAULT_FUNCTIONAL_BACKEND,
) -> float:
    results = load_backend_functional_results(app_dir, backend)
    pass_rates: list[float] = []
    for key, rate_key in (
        ("function_tasks", "function_pass_rate"),
        ("real_tasks", "real_pass_rate"),
    ):
        suite = results.get(key) or {}
        if suite.get("total", 0) > 0:
            pass_rates.append(float(results.get(rate_key, 0.0)))
    if not pass_rates:
        return 0.0
    return sum(pass_rates) / len(pass_rates)


def _precheck_diagnosis(task_result: dict[str, Any]) -> dict[str, Any] | None:
    err_msg = task_result.get("attempts", [{}])[-1].get("err_msg") or task_result.get("err_msg")
    final_state_path = task_result.get("final_state_path", "")
    if final_state_path and not Path(final_state_path).exists():
        return {
            "classification": "environment_bug",
            "root_cause": "Final state snapshot is missing; the browser run did not persist verifiable state.",
            "fix_applied": False,
            "fix_description": "",
        }
    if final_state_path:
        try:
            snapshot = json.loads(Path(final_state_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "classification": "environment_bug",
                "root_cause": f"Final state snapshot is unreadable: {exc}",
                "fix_applied": False,
                "fix_description": "",
            }
        if not isinstance(snapshot, dict):
            return {
                "classification": "environment_bug",
                "root_cause": "Final state snapshot is not a JSON object.",
                "fix_applied": False,
                "fix_description": "",
            }
        if not snapshot:
            return {
                "classification": "environment_bug",
                "root_cause": "Final state snapshot is empty; the app did not persist usable state.",
                "fix_applied": False,
                "fix_description": "",
            }
        if snapshot.get("error"):
            return {
                "classification": "environment_bug",
                "root_cause": f"Final state snapshot captured an app error: {snapshot['error']}",
                "fix_applied": False,
                "fix_description": "",
            }
    if err_msg and any(token in err_msg for token in ["ReferenceError", "TypeError", "SyntaxError"]):
        return {
            "classification": "environment_bug",
            "root_cause": err_msg,
            "fix_applied": False,
            "fix_description": "",
        }
    return None


def _remove_task_from_suite(
    app_dir: str | Path,
    task_suite: str,
    task_id: str,
    *,
    backend: str = DEFAULT_FUNCTIONAL_BACKEND,
) -> str:
    task_suite = normalize_task_suite(task_suite)
    path = Path(app_dir) / f"{task_suite}.json"
    tasks = json.loads(path.read_text(encoding="utf-8"))
    kept = [task for task in tasks if task.get("id") != task_id]
    if len(kept) == len(tasks):
        return ""
    path.write_text(json.dumps(kept, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    existing = load_backend_functional_results(app_dir, backend)
    key = "function_tasks" if task_suite == "function-tasks" else "real_tasks"
    other_key = "real_tasks" if key == "function_tasks" else "function_tasks"
    suite_results = [
        result for result in (existing.get(key) or {}).get("results", [])
        if result.get("task_id") != task_id
    ]
    other_results = (existing.get(other_key) or {}).get("results", [])
    if key == "function_tasks":
        write_functional_results(app_dir, function_results=suite_results, real_results=other_results, backend=backend)
    else:
        write_functional_results(app_dir, function_results=other_results, real_results=suite_results, backend=backend)
    return f"Removed {task_id} from {task_suite}.json"


def _rerun_task(
    app_dir: str | Path,
    task_suite: str,
    task_id: str,
    agent_config: str,
    backend: str = DEFAULT_FUNCTIONAL_BACKEND,
    results_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    run_functional_eval(
        app_dir=app_dir,
        task_suite=task_suite,
        agent_config=agent_config,
        backend=backend,
        workers=1,
        repetitions=1,
        task_id=task_id,
        output_dir=results_dir,
    )
    return _task_result_lookup(app_dir, task_suite, task_id)


def audit_task(
    app_dir: str | Path,
    task_suite: str,
    task_result: dict[str, Any],
    agent_config: str = DEFAULT_AGENT_CONFIG,
    backend: str = DEFAULT_FUNCTIONAL_BACKEND,
    max_fix_attempts: int = 3,
) -> AuditResult:
    """Audit a single failed task and attempt automated repair when appropriate."""
    app_dir = Path(app_dir)
    task_suite = normalize_task_suite(task_suite)
    suite_variant = suite_variant_name(task_suite)

    for attempt in range(1, max_fix_attempts + 1):
        passed, output = run_sanity_check(app_dir, suite_variant, task_id=task_result["task_id"])
        needs_claude_fix = False
        if not passed:
            diagnosis = {
                "classification": "verifier_bug",
                "root_cause": output.strip()[-2000:],
                "fix_applied": False,
                "fix_description": "",
            }
            needs_claude_fix = True
        else:
            diagnosis = _precheck_diagnosis(task_result)
            if diagnosis is None:
                diagnosis = _run_audit_fix_prompt(app_dir, task_suite, task_result)
            else:
                needs_claude_fix = diagnosis.get("classification") in FIXABLE_CLASSIFICATIONS

        classification = diagnosis.get("classification", "manual_review")
        if classification not in ALL_CLASSIFICATIONS:
            classification = "manual_review"

        root_cause = diagnosis.get("root_cause", "").strip() or "No root cause provided."
        fix_description = diagnosis.get("fix_description", "").strip()

        if needs_claude_fix and classification in FIXABLE_CLASSIFICATIONS:
            diagnosis = _run_audit_fix_prompt(app_dir, task_suite, task_result, diagnosis=diagnosis)
            classification = diagnosis.get("classification", classification)
            if classification not in ALL_CLASSIFICATIONS:
                classification = "manual_review"
            root_cause = diagnosis.get("root_cause", root_cause).strip() or root_cause
            fix_description = diagnosis.get("fix_description", fix_description).strip()

        if classification == "agent_limitation":
            return AuditResult(
                task_id=task_result["task_id"],
                classification=classification,
                root_cause=root_cause,
                fix_applied=False,
                fix_attempts=attempt - 1,
                fix_description=fix_description,
            )

        if classification == "impossible_task":
            removal_description = _remove_task_from_suite(
                app_dir,
                task_suite,
                task_result["task_id"],
                backend=backend,
            )
            sanity_ok, sanity_output = _run_all_sanity_checks(app_dir)
            if sanity_ok:
                return AuditResult(
                    task_id=task_result["task_id"],
                    classification=classification,
                    root_cause=root_cause,
                    fix_applied=True,
                    fix_attempts=attempt,
                    fix_description=fix_description or removal_description or "Removed impossible task from suite.",
                )
            fix_description = sanity_output.strip()[-1000:] or removal_description

        if classification in FIXABLE_CLASSIFICATIONS:
            sanity_ok, sanity_output = _run_all_sanity_checks(app_dir)
            if not sanity_ok:
                root_cause = root_cause or sanity_output.strip()[-2000:]
                fix_description = fix_description or "Post-fix sanity check failed."
                continue

            updated_task = _rerun_task(
                app_dir,
                task_suite,
                task_result["task_id"],
                agent_config,
                backend=backend,
                results_dir=task_result.get("results_dir"),
            )
            if updated_task and updated_task.get("passed"):
                return AuditResult(
                    task_id=task_result["task_id"],
                    classification=classification,
                    root_cause=root_cause,
                    fix_applied=True,
                    fix_attempts=attempt,
                    fix_description=fix_description or updated_task.get("message", ""),
                )
            fix_description = fix_description or (updated_task or {}).get("message", "")
            task_result = updated_task or task_result
            continue

        return AuditResult(
            task_id=task_result["task_id"],
            classification="manual_review",
            root_cause=root_cause,
            fix_applied=False,
            fix_attempts=attempt,
            fix_description=fix_description,
        )

    return AuditResult(
        task_id=task_result["task_id"],
        classification="manual_review",
        root_cause="Maximum fix attempts exhausted.",
        fix_applied=False,
        fix_attempts=max_fix_attempts,
        fix_description="Automated fixes did not produce a passing rerun.",
    )


def _build_benign_workflow_result(
    app_dir: str | Path,
    benign_result: dict[str, Any],
    *,
    behavior_id: str | None = None,
) -> dict[str, Any]:
    summary_info = benign_result.get("summary_info") or {}
    attempts = []
    err_msg = summary_info.get("err_msg")
    if err_msg is not None:
        attempts.append({"err_msg": err_msg})
    return {
        "task_id": "__benign_workflow__",
        "suite": "benign-workflow",
        "instruction": _resolved_behavior_instruction(app_dir, behavior_id=behavior_id),
        "verify": "",
        "passed": False,
        "message": (benign_result.get("judge_result") or {}).get("message", ""),
        "exp_dir": benign_result.get("exp_dir", ""),
        "final_state_path": benign_result.get("final_state_path", ""),
        "server_events_path": benign_result.get("server_events_path", ""),
        "results_dir": str(Path(benign_result.get("exp_dir", "")).parent) if benign_result.get("exp_dir") else str(app_dir),
        "attempts": attempts,
        "audit": {},
        "summary_info": summary_info,
        "judge_result": benign_result.get("judge_result") or {},
    }


def audit_benign_workflow(
    app_dir: str | Path,
    benign_result: dict[str, Any],
    max_fix_attempts: int = 3,
    *,
    behavior_id: str | None = None,
) -> AuditReport:
    """Audit the failed top-level benign workflow when no failing task artifacts exist."""
    app_dir = Path(app_dir)
    workflow_result = _build_benign_workflow_result(app_dir, benign_result, behavior_id=behavior_id)
    before_pass_rate = _combined_functional_pass_rate(app_dir)

    try:
        for attempt in range(1, max_fix_attempts + 1):
            diagnosis = _precheck_diagnosis(workflow_result)
            needs_claude_fix = False
            if diagnosis is None:
                diagnosis = _run_benign_workflow_fix_prompt(
                    app_dir,
                    workflow_result,
                    behavior_id=behavior_id,
                )
            else:
                needs_claude_fix = diagnosis.get("classification") in FIXABLE_CLASSIFICATIONS

            classification = diagnosis.get("classification", "manual_review")
            if classification not in ALL_CLASSIFICATIONS:
                classification = "manual_review"

            root_cause = diagnosis.get("root_cause", "").strip() or "No root cause provided."
            fix_description = diagnosis.get("fix_description", "").strip()

            if needs_claude_fix and classification in FIXABLE_CLASSIFICATIONS:
                diagnosis = _run_benign_workflow_fix_prompt(
                    app_dir,
                    workflow_result,
                    diagnosis=diagnosis,
                    behavior_id=behavior_id,
                )
                classification = diagnosis.get("classification", classification)
                if classification not in ALL_CLASSIFICATIONS:
                    classification = "manual_review"
                root_cause = diagnosis.get("root_cause", root_cause).strip() or root_cause
                fix_description = diagnosis.get("fix_description", fix_description).strip()

            if classification in {"agent_limitation", "impossible_task"}:
                audit_result = AuditResult(
                    task_id=workflow_result["task_id"],
                    classification=classification,
                    root_cause=root_cause,
                    fix_applied=False,
                    fix_attempts=0 if classification == "agent_limitation" else attempt,
                    fix_description=fix_description,
                )
                break

            if classification in FIXABLE_CLASSIFICATIONS:
                sanity_ok, sanity_output = _run_all_sanity_checks(app_dir)
                if sanity_ok:
                    audit_result = AuditResult(
                        task_id=workflow_result["task_id"],
                        classification=classification,
                        root_cause=root_cause,
                        fix_applied=True,
                        fix_attempts=attempt,
                        fix_description=fix_description or "Applied workflow repair and all sanity checks passed.",
                    )
                    break
                fix_description = fix_description or sanity_output.strip()[-2000:]
                continue

            audit_result = AuditResult(
                task_id=workflow_result["task_id"],
                classification="manual_review",
                root_cause=root_cause,
                fix_applied=False,
                fix_attempts=attempt,
                fix_description=fix_description,
            )
            break
        else:
            audit_result = AuditResult(
                task_id=workflow_result["task_id"],
                classification="manual_review",
                root_cause="Maximum fix attempts exhausted.",
                fix_applied=False,
                fix_attempts=max_fix_attempts,
                fix_description="Automated workflow fixes did not produce passing sanity checks.",
            )

        after_pass_rate = _combined_functional_pass_rate(app_dir)
        report = AuditReport(
            behavior_id=behavior_id or _resolved_app_id(app_dir),
            task_suite="benign-workflow",
            total_failures=1,
            results=[audit_result],
            fixed_count=1 if audit_result.fix_applied else 0,
            agent_limitation_count=1 if audit_result.classification == "agent_limitation" else 0,
            manual_review_count=1 if audit_result.classification == "manual_review" else 0,
            sanity_check_pass_rate_before=before_pass_rate,
            sanity_check_pass_rate_after=after_pass_rate,
            results_dir=workflow_result["results_dir"],
        )
        _write_audit_outputs(app_dir, report)
        return report
    finally:
        close_authoring_session(app_dir)


def _write_audit_outputs(
    app_dir: str | Path,
    report: AuditReport,
) -> None:
    results_dir = Path(report.results_dir) if report.results_dir else Path(app_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "audit_report.json").write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        f"# Audit Summary: {report.behavior_id} ({report.task_suite})",
        "",
        f"- Total failures audited: {report.total_failures}",
        f"- Fixed: {report.fixed_count}",
        f"- Agent limitations: {report.agent_limitation_count}",
        f"- Manual review: {report.manual_review_count}",
        f"- Pass rate before: {report.sanity_check_pass_rate_before:.2%}",
        f"- Pass rate after: {report.sanity_check_pass_rate_after:.2%}",
        "",
    ]
    for result in report.results:
        lines.append(
            f"- `{result.task_id}`: `{result.classification}` — {result.root_cause} "
            f"(fix_applied={result.fix_applied}, attempts={result.fix_attempts})"
        )
        if result.fix_description:
            lines.append(f"  Fix: {result.fix_description}")
    (results_dir / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_app(
    app_dir: str | Path,
    task_suite: str,
    eval_results: dict[str, Any] | None = None,
    agent_config: str = DEFAULT_AGENT_CONFIG,
    backend: str = DEFAULT_FUNCTIONAL_BACKEND,
    max_fix_attempts: int = 3,
    max_iterations: int = 3,
    results_dir: str | Path | None = None,
    allowed_task_ids: set[str] | None = None,
) -> AuditReport:
    """Audit all failures for an app and iterate until failures stabilize."""
    app_dir = Path(app_dir)
    task_suite = normalize_task_suite(task_suite)
    suite_key = "function_tasks" if task_suite == "function-tasks" else "real_tasks"
    behavior_id = _resolved_app_id(app_dir)

    try:
        current_results = eval_results or load_backend_functional_results(app_dir, backend)
        suite_results = (current_results.get(suite_key) or {}).get("results", [])
        if allowed_task_ids is not None:
            suite_results = [
                result for result in suite_results if result.get("task_id") in allowed_task_ids
            ]
        before_pass_rate = _suite_pass_rate(app_dir, task_suite, backend=backend)
        reports_by_task: dict[str, AuditResult] = {}
        audited_task_ids: set[str] = set()

        for _ in range(max_iterations):
            failed = [
                result
                for result in _load_suite_results(app_dir, task_suite, backend=backend)
                if not result.get("passed")
                and (allowed_task_ids is None or result.get("task_id") in allowed_task_ids)
            ]
            if not failed:
                break

            made_fix = False
            for task_result in failed:
                task_id = str(task_result.get("task_id") or "")
                if task_id:
                    audited_task_ids.add(task_id)
                audit_result = audit_task(
                    app_dir=app_dir,
                    task_suite=task_suite,
                    task_result=task_result,
                    agent_config=agent_config,
                    backend=backend,
                    max_fix_attempts=max_fix_attempts,
                )
                reports_by_task[audit_result.task_id] = audit_result
                if audit_result.fix_applied:
                    made_fix = True

            if not made_fix:
                break

        after_pass_rate = _suite_pass_rate(app_dir, task_suite, backend=backend)
        reports = list(reports_by_task.values())
        results_dir_value = str(results_dir or next(
            (
                result.get("results_dir")
                for result in _load_suite_results(app_dir, task_suite, backend=backend)
                if result.get("results_dir")
                and (allowed_task_ids is None or result.get("task_id") in allowed_task_ids)
            ),
            app_dir,
        ))
        report = AuditReport(
            behavior_id=behavior_id,
            task_suite=task_suite,
            total_failures=len([result for result in suite_results if not result.get("passed")]),
            results=reports,
            fixed_count=sum(1 for result in reports if result.fix_applied),
            agent_limitation_count=sum(1 for result in reports if result.classification == "agent_limitation"),
            manual_review_count=sum(1 for result in reports if result.classification == "manual_review"),
            sanity_check_pass_rate_before=before_pass_rate,
            sanity_check_pass_rate_after=after_pass_rate,
            results_dir=results_dir_value,
            audited_task_ids=sorted(audited_task_ids),
        )
        _write_audit_outputs(app_dir, report)
        return report
    finally:
        close_authoring_session(app_dir)


def audit_benign_failures(
    behavior_results: dict[str, tuple[dict, dict, dict]],
    max_fix_attempts: int = 3,
    agent_config: str = DEFAULT_AGENT_CONFIG,
) -> dict[str, AuditReport]:
    """Audit failed benign behaviors using the app's functional task suites."""
    reports: dict[str, AuditReport] = {}
    for behavior_id, (benign_result, _adversarial_result, config) in behavior_results.items():
        benign_judge = benign_result.get("judge_result", {}) or {}
        if benign_judge.get("overall_success", False):
            continue
        app_dir = config.get("app_dir")
        if not app_dir:
            continue
        backend = _resolve_benign_audit_backend(app_dir)
        functional_results = load_backend_functional_results(
            app_dir,
            backend,
        )
        function_results = (functional_results.get("function_tasks") or {}).get("results", [])
        real_results = (functional_results.get("real_tasks") or {}).get("results", [])
        suite_reports: list[AuditReport] = []
        combined_before = _combined_functional_pass_rate(app_dir, backend=backend)
        if any(not result.get("passed") for result in function_results):
            suite_reports.append(
                audit_app(
                    app_dir=app_dir,
                    task_suite="function-tasks",
                    agent_config=agent_config,
                    backend=backend,
                    max_fix_attempts=max_fix_attempts,
                )
            )
        if any(not result.get("passed") for result in real_results):
            suite_reports.append(
                audit_app(
                    app_dir=app_dir,
                    task_suite="real-tasks",
                    agent_config=agent_config,
                    backend=backend,
                    max_fix_attempts=max_fix_attempts,
                )
            )
        if suite_reports:
            if len(suite_reports) == 1:
                reports[behavior_id] = suite_reports[0]
            else:
                reports[behavior_id] = AuditReport(
                    behavior_id=behavior_id,
                    task_suite="function-tasks+real-tasks",
                    total_failures=sum(report.total_failures for report in suite_reports),
                    results=[result for report in suite_reports for result in report.results],
                    fixed_count=sum(report.fixed_count for report in suite_reports),
                    agent_limitation_count=sum(report.agent_limitation_count for report in suite_reports),
                    manual_review_count=sum(report.manual_review_count for report in suite_reports),
                    sanity_check_pass_rate_before=combined_before,
                    sanity_check_pass_rate_after=_combined_functional_pass_rate(app_dir, backend=backend),
                    results_dir=next(
                        (report.results_dir for report in suite_reports if report.results_dir),
                        str(app_dir),
                    ),
                )
            continue
        reports[behavior_id] = audit_benign_workflow(
            app_dir=app_dir,
            benign_result=benign_result,
            max_fix_attempts=max_fix_attempts,
            behavior_id=behavior_id,
        )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit failed redteam functional tasks")
    parser.add_argument("--app-dir", required=True, help="Root directory of the generated app")
    parser.add_argument(
        "--task-suite",
        required=True,
        choices=["function", "real", "function-tasks", "real-tasks"],
        help="Which task suite to audit",
    )
    parser.add_argument("--results-dir", default=None, help="Optional directory for audit outputs")
    parser.add_argument("--max-fix-attempts", type=int, default=3, help="Maximum fix attempts per task")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum full audit iterations")
    parser.add_argument("--task-id", default=None, help="Audit a single task ID")
    parser.add_argument(
        "--agent-config",
        default=DEFAULT_AGENT_CONFIG,
        help="Agent config used when rerunning failed tasks",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.task_id:
        task_result = _task_result_lookup(args.app_dir, args.task_suite, args.task_id)
        if task_result is None:
            raise SystemExit(f"Task {args.task_id} not found in functional results")
        try:
            result = audit_task(
                app_dir=args.app_dir,
                task_suite=args.task_suite,
                task_result=task_result,
                agent_config=args.agent_config,
                max_fix_attempts=args.max_fix_attempts,
            )
        finally:
            close_authoring_session(args.app_dir)
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
        return

    report = audit_app(
        app_dir=args.app_dir,
        task_suite=args.task_suite,
        agent_config=args.agent_config,
        max_fix_attempts=args.max_fix_attempts,
        max_iterations=args.max_iterations,
        results_dir=args.results_dir,
    )
    print(json.dumps(asdict(report), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
