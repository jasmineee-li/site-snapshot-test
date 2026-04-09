"""Controller-visible audit-step wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentlab.benchmarks.redteam.audit import audit_app


@dataclass
class AuditStepOutcome:
    phase: str
    suite: str
    iteration: int
    prompt_name: str
    results_dir: str
    audited_task_ids: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    fix_applied: bool = False
    sanity_reruns_passed: bool = True
    targeted_reruns_passed: bool = True
    updated_pass_rate: float | None = None
    stop_classification: str = "no_change"
    audit_summary_path: str = ""


def run_audit_step(
    app_dir: str | Path,
    *,
    phase: str,
    suite: str,
    iteration: int,
    results_dir: str | Path,
    backend: str,
    agent_config: str | None,
    allowed_task_ids: set[str] | None = None,
) -> AuditStepOutcome:
    task_suite = "function-tasks" if suite == "function" else "real-tasks"
    prompt_name = (
        "repair-from-function-audit.md"
        if suite == "function"
        else "repair-from-real-audit.md"
    )
    report = audit_app(
        app_dir=app_dir,
        task_suite=task_suite,
        agent_config=agent_config,
        backend=backend,
        max_fix_attempts=3,
        max_iterations=1,
        results_dir=results_dir,
        allowed_task_ids=allowed_task_ids,
    )
    audit_summary_path = str(Path(report.results_dir) / "audit_summary.md")
    fix_applied = report.fixed_count > 0
    stop_classification = "fixed" if fix_applied else "no_change"
    return AuditStepOutcome(
        phase=phase,
        suite=suite,
        iteration=iteration,
        prompt_name=prompt_name,
        results_dir=str(results_dir),
        audited_task_ids=list(report.audited_task_ids),
        fix_applied=fix_applied,
        updated_pass_rate=report.sanity_check_pass_rate_after,
        stop_classification=stop_classification,
        audit_summary_path=audit_summary_path,
    )
