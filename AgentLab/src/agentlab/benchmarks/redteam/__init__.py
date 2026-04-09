"""Redteam benchmark for security testing with dynamic synthetic pages."""

from agentlab.benchmarks.redteam.config import Page, RedteamEnvArgs
from agentlab.benchmarks.redteam.env import RedteamEnv
from agentlab.benchmarks.redteam.benchmark import RedteamBenchmark
from agentlab.benchmarks.redteam.analysis import analyze_results
from agentlab.benchmarks.redteam.controller import (
    ControllerConfig,
    ControllerResult,
    run_behavior,
    resume_behavior,
    rerun_behavior_from_phase,
)
from agentlab.benchmarks.redteam.app_pipeline import (
    AppGenerationResult,
    generate_app,
    generate_app_scaffold,
    generate_task_suite,
    generate_variants,
    load_prompt_template,
    run_claude_code,
    run_task_validation_loop,
    validate_app,
)
from agentlab.benchmarks.redteam.data_validator import (
    CheckResult,
    DataValidationResult,
    DataValidator,
)
from agentlab.benchmarks.redteam.audit import (
    AuditReport,
    AuditResult,
    audit_app,
    audit_benign_failures,
    audit_task,
)

__all__ = [
    "Page",
    "RedteamEnvArgs",
    "RedteamEnv",
    "RedteamBenchmark",
    "analyze_results",
    "ControllerConfig",
    "ControllerResult",
    "run_behavior",
    "resume_behavior",
    "rerun_behavior_from_phase",
    # App generation pipeline
    "AppGenerationResult",
    "generate_app",
    "generate_app_scaffold",
    "generate_task_suite",
    "generate_variants",
    "validate_app",
    "run_claude_code",
    "run_task_validation_loop",
    "load_prompt_template",
    "CheckResult",
    "DataValidationResult",
    "DataValidator",
    "AuditResult",
    "AuditReport",
    "audit_task",
    "audit_app",
    "audit_benign_failures",
]
