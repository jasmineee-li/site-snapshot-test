"""Centralized configuration for the redteam generation pipeline.

All tunable constants live here with environment-variable overrides.
Call :func:`PipelineConfig.validate` at startup to fail fast on
missing prerequisites (API keys, CLI binaries, paths).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentlab.benchmarks.redteam.errors import (
    PipelineError,
    config_error,
    SEVERITY_WARNING,
    CATEGORY_CONFIG,
)


# ---------------------------------------------------------------------------
# Env-var helpers (same pattern as execution.py)
# ---------------------------------------------------------------------------


def _env(key: str, default: str) -> str:
    return os.getenv(key, default).strip()


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Canonical default values — importable by eval_loops, app_pipeline, etc.
# ---------------------------------------------------------------------------

DEFAULT_FUNCTIONAL_THRESHOLD = 0.80
DEFAULT_REAL_TASK_THRESHOLD = 1.0
DEFAULT_MAX_EVAL_ITERATIONS = 3
DEFAULT_HARDENING_ROUNDS = 3
DEFAULT_TASKS_PER_HARDENING_ROUND = 20
DEFAULT_AUDIT_EVERY = 0
DEFAULT_READINESS_BACKEND = "agent-browser"


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """All tunable parameters for app generation, with env-var overrides.

    Environment variables follow the ``AGENTLAB_REDTEAM_*`` convention
    established in ``execution.py``.
    """

    # -- Execution backend --------------------------------------------------
    execution_backend: str = field(
        default_factory=lambda: _env("AGENTLAB_REDTEAM_EXECUTION_BACKEND", "modal")
    )
    unsafe_local_execution: bool = field(
        default_factory=lambda: _env_bool("AGENTLAB_REDTEAM_UNSAFE_LOCAL_EXECUTION", False)
    )

    # -- Timeouts (seconds) -------------------------------------------------
    scaffold_timeout: int = field(
        default_factory=lambda: _env_int("AGENTLAB_REDTEAM_SCAFFOLD_TIMEOUT", 600)
    )
    task_gen_timeout: int = field(
        default_factory=lambda: _env_int("AGENTLAB_REDTEAM_TASK_GEN_TIMEOUT", 3600)
    )
    variant_timeout: int = field(
        default_factory=lambda: _env_int("AGENTLAB_REDTEAM_VARIANT_TIMEOUT", 1800)
    )
    validation_timeout: int = field(
        default_factory=lambda: _env_int("AGENTLAB_REDTEAM_VALIDATION_TIMEOUT", 60)
    )

    # -- Quality thresholds -------------------------------------------------
    functional_threshold: float = field(
        default_factory=lambda: _env_float("AGENTLAB_REDTEAM_FUNCTIONAL_THRESHOLD", DEFAULT_FUNCTIONAL_THRESHOLD)
    )
    real_task_threshold: float = field(
        default_factory=lambda: _env_float("AGENTLAB_REDTEAM_REAL_TASK_THRESHOLD", DEFAULT_REAL_TASK_THRESHOLD)
    )

    # -- Iteration limits ---------------------------------------------------
    max_eval_iterations: int = field(
        default_factory=lambda: _env_int("AGENTLAB_REDTEAM_MAX_EVAL_ITERATIONS", DEFAULT_MAX_EVAL_ITERATIONS)
    )
    hardening_rounds: int = field(
        default_factory=lambda: _env_int("AGENTLAB_REDTEAM_HARDENING_ROUNDS", DEFAULT_HARDENING_ROUNDS)
    )
    tasks_per_hardening_round: int = field(
        default_factory=lambda: _env_int("AGENTLAB_REDTEAM_TASKS_PER_ROUND", DEFAULT_TASKS_PER_HARDENING_ROUND)
    )
    audit_every: int = field(
        default_factory=lambda: _env_int("AGENTLAB_REDTEAM_AUDIT_EVERY", DEFAULT_AUDIT_EVERY)
    )

    # -- Functional evaluation backend --------------------------------------
    readiness_backend: str = field(
        default_factory=lambda: _env("AGENTLAB_REDTEAM_FUNCTIONAL_BACKEND", DEFAULT_READINESS_BACKEND)
    )

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def validate(self, *, check_cli: bool = True) -> list[PipelineError]:
        """Fail-fast validation of prerequisites.

        Returns a list of ``PipelineError`` objects. An empty list means
        everything looks good. Callers should check for fatal errors and
        abort early.
        """
        errors: list[PipelineError] = []

        # 1. API key check
        from agentlab.benchmarks.redteam.execution import build_claude_env

        claude_env = build_claude_env()
        if not claude_env:
            errors.append(
                config_error(
                    "E_CONFIG_API_KEY_MISSING",
                    "No API key found. Set ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN.",
                )
            )

        # 2. Claude CLI check (local backend only)
        if check_cli and self.execution_backend == "local":
            if not shutil.which("claude"):
                errors.append(
                    config_error(
                        "E_CONFIG_CLI_MISSING",
                        "Claude Code CLI not found on PATH.",
                    )
                )

        # 3. Modal import check (modal backend only)
        if self.execution_backend == "modal":
            try:
                import modal  # noqa: F401
            except ImportError:
                errors.append(
                    config_error(
                        "E_CONFIG_MODAL_MISSING",
                        "Modal package not installed but execution_backend='modal'.",
                    )
                )

        # 4. Threshold sanity
        if not (0.0 <= self.functional_threshold <= 1.0):
            errors.append(
                config_error(
                    "E_CONFIG_INVALID_VALUE",
                    f"functional_threshold={self.functional_threshold} must be in [0, 1].",
                )
            )
        if not (0.0 <= self.real_task_threshold <= 1.0):
            errors.append(
                config_error(
                    "E_CONFIG_INVALID_VALUE",
                    f"real_task_threshold={self.real_task_threshold} must be in [0, 1].",
                )
            )

        # 5. Timeout sanity
        for name in ("scaffold_timeout", "task_gen_timeout", "variant_timeout", "validation_timeout"):
            val = getattr(self, name)
            if val <= 0:
                errors.append(
                    config_error(
                        "E_CONFIG_INVALID_VALUE",
                        f"{name}={val} must be a positive integer.",
                    )
                )

        return errors

    def describe(self) -> dict[str, Any]:
        """Return a dict of all settings for logging / manifest embedding."""
        return {
            "execution_backend": self.execution_backend,
            "scaffold_timeout": self.scaffold_timeout,
            "task_gen_timeout": self.task_gen_timeout,
            "variant_timeout": self.variant_timeout,
            "validation_timeout": self.validation_timeout,
            "functional_threshold": self.functional_threshold,
            "real_task_threshold": self.real_task_threshold,
            "max_eval_iterations": self.max_eval_iterations,
            "hardening_rounds": self.hardening_rounds,
            "tasks_per_hardening_round": self.tasks_per_hardening_round,
            "audit_every": self.audit_every,
            "readiness_backend": self.readiness_backend,
        }
