"""Structured error handling for the redteam generation pipeline.

Provides ``PipelineError`` (a structured, immutable error record) and
``PipelineErrorRegistry`` (an accumulator that replaces the old
``errors: list[str]`` pattern while remaining backward-compatible with
manifest serialization).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agentlab.benchmarks.redteam.utils import utc_timestamp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error categories
# ---------------------------------------------------------------------------

CATEGORY_VALIDATION = "validation"
CATEGORY_LLM = "llm"
CATEGORY_TIMEOUT = "timeout"
CATEGORY_RESOURCE = "resource"
CATEGORY_CONFIG = "config"
CATEGORY_STATE = "state"

# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------

SEVERITY_FATAL = "fatal"
SEVERITY_DEGRADED = "degraded"
SEVERITY_WARNING = "warning"

# ---------------------------------------------------------------------------
# Error codes and default recovery suggestions
# ---------------------------------------------------------------------------

_RECOVERY_SUGGESTIONS: dict[str, str] = {
    # LLM / Claude Code errors
    "E_LLM_TIMEOUT": (
        "Increase timeout via AGENTLAB_REDTEAM_SCAFFOLD_TIMEOUT env var, "
        "or simplify the behavior spec."
    ),
    "E_LLM_NONZERO_EXIT": "Check Claude Code logs in .generation-logs/ for details.",
    "E_LLM_CLI_MISSING": "Install Claude Code: npm install -g @anthropic-ai/claude-code",
    "E_LLM_OUTPUT_MISSING": "Claude Code ran but did not produce expected files; check prompt.",
    # Configuration errors
    "E_CONFIG_API_KEY_MISSING": (
        "Set ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN in your environment."
    ),
    "E_CONFIG_TEMPLATE_MISSING": "Check --design-guides-dir and --template-dir paths.",
    "E_CONFIG_GUIDE_MISSING": "Required design guide file not found; verify guides directory.",
    "E_CONFIG_CLI_MISSING": "Claude Code CLI not found on PATH.",
    "E_CONFIG_MODAL_MISSING": "Install modal: pip install modal",
    "E_CONFIG_INVALID_VALUE": "Check the configuration value and correct it.",
    # Validation errors
    "E_VALIDATION_FILE_MISSING": "Required app file not generated; check scaffold prompt.",
    "E_VALIDATION_RUNTIME_FAIL": "App server failed to start or HTTP health check failed.",
    "E_VALIDATION_DATA_JS": "data.js validation failed; check the generated content.",
    "E_VALIDATION_MANIFEST": "Manifest contract violation; check app_manifest.json.",
    # State / resume errors
    "E_STATE_CORRUPT": "Pipeline state file is unreadable; delete and restart.",
    "E_STATE_BACKEND_MISMATCH": (
        "Delete pipeline_state.json to start fresh, or pass the original "
        "--functional-backend value."
    ),
    "E_STATE_PHASE_INVARIANT": "Internal phase ordering violation; report this as a bug.",
    # Resource errors
    "E_RESOURCE_MODAL_SANDBOX": "Modal sandbox creation failed; check Modal dashboard.",
    "E_RESOURCE_PORT_BINDING": "Could not bind to a free TCP port.",
    "E_RESOURCE_DISK": "Disk space may be insufficient for app generation.",
    # Generation errors
    "E_GENERATION_UNEXPECTED": "Unexpected error during generation; check the full traceback.",
    "E_GENERATION_SCAFFOLD_FAILED": "App scaffold generation did not produce a valid app.",
    "E_GENERATION_TASK_FAILED": "Task suite generation failed.",
    "E_GENERATION_VARIANT_FAILED": "Variant generation failed.",
}


def _default_recovery(code: str) -> str:
    """Look up the default recovery suggestion for an error code."""
    return _RECOVERY_SUGGESTIONS.get(code, "")


# ---------------------------------------------------------------------------
# PipelineError
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineError:
    """A single structured error from the generation pipeline.

    Attributes:
        code: Machine-readable identifier (e.g. ``E_LLM_TIMEOUT``).
        category: One of the ``CATEGORY_*`` constants.
        severity: One of the ``SEVERITY_*`` constants.
        message: Human-readable description of what went wrong.
        phase: Phase ID where the error occurred (e.g. ``phase_1a``).
        context: Structured details (paths, return codes, etc.).
        recovery: Actionable suggestion for the developer.
        timestamp: ISO-8601 UTC timestamp.
    """

    code: str
    category: str
    severity: str
    message: str
    phase: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    recovery: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            object.__setattr__(self, "timestamp", utc_timestamp())
        if not self.recovery:
            suggestion = _default_recovery(self.code)
            if suggestion:
                object.__setattr__(self, "recovery", suggestion)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "category": self.category,
            "severity": self.severity,
            "message": self.message,
            "phase": self.phase,
            "context": self.context,
            "recovery": self.recovery,
            "timestamp": self.timestamp,
        }

    def to_log_line(self) -> str:
        """Format as a single log line: ``[CODE] message (recovery: ...)``."""
        parts = [f"[{self.code}] {self.message}"]
        if self.recovery:
            parts.append(f"Recovery: {self.recovery}")
        return " | ".join(parts)

    def to_manifest_string(self) -> str:
        """Backward-compatible string for ``manifest["errors"]``."""
        prefix = f"[{self.code}] " if self.code else ""
        suffix = f" Recovery: {self.recovery}" if self.recovery else ""
        return f"{prefix}{self.message}{suffix}"


# ---------------------------------------------------------------------------
# PipelineErrorRegistry
# ---------------------------------------------------------------------------


class PipelineErrorRegistry:
    """Accumulator for ``PipelineError`` objects.

    Replaces the old ``errors: list[str]`` pattern while providing
    backward-compatible serialization via :meth:`to_string_list`.
    """

    def __init__(self) -> None:
        self.errors: list[PipelineError] = []

    def add(self, error: PipelineError) -> None:
        """Record an error and log it."""
        self.errors.append(error)
        log_fn = logger.error if error.severity == SEVERITY_FATAL else logger.warning
        log_fn("%s", error.to_log_line())

    def has_fatal(self) -> bool:
        return any(e.severity == SEVERITY_FATAL for e in self.errors)

    def by_category(self, category: str) -> list[PipelineError]:
        return [e for e in self.errors if e.category == category]

    def by_phase(self, phase: str) -> list[PipelineError]:
        return [e for e in self.errors if e.phase == phase]

    def to_string_list(self) -> list[str]:
        """Backward-compatible: convert to ``list[str]`` for manifest."""
        return [e.to_manifest_string() for e in self.errors]

    def to_dict_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.errors]

    def summary(self) -> str:
        """Human-readable error summary grouped by phase."""
        if not self.errors:
            return "No errors."
        lines: list[str] = []
        phases_seen: dict[str, list[PipelineError]] = {}
        for err in self.errors:
            phase_key = err.phase or "(no phase)"
            phases_seen.setdefault(phase_key, []).append(err)

        for phase, errs in phases_seen.items():
            lines.append(f"\n  {phase}:")
            for err in errs:
                marker = "FATAL" if err.severity == SEVERITY_FATAL else err.severity.upper()
                lines.append(f"    [{marker}] {err.code}: {err.message}")
                if err.recovery:
                    lines.append(f"           Recovery: {err.recovery}")

        header = f"Pipeline errors ({len(self.errors)} total):"
        return header + "".join(lines)

    def __len__(self) -> int:
        return len(self.errors)

    def __bool__(self) -> bool:
        return bool(self.errors)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def config_error(
    code: str,
    message: str,
    *,
    phase: str = "",
    context: dict[str, Any] | None = None,
    recovery: str = "",
) -> PipelineError:
    """Create a configuration error."""
    return PipelineError(
        code=code,
        category=CATEGORY_CONFIG,
        severity=SEVERITY_FATAL,
        message=message,
        phase=phase,
        context=context or {},
        recovery=recovery,
    )


def validation_error(
    code: str,
    message: str,
    *,
    phase: str = "",
    context: dict[str, Any] | None = None,
    severity: str = SEVERITY_FATAL,
    recovery: str = "",
) -> PipelineError:
    """Create a validation error."""
    return PipelineError(
        code=code,
        category=CATEGORY_VALIDATION,
        severity=severity,
        message=message,
        phase=phase,
        context=context or {},
        recovery=recovery,
    )


def llm_error(
    code: str,
    message: str,
    *,
    phase: str = "",
    context: dict[str, Any] | None = None,
    severity: str = SEVERITY_FATAL,
    recovery: str = "",
) -> PipelineError:
    """Create an LLM/Claude Code error."""
    return PipelineError(
        code=code,
        category=CATEGORY_LLM,
        severity=severity,
        message=message,
        phase=phase,
        context=context or {},
        recovery=recovery,
    )
