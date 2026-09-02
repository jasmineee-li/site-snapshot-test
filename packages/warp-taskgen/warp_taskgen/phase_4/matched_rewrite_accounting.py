"""Call policy and usage accounting owned by the matched-rewrite study."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class MatchedCallPolicy:
    """The bounded host call policy shared by both diagnosis arms.

    ``max_tokens`` is the enforceable per-completion output ceiling. The
    retry fields bound semantic and transport retry loops; a dollar ceiling is
    not guessed because it depends on provider usage and pricing responses.
    Actual usage and cost are recorded on each attempt instead.
    """

    model: str
    max_tokens: int = 8192
    semantic_retries: int = 2
    transport_retries: int = 3
    temperature: float = 0.2
    provider: str = "unconfigured"
    runner: str = "unconfigured"

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("matched call policy model must be non-empty")
        object.__setattr__(self, "model", self.model.strip())
        for name in ("provider", "runner"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value.strip()) > 128:
                raise ValueError(
                    f"matched call policy {name} must be a non-empty string of at most 128 characters"
                )
            object.__setattr__(self, name, value.strip())
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 256_000:
            raise ValueError("matched call policy max_tokens must be between 1 and 256000")
        for name in ("semantic_retries", "transport_retries"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 8:
                raise ValueError(f"matched call policy {name} must be between 0 and 8")
        if (
            type(self.temperature) not in (int, float)
            or not math.isfinite(self.temperature)
            or not 0 <= self.temperature <= 1
        ):
            raise ValueError("matched call policy temperature must be between 0 and 1")

    @classmethod
    def for_model(
        cls,
        model: str,
        *,
        provider: str = "unconfigured",
        runner: str = "unconfigured",
    ) -> MatchedCallPolicy:
        """Use the current TP diagnosis defaults for a given model identity."""

        return cls(model=model, provider=provider, runner=runner)

    @property
    def max_output_tokens(self) -> int:
        """Human-readable alias for the Messages API ``max_tokens`` field."""

        return self.max_tokens

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "provider": self.provider,
            "runner": self.runner,
            "max_tokens": self.max_tokens,
            "semantic_retries": self.semantic_retries,
            "transport_retries": self.transport_retries,
            "temperature": self.temperature,
        }


@dataclass(frozen=True, slots=True)
class MatchedStudyBudget:
    """Operator ceilings for one fixed two-arm matched study.

    The token and dollar values are aggregate backpressure limits for all
    completions, including retries, in each arm and across the pair. Dollar
    values are checked only against measured usage after a completion; they do
    not pretend to be a provider-side pre-call reservation.
    """

    per_arm_max_tokens: int
    total_max_tokens: int
    per_arm_max_cost_usd: float
    total_max_cost_usd: float

    def __post_init__(self) -> None:
        for name in ("per_arm_max_tokens", "total_max_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"matched study budget {name} must be a positive integer")
        for name in ("per_arm_max_cost_usd", "total_max_cost_usd"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"matched study budget {name} must be a finite non-negative number"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "per_arm_max_tokens": self.per_arm_max_tokens,
            "total_max_tokens": self.total_max_tokens,
            "per_arm_max_cost_usd": self.per_arm_max_cost_usd,
            "total_max_cost_usd": self.total_max_cost_usd,
        }


@dataclass(frozen=True, slots=True)
class Usage:
    """Measured token and cost usage for one provider stage."""

    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    unavailable_reason: str | None = None
    attempts: int = 1

    @classmethod
    def unavailable(cls, reason: str, *, attempts: int = 1) -> Usage:
        return cls(None, None, None, reason.strip() or "usage_unavailable", attempts)

    @property
    def available(self) -> bool:
        return (
            self.input_tokens is not None
            and self.output_tokens is not None
            and self.cost_usd is not None
            and self.unavailable_reason is None
        )

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"usage {name} must be a non-negative integer or null")
        if self.cost_usd is not None and (
            type(self.cost_usd) not in (int, float)
            or not math.isfinite(self.cost_usd)
            or self.cost_usd < 0
        ):
            raise ValueError("usage cost_usd must be non-negative or null")
        if type(self.attempts) is not int or self.attempts < 1:
            raise ValueError("usage attempts must be a positive integer")


@dataclass(slots=True)
class PairAccounting:
    """Aggregate usage while retaining explicit unavailable-stage reasons."""

    diagnosis_calls: int = 0
    proposal_calls: int = 0
    repair_calls: int = 0
    browser_attempts: int = 0
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    total_tokens: int | None = 0
    cost_usd: float | None = 0.0
    retry_attempts: int = 0
    usage_unavailable_reasons: list[str] | None = None

    def __post_init__(self) -> None:
        if self.usage_unavailable_reasons is None:
            self.usage_unavailable_reasons = []

    def record(self, stage: str, outcome: Any, *, browser_counted: bool = True) -> None:
        if stage in {"tp_diagnosis", "ordinary_critique"}:
            self.diagnosis_calls += 1
        elif stage == "proposal":
            self.proposal_calls += 1
        elif stage == "repair":
            self.repair_calls += 1
        elif stage == "browser" and browser_counted:
            self.browser_attempts += 1

        usage = outcome.usage
        self.retry_attempts += max(0, usage.attempts - 1)
        if not usage.available:
            reason = usage.unavailable_reason or f"{stage}_usage_unavailable"
            assert self.usage_unavailable_reasons is not None
            self.usage_unavailable_reasons.append(reason)
            # Browser execution owns its own accounting artifact. Preserve
            # measured model usage when only that artifact is absent.
            if stage != "browser":
                self.input_tokens = self.output_tokens = self.total_tokens = self.cost_usd = None
            return
        if self.input_tokens is not None:
            self.input_tokens += cast(int, usage.input_tokens)
        if self.output_tokens is not None:
            self.output_tokens += cast(int, usage.output_tokens)
        if self.total_tokens is not None:
            self.total_tokens += cast(int, usage.input_tokens) + cast(int, usage.output_tokens)
        if self.cost_usd is not None:
            self.cost_usd += cast(float, usage.cost_usd)

    def to_dict(self) -> dict[str, object]:
        unavailable = list(self.usage_unavailable_reasons or [])
        return {
            "diagnosis_calls": self.diagnosis_calls,
            "proposal_calls": self.proposal_calls,
            "repair_calls": self.repair_calls,
            "browser_attempts": self.browser_attempts,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "retry_attempts": self.retry_attempts,
            "usage_status": "available" if self.input_tokens is not None else "unavailable",
            "usage_unavailable_reasons": unavailable,
        }


__all__ = ["MatchedCallPolicy", "MatchedStudyBudget", "PairAccounting", "Usage"]
