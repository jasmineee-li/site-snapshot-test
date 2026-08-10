"""Phase 4 variant configuration used by the CLI and runtime.

This module intentionally contains only option values and small normalization
helpers. Keeping the parser-facing choices here lets the CLI
avoid importing Phase 4's linked execution context merely to build argparse
choices.
"""

from __future__ import annotations

PHASE_4_VARIANT_SYSTEMS: tuple[str, ...] = (
    "eval-awareness-iterator",
    "strategy-variation",
    "none",
)
DEFAULT_PHASE_4_VARIANT_SYSTEM = "eval-awareness-iterator"
DEFAULT_PHASE_4_EVAL_AWARENESS_MAX_ITERATIONS = 3
PHASE_4_EVAL_AWARENESS_ITERATOR_VERSION = "eval-awareness-iterator-v1"

ADAPTIVE_VARIANT_BUDGET: tuple[int, ...] = (3, 3, 1)
PHASE_4_VARIANT_BUDGET_PRESETS: dict[str, tuple[int, ...]] = {
    "adaptive-3-3-1": ADAPTIVE_VARIANT_BUDGET,
    "smoke-3-probe": (3,),
}
DEFAULT_PHASE_4_VARIANT_BUDGET_PRESET = "adaptive-3-3-1"


def phase_4_variant_budget_choices() -> tuple[str, ...]:
    """Return the accepted Phase 4 variant budget preset names."""

    return tuple(PHASE_4_VARIANT_BUDGET_PRESETS)


def phase_4_variant_system_choices() -> tuple[str, ...]:
    """Return the accepted Phase 4 variant system names."""

    return PHASE_4_VARIANT_SYSTEMS


def normalize_phase_4_variant_system(value: str | None) -> str:
    """Resolve an optional variant-system value to the supported default."""

    normalized = (value or DEFAULT_PHASE_4_VARIANT_SYSTEM).strip()
    if normalized in PHASE_4_VARIANT_SYSTEMS:
        return normalized
    return DEFAULT_PHASE_4_VARIANT_SYSTEM


def normalize_eval_awareness_max_iterations(value: int | None) -> int:
    """Resolve an optional iterator limit while rejecting invalid values."""

    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return DEFAULT_PHASE_4_EVAL_AWARENESS_MAX_ITERATIONS


def phase_4_variant_budget_shape(preset: str | None) -> tuple[int, ...]:
    """Return the round budget for a preset, falling back to the default."""

    normalized = (preset or DEFAULT_PHASE_4_VARIANT_BUDGET_PRESET).strip()
    return PHASE_4_VARIANT_BUDGET_PRESETS.get(
        normalized,
        PHASE_4_VARIANT_BUDGET_PRESETS[DEFAULT_PHASE_4_VARIANT_BUDGET_PRESET],
    )


__all__ = [
    "ADAPTIVE_VARIANT_BUDGET",
    "DEFAULT_PHASE_4_EVAL_AWARENESS_MAX_ITERATIONS",
    "DEFAULT_PHASE_4_VARIANT_BUDGET_PRESET",
    "DEFAULT_PHASE_4_VARIANT_SYSTEM",
    "PHASE_4_EVAL_AWARENESS_ITERATOR_VERSION",
    "PHASE_4_VARIANT_BUDGET_PRESETS",
    "PHASE_4_VARIANT_SYSTEMS",
    "normalize_eval_awareness_max_iterations",
    "normalize_phase_4_variant_system",
    "phase_4_variant_budget_choices",
    "phase_4_variant_budget_shape",
    "phase_4_variant_system_choices",
]
