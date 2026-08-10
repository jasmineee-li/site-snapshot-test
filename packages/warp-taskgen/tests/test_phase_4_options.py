"""Contracts for the lightweight Phase 4 variant option surface."""

from __future__ import annotations

import pytest

from worldsim.main import build_parser
from worldsim.phase_4.options import (
    PHASE_4_VARIANT_BUDGET_PRESETS,
    PHASE_4_VARIANT_SYSTEMS,
    phase_4_variant_budget_choices,
    phase_4_variant_system_choices,
)


def test_variant_choice_helpers_expose_configured_values() -> None:
    assert phase_4_variant_system_choices() == PHASE_4_VARIANT_SYSTEMS
    assert phase_4_variant_budget_choices() == tuple(PHASE_4_VARIANT_BUDGET_PRESETS)


@pytest.mark.parametrize("value", PHASE_4_VARIANT_SYSTEMS)
def test_phase_command_accepts_each_variant_system(value: str) -> None:
    args = build_parser().parse_args(["phase", "4", "--phase-4-variant-system", value])

    assert args.phase_4_variant_system == value


@pytest.mark.parametrize("value", tuple(PHASE_4_VARIANT_BUDGET_PRESETS))
def test_phase_command_accepts_each_variant_budget(value: str) -> None:
    args = build_parser().parse_args(["phase", "4", "--phase-4-variant-budget", value])

    assert args.phase_4_variant_budget == value
