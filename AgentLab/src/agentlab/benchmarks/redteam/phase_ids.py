"""Shared canonical phase identifiers for the redteam controller."""

PHASE_1A = "phase_1a_scaffold_generation"
PHASE_1B = "phase_1b_variant_generation"
PHASE_1C = "phase_1c_runtime_validation"
PHASE_2A = "phase_2a_function_suite_generation"
PHASE_2B = "phase_2b_function_benign_readiness"
PHASE_3A = "phase_3a_real_suite_generation"
PHASE_3B = "phase_3b_real_benign_readiness"
PHASE_4A = "phase_4a_hardening_generation"
PHASE_4B = "phase_4b_hardening_eval_audit"
PHASE_5 = "phase_5_final_regression"
PHASE_COMPLETED = "completed"

CANONICAL_PHASES = (
    PHASE_1A,
    PHASE_1B,
    PHASE_1C,
    PHASE_2A,
    PHASE_2B,
    PHASE_3A,
    PHASE_3B,
    PHASE_4A,
    PHASE_4B,
    PHASE_5,
    PHASE_COMPLETED,
)

_SHORT_PHASE_ALIASES = {
    "phase_1a": PHASE_1A,
    "phase_1b": PHASE_1B,
    "phase_1c": PHASE_1C,
    "phase_2a": PHASE_2A,
    "phase_2b": PHASE_2B,
    "phase_3a": PHASE_3A,
    "phase_3b": PHASE_3B,
    "phase_4a": PHASE_4A,
    "phase_4b": PHASE_4B,
    "phase_5": PHASE_5,
}

def normalize_phase_id(phase: str | None) -> str:
    if not phase:
        return PHASE_1A
    if phase in _SHORT_PHASE_ALIASES:
        return _SHORT_PHASE_ALIASES[phase]
    if phase not in CANONICAL_PHASES:
        raise ValueError(
            f"Unsupported legacy phase id {phase!r}. Regenerate or rerun with current app artifacts."
        )
    return phase
