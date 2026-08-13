"""Shared Phase 4 exposure admission predicate for adversarial tasks.

Used by :mod:`warp_taskgen.phases.phase_4_adversarial` and offline validators so
IPI/encounter invariants stay aligned without importing the full Phase 4
runner (browser agent, judges, etc.).
"""

from __future__ import annotations

from typing import Any


def exposure_admission_error(task: dict[str, Any]) -> str | None:
    """Return a skip reason unless Phase 2c proved the exposure contract
    the runtime requires.

    The witnessed ``feasibility.exposure`` evidence must match
    ``exposure_contract`` (contract_id + verification) exactly; eligibility
    and ``phase4_exposure.admissible`` must hold.
    """
    contract = task.get("exposure_contract")
    if not isinstance(contract, dict):
        return "missing_exposure_contract"

    eligibility = contract.get("eligibility")
    if not isinstance(eligibility, dict) or eligibility.get("status") != "eligible":
        return "exposure_contract_ineligible"

    phase4_exposure = contract.get("phase4_exposure")
    if not isinstance(phase4_exposure, dict):
        return "missing_phase4_exposure_capability"
    if phase4_exposure.get("admissible") is not True:
        reason = phase4_exposure.get("reason")
        if isinstance(reason, str) and reason.strip():
            return f"phase4_exposure_inadmissible:{reason.strip()}"
        return "phase4_exposure_inadmissible"

    contract_id = contract.get("contract_id")
    if not isinstance(contract_id, str) or not contract_id.strip():
        return "missing_exposure_contract_id"

    contract_verification = contract.get("verification")
    if not isinstance(contract_verification, dict):
        return "missing_exposure_contract_verification"

    feasibility = task.get("feasibility")
    exposure = feasibility.get("exposure") if isinstance(feasibility, dict) else None
    if not isinstance(exposure, dict):
        return "missing_exposure_evidence"
    if exposure.get("reachable") is not True:
        return "exposure_not_reachable"
    if exposure.get("contract_id") != contract_id:
        return "exposure_contract_id_mismatch"
    if exposure.get("verification") != contract_verification:
        return "exposure_verification_mismatch"
    comment_visibility_error = _reddit_comment_body_painted_visibility_error(
        contract,
        phase4_exposure,
        exposure,
    )
    if comment_visibility_error is not None:
        return comment_visibility_error
    return None


def _reddit_comment_body_painted_visibility_error(
    contract: dict[str, Any],
    phase4_exposure: dict[str, Any],
    exposure: dict[str, Any],
) -> str | None:
    """Require seed-specific painted visibility for Reddit comment carriers.

    Postmill's default submission view sorts top-level comments by score. A
    newly seeded carrier comment can be present in the DOM while not being
    painted in the viewport or selected comment region. Those rows are
    placement failures and must not enter Phase 4 ASR denominators.
    """
    if str(contract.get("site") or "").strip().lower() != "reddit":
        return None
    if str(contract.get("target_surface_id") or "").strip() != "comment.body":
        return None
    if phase4_exposure.get("requires_visual_comment_region") is not True:
        return None
    if exposure.get("layout_visible_at_entry") is not True:
        return "reddit_comment_seed_not_painted_at_entry"
    if exposure.get("requires_expand") is True:
        return "reddit_comment_seed_requires_expand"
    return None
