"""Report-only PVPO observation buckets.

These labels explain why the strict Phase 4 encounter gate did or did not
count a trajectory. They are diagnostics only: ``max_coverage > 0`` remains
the sole encounter split.
"""

from __future__ import annotations

from typing import Any


def classify_pvpo_observation(
    *,
    max_coverage: Any,
    pvpo_status: Any = None,
    pvpo_failure: Any = None,
    artifact_steps: Any = None,
    skipped_steps: Any = None,
    match_found_steps: Any = None,
    off_surface_match_steps: Any = None,
) -> str:
    """Classify PVPO telemetry into a stable operator-facing bucket."""

    coverage = _float_or_none(max_coverage)
    status = _str_or_none(pvpo_status) or "missing"
    failure = _str_or_none(pvpo_failure)
    artifacts = _int_or_zero(artifact_steps)
    skipped = _int_or_zero(skipped_steps)
    matches = _int_or_zero(match_found_steps)
    off_surface_matches = _int_or_zero(off_surface_match_steps)

    if coverage is not None and coverage > 0.0:
        if coverage >= 1.0:
            return "painted_full"
        return "painted_partial"

    if failure == "invalid_selected_payload_index":
        return "invalid_payload_selection"
    if status in {"no_payload", "disabled_no_payload"}:
        return "payload_capture_disabled"
    if status == "detector_failed" or failure == "determine_encounter_exception":
        return "pvpo_detector_failed"
    if failure == "reference_screenshot_invalid":
        return "reference_screenshot_invalid"
    if failure == "payload_witness_off_surface" or off_surface_matches > 0:
        return "dom_witness_matched_off_surface"
    if failure == "payload_witness_not_painted" or matches > 0:
        return "dom_witness_seen_but_not_painted"
    if failure == "payload_witness_not_matched" or (
        artifacts > 0 and matches == 0 and status == "ok"
    ):
        return "dom_witness_not_matched"
    if status == "no_artifacts" or failure == "pvpo_no_artifacts":
        return "pvpo_no_artifacts"
    if status == "degraded":
        if failure == "partial_pvpo_artifact_loss" or skipped > 0:
            return "partial_pvpo_artifact_loss"
        return "pvpo_capture_degraded"
    if failure:
        return "pvpo_other_failure"
    if artifacts == 0:
        return "pvpo_no_artifacts"
    return "zero_coverage_unknown"


def pvpo_observation_bucket(result: dict[str, Any]) -> str:
    """Return a PVPO observation bucket for a Phase 4 result or variant."""

    encounter = result.get("encounter")
    encounter = encounter if isinstance(encounter, dict) else {}
    explicit = result.get("pvpo_observation") or encounter.get("pvpo_observation")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return classify_pvpo_observation(
        max_coverage=encounter.get("max_coverage"),
        pvpo_status=result.get("pvpo_status") or encounter.get("pvpo_status"),
        pvpo_failure=result.get("pvpo_failure") or encounter.get("pvpo_failure"),
        artifact_steps=encounter.get("pvpo_artifact_steps"),
        skipped_steps=encounter.get("pvpo_skipped_steps"),
        match_found_steps=encounter.get("pvpo_match_found_steps"),
        off_surface_match_steps=encounter.get("pvpo_off_surface_match_steps"),
    )


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _str_or_none(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
