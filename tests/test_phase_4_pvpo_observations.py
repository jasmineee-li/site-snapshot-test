from __future__ import annotations

import pytest

from worldsim.phase_4.pvpo_observations import classify_pvpo_observation


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"max_coverage": 1.0}, "painted_full"),
        ({"max_coverage": 0.5}, "painted_partial"),
        (
            {
                "max_coverage": 0.0,
                "pvpo_failure": "payload_witness_not_painted",
                "match_found_steps": 1,
            },
            "dom_witness_seen_but_not_painted",
        ),
        (
            {"max_coverage": 0.0, "pvpo_failure": "payload_witness_not_matched"},
            "dom_witness_not_matched",
        ),
        (
            {
                "max_coverage": 0.0,
                "pvpo_failure": "payload_witness_off_surface",
                "off_surface_match_steps": 1,
            },
            "dom_witness_matched_off_surface",
        ),
        (
            {"max_coverage": 0.0, "pvpo_status": "degraded", "skipped_steps": 1},
            "partial_pvpo_artifact_loss",
        ),
        (
            {"max_coverage": 0.0, "pvpo_status": "degraded", "pvpo_failure": "capture_failed"},
            "pvpo_capture_degraded",
        ),
        (
            {"max_coverage": 0.0, "pvpo_failure": "reference_screenshot_invalid"},
            "reference_screenshot_invalid",
        ),
        (
            {"max_coverage": 0.0, "pvpo_failure": "invalid_selected_payload_index"},
            "invalid_payload_selection",
        ),
        (
            {"max_coverage": 0.0, "pvpo_status": "detector_failed"},
            "pvpo_detector_failed",
        ),
        (
            {"max_coverage": 0.0, "pvpo_status": "no_artifacts"},
            "pvpo_no_artifacts",
        ),
    ],
)
def test_classify_pvpo_observation_covers_report_only_edge_buckets(
    kwargs: dict[str, object],
    expected: str,
) -> None:
    assert classify_pvpo_observation(**kwargs) == expected
