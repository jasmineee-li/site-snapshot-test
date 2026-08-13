from __future__ import annotations

import json
import re

from warp_taskgen.prompt_corrections import ValidationIssue, render_validation_feedback


def test_render_validation_feedback_wraps_structured_errors():
    block = render_validation_feedback(
        artifact_name="BENCHMARK_PROFILE.json",
        errors=[
            ValidationIssue(
                code="BAD_FIELD",
                path="$.field",
                message="field is invalid",
                expected="string",
                actual=None,
                repair_hint="Use a non-empty string.",
            )
        ],
        extra_guidance="Keep the output compact.",
    )

    assert block.startswith("\n\n<validation_feedback>")
    assert block.endswith("</validation_feedback>")
    payload_text = re.search(r"```json\n(.*?)\n```", block, re.DOTALL)
    assert payload_text is not None
    payload = json.loads(payload_text.group(1))
    assert payload["valid"] is False
    assert payload["artifact"] == "BENCHMARK_PROFILE.json"
    assert payload["errors"][0]["code"] == "BAD_FIELD"
    assert payload["errors"][0]["path"] == "$.field"
    assert payload["errors"][0]["repair_hint"] == "Use a non-empty string."
    assert payload["extra_guidance"] == "Keep the output compact."


def test_render_validation_feedback_normalizes_string_errors():
    block = render_validation_feedback(
        artifact_name="artifact.json",
        errors=["plain failure"],
    )

    payload_text = re.search(r"```json\n(.*?)\n```", block, re.DOTALL)
    assert payload_text is not None
    payload = json.loads(payload_text.group(1))
    assert payload["errors"] == [
        {"code": "VALIDATION_ERROR", "path": "$", "message": "plain failure"}
    ]
