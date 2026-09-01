"""Canonical neutral prompt projections shared by Phase 4 study/API code."""

from __future__ import annotations

type JsonObject = dict[str, object]


def trajectory_summary(result: JsonObject) -> JsonObject:
    """Project the bounded trajectory evidence used by rewrite prompts."""

    keys = ("task_id", "outcome", "final_status", "final_result")
    summary = {key: result.get(key) for key in keys if key in result}
    encounter = result.get("encounter")
    if isinstance(encounter, dict):
        summary["encounter"] = {
            key: encounter.get(key)
            for key in ("max_coverage", "coverage_basis", "pvpo_observation")
            if key in encounter
        }
    return summary


def rewrite_constraints(task: JsonObject) -> JsonObject:
    """Project the canonical task constraints shared by cue/rewrite prompts."""

    return {
        "preserve_task": True,
        "preserve_editor_call_structure": True,
        "preserve_required_anchors": True,
        "preserve_target_action": True,
        "preserve_placement": True,
        "respect_field_length_budget": True,
        "required_tokens": task.get("required_tokens"),
        "length_budget": task.get("length_budget"),
        "delivery_channel": task.get("delivery_channel"),
        "selected_payload_index": task.get("selected_payload_index"),
    }


__all__ = ["rewrite_constraints", "trajectory_summary"]
