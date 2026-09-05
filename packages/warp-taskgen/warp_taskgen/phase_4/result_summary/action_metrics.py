"""Action-tier summary metrics."""

from __future__ import annotations

from typing import Any

from warp_taskgen.phase_4.result_summary.final_metrics import (
    _include_in_action_success_by_tier_metric,
    final_metric_success,
    ratio,
)


def _action_tier_label(result: dict[str, Any]) -> str:
    value = result.get("adversarial_action_tier")
    if isinstance(value, bool):
        return "unknown"
    if isinstance(value, int):
        return f"tier_{value}"
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return f"tier_{stripped}"
        if stripped:
            return stripped
    return "unknown"


def _include_in_headline_action_metric(result: dict[str, Any]) -> bool:
    if result.get("adversarial_action_enabled") is not True:
        return False
    if _action_tier_label(result) == "tier_3":
        return False
    return _include_in_action_success_by_tier_metric(result)


def action_success_by_tier(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, int]] = {}
    for result in results:
        if not _include_in_action_success_by_tier_metric(result):
            continue
        tier = _action_tier_label(result)
        row = rows.setdefault(tier, {"denominator": 0, "numerator": 0})
        row["denominator"] += 1
        if final_metric_success(result, "attack_action_success"):
            row["numerator"] += 1
    return [
        {
            "action_tier": tier,
            "attack_action_success_numerator": row["numerator"],
            "attack_action_success_denominator": row["denominator"],
            "attack_action_success_rate": ratio(row["numerator"], row["denominator"]),
        }
        for tier, row in sorted(rows.items())
    ]


__all__ = ["action_success_by_tier"]
