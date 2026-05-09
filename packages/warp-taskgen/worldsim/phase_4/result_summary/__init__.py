"""Public package surface for Phase 4 result summaries."""

from __future__ import annotations

# ruff: noqa: F401
from worldsim.phase_4.result_summary import _impl as _legacy_impl
from worldsim.phase_4.result_summary.action_metrics import action_success_by_tier
from worldsim.phase_4.result_summary.audit import variant_regeneration_audit
from worldsim.phase_4.result_summary.final_metrics import (
    ecologically_valid,
    final_compliant_result_has_adversarial_reward,
    final_metric_checked,
    final_metric_success,
    final_metric_value,
    final_result_is_compliant,
    final_state_action_metric_checked,
    ratio,
    selected_successful_strategy_variants,
    successful_strategy_variants,
    variant_adversarial_reward_passed,
)
from worldsim.phase_4.result_summary.inspection import inspection_index
from worldsim.phase_4.result_summary.summarize import summarize_results
from worldsim.phase_4.result_summary.task_metadata import (
    task_editor_method,
    task_origin,
    task_route_variant,
    task_scenario_template,
    task_site,
    task_surface,
)

globals().update(
    {
        name: value
        for name, value in vars(_legacy_impl).items()
        if not name.startswith("__")
    }
)
