"""Semantic accounting for Phase 4 variant/iterator attempts."""

from __future__ import annotations

from typing import Any

_TP_REGRESSION = "tp_regression"
_POST_EVAL_FAILURES = {
    _TP_REGRESSION,
    "task_broke",
    "lost_pvpo_encounter",
    "worker_timeout_salvaged",
    "process_pool_worker_timeout",
}


def generation_failure_class(error: Any) -> str:
    if isinstance(error, dict):
        value = error.get("failure_class") or error.get("status") or error.get("reason")
        if value not in (None, ""):
            return str(value)
    if error not in (None, ""):
        return str(error)
    return "unknown"


def is_post_eval_rejection(error: Any) -> bool:
    failure_class = generation_failure_class(error)
    return failure_class in _POST_EVAL_FAILURES


def semantic_variant_accounting(
    *,
    variant_results: list[dict[str, Any]],
    generation_errors: list[Any],
) -> dict[str, int]:
    """Return operator-facing counters that avoid overloading "generation failed".

    Legacy progress treated every item in ``variant_generation_errors`` as a
    failed generation. For eval-awareness iteration that is imprecise because
    ``tp_regression`` is a browser-evaluated variant rejected for research
    validity, not an API/schema generation failure. These counters split
    pre-browser rewrite/contract failures from post-evaluation rejections while
    keeping the legacy totals available elsewhere.
    """

    pre_browser_errors = [
        error for error in generation_errors if not is_post_eval_rejection(error)
    ]
    post_eval_errors = [
        error for error in generation_errors if is_post_eval_rejection(error)
    ]
    schema_failures = [
        error for error in generation_errors if generation_failure_class(error) == "schema_violation"
    ]
    contract_rejections = [
        error
        for error in generation_errors
        if generation_failure_class(error).startswith("rewrite_inapplicable")
    ]
    tp_regressions = [
        error for error in generation_errors if generation_failure_class(error) == _TP_REGRESSION
    ]
    return {
        "rewrite_attempted": len(variant_results) + len(pre_browser_errors),
        "variant_evaluated": len(variant_results),
        "variant_rejection_records": len(generation_errors),
        "pre_browser_rejections": len(pre_browser_errors),
        "post_eval_rejections": len(post_eval_errors),
        "tp_regression_rejections": len(tp_regressions),
        "schema_validation_failures": len(schema_failures),
        "contract_inapplicable_rejections": len(contract_rejections),
    }


__all__ = [
    "generation_failure_class",
    "is_post_eval_rejection",
    "semantic_variant_accounting",
]
