from __future__ import annotations

from warp_taskgen.phase_4.variant_accounting import semantic_variant_accounting


def test_semantic_variant_accounting_splits_research_rejections() -> None:
    accounting = semantic_variant_accounting(
        variant_results=[{"outcome": "complied"}, {"outcome": "refused_or_ignored"}],
        generation_errors=[
            {"failure_class": "tp_regression"},
            {"failure_class": "schema_violation"},
            {"failure_class": "rewrite_inapplicable_irreconcilable_contract"},
        ],
    )

    assert accounting == {
        "rewrite_attempted": 4,
        "variant_evaluated": 2,
        "variant_rejection_records": 3,
        "pre_browser_rejections": 2,
        "post_eval_rejections": 1,
        "tp_regression_rejections": 1,
        "schema_validation_failures": 1,
        "contract_inapplicable_rejections": 1,
    }
