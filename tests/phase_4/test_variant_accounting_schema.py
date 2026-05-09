from __future__ import annotations

import json
from pathlib import Path

from scripts.normalize_phase4_variant_schema import normalize_run
from worldsim.phase_4.variant_accounting import semantic_variant_accounting


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


def test_normalize_run_adds_semantic_accounting_to_existing_artifacts(tmp_path: Path) -> None:
    phase4 = tmp_path / "phase_4"
    phase4.mkdir()
    results = [
        {
            "task_id": "t1",
            "eval_awareness_iterator": {
                "variant_results": [
                    {"outcome": "complied", "encounter": {"max_coverage": 1.0}}
                ],
                "variant_generation_errors": [
                    {"failure_class": "tp_regression"},
                    {"failure_class": "schema_violation"},
                ],
            },
        },
        {
            "task_id": "t2",
            "eval_awareness_iterator": {
                "status": "skipped",
                "reason": "injection_not_encountered",
            },
        },
    ]
    (phase4 / "results.json").write_text(json.dumps(results), encoding="utf-8")
    (phase4 / "progress.json").write_text(
        json.dumps({"variant_progress": {"generation_attempted": 3}}),
        encoding="utf-8",
    )

    dry_run = normalize_run(tmp_path, write=False)
    assert dry_run["rows_with_added_accounting"] == 1
    assert dry_run["progress_updated"] is True

    report = normalize_run(tmp_path, write=True)
    assert report["entered_tasks"] == 1
    assert report["rewrite_attempted"] == 2
    assert report["variant_evaluated"] == 1
    assert report["post_eval_rejections"] == 1
    updated_results = json.loads((phase4 / "results.json").read_text(encoding="utf-8"))
    assert updated_results[0]["eval_awareness_iterator"]["variant_outcome_accounting"][
        "schema_validation_failures"
    ] == 1
    assert "variant_outcome_accounting" not in updated_results[1]["eval_awareness_iterator"]
    updated_progress = json.loads((phase4 / "progress.json").read_text(encoding="utf-8"))
    assert updated_progress["variant_progress"]["rewrite_attempted"] == 2
    assert updated_progress["variant_progress"]["post_eval_rejections"] == 1
