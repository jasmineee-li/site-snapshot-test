"""Offline unit tests for the shared summary-table helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_awareness_experiments.summary_tables import finalize_summary_row, read_jsonl


def _write_mixed_jsonl(path: Path) -> None:
    path.write_text('{"a": 1}\n\n{not json}\n{"a": 2}\n', encoding="utf-8")


def test_read_jsonl_strict_raises_on_malformed_line(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    _write_mixed_jsonl(path)
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)


def test_read_jsonl_skip_malformed_drops_only_the_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    _write_mixed_jsonl(path)
    assert read_jsonl(path, skip_malformed=True) == [{"a": 1}, {"a": 2}]


def test_read_jsonl_missing_file_is_empty(tmp_path: Path) -> None:
    assert read_jsonl(tmp_path / "absent.jsonl") == []
    assert read_jsonl(tmp_path / "absent.jsonl", skip_malformed=True) == []


def test_finalize_summary_row_derives_rates_and_leaves_input_untouched() -> None:
    row = {
        "user_task_pass_count": 3,
        "user_task_pass_n": 4,
        "asr_success_count": 0,
        "asr_n": 0,
        "verbalized_n": 8,
        "verbalized_eval_count": 2,
        "verbalized_alignment_eval_count": 1,
        "verbalized_capability_eval_count": 1,
        "verbalized_other_eval_count": 0,
        "verbalized_real_count": 6,
        "purpose_n": 8,
        "purpose_eval_class_count": 4,
        "purpose_alignment_eval_class_count": 2,
        "purpose_capability_eval_class_count": 1,
        "purpose_other_eval_class_count": 1,
        "purpose_real_class_count": 4,
        "purpose_alignment_eval_share_sum": 1.5,
        "purpose_refused_share_sum": 0.5,
        "purpose_share_n": 5,
    }
    before = dict(row)

    out = finalize_summary_row(row)

    assert row == before
    assert out["user_task_pass_rate"] == 0.75
    assert out["asr_rate"] is None
    assert out["verbalized_eval_rate"] == 0.25
    assert out["verbalized_alignment_eval_rate"] == 0.125
    assert out["verbalized_real_rate"] == 0.75
    assert out["purpose_eval_class_rate"] == 0.5
    assert out["purpose_real_class_rate"] == 0.5
    assert out["purpose_alignment_eval_share_mean"] == 0.3
    assert out["purpose_refused_share_mean"] == 0.1
