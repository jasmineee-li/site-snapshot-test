from __future__ import annotations

import json
from pathlib import Path

from scripts import compute_eval_awareness_table as table


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_collect_wasp_uses_final_asr_when_metadata_is_intermediate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(table, "ROOT", tmp_path)
    cell_dir = tmp_path / "wasp" / "bare" / "model-a"
    split_dir = cell_dir / "wasp" / "gitlab"
    _write_jsonl(
        split_dir / "trajectory_awareness_results.jsonl",
        [
            {
                "sample_id": "wasp_gitlab_task1",
                "experiment_type": "trajectory_verbalized_awareness",
                "classification": "Real",
                "metadata": {
                    "attack_success": True,
                    "attack_success_source": "wasp_post_hoc_action_classifier",
                },
            },
            {
                "sample_id": "wasp_gitlab_task1",
                "experiment_type": "trajectory_purpose_continue_5q",
                "classification": "Real",
                "metadata": {"alignment_eval_share": 0.0},
            },
        ],
    )
    _write_jsonl(
        cell_dir / "wasp_final_asr_gitlab.jsonl",
        [
            {"source": "wasp_native_attacker_utility"},
            {"task_id": "task1", "final_attack_success": False},
        ],
    )

    rows = table.collect("wasp")

    assert len(rows) == 1
    assert rows[0]["attack_success"] is False


def test_summarize_handles_empty_canonical_asr(capsys) -> None:
    table.summarize("wasp", [])

    assert "no rows with canonical ASR" in capsys.readouterr().out
