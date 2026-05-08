from __future__ import annotations

import json
from pathlib import Path

from eval_awareness_experiments.trajectory_converter import parse_eia_trajectory


def test_parse_eia_trajectory_preserves_top_level_imported_attack_success(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "eia_task"
    task_dir.mkdir()
    (task_dir / "trajectory.json").write_text(
        json.dumps(
            [
                {
                    "thought": "I should complete the current step.",
                    "action": "TYPE(secret)",
                    "observation": "A form field is visible.",
                }
            ]
        ),
        encoding="utf-8",
    )
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": "eia_task",
                "instruction": "Fill the form.",
                "attack_success": True,
            }
        ),
        encoding="utf-8",
    )

    traj = parse_eia_trajectory(task_dir)

    assert traj is not None
    assert traj.attack_success is True


def test_parse_eia_trajectory_falls_back_when_attack_metadata_success_is_null(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "eia_task"
    task_dir.mkdir()
    (task_dir / "trajectory.json").write_text(
        json.dumps(
            [
                {
                    "thought": "I should complete the current step.",
                    "action": "TYPE(secret)",
                    "observation": "A form field is visible.",
                }
            ]
        ),
        encoding="utf-8",
    )
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": "eia_task",
                "instruction": "Fill the form.",
                "attack_metadata": {"attack_success": None},
                "attack_success": True,
            }
        ),
        encoding="utf-8",
    )

    traj = parse_eia_trajectory(task_dir)

    assert traj is not None
    assert traj.attack_success is True
