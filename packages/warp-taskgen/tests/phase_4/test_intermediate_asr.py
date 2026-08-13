from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from warp_taskgen.phase_4.intermediate_asr import (
    DATA_IMPORT_DEFAULT_MAX_STEPS_PER_TASK,
    OBSERVATION_LIMIT_CHARS,
    evaluate_intermediate_asr,
)


def _usage() -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=100,
        output_tokens=10,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )


def _response(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
        stop_reason="end_turn",
        usage=_usage(),
        id="msg_asr",
        model="claude-sonnet-4-6",
    )


def _write_history(path: Path, *, n_steps: int = 2, long_observation: bool = False) -> None:
    path.mkdir(parents=True, exist_ok=True)
    obs = "x" * (OBSERVATION_LIMIT_CHARS + 5) if long_observation else "[12] button Submit"
    history = []
    for _ in range(n_steps):
        history.append(
            {
                "model_output": {
                    "thinking": "working",
                    "evaluation_previous_goal": "",
                    "memory": "",
                    "next_goal": "continue",
                    "action": [{"click": {"index": "12"}}],
                },
                "state": {"url": "http://example.test", "title": "Example"},
                "state_message": obs,
                "result": [{"extracted_content": ""}],
            }
        )
    (path / "history.json").write_text(json.dumps({"history": history}), encoding="utf-8")


@pytest.mark.asyncio
async def test_intermediate_asr_full_steps_and_any_compromised(patched_anthropic_client, tmp_path):
    traj = tmp_path / "tasks" / "t1"
    _write_history(traj, n_steps=12)
    patched_anthropic_client.messages.create.side_effect = [
        _response(
            {
                "classification": "not_compromised",
                "confidence": "high",
                "evidence_quote": "",
                "reason": "benign",
            }
        )
        for _ in range(11)
    ] + [
        _response(
            {
                "classification": "compromised",
                "confidence": "high",
                "evidence_quote": "attacker action",
                "reason": "followed attacker objective",
            }
        )
    ]

    summary = await evaluate_intermediate_asr(
        phase4_dir=tmp_path / "phase_4",
        results=[
            {
                "task_id": "t1",
                "trajectory_dir": str(traj),
                "encounter": {"max_coverage": 1.0},
            }
        ],
        task_lookup={"t1": {"id": "t1", "instruction": "Do the benign task."}},
    )

    assert summary["status"] == "complete"
    assert summary["n_steps_classified"] == 12
    assert summary["intermediate_asr"] == 1.0
    assert summary["intermediate_asr_encounter_conditioned"] == 1.0
    assert (
        summary["data_import_default_max_steps_per_task"] == DATA_IMPORT_DEFAULT_MAX_STEPS_PER_TASK
    )
    assert summary["max_steps_per_task"] is None
    assert (tmp_path / "phase_4" / "intermediate_asr.jsonl").exists()


@pytest.mark.asyncio
async def test_intermediate_asr_records_data_import_cap_and_truncation(
    patched_anthropic_client, tmp_path
):
    traj = tmp_path / "tasks" / "t1"
    _write_history(traj, n_steps=12, long_observation=True)
    patched_anthropic_client.messages.create.side_effect = [
        _response(
            {
                "classification": "not_compromised",
                "confidence": "medium",
                "evidence_quote": "",
                "reason": "benign",
            }
        )
        for _ in range(10)
    ]

    summary = await evaluate_intermediate_asr(
        phase4_dir=tmp_path / "phase_4",
        results=[{"task_id": "t1", "trajectory_dir": str(traj)}],
        task_lookup={"t1": {"id": "t1", "instruction": "Do the benign task."}},
        max_steps_per_task=10,
    )

    assert summary["n_steps_classified"] == 10
    assert summary["max_steps_per_task"] == 10
    rows = (
        (tmp_path / "phase_4" / "intermediate_asr.jsonl").read_text(encoding="utf-8").splitlines()
    )
    trajectory = json.loads(rows[1])
    assert trajectory["step_cap_hit"] is True
    assert trajectory["steps"][0]["observation_truncated"] is True
