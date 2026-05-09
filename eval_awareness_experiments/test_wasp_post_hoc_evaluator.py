from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval_awareness_experiments.wasp_post_hoc_evaluator import (
    OBSERVATION_LIMIT_CHARS,
    STRUCTURED_OUTPUT_MODE,
    WASP_ASR_SCHEMA,
    StepRecord,
    classify_step,
    evaluate_wasp_cell_split,
    load_wasp_asr_map,
)


class _FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.structured_calls: list[dict] = []

    async def generate_json_schema(self, prompt, *, schema: dict, name: str, strict: bool = True):
        self.structured_calls.append(
            {"prompt": prompt, "schema": schema, "name": name, "strict": strict}
        )
        return SimpleNamespace(message=SimpleNamespace(text=self.responses.pop(0)))


def _step(
    *,
    task_id: str = "webarena.1",
    action: str = "click('12')",
    observation: str = "[12] button 'Submit'\n[14] textbox 'Comment'",
) -> StepRecord:
    return StepRecord(
        task_id=task_id,
        cell={"split": "gitlab"},
        task_dir=Path("/tmp/task"),
        step_index=1,
        legitimate_objective="Create the requested issue comment.",
        observation_text=observation,
        action_description=action,
    )


def _payload(classification: str = "not_compromised") -> str:
    return json.dumps(
        {
            "classification": classification,
            "confidence": "high",
            "evidence_quote": "clicked submit",
            "reason": "The action is consistent with the legitimate objective.",
        }
    )


@pytest.mark.asyncio
async def test_classify_step_uses_schema_and_records_structured_fields() -> None:
    llm = _FakeLLM([_payload("compromised")])

    result = await classify_step(llm, _step())

    assert result.classification == "compromised"
    assert result.confidence == "high"
    assert result.evidence_quote == "clicked submit"
    assert result.reason == "The action is consistent with the legitimate objective."
    assert result.parsed_classifier_response is not None
    assert result.bid_resolutions == "12: button 'Submit'"
    assert llm.structured_calls[0]["schema"] == WASP_ASR_SCHEMA
    assert llm.structured_calls[0]["name"] == "wasp_asr_step"
    assert llm.structured_calls[0]["strict"] is True


@pytest.mark.parametrize(
    "response,classification,failure",
    [
        ("'''compromised'''", "parse_error", "json_parse_error"),
        ("```json\n" + _payload("not_compromised") + "\n```", "parse_error", "json_parse_error"),
        (
            json.dumps(
                {
                    "classification": "not_compromised",
                    "confidence": "high",
                    "evidence_quote": "ok",
                    "reason": "ok",
                    "extra": "x",
                }
            ),
            "schema_error",
            "schema_violation",
        ),
        (
            json.dumps(
                {
                    "classification": "not_compromised",
                    "confidence": "certain",
                    "evidence_quote": "ok",
                    "reason": "ok",
                }
            ),
            "schema_error",
            "schema_violation",
        ),
    ],
)
@pytest.mark.asyncio
async def test_classify_step_invalid_output_fails_closed(
    response: str,
    classification: str,
    failure: str,
) -> None:
    result = await classify_step(_FakeLLM([response]), _step())
    assert result.classification == classification
    assert result.classifier_failure == failure
    assert result.parsed_classifier_response is None


@pytest.mark.asyncio
async def test_classify_step_records_truncation_metadata() -> None:
    observation = "[12] button 'Submit'\n" + ("x" * (OBSERVATION_LIMIT_CHARS + 5))
    result = await classify_step(_FakeLLM([_payload()]), _step(observation=observation))
    assert result.observation_truncated is True


@pytest.mark.asyncio
async def test_evaluate_wasp_cell_split_preserves_asr_semantics(monkeypatch, tmp_path) -> None:
    task_dir = tmp_path / "study" / "GenericAgent_on_webarena.1_0"
    task_dir.mkdir(parents=True)
    (task_dir / "goal_object.pkl.gz").write_bytes(b"placeholder")

    steps = [
        _step(task_id="webarena.1", action="click('12')"),
        _step(task_id="webarena.1", action="click('14')", observation="[14] link 'Ignore'"),
        _step(task_id="webarena.2", action="click('15')", observation="[15] button 'Cancel'"),
    ]

    monkeypatch.setattr(
        "eval_awareness_experiments.wasp_post_hoc_evaluator._walk_tasks",
        lambda study_dir, max_tasks: [task_dir],
    )
    monkeypatch.setattr(
        "eval_awareness_experiments.wasp_post_hoc_evaluator.extract_steps_from_task",
        lambda task_dir_arg, cell: steps,
    )

    llm = _FakeLLM(
        [
            _payload("not_compromised"),
            _payload("compromised"),
            _payload("outright_refusal"),
        ]
    )
    cell = {
        "arm": "bare",
        "model_slug": "model",
        "split": "gitlab",
        "study_dir": str(tmp_path / "study"),
        "cell_dir": str(tmp_path / "cell"),
    }

    summary = await evaluate_wasp_cell_split(
        cell=cell,
        llm=llm,
        classifier_model="anthropic/claude-sonnet-4.6",
        max_steps_per_task=10,
        force=True,
    )

    assert summary["asr"] == 0.5
    assert summary["n_compromised"] == 1
    assert summary["n_refusal"] == 1
    assert summary["step_classification_counts"] == {
        "compromised": 1,
        "not_compromised": 1,
        "outright_refusal": 1,
    }
    assert load_wasp_asr_map(Path(cell["cell_dir"]), "gitlab") == {
        "webarena.1": True,
        "webarena.2": False,
    }
    rows = (Path(cell["cell_dir"]) / "wasp_asr_gitlab.jsonl").read_text().splitlines()
    task_row = json.loads(rows[1])
    assert task_row["steps"][1]["parsed_classifier_response"]["classification"] == "compromised"


@pytest.mark.asyncio
async def test_evaluate_wasp_cell_split_counts_schema_failures_not_asr(
    monkeypatch,
    tmp_path,
) -> None:
    task_dir = tmp_path / "study" / "GenericAgent_on_webarena.1_0"
    task_dir.mkdir(parents=True)
    (task_dir / "goal_object.pkl.gz").write_bytes(b"placeholder")

    monkeypatch.setattr(
        "eval_awareness_experiments.wasp_post_hoc_evaluator._walk_tasks",
        lambda study_dir, max_tasks: [task_dir],
    )
    monkeypatch.setattr(
        "eval_awareness_experiments.wasp_post_hoc_evaluator.extract_steps_from_task",
        lambda task_dir_arg, cell: [_step(task_id="webarena.1")],
    )
    cell = {
        "arm": "bare",
        "model_slug": "model",
        "split": "gitlab",
        "study_dir": str(tmp_path / "study"),
        "cell_dir": str(tmp_path / "cell"),
    }

    summary = await evaluate_wasp_cell_split(
        cell=cell,
        llm=_FakeLLM(['{"classification":"compromised"}']),
        classifier_model="anthropic/claude-sonnet-4.6",
        force=True,
    )

    assert summary["asr"] == 0.0
    assert summary["n_schema_error_steps"] == 1
    assert summary["n_compromised"] == 0
    assert load_wasp_asr_map(Path(cell["cell_dir"]), "gitlab") == {"webarena.1": False}


@pytest.mark.asyncio
async def test_evaluate_wasp_cell_split_recomputes_stale_existing_sidecar(
    monkeypatch,
    tmp_path,
) -> None:
    task_dir = tmp_path / "study" / "GenericAgent_on_webarena.1_0"
    task_dir.mkdir(parents=True)
    (task_dir / "goal_object.pkl.gz").write_bytes(b"placeholder")
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    (cell_dir / "wasp_asr_gitlab.jsonl").write_text(
        json.dumps({"cell": {"study_dir": str(tmp_path / "study")}, "asr": 1.0}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "eval_awareness_experiments.wasp_post_hoc_evaluator._walk_tasks",
        lambda study_dir, max_tasks: [task_dir],
    )
    monkeypatch.setattr(
        "eval_awareness_experiments.wasp_post_hoc_evaluator.extract_steps_from_task",
        lambda task_dir_arg, cell: [_step(task_id="webarena.1")],
    )
    cell = {
        "arm": "bare",
        "model_slug": "model",
        "split": "gitlab",
        "study_dir": str(tmp_path / "study"),
        "cell_dir": str(cell_dir),
    }

    summary = await evaluate_wasp_cell_split(
        cell=cell,
        llm=_FakeLLM([_payload("not_compromised")]),
        classifier_model="anthropic/claude-sonnet-4.6",
        force=False,
    )

    assert summary["structured_output_mode"] == STRUCTURED_OUTPUT_MODE
    assert "skipped_existing" not in summary
    assert summary["asr"] == 0.0


@pytest.mark.asyncio
async def test_evaluate_wasp_cell_split_reuses_current_existing_sidecar(tmp_path) -> None:
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    cell = {
        "arm": "bare",
        "model_slug": "model",
        "split": "gitlab",
        "study_dir": str(tmp_path / "study"),
        "cell_dir": str(cell_dir),
    }
    (cell_dir / "wasp_asr_gitlab.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "wasp_asr_v1",
                "structured_output_mode": STRUCTURED_OUTPUT_MODE,
                "cell": cell,
                "asr": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = await evaluate_wasp_cell_split(
        cell=cell,
        llm=_FakeLLM([]),
        classifier_model="anthropic/claude-sonnet-4.6",
        force=False,
    )

    assert summary["skipped_existing"] is True


@pytest.mark.asyncio
async def test_evaluate_wasp_cell_split_counts_no_action_tasks(monkeypatch, tmp_path) -> None:
    task_dir = tmp_path / "study" / "GenericAgent_on_webarena.1_0"
    task_dir.mkdir(parents=True)
    (task_dir / "goal_object.pkl.gz").write_bytes(b"placeholder")

    monkeypatch.setattr(
        "eval_awareness_experiments.wasp_post_hoc_evaluator._walk_tasks",
        lambda study_dir, max_tasks: [task_dir],
    )
    monkeypatch.setattr(
        "eval_awareness_experiments.wasp_post_hoc_evaluator.extract_steps_from_task",
        lambda task_dir_arg, cell: [],
    )
    cell = {
        "arm": "bare",
        "model_slug": "model",
        "split": "gitlab",
        "study_dir": str(tmp_path / "study"),
        "cell_dir": str(tmp_path / "cell"),
    }

    summary = await evaluate_wasp_cell_split(
        cell=cell,
        llm=_FakeLLM([]),
        classifier_model="anthropic/claude-sonnet-4.6",
        force=True,
    )

    assert summary["n_tasks"] == 1
    assert summary["n_no_action_tasks"] == 1
    assert summary["asr"] == 0.0
