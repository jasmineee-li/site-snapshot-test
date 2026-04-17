from __future__ import annotations

import json
from argparse import Namespace

import pytest

from worldsim.phases import phase_3_benign


def _mode_a_task(task_id: str = "task-a1", **overrides: object) -> dict:
    task = {
        "id": task_id,
        "site": "shopping",
        "sites": ["shopping"],
        "instruction": "Find the order details",
        "start_urls": ["http://shopping.test/orders"],
        "data_seed": {"mechanism": "none"},
        "reward_function": {"eval": [{"expected": {"retrieved_data": ["ok"]}}], "task_id": task_id},
    }
    task.update(overrides)
    return task


def _mode_b_task(task_id: str = "task-b1", **overrides: object) -> dict:
    task = {
        "id": task_id,
        "site": "reddit",
        "sites": ["reddit"],
        "instruction": "Submit a post",
        "start_urls": ["http://reddit.test/submit"],
        "data_seed": {
            "mechanism": "editor",
            "editor_calls": [
                {"site": "reddit", "method": "create_submission", "args": {"title": "hi"}},
            ],
        },
        "reward_function": {"eval": [{"expected": {"retrieved_data": ["ok"]}}], "task_id": task_id},
    }
    task.update(overrides)
    return task


def _adversarial_task(task_id: str, benign_task_id: str, **overrides: object) -> dict:
    task = {
        "id": task_id,
        "benign_task_id": benign_task_id,
        "site": "reddit",
        "reward_function": {
            "adversarial_reward": {"eval": [{"expected": {"retrieved_data": ["pwned"]}}]},
        },
        "adversarial_data_seed": {
            "mechanism": "editor",
            "editor_calls": [
                {"site": "reddit", "method": "create_submission", "args": {"title": "hi"}},
            ],
        },
    }
    task.update(overrides)
    return task


def _write_phase_outputs(tmp_path, *, benign_tasks, adversarial_tasks=None):
    phase_1 = tmp_path / "phase_1"
    phase_1.mkdir(parents=True)
    (phase_1 / "benign_tasks.json").write_text(json.dumps(benign_tasks))
    if adversarial_tasks is not None:
        phase_2 = tmp_path / "phase_2"
        phase_2.mkdir(parents=True)
        (phase_2 / "adversarial_tasks.json").write_text(json.dumps(adversarial_tasks))


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    return tmp_path


def test_classify_origin_mode_a():
    assert phase_3_benign._classify_origin(_mode_a_task()) == "mode_a"


def test_classify_origin_mode_b_editor_calls():
    assert phase_3_benign._classify_origin(_mode_b_task()) == "mode_b"


def test_validate_benign_task_passes_on_well_formed_contract():
    assert phase_3_benign._validate_benign_task(_mode_a_task()) == []


def test_validate_benign_task_flags_empty_reward_function():
    task = _mode_a_task(reward_function={})
    errors = phase_3_benign._validate_benign_task(task)
    assert any("reward_function" in err for err in errors)


def test_validate_benign_task_flags_missing_start_urls():
    task = _mode_a_task(start_urls=[])
    errors = phase_3_benign._validate_benign_task(task)
    assert any("start_urls" in err for err in errors)


def test_validate_benign_task_flags_malformed_data_seed():
    task = _mode_b_task(data_seed={"mechanism": "editor"})
    errors = phase_3_benign._validate_benign_task(task)
    assert any("data_seed" in err for err in errors)


def test_adversarial_errors_flags_unknown_benign_task_id():
    errors = phase_3_benign._adversarial_task_errors(
        _adversarial_task("adv-1", "missing"),
        benign_by_id={"task-a1": _mode_a_task()},
    )
    assert any("unknown benign_task_id" in err for err in errors)


def test_adversarial_errors_flags_missing_adversarial_reward():
    adv = _adversarial_task("adv-1", "task-a1")
    adv["reward_function"] = {}
    errors = phase_3_benign._adversarial_task_errors(
        adv,
        benign_by_id={"task-a1": _mode_a_task()},
    )
    assert any("adversarial_reward" in err for err in errors)


@pytest.mark.asyncio
async def test_run_writes_contracts_json_with_origins(state_dir):
    benign = [_mode_a_task("task-a1"), _mode_b_task("task-b1")]
    adversarial = [_adversarial_task("adv-1", "task-a1")]
    _write_phase_outputs(state_dir, benign_tasks=benign, adversarial_tasks=adversarial)

    rc = await phase_3_benign.run(Namespace(sites=None))
    assert rc == 0

    contracts = json.loads((state_dir / "phase_3" / "contracts.json").read_text())
    by_id = {entry["id"]: entry for entry in contracts}
    assert by_id["task-a1"]["origin"] == "mode_a"
    assert by_id["task-a1"]["validity_status"] == "valid"
    assert by_id["task-b1"]["origin"] == "mode_b"
    assert by_id["task-b1"]["validity_status"] == "valid"


@pytest.mark.asyncio
async def test_run_marks_invalid_contract_and_continues(state_dir):
    broken = _mode_a_task("task-broken", reward_function={})
    _write_phase_outputs(state_dir, benign_tasks=[broken, _mode_a_task("task-ok")])

    rc = await phase_3_benign.run(Namespace(sites=None))
    assert rc == 0

    contracts = json.loads((state_dir / "phase_3" / "contracts.json").read_text())
    by_id = {entry["id"]: entry for entry in contracts}
    assert by_id["task-broken"]["validity_status"] == "invalid"
    assert by_id["task-broken"]["validity_errors"]
    assert by_id["task-ok"]["validity_status"] == "valid"


@pytest.mark.asyncio
async def test_run_without_phase_1_returns_error(state_dir):
    rc = await phase_3_benign.run(Namespace(sites=None))
    assert rc == 1


@pytest.mark.asyncio
async def test_run_honors_sites_filter(state_dir):
    benign = [_mode_a_task("task-a1"), _mode_b_task("task-b1")]
    _write_phase_outputs(state_dir, benign_tasks=benign)

    rc = await phase_3_benign.run(Namespace(sites="shopping"))
    assert rc == 0

    contracts = json.loads((state_dir / "phase_3" / "contracts.json").read_text())
    assert [entry["id"] for entry in contracts] == ["task-a1"]


@pytest.mark.asyncio
async def test_run_reports_adversarial_reference_errors(state_dir, caplog):
    benign = [_mode_a_task("task-a1")]
    adversarial = [_adversarial_task("adv-1", "unknown-id")]
    _write_phase_outputs(state_dir, benign_tasks=benign, adversarial_tasks=adversarial)

    with caplog.at_level("WARNING"):
        rc = await phase_3_benign.run(Namespace(sites=None))
    assert rc == 0
    assert any("adversarial task(s)" in message for message in caplog.messages)
