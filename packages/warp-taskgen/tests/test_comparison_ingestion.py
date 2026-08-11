from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from worldsim.benchmark_capabilities import infer_benchmark_from_metadata
from worldsim.comparison_ingestion import (
    ComparisonRecord,
    ingest_comparison_payload,
    write_comparison_result,
)
from worldsim.runners import agentlab as agentlab_runner

_OUTCOME_MODES = {
    "wasp": "resistance",
    "stwebagentbench": "capability",
    "doomarena": "attack_success",
}


def _payload(
    task_id: str = "comparison-1",
    benchmark_name: str = "wasp",
    **overrides: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "task_id": task_id,
        "status": "success",
        "passed": True,
        "reward": 1.0,
        "steps": 3,
        "elapsed": 1.5,
        "error": None,
        "summary_info": {
            "n_steps": 3,
            "cum_reward": 1.0,
            "err_msg": None,
            "terminated": True,
            "truncated": False,
        },
        "artifacts": {"summary_info": "summary_info.json"},
        "versions": {"agentlab": "test"},
        "model": {"name": "demo"},
        "benchmark_config": {
            "status": "applied",
            "benchmark_name": benchmark_name,
        },
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize("benchmark_name", tuple(_OUTCOME_MODES))
def test_ingest_comparison_payload_uses_catalog_outcome_mode(benchmark_name: str):
    record = ingest_comparison_payload(
        {"id": "comparison-1", "benchmark_name": benchmark_name},
        _payload(benchmark_name=benchmark_name),
    )

    assert isinstance(record, ComparisonRecord)
    assert record.result_kind == "comparison"
    assert record.benchmark_name == benchmark_name
    assert record.comparison_outcome_mode == _OUTCOME_MODES[benchmark_name]
    assert record.evaluator_authority == "comparison_runner"
    assert record.native_reward == 1.0
    assert record.to_dict()["native_reward"] == 1.0
    assert "reward" not in record.to_dict()


def test_comparison_record_freezes_nested_artifacts_and_writes_atomic_json(tmp_path: Path):
    record = ingest_comparison_payload(
        {"id": "comparison-1", "benchmark_name": "wasp"},
        _payload(),
        artifact_dir=tmp_path,
    )

    with pytest.raises(TypeError):
        record.artifact_refs["new"] = "bad"  # type: ignore[index]
    with pytest.raises(ValueError, match="nested WARP-only"):
        replace(record, provenance={"asr": 1})

    output_path = tmp_path / "comparison_result.json"
    write_comparison_result(output_path, record)
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["result_kind"] == "comparison"
    assert data["native_reward"] == 1.0
    assert data["provenance"]["artifact_dir"] == str(tmp_path)
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"status": "running"}, "status"),
        ({"passed": 1}, "passed"),
        ({"steps": -1}, "steps"),
        ({"elapsed": float("nan")}, "elapsed"),
        ({"artifacts": []}, "artifacts"),
        ({"task_id": "other"}, "identity"),
        ({"comparison_outcome_mode": "attack_success"}, "outcome mode"),
        ({"adversarial_passed": False}, "WARP-only"),
        ({"asr": None}, "WARP-only"),
        (
            {
                "benchmark_config": {
                    "status": "applied",
                    "benchmark_name": "wasp",
                    "warp_evaluation": {"asr": None},
                }
            },
            "nested WARP-only",
        ),
        ({"reward": None}, "reward"),
        ({"summary_info": {}}, "summary_info"),
        ({"model": {"bad": object()}}, "JSON-shaped"),
        ({"status": "failure", "passed": True}, "status and passed"),
        ({"reward": 0.5}, "summary_info.cum_reward"),
        (
            {
                "summary_info": {
                    "n_steps": 3,
                    "cum_reward": 1.0,
                    "err_msg": "unexpected",
                    "terminated": True,
                    "truncated": False,
                }
            },
            "summary_info.err_msg",
        ),
        (
            {"benchmark_config": {"status": "applied", "benchmark_name": "doomarena"}},
            "mixed benchmark metadata",
        ),
    ],
)
def test_ingest_comparison_payload_fails_closed(overrides: dict[str, object], message: str):
    with pytest.raises(ValueError, match=message):
        ingest_comparison_payload(
            {"id": "comparison-1", "benchmark_name": "wasp"},
            _payload(**overrides),
        )


def test_ingest_rejects_warp_and_comparison_benchmark_mix():
    with pytest.raises(ValueError, match="mixed benchmark metadata"):
        ingest_comparison_payload(
            {"id": "comparison-1", "benchmark_name": "wasp"},
            _payload(benchmark_name="webarena_verified"),
        )


@pytest.mark.parametrize(
    "field",
    ("summary_info", "artifacts", "versions", "model", "benchmark_config"),
)
def test_ingest_requires_pinned_native_sections(field: str):
    payload = _payload()
    payload.pop(field)

    with pytest.raises(ValueError, match=field):
        ingest_comparison_payload(
            {"id": "comparison-1", "benchmark_name": "wasp"},
            payload,
        )


def test_ingest_requires_native_reward_evidence():
    payload = _payload()
    payload.pop("reward")

    with pytest.raises(ValueError, match="reward or native_reward is required"):
        ingest_comparison_payload(
            {"id": "comparison-1", "benchmark_name": "wasp"},
            payload,
        )


@pytest.mark.parametrize("value", (None, "", "   ", []))
def test_benchmark_metadata_rejects_explicit_blank(value: object):
    with pytest.raises(ValueError, match="benchmark metadata is empty"):
        infer_benchmark_from_metadata(({"benchmark_name": value},))

    with pytest.raises(ValueError, match="does not support capability"):
        ingest_comparison_payload(
            {"id": "comparison-1", "benchmark_name": "webarena_verified"},
            _payload(benchmark_name="webarena_verified"),
        )


def test_ingest_accepts_native_error_payload_without_reclassifying_it():
    payload = _payload(
        status="error",
        passed=False,
        reward=0.0,
        error="native failure",
        summary_info={
            "n_steps": 3,
            "cum_reward": 0.0,
            "err_msg": "native failure",
            "terminated": False,
            "truncated": True,
        },
    )

    record = ingest_comparison_payload(
        {"id": "comparison-1", "benchmark_name": "wasp"},
        payload,
    )

    assert record.status == "error"
    assert record.passed is False
    assert record.error == "native failure"
    assert record.native_reward == 0.0


def test_agentlab_comparison_runner_writes_separate_envelope(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}
    stale_warp_result = tmp_path / "result.json"
    stale_warp_result.write_text('{"stale": true}', encoding="utf-8")

    async def fake_reset(task):
        captured["reset"] = task["id"]

    def fake_sidecar(request, task_dir, subcommand="run", timeout=None):
        captured["request"] = request
        captured["subcommand"] = subcommand
        return _payload(task_id="comparison-1")

    monkeypatch.setattr(agentlab_runner, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(agentlab_runner, "_run_sidecar_request", fake_sidecar)

    runner = agentlab_runner.make_task_runner(max_steps=4)
    result = asyncio.run(
        runner(
            {
                "id": "comparison-1",
                "site": "gitlab",
                "sites": ["gitlab"],
                "benchmark_name": "wasp",
                "agentlab_task_name": "wasp.comparison-1",
                "data_seed": {"mechanism": "none"},
            },
            agentlab_runner.AgentLabAgentWrapper(model="demo-model"),
            SimpleNamespace(
                site_name="gitlab",
                site_url="http://gitlab.test",
                model_dump=lambda: {
                    "site_name": "gitlab",
                    "site_url": "http://gitlab.test",
                },
            ),
            tmp_path,
        )
    )

    assert result["result_kind"] == "comparison"
    assert result["native_reward"] == 1.0
    assert captured["subcommand"] == "run"
    assert (tmp_path / "comparison_result.json").is_file()
    assert not stale_warp_result.exists()


def test_agentlab_comparison_runner_uses_nonempty_task_id_fallback(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    async def fake_reset(_task):
        return None

    def fake_sidecar(request, _task_dir, subcommand="run", timeout=None):
        captured["task_id"] = request["task_id"]
        return _payload(task_id="comparison-1")

    monkeypatch.setattr(agentlab_runner, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(agentlab_runner, "_run_sidecar_request", fake_sidecar)

    result = asyncio.run(
        agentlab_runner.make_task_runner()(
            {
                "id": "",
                "task_id": "comparison-1",
                "benchmark_name": "wasp",
                "agentlab_task_name": "wasp.comparison-1",
                "data_seed": {"mechanism": "none"},
            },
            agentlab_runner.AgentLabAgentWrapper(model="demo-model"),
            SimpleNamespace(
                site_name="gitlab",
                site_url="http://gitlab.test",
                model_dump=lambda: {
                    "site_name": "gitlab",
                    "site_url": "http://gitlab.test",
                },
            ),
            tmp_path,
        )
    )

    assert captured["task_id"] == "comparison-1"
    assert result["task_id"] == "comparison-1"


def test_agentlab_comparison_runner_rejects_conflicting_task_identity(tmp_path: Path):
    runner = agentlab_runner.make_task_runner()

    with pytest.raises(ValueError, match="conflicting id/task_id"):
        asyncio.run(
            runner(
                {
                    "id": "comparison-1",
                    "task_id": "comparison-2",
                    "benchmark_name": "wasp",
                    "agentlab_task_name": "wasp.comparison-1",
                },
                agentlab_runner.AgentLabAgentWrapper(model="demo-model"),
                SimpleNamespace(
                    site_name="gitlab",
                    site_url="http://gitlab.test",
                    model_dump=lambda: {
                        "site_name": "gitlab",
                        "site_url": "http://gitlab.test",
                    },
                ),
                tmp_path,
            )
        )


@pytest.mark.parametrize("malformed_id", (True, 1.5, {"id": 1}, ["id"]))
def test_agentlab_comparison_runner_rejects_malformed_task_identity(
    malformed_id: object,
    tmp_path: Path,
):
    runner = agentlab_runner.make_task_runner()

    with pytest.raises(ValueError, match="non-empty string or integer"):
        asyncio.run(
            runner(
                {
                    "id": malformed_id,
                    "benchmark_name": "wasp",
                    "agentlab_task_name": "wasp.comparison-1",
                },
                agentlab_runner.AgentLabAgentWrapper(model="demo-model"),
                SimpleNamespace(
                    site_name="gitlab",
                    site_url="http://gitlab.test",
                    model_dump=lambda: {
                        "site_name": "gitlab",
                        "site_url": "http://gitlab.test",
                    },
                ),
                tmp_path,
            )
        )


def test_agentlab_webarena_legacy_runner_still_allows_explicit_task_without_id(
    monkeypatch,
    tmp_path: Path,
):
    async def fake_reset(_task):
        return None

    def fake_sidecar(_request, _task_dir, subcommand="run", timeout=None):
        return _payload(task_id="unknown", benchmark_name="webarena_verified")

    monkeypatch.setattr(agentlab_runner, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(agentlab_runner, "_run_sidecar_request", fake_sidecar)

    runner = agentlab_runner.make_task_runner()
    result = asyncio.run(
        runner(
            {
                "benchmark_name": "webarena_verified",
                "agentlab_task_name": "webarena_verified.1",
                "data_seed": {"mechanism": "none"},
            },
            agentlab_runner.AgentLabAgentWrapper(model="demo-model"),
            SimpleNamespace(
                site_name="gitlab",
                site_url="http://gitlab.test",
                model_dump=lambda: {
                    "site_name": "gitlab",
                    "site_url": "http://gitlab.test",
                },
            ),
            tmp_path,
        )
    )

    assert result["task_id"] == "unknown"
    assert (tmp_path / "result.json").is_file()
    assert not (tmp_path / "comparison_result.json").exists()


def test_agentlab_runner_clears_stale_comparison_envelope_before_failed_ingestion(
    monkeypatch,
    tmp_path: Path,
):
    stale_path = tmp_path / "comparison_result.json"
    stale_path.write_text('{"stale": true}', encoding="utf-8")

    async def fake_reset(_task):
        return None

    def fake_sidecar(_request, _task_dir, subcommand="run", timeout=None):
        payload = _payload()
        payload.pop("summary_info")
        return payload

    monkeypatch.setattr(agentlab_runner, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(agentlab_runner, "_run_sidecar_request", fake_sidecar)

    runner = agentlab_runner.make_task_runner()
    with pytest.raises(ValueError, match="summary_info"):
        asyncio.run(
            runner(
                {
                    "id": "comparison-1",
                    "benchmark_name": "wasp",
                    "agentlab_task_name": "wasp.comparison-1",
                    "data_seed": {"mechanism": "none"},
                },
                agentlab_runner.AgentLabAgentWrapper(model="demo-model"),
                SimpleNamespace(
                    site_name="gitlab",
                    site_url="http://gitlab.test",
                    model_dump=lambda: {
                        "site_name": "gitlab",
                        "site_url": "http://gitlab.test",
                    },
                ),
                tmp_path,
            )
        )

    assert not stale_path.exists()
