from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from worldsim.phase_2.phase_2c import _impl
from worldsim.phase_2.phase_2c.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointValidationError,
    checkpoint_context,
    checkpoint_is_fresh,
    checkpoint_path,
    load_checkpoint,
    task_fingerprint,
    write_checkpoint,
)


def _task() -> dict[str, object]:
    return {
        "id": "task/one",
        "site": "gitlab",
        "adversarial_data_seed": {"editor_calls": [{"method": "create_issue"}]},
    }


def _context(task: dict[str, object], *, topology: str = "topology-a"):
    return checkpoint_context(
        run_id="run-123",
        definition_digest="a" * 64,
        task=task,
        task_content_hash="content-a",
        topology_fingerprint={"instances_digest": topology},
    )


def test_checkpoint_is_atomic_and_bound_to_exact_identity(tmp_path: Path) -> None:
    task = _task()
    context = _context(task)
    assert context is not None

    path = write_checkpoint(
        tmp_path,
        context=context,
        result={**task, "feasibility": {"status": "verified"}},
    )

    assert path == checkpoint_path(tmp_path, "task/one")
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert payload["run_id"] == "run-123"
    assert payload["definition_digest"] == "a" * 64
    assert payload["cleanup_completed"] is True
    assert payload["work_unit"]["seed_applied"] is True
    assert payload["work_unit"]["render_completed"] is False
    assert payload["work_unit"]["reachability_completed"] is False
    assert load_checkpoint(tmp_path, context=context).reason == "compatible"


def test_checkpoint_redacts_agent_context_secrets(tmp_path: Path) -> None:
    task = {
        **_task(),
        "agent_context": {
            "gitlab": {
                "credentials": {"password": "do-not-persist"},
                "user_handles": ["reviewer"],
            },
        },
    }
    context = _context(task)
    assert context is not None
    path = write_checkpoint(
        tmp_path,
        context=context,
        result={**task, "feasibility": {"status": "verified"}},
    )

    payload = json.loads(path.read_text())
    serialized = json.dumps(payload)
    assert "do-not-persist" not in serialized
    assert payload["result"]["agent_context"]["gitlab"]["credentials"]["password"] == ("<redacted>")
    sanitized = json.loads(json.dumps(task))
    sanitized["agent_context"]["gitlab"]["credentials"]["password"] = "<redacted>"
    assert task_fingerprint(task) == task_fingerprint(sanitized)


def test_checkpoint_write_requires_matching_complete_result(tmp_path: Path) -> None:
    task = _task()
    context = _context(task)
    assert context is not None
    with pytest.raises(CheckpointValidationError, match="does not match"):
        write_checkpoint(
            tmp_path,
            context=context,
            result={"id": "other", "feasibility": {"status": "verified"}},
        )
    with pytest.raises(CheckpointValidationError, match="missing feasibility"):
        write_checkpoint(tmp_path, context=context, result=task)


def test_checkpoint_rejects_tamper_and_topology_drift(tmp_path: Path) -> None:
    task = _task()
    context = _context(task)
    assert context is not None
    path = write_checkpoint(
        tmp_path,
        context=context,
        result={**task, "feasibility": {"status": "infeasible"}},
    )
    payload = json.loads(path.read_text())
    payload["result"]["feasibility"]["status"] = "verified"
    path.write_text(json.dumps(payload))
    assert load_checkpoint(tmp_path, context=context).reason == "tampered"

    # Re-write valid evidence, then bind a new topology. The old evidence is
    # intentionally not accepted even when its result is otherwise recent.
    write_checkpoint(
        tmp_path,
        context=context,
        result={**task, "feasibility": {"status": "infeasible"}},
    )
    drifted = _context(task, topology="topology-b")
    assert drifted is not None
    assert load_checkpoint(tmp_path, context=drifted).reason == "topology_drift"


def test_checkpoint_rejects_redigested_unknown_outcome(tmp_path: Path) -> None:
    task = _task()
    context = _context(task)
    assert context is not None
    path = write_checkpoint(
        tmp_path,
        context=context,
        result={**task, "feasibility": {"status": "verified"}},
    )
    payload = json.loads(path.read_text())
    payload["result"]["feasibility"]["status"] = "poisoned"
    payload["work_unit"]["outcome"] = "poisoned"
    without_digest = {key: value for key, value in payload.items() if key != "checkpoint_digest"}
    payload["checkpoint_digest"] = hashlib.sha256(
        json.dumps(
            without_digest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload))

    loaded = load_checkpoint(tmp_path, context=context)
    assert not loaded.reusable
    assert loaded.reason == "malformed: invalid outcome"


def test_phase2c_star_exports_preserve_artifact_failpoints() -> None:
    namespace: dict[str, Any] = {}
    exec("from worldsim.phase_2.phase_2c import *", namespace)
    assert namespace["FAILPOINT_DATASET"] == "phase_2.output.feasibility_dataset"
    assert namespace["FAILPOINT_QUARANTINE"] == "phase_2.output.feasibility_quarantine"
    assert namespace["FAILPOINT_REPORT"] == "phase_2.output.feasibility_report"
    assert namespace["FAILPOINT_DROPPED_SOURCE_DATA"] == (
        "phase_2.output.feasibility_dropped_source_data"
    )


def test_checkpoint_ttl_expires_only_when_requested(tmp_path: Path) -> None:
    task = _task()
    context = _context(task)
    assert context is not None
    path = write_checkpoint(
        tmp_path,
        context=context,
        result={**task, "feasibility": {"status": "verified"}},
    )
    loaded = load_checkpoint(tmp_path, context=context)
    assert loaded.reusable
    assert checkpoint_is_fresh(loaded, ttl_hours=None)
    assert checkpoint_is_fresh(loaded, ttl_hours=1.0)
    assert not checkpoint_is_fresh(loaded, ttl_hours=0.0)

    payload = json.loads(path.read_text())
    payload["completed_at"] = "2000-01-01T00:00:00Z"
    without_digest = {key: value for key, value in payload.items() if key != "checkpoint_digest"}
    payload["checkpoint_digest"] = hashlib.sha256(
        json.dumps(
            without_digest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload))
    expired = load_checkpoint(tmp_path, context=context)
    assert expired.reusable
    assert not checkpoint_is_fresh(expired, ttl_hours=1.0)


def test_task_fingerprint_ignores_only_mutable_feasibility_stanza() -> None:
    task = _task()
    with_result = {**task, "feasibility": {"status": "verified"}}
    assert task_fingerprint(task) == task_fingerprint(with_result)
    changed = {**task, "site": "reddit"}
    assert task_fingerprint(task) != task_fingerprint(changed)


def test_task_fingerprint_ignores_phase2c_reward_probe_annotations() -> None:
    task = {
        **_task(),
        "reward_function": {
            "adversarial_reward": {
                "evaluator": "FinalStateEvaluator",
                "expected": {
                    "state_probe": {
                        "kind": "reddit_comment_contains",
                        "requires_attribution": False,
                    }
                },
            }
        },
    }
    annotated = json.loads(json.dumps(task))
    probe = annotated["reward_function"]["adversarial_reward"]["expected"]["state_probe"]
    probe.update(
        {
            "requires_attribution": True,
            "excluded_comment_ids": ["seed-1"],
            "seed_comment_id": "seed-1",
            "project_id": "42",
            "project_path": "owner/project",
            "issue_iid": "7",
        }
    )
    assert task_fingerprint(task) == task_fingerprint(annotated)

    changed = json.loads(json.dumps(task))
    changed["reward_function"]["adversarial_reward"]["expected"]["state_probe"]["kind"] = (
        "reddit_submission_contains"
    )
    assert task_fingerprint(task) != task_fingerprint(changed)


def test_legacy_context_cannot_be_created() -> None:
    assert (
        checkpoint_context(
            run_id=None,
            definition_digest=None,
            task=_task(),
            task_content_hash="content-a",
            topology_fingerprint={},
        )
        is None
    )


@pytest.mark.asyncio
async def test_verify_reconstructs_verified_outcome_without_seed_after_crash_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete task unit is durable before aggregate artifacts are written."""

    task = {
        "id": "checkpoint-task",
        "site": "gitlab",
        "benchmark": "webarena_verified",
        "adversarial_data_seed": {
            "editor_calls": [{"site": "gitlab", "method": "create_issue", "args": {}}]
        },
    }
    tasks_path = tmp_path / "adversarial_tasks.json"
    tasks_path.write_text(json.dumps([task]))
    instance = {
        "site_name": "gitlab",
        "site_url": "https://gitlab.example",
        "benchmark": "webarena_verified",
    }

    class _Editor:
        @classmethod
        def probe_base_state(cls, _instance: dict[str, Any]) -> None:
            return None

    class _Handle:
        def cleanup(self) -> None:
            raise RuntimeError("cleanup boom")

    calls = {"seed": 0}

    async def apply(_seed: dict[str, Any], _instance: dict[str, Any]):
        calls["seed"] += 1
        return _Handle(), {}

    async def no_preflight(raw: list[dict[str, Any]], **_kwargs: Any):
        return []

    monkeypatch.setenv("WORLDSIM_PHASE_2C_SKIP_RENDER_CHECK", "1")
    monkeypatch.setattr(_impl, "acquire_tokens_for_instances", lambda _instances: [])
    monkeypatch.setattr(
        _impl,
        "EDITOR_REGISTRY",
        {("webarena_verified", "gitlab"): _Editor},
    )
    monkeypatch.setattr(_impl, "_run_preflight_and_filter_raw", no_preflight)
    monkeypatch.setattr(_impl, "apply_data_seed_async", apply)

    checkpoint_dir = tmp_path / "checkpoints"
    first = await _impl.verify_feasibility(
        tasks_path,
        instances=[instance],
        concurrency=1,
        retry_count=0,
        checkpoint_dir=checkpoint_dir,
        run_id="run-123",
        definition_digest="a" * 64,
    )
    assert calls["seed"] == 1
    assert len(first.verified) == 1
    assert first.cleanup_warnings == [
        "task=checkpoint-task cleanup_raised: RuntimeError: cleanup boom"
    ]
    checkpoint_files = list(checkpoint_dir.glob("*.json"))
    assert checkpoint_files
    checkpoint_payload = json.loads(checkpoint_files[0].read_text())
    assert checkpoint_payload["work_unit"]["seed_applied"] is True
    assert checkpoint_payload["work_unit"]["cleanup_completed"] is True
    assert checkpoint_payload["verifier_version"].endswith("-render-disabled")

    async def should_not_seed(*_args: Any, **_kwargs: Any):
        raise AssertionError("compatible checkpoint should be reused")

    monkeypatch.setattr(_impl, "apply_data_seed_async", should_not_seed)
    second = await _impl.verify_feasibility(
        tasks_path,
        instances=[instance],
        concurrency=1,
        retry_count=0,
        checkpoint_dir=checkpoint_dir,
        run_id="run-123",
        definition_digest="a" * 64,
    )
    assert second.reused_checkpoints == 1
    assert second.verified[0]["feasibility"]["status"] == "verified"
    assert second.cleanup_warnings == first.cleanup_warnings
