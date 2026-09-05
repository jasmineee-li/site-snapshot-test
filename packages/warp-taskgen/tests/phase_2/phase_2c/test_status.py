"""Phase 2c feasibility checkpoint status regressions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from warp_taskgen.cli.status import build_status_payload, format_status_payload
from warp_taskgen.phase_2.phase_2c.checkpoints import (
    VERIFIER_VERSION,
    checkpoint_context,
    write_checkpoint,
)
from warp_taskgen.phase_2.phase_2c.fingerprints import _host_fingerprint, _task_content_hash
from warp_taskgen.run_control_status import build_run_control_projection
from warp_taskgen.run_definition_contracts import RunDefinition
from warp_taskgen.run_transition import resolve_run_request


def _task() -> dict[str, object]:
    return {
        "id": "adv-gitlab-1",
        "benchmark": "webarena_verified",
        "site": "gitlab",
        "adversarial_data_seed": {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "gitlab",
                    "method": "create_issue",
                    "args": {"project": "group/project", "description": "{{PAYLOAD_TEXT}}"},
                }
            ],
        },
    }


def _instances() -> list[dict[str, object]]:
    return [
        {
            "site_name": "gitlab",
            "site_url": "https://gitlab.example",
            "benchmark_name": "webarena_verified",
        }
    ]


def _definition(instances_path: Path) -> RunDefinition:
    transition = resolve_run_request(
        {
            "sandbox_model": "model-a",
            "feasibility_instances": str(instances_path),
            "feasibility_retry_count": 0,
            "feasibility_ttl_hours": None,
            "force_reverify": False,
            "skip_feasibility": False,
        },
        existing_state=None,
        new_run_id="run-phase-2c-status",
    )
    assert transition.definition is not None
    return transition.definition


def _state(root: Path, instances_path: Path) -> tuple[dict[str, object], RunDefinition]:
    definition = _definition(instances_path)
    state: dict[str, object] = {
        "step": "phase_2",
        "status": "running",
        "phase_2_stage": "feasibility",
        "logs_dir": str(root),
        "run_definition": definition.to_dict(),
        "feasibility_instances": str(instances_path),
        "feasibility_retry_count": 0,
        "feasibility_ttl_hours": None,
        "force_reverify": False,
        "skip_feasibility": False,
        "sites": None,
    }
    (root / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")
    return state, definition


def _write_checkpoint_fixture(
    root: Path,
    *,
    task: dict[str, object],
    instances: list[dict[str, object]],
    definition: RunDefinition,
    outcome: str,
) -> Path:
    normalized_instances = [
        {**instance, "benchmark": "webarena_verified"} for instance in instances
    ]
    topology = _host_fingerprint("instances.smoke.json", normalized_instances)
    context = checkpoint_context(
        run_id=definition.run_id,
        definition_digest=definition.definition_digest,
        task=task,
        task_content_hash=_task_content_hash(
            task["adversarial_data_seed"]["editor_calls"],  # type: ignore[index]
        ),
        topology_fingerprint=topology,
    )
    assert context is not None
    return write_checkpoint(
        root / "phase_2" / "feasibility_checkpoints",
        context=context,
        result={**task, "feasibility": {"status": outcome}},
    )


def _redigest(path: Path, payload: dict[str, object]) -> None:
    without_digest = {key: value for key, value in payload.items() if key != "checkpoint_digest"}
    payload["checkpoint_digest"] = hashlib.sha256(
        json.dumps(
            without_digest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_phase_2c_status_reports_unknown_counts_before_checkpoint_inspection(
    tmp_path: Path,
) -> None:
    instances = _instances()
    instances_path = tmp_path / "instances.smoke.json"
    instances_path.write_text(
        json.dumps({"benchmark_name": "webarena_verified", "instances": instances}),
        encoding="utf-8",
    )
    state, definition = _state(tmp_path, instances_path)
    task = _task()
    tasks_path = tmp_path / "phase_2" / "adversarial_tasks.json"
    tasks_path.parent.mkdir(parents=True)
    tasks_path.write_text(json.dumps([task]), encoding="utf-8")

    topology = _host_fingerprint(
        instances_path.name,
        [{**instance, "benchmark": "webarena_verified"} for instance in instances],
    )
    task["feasibility"] = {"status": "verified", "host_fingerprint": topology}
    content_hash = _task_content_hash(
        task["adversarial_data_seed"]["editor_calls"],  # type: ignore[index]
        exposure_contract=task.get("exposure_contract"),
    )
    context = checkpoint_context(
        run_id=definition.run_id,
        definition_digest=definition.definition_digest,
        task=task,
        task_content_hash=content_hash,
        topology_fingerprint=topology,
    )
    assert context is not None
    checkpoint_dir = tmp_path / "phase_2" / "feasibility_checkpoints"
    write_checkpoint(
        checkpoint_dir,
        context=context,
        result={**task, "feasibility": {"status": "verified"}},
    )

    projection = build_run_control_projection(tmp_path, state)

    # This is the red acceptance case: durable task evidence exists but the
    # pre-feature status surface only exposes absent state counters.
    assert projection["feasibility_checkpoint_inspection"]["status"] == "inspected"
    assert projection["feasibility_checkpoint_inspection"]["compatible_count"] == 1


def test_phase_2c_status_matches_production_fingerprint_with_relative_benchmark_root(
    tmp_path: Path,
) -> None:
    instances = _instances()
    instances_path = tmp_path / "instances.smoke.json"
    instances_path.write_text(
        json.dumps(
            {
                "benchmark_name": "webarena_verified",
                "benchmark_codebase": "benchmarks/webarena",
                "instances": instances,
            }
        ),
        encoding="utf-8",
    )
    state, definition = _state(tmp_path, instances_path)
    task = _task()
    tasks_path = tmp_path / "phase_2" / "adversarial_tasks.json"
    tasks_path.parent.mkdir(parents=True)
    tasks_path.write_text(json.dumps([task]), encoding="utf-8")
    _write_checkpoint_fixture(
        tmp_path,
        task=task,
        instances=instances,
        definition=definition,
        outcome="verified",
    )

    inspection = build_run_control_projection(tmp_path, state)["feasibility_checkpoint_inspection"]

    assert inspection["status"] == "inspected"
    assert inspection["compatible_count"] == 1


def test_phase_2c_status_separates_verified_and_infeasible_outcomes(tmp_path: Path) -> None:
    instances = _instances()
    instances_path = tmp_path / "instances.smoke.json"
    instances_path.write_text(
        json.dumps({"benchmark_name": "webarena_verified", "instances": instances}),
        encoding="utf-8",
    )
    state, definition = _state(tmp_path, instances_path)
    tasks = [_task(), {**_task(), "id": "adv-gitlab-2"}]
    tasks_path = tmp_path / "phase_2" / "adversarial_tasks.json"
    tasks_path.parent.mkdir(parents=True)
    tasks_path.write_text(json.dumps(tasks), encoding="utf-8")
    _write_checkpoint_fixture(
        tmp_path,
        task=tasks[0],
        instances=instances,
        definition=definition,
        outcome="verified",
    )
    _write_checkpoint_fixture(
        tmp_path,
        task=tasks[1],
        instances=instances,
        definition=definition,
        outcome="infeasible",
    )

    inspection = build_run_control_projection(tmp_path, state)["feasibility_checkpoint_inspection"]

    assert inspection["compatible_count"] == 2
    assert inspection["verified_count"] == 1
    assert inspection["infeasible_count"] == 1
    assert inspection["compatible_verified_count"] == 1
    assert inspection["compatible_infeasible_count"] == 1


def test_phase_2c_status_uses_terminal_verified_and_infeasible_sources(
    tmp_path: Path,
) -> None:
    instances = _instances()
    instances_path = tmp_path / "instances.smoke.json"
    instances_path.write_text(
        json.dumps({"benchmark_name": "webarena_verified", "instances": instances}),
        encoding="utf-8",
    )
    state, definition = _state(tmp_path, instances_path)
    verified = _task()
    infeasible = {**_task(), "id": "adv-gitlab-infeasible"}
    tasks_path = tmp_path / "phase_2" / "adversarial_tasks.json"
    tasks_path.parent.mkdir(parents=True)
    tasks_path.write_text(json.dumps([verified]), encoding="utf-8")
    infeasible_path = tasks_path.with_name("adversarial_tasks.infeasible.json")
    infeasible_path.write_text(json.dumps([infeasible]), encoding="utf-8")
    _write_checkpoint_fixture(
        tmp_path,
        task=verified,
        instances=instances,
        definition=definition,
        outcome="verified",
    )
    _write_checkpoint_fixture(
        tmp_path,
        task=infeasible,
        instances=instances,
        definition=definition,
        outcome="infeasible",
    )
    state.update(
        status="complete",
        phase_2_stage="complete",
        feasibility_completed_at="2026-08-31T00:00:00+00:00",
    )
    (tmp_path / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")

    projection = build_run_control_projection(tmp_path, state)

    inspection = projection["feasibility_checkpoint_inspection"]
    assert inspection["status"] == "inspected"
    assert inspection["expected_count"] == 2
    assert inspection["compatible_count"] == 2
    assert inspection["verified_count"] == 1
    assert inspection["infeasible_count"] == 1


def test_phase_2c_status_requires_terminal_infeasible_sidecar(tmp_path: Path) -> None:
    instances_path = tmp_path / "instances.smoke.json"
    instances_path.write_text(
        json.dumps({"benchmark_name": "webarena_verified", "instances": _instances()}),
        encoding="utf-8",
    )
    state, _ = _state(tmp_path, instances_path)
    tasks_path = tmp_path / "phase_2" / "adversarial_tasks.json"
    tasks_path.parent.mkdir(parents=True)
    tasks_path.write_text(json.dumps([_task()]), encoding="utf-8")
    state.update(
        status="complete",
        phase_2_stage="complete",
        feasibility_completed_at="2026-08-31T00:00:00+00:00",
    )
    (tmp_path / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")

    inspection = build_run_control_projection(tmp_path, state)["feasibility_checkpoint_inspection"]

    assert inspection["status"] == "not_inspected"
    assert inspection["expected_count"] is None
    assert inspection["reason_code"] == "feasibility_infeasible_missing"


def test_phase_2c_status_fails_closed_for_missing_tasks_or_relative_instances(
    tmp_path: Path,
) -> None:
    instances_path = tmp_path / "instances.smoke.json"
    instances_path.write_text(
        json.dumps({"benchmark_name": "webarena_verified", "instances": _instances()}),
        encoding="utf-8",
    )
    state, _ = _state(tmp_path, instances_path)
    missing = build_run_control_projection(tmp_path, state)["feasibility_checkpoint_inspection"]
    assert missing["status"] == "not_inspected"
    assert missing["expected_count"] is None
    assert missing["reason_code"] == "feasibility_tasks_missing"

    state["feasibility_instances"] = instances_path.name
    tasks_path = tmp_path / "phase_2" / "adversarial_tasks.json"
    tasks_path.parent.mkdir(parents=True)
    tasks_path.write_text(json.dumps([_task()]), encoding="utf-8")
    relative = build_run_control_projection(tmp_path, state)["feasibility_checkpoint_inspection"]
    assert relative["status"] == "not_inspected"
    assert relative["reason_code"] == "feasibility_instances_unresolved"


def test_phase_2c_status_classifies_drift_and_malformed_envelopes(tmp_path: Path) -> None:
    instances = _instances()
    instances_path = tmp_path / "instances.smoke.json"
    instances_path.write_text(
        json.dumps({"benchmark_name": "webarena_verified", "instances": instances}),
        encoding="utf-8",
    )
    state, definition = _state(tmp_path, instances_path)
    task = _task()
    tasks_path = tmp_path / "phase_2" / "adversarial_tasks.json"
    tasks_path.parent.mkdir(parents=True)
    tasks_path.write_text(json.dumps([task]), encoding="utf-8")
    checkpoint = _write_checkpoint_fixture(
        tmp_path,
        task=task,
        instances=instances,
        definition=definition,
        outcome="verified",
    )

    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["run_id"] = "other-run"
    _redigest(checkpoint, payload)
    inspection = build_run_control_projection(tmp_path, state)["feasibility_checkpoint_inspection"]
    assert inspection["stale_count"] == 1
    assert inspection["units"][0]["reason_code"] == "checkpoint_run_or_definition_mismatch"

    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["checkpoint_digest"] = "broken"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    inspection = build_run_control_projection(tmp_path, state)["feasibility_checkpoint_inspection"]
    assert inspection["malformed_count"] == 1
    assert inspection["units"][0]["reason_code"] == "checkpoint_digest_invalid"


def test_phase_2c_status_classifies_topology_version_cleanup_and_outcome_drift(
    tmp_path: Path,
) -> None:
    instances = _instances()
    instances_path = tmp_path / "instances.smoke.json"
    instances_path.write_text(
        json.dumps({"benchmark_name": "webarena_verified", "instances": instances}),
        encoding="utf-8",
    )
    state, definition = _state(tmp_path, instances_path)
    task = _task()
    tasks_path = tmp_path / "phase_2" / "adversarial_tasks.json"
    tasks_path.parent.mkdir(parents=True)
    tasks_path.write_text(json.dumps([task]), encoding="utf-8")

    checkpoint = _write_checkpoint_fixture(
        tmp_path,
        task=task,
        instances=instances,
        definition=definition,
        outcome="verified",
    )
    changed_instances = [{**instances[0], "site_url": "https://gitlab-new.example"}]
    instances_path.write_text(
        json.dumps({"benchmark_name": "webarena_verified", "instances": changed_instances}),
        encoding="utf-8",
    )
    inspection = build_run_control_projection(tmp_path, state)["feasibility_checkpoint_inspection"]
    assert inspection["stale_count"] == 1
    assert inspection["units"][0]["reason_code"] == "checkpoint_topology_drift"

    # Restore topology so the remaining envelope mutations are isolated.
    instances_path.write_text(
        json.dumps({"benchmark_name": "webarena_verified", "instances": instances}),
        encoding="utf-8",
    )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["verifier_version"] = "phase_2c-verifier-old"
    _redigest(checkpoint, payload)
    inspection = build_run_control_projection(tmp_path, state)["feasibility_checkpoint_inspection"]
    assert inspection["stale_count"] == 1
    assert inspection["units"][0]["reason_code"] == "checkpoint_verifier_version_drift"

    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["verifier_version"] = VERIFIER_VERSION
    payload["cleanup_completed"] = False
    _redigest(checkpoint, payload)
    inspection = build_run_control_projection(tmp_path, state)["feasibility_checkpoint_inspection"]
    assert inspection["malformed_count"] == 1
    assert inspection["units"][0]["reason_code"] == "checkpoint_cleanup_incomplete"

    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["verifier_version"] = VERIFIER_VERSION
    payload["cleanup_completed"] = True
    payload["result"]["feasibility"]["status"] = "poisoned"  # type: ignore[index]
    payload["work_unit"]["outcome"] = "poisoned"  # type: ignore[index]
    _redigest(checkpoint, payload)
    inspection = build_run_control_projection(tmp_path, state)["feasibility_checkpoint_inspection"]
    assert inspection["malformed_count"] == 1
    assert inspection["units"][0]["reason_code"] == "checkpoint_invalid_outcome"


def test_phase_2c_status_marks_expired_and_forced_checkpoints_stale(tmp_path: Path) -> None:
    instances = _instances()
    instances_path = tmp_path / "instances.smoke.json"
    instances_path.write_text(
        json.dumps({"benchmark_name": "webarena_verified", "instances": instances}),
        encoding="utf-8",
    )
    state, definition = _state(tmp_path, instances_path)
    task = _task()
    tasks_path = tmp_path / "phase_2" / "adversarial_tasks.json"
    tasks_path.parent.mkdir(parents=True)
    tasks_path.write_text(json.dumps([task]), encoding="utf-8")
    checkpoint = _write_checkpoint_fixture(
        tmp_path,
        task=task,
        instances=instances,
        definition=definition,
        outcome="verified",
    )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["completed_at"] = "2000-01-01T00:00:00Z"
    _redigest(checkpoint, payload)
    state["feasibility_ttl_hours"] = 1
    inspection = build_run_control_projection(tmp_path, state)["feasibility_checkpoint_inspection"]
    assert inspection["stale_count"] == 1
    assert inspection["units"][0]["reason_code"] == "checkpoint_ttl_expired"

    state["feasibility_ttl_hours"] = None
    state["force_reverify"] = True
    inspection = build_run_control_projection(tmp_path, state)["feasibility_checkpoint_inspection"]
    assert inspection["stale_count"] == 1
    assert inspection["units"][0]["reason_code"] == "checkpoint_reverify_required"


def test_phase_2c_status_skip_is_not_inspected_and_preserves_files(tmp_path: Path) -> None:
    instances_path = tmp_path / "instances.smoke.json"
    instances_path.write_text(
        json.dumps({"benchmark_name": "webarena_verified", "instances": _instances()}),
        encoding="utf-8",
    )
    state, _ = _state(tmp_path, instances_path)
    state["skip_feasibility"] = True
    (tmp_path / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")
    tasks_path = tmp_path / "phase_2" / "adversarial_tasks.json"
    tasks_path.parent.mkdir(parents=True)
    tasks_path.write_text(json.dumps([_task()]), encoding="utf-8")
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    payload = build_status_payload(tmp_path)
    inspection = payload["run_control"]["feasibility_checkpoint_inspection"]
    text = format_status_payload(payload)

    assert inspection["status"] == "not_inspected"
    assert inspection["expected_count"] is None
    assert inspection["reason_code"] == "feasibility_skipped"
    assert "Phase 2c feasibility checkpoints: status=not_inspected" in text
    assert {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == before


def test_phase_2c_status_applies_persisted_site_filter(tmp_path: Path) -> None:
    instances = _instances()
    instances_path = tmp_path / "instances.smoke.json"
    instances_path.write_text(
        json.dumps({"benchmark_name": "webarena_verified", "instances": instances}),
        encoding="utf-8",
    )
    state, definition = _state(tmp_path, instances_path)
    state["sites"] = "gitlab"
    tasks = [_task(), {**_task(), "id": "adv-reddit-1", "site": "reddit"}]
    tasks_path = tmp_path / "phase_2" / "adversarial_tasks.json"
    tasks_path.parent.mkdir(parents=True)
    tasks_path.write_text(json.dumps(tasks), encoding="utf-8")
    _write_checkpoint_fixture(
        tmp_path,
        task=tasks[0],
        instances=instances,
        definition=definition,
        outcome="verified",
    )

    inspection = build_run_control_projection(tmp_path, state)["feasibility_checkpoint_inspection"]

    assert inspection["expected_count"] == 1
    assert [unit["task_id"] for unit in inspection["units"]] == [tasks[0]["id"]]


def test_phase_2c_status_rejects_malformed_task_seed_shape(tmp_path: Path) -> None:
    instances_path = tmp_path / "instances.smoke.json"
    instances_path.write_text(
        json.dumps({"benchmark_name": "webarena_verified", "instances": _instances()}),
        encoding="utf-8",
    )
    state, _ = _state(tmp_path, instances_path)
    tasks_path = tmp_path / "phase_2" / "adversarial_tasks.json"
    tasks_path.parent.mkdir(parents=True)
    malformed = _task()
    malformed["adversarial_data_seed"] = {"editor_calls": "not-a-list"}
    tasks_path.write_text(json.dumps([malformed]), encoding="utf-8")

    inspection = build_run_control_projection(tmp_path, state)["feasibility_checkpoint_inspection"]

    assert inspection["status"] == "not_inspected"
    assert inspection["expected_count"] is None
    assert inspection["reason_code"] == "feasibility_tasks_invalid"
