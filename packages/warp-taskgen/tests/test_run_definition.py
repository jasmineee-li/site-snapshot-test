from __future__ import annotations

import json
from pathlib import Path

import pytest

from warp_taskgen.cli.status import build_status_payload, format_status_payload
from warp_taskgen.run_definition import define_run, plan_resume
from warp_taskgen.run_definition_contracts import RunDefinition
from warp_taskgen.run_transition import resolve_run_request


def _state(tmp_path: Path, **overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        "step": "phase_2",
        "status": "complete",
        "timestamp": "2026-08-11T12:00:00",
        "logs_dir": str(tmp_path),
        "benchmark_name": "webarena_verified",
        "benchmark_path": str(tmp_path / "benchmark"),
        "instances_path": str(tmp_path / "instances.json"),
        "sandbox_model": "sandbox-model",
        "phase_2_text_model": "text-model",
        "agent_model": "agent-model",
        "sites": ["reddit", "gitlab"],
    }
    state.update(overrides)
    return state


def test_define_run_is_deterministic_immutable_and_path_normalized(tmp_path: Path):
    first = define_run(_state(tmp_path, sites=["reddit", "gitlab"]))
    second = define_run(
        _state(
            tmp_path,
            sites=["gitlab", "reddit", "reddit"],
            benchmark_path=tmp_path / "." / "benchmark",
        )
    )

    assert first.definition_digest == second.definition_digest
    assert first.legacy is True
    assert first.run_id is None
    assert first.contributions["pipeline"]["benchmark_path"] == str(
        (tmp_path / "benchmark").resolve()
    )
    with pytest.raises(TypeError):
        first.contributions["pipeline"]["sites"] = ()  # type: ignore[index]

    set_backed = define_run(
        _state(
            tmp_path,
            phase_2a_resolution_signature={"labels": {"gamma", "alpha", "beta"}},
        )
    )
    assert set_backed.contributions["phase_2"]["phase_2a_resolution_signature"] == {
        "labels": ("alpha", "beta", "gamma")
    }


def test_define_run_digest_changes_only_for_allowlisted_semantic_inputs(tmp_path: Path):
    base = define_run(_state(tmp_path, agent_model="one", api_token="secret"))
    secret_changed = define_run(_state(tmp_path, agent_model="one", api_token="other"))
    model_changed = define_run(_state(tmp_path, agent_model="two", api_token="secret"))

    assert base.definition_digest == secret_changed.definition_digest
    assert base.definition_digest != model_changed.definition_digest
    serialized = json.dumps(base.to_dict(), sort_keys=True)
    assert "api_token" not in serialized
    assert "secret" not in serialized


def test_matched_rewrite_baseline_contracts_are_allowlisted_and_projected(tmp_path: Path):
    task = {
        "id": "matched-task",
        "instruction": "perform the task",
        "reward_function": {"adversarial_reward": {"type": "host"}},
        "exposure_contract": {"surface": "gitlab.issue"},
    }
    result = {
        "task_id": "matched-task",
        "final_status": "refused_or_ignored",
        "encounter": {"max_coverage": 0.5},
        "score": {"adversarial": 0},
    }
    base = _state(
        tmp_path,
        phase_4_matched_rewrite_study_condition="tp_guided_vs_ordinary",
        phase_4_matched_rewrite_study_schedule="one_opportunity",
        phase_4_matched_rewrite_study_baseline_task=task,
        phase_4_matched_rewrite_study_baseline_result=result,
        phase_4_matched_rewrite_study_selected_payload={"rendered_payload": "same payload"},
        phase_4_matched_rewrite_study_witness=[{"value": "anchor"}],
        phase_4_matched_rewrite_study_constraints={"preserve_required_anchors": True},
    )
    definition = define_run(base)
    projection = definition.input_projection()
    assert projection["phase_4_matched_rewrite_study_baseline_task"] == task
    assert projection["phase_4_matched_rewrite_study_baseline_result"] == result
    assert projection["phase_4_matched_rewrite_study_selected_payload"] == {
        "rendered_payload": "same payload"
    }
    assert projection["phase_4_matched_rewrite_study_witness"] == [{"value": "anchor"}]
    assert projection["phase_4_matched_rewrite_study_constraints"] == {
        "preserve_required_anchors": True
    }

    changed = define_run(
        {
            **base,
            "phase_4_matched_rewrite_study_baseline_result": {
                **result,
                "final_status": "complied",
            },
        }
    )
    assert changed.definition_digest != definition.definition_digest

    non_study_change = define_run(
        {
            **base,
            "unrelated_matched_rewrite_baseline_result": {"final_status": "complied"},
        }
    )
    assert non_study_change.definition_digest == definition.definition_digest


def test_define_run_redacts_nested_secrets_but_tracks_their_identity(tmp_path: Path):
    first = define_run(
        _state(
            tmp_path,
            phase_2a_resolution_signature={
                "X-Api-Key": "root-api-secret",
                "auth": {
                    "headers": {
                        "Authorization": "Bearer nested-secret",
                        "X-Signature": "first",
                    },
                    "password": "nested-password",
                },
            },
        )
    )
    second = define_run(
        _state(
            tmp_path,
            phase_2a_resolution_signature={
                "X-Api-Key": "changed-root-api-secret",
                "auth": {
                    "headers": {
                        "Authorization": "Bearer changed-secret",
                        "X-Signature": "first",
                    },
                    "password": "changed-password",
                    "Authorization_digest": "raw-secret-under-digest-name",
                    "sensitive-set": {"gamma", "alpha", "beta"},
                },
            },
        )
    )

    serialized = json.dumps(first.to_dict(), sort_keys=True)
    assert "nested-secret" not in serialized
    assert "nested-password" not in serialized
    assert "raw-secret-under-digest-name" not in json.dumps(second.to_dict())
    assert "root-api-secret" not in serialized
    assert first.definition_digest != second.definition_digest

    assert define_run({"run_definition": second.to_dict()}) == second

    ambiguous_hex = define_run(
        _state(
            tmp_path,
            phase_2a_resolution_signature={"Authorization": {"value_sha256": "a" * 64}},
        )
    )
    assert "a" * 64 not in json.dumps(ambiguous_hex.to_dict())


def test_define_run_reads_state_path_and_nested_envelope(tmp_path: Path):
    state = _state(tmp_path)
    state_path = tmp_path / "pipeline_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    from_mapping = define_run(state)
    assert define_run(tmp_path).definition_digest == from_mapping.definition_digest
    assert define_run(state_path).definition_digest == from_mapping.definition_digest
    assert define_run({"run_definition": from_mapping.to_dict()}) == from_mapping


def test_define_run_validates_persisted_definition_metadata(tmp_path: Path):
    state = _state(
        tmp_path,
        run_id="run-1",
        source_run_id="run-0",
        run_definition_schema_version=1,
    )
    definition = define_run(state)

    assert definition.legacy is False
    assert definition.run_id == "run-1"
    assert definition.source_run_id == "run-0"

    with pytest.raises(ValueError, match="persisted definition digest"):
        define_run({**state, "definition_digest": "0" * 64})
    with pytest.raises(ValueError, match="schema_version"):
        define_run({**state, "run_definition_schema_version": 2})
    with pytest.raises(ValueError, match="schema_version"):
        define_run({**state, "run_definition_schema_version": True})

    legacy_with_stray_identity = define_run(
        _state(tmp_path, run_id="ignored-legacy-id", source_run_id="ignored-source")
    )
    assert legacy_with_stray_identity.legacy is True
    assert legacy_with_stray_identity.run_id is None
    assert legacy_with_stray_identity.source_run_id is None


def test_run_definition_rejects_unsafe_or_self_referential_identity(tmp_path: Path):
    projected = define_run(_state(tmp_path))

    for run_id in ("../escape", "contains space", "line\nbreak", "a" * 129):
        with pytest.raises(ValueError, match="safe opaque"):
            RunDefinition(
                schema_version=1,
                run_id=run_id,
                source_run_id=None,
                definition_digest=projected.definition_digest,
                contributions=projected.contributions,
                legacy=False,
            )

    with pytest.raises(ValueError, match="must not equal"):
        RunDefinition(
            schema_version=1,
            run_id="run-child",
            source_run_id="run-child",
            definition_digest=projected.definition_digest,
            contributions=projected.contributions,
            legacy=False,
        )


def test_resolve_run_request_creates_exact_and_derived_transitions(tmp_path: Path):
    inputs = _state(
        tmp_path,
        agent_model="one",
        allow_unknown_auth=False,
        skip_host_bound_storage_state_auth=False,
    )
    first = resolve_run_request(inputs, existing_state=None, new_run_id="run-one")
    second = resolve_run_request(inputs, existing_state=None, new_run_id="run-two")

    assert first.kind == second.kind == "new"
    assert first.definition is not None and second.definition is not None
    assert first.definition.run_id == "run-one"
    assert second.definition.run_id == "run-two"
    assert first.definition.definition_digest == second.definition.definition_digest

    persisted = {**inputs, "run_definition": first.definition.to_dict()}
    exact = resolve_run_request({}, existing_state=persisted)
    derived = resolve_run_request({"agent_model": "two"}, existing_state=persisted)

    assert exact.kind == "exact"
    assert exact.definition == first.definition
    assert derived.kind == "derived_required"
    assert derived.definition is not None and derived.definition.run_id is None
    assert derived.drift_fields == ("phase_4.agent_model",)


def test_resolve_run_request_keeps_legacy_resume_mutable_without_identity(tmp_path: Path):
    state = _state(tmp_path, agent_model="one")

    transition = resolve_run_request({"agent_model": "two"}, existing_state=state)

    assert transition.kind == "legacy"
    assert transition.definition is not None
    assert transition.definition.run_id is None


def test_resolve_run_request_rejects_envelope_metadata_conflict(tmp_path: Path):
    transition = resolve_run_request(
        _state(tmp_path, agent_model="one"),
        existing_state=None,
        new_run_id="run-source",
    )
    assert transition.definition is not None
    persisted = {
        **_state(tmp_path, agent_model="two"),
        "run_definition": transition.definition.to_dict(),
    }

    with pytest.raises(ValueError, match="conflicts with run_definition"):
        resolve_run_request({}, existing_state=persisted)


def test_definition_and_status_projection_hide_credential_urls(tmp_path: Path):
    state = _state(
        tmp_path,
        phase_2a_resolution_signature={"callback": "https://user:password@example.invalid/path"},
    )
    (tmp_path / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")

    definition = define_run(state)
    status = build_status_payload(tmp_path)

    assert "password" not in json.dumps(definition.to_dict())
    assert "password" not in json.dumps(status)


@pytest.mark.parametrize(
    ("status", "action", "target", "reason"),
    [
        ("complete", "advance_phase", "phase_3", "pipeline_checkpoint_complete"),
        ("partial_complete", "advance_phase", "phase_3", "pipeline_checkpoint_complete"),
        ("running", "rerun_phase", "phase_2", "pipeline_checkpoint_running"),
        ("failed", "rerun_phase", "phase_2", "pipeline_checkpoint_failed"),
        ("paused", "rerun_phase", "phase_2", "pipeline_checkpoint_paused"),
        ("interrupted", "rerun_phase", "phase_2", "pipeline_checkpoint_interrupted"),
    ],
)
def test_plan_resume_preserves_pipeline_lifecycle_rules(
    tmp_path: Path,
    status: str,
    action: str,
    target: str,
    reason: str,
):
    state = _state(tmp_path, status=status)
    (tmp_path / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")
    plan = plan_resume(define_run(state), state, run_root=tmp_path)

    assert plan.mode == "legacy"
    assert plan.lifecycle_action == action
    assert plan.target_step == target
    assert plan.checkpoint_decisions[0].reason_code == reason


def test_plan_resume_rejects_unknown_status_without_writing(tmp_path: Path):
    state = _state(tmp_path, status="mystery")
    before = set(tmp_path.rglob("*"))

    plan = plan_resume(define_run(state), state, run_root=tmp_path)

    assert plan.mode == "rejected"
    assert plan.lifecycle_action == "reject"
    assert plan.errors == ("unknown_status",)
    assert set(tmp_path.rglob("*")) == before


def test_plan_resume_reports_definition_drift_conservatively(tmp_path: Path):
    (tmp_path / "phase_2").mkdir()
    state = _state(tmp_path, agent_model="one")
    requested = _state(tmp_path, agent_model="two")
    (tmp_path / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")

    plan = plan_resume(
        define_run(state),
        state,
        run_root=tmp_path,
        requested_inputs=requested,
    )

    assert plan.mode == "derived_required"
    assert plan.drift_fields == ("phase_4.agent_model",)
    assert plan.checkpoint_decisions[0].action == "rerun"
    assert plan.checkpoint_decisions[1].action == "not_inspected"
    assert plan.checkpoint_decisions[1].reason_code == "definition_drift_feature_validator_required"
    assert plan.source_digest != plan.requested_digest


def test_plan_resume_does_not_claim_reuse_without_authoritative_state(tmp_path: Path):
    state = _state(tmp_path)

    plan = plan_resume(define_run(state), state, run_root=tmp_path)

    assert plan.lifecycle_action == "advance_phase"
    assert plan.checkpoint_decisions[0].action == "not_inspected"
    assert plan.checkpoint_decisions[0].reason_code == "pipeline_state_not_verified"

    drifted = plan_resume(
        define_run(state),
        state,
        run_root=tmp_path,
        requested_inputs=_state(tmp_path, agent_model="changed"),
    )
    assert drifted.mode == "derived_required"
    assert drifted.checkpoint_decisions[0].action == "not_inspected"
    assert (
        drifted.checkpoint_decisions[0].reason_code
        == "definition_drift_pipeline_state_not_verified"
    )


def test_plan_resume_rejects_authoritative_run_identity_mismatch(tmp_path: Path):
    source = _state(
        tmp_path,
        run_id="run-one",
        run_definition_schema_version=1,
    )
    observed = {**source, "run_id": "run-two"}
    (tmp_path / "pipeline_state.json").write_text(json.dumps(observed), encoding="utf-8")

    plan = plan_resume(define_run(source), source, run_root=tmp_path)

    assert plan.mode == "exact"
    assert plan.checkpoint_decisions[0].action == "not_inspected"
    assert plan.checkpoint_decisions[0].reason_code == "pipeline_state_not_verified"


def test_plan_resume_rejects_caller_state_drift_from_authoritative_file(tmp_path: Path):
    observed = _state(tmp_path, agent_model="source")
    (tmp_path / "pipeline_state.json").write_text(json.dumps(observed), encoding="utf-8")
    caller_state = {**observed, "agent_model": "caller-drift"}

    plan = plan_resume(
        define_run(observed),
        caller_state,
        run_root=tmp_path,
    )

    assert plan.checkpoint_decisions[0].action == "not_inspected"
    assert plan.checkpoint_decisions[0].reason_code == "pipeline_state_not_verified"


def test_definition_tracks_persisted_feature_signatures(tmp_path: Path):
    base = define_run(
        _state(
            tmp_path,
            action_counts={"gitlab": 1},
            exposure_contract_signature="first",
        )
    )
    changed = define_run(
        _state(
            tmp_path,
            action_counts={"gitlab": 2},
            exposure_contract_signature="second",
        )
    )

    assert base.definition_digest != changed.definition_digest

    inventory_changed = define_run(
        _state(
            tmp_path,
            action_counts={"gitlab": 1},
            exposure_contract_signature="first",
            host_inventory_instances_sha256="inventory-two",
        )
    )
    assert base.definition_digest != inventory_changed.definition_digest


def test_status_projects_definition_and_resume_plan_without_writes(tmp_path: Path):
    state_path = tmp_path / "pipeline_state.json"
    state_path.write_text(json.dumps(_state(tmp_path, api_token="secret")), encoding="utf-8")
    before = state_path.read_bytes()

    payload = build_status_payload(tmp_path)
    text = format_status_payload(payload)

    assert payload["run_definition"]["legacy"] is True
    assert payload["resume_plan"]["lifecycle_action"] == "advance_phase"
    assert "Run definition:" in text
    assert "Resume plan:" in text
    projected = json.dumps(
        {"definition": payload["run_definition"], "plan": payload["resume_plan"]},
        sort_keys=True,
    )
    assert "api_token" not in projected
    assert "secret" not in projected
    assert "secret" not in json.dumps(payload, sort_keys=True)
    assert payload["pipeline_state"]["api_token"] == "<redacted>"
    assert state_path.read_bytes() == before


def test_status_keeps_malformed_definition_advisory(tmp_path: Path):
    state_path = tmp_path / "pipeline_state.json"
    state_path.write_text(
        json.dumps(_state(tmp_path, run_definition_schema_version=99)),
        encoding="utf-8",
    )

    payload = build_status_payload(tmp_path)

    assert "pipeline_state" in payload
    assert "run_definition" not in payload
    assert "unsupported persisted run definition" in payload["run_definition_error"]
