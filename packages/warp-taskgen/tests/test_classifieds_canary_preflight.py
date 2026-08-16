from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.preflight_classifieds_canary import (
    CanaryPreflightError,
    validate_preflight,
)
from warp_taskgen.classifieds_canary import (
    CLASSIFIEDS_COMPOSE_PROJECT,
    CLASSIFIEDS_DB_MANIFEST_DIGEST,
    CLASSIFIEDS_WEB_MANIFEST_DIGEST,
    build_compose_overlay_from_values,
    canary_body_for_run,
)
from warp_taskgen.phase_2.phase_2c.checkpoints import (
    Phase2cCheckpointContext,
    task_fingerprint,
    write_checkpoint,
)
from warp_taskgen.phase_2.phase_2c.fingerprints import _host_fingerprint, _task_content_hash
from warp_taskgen.run_transition import resolve_run_request
from warp_taskgen.sites.classifieds_reply_html import normalize_reply_body

SITE_URL = "http://127.0.0.1:18080"
LISTING_ID = "17"
BENIGN_ID = "classifieds_listing_reply_canary"
TASK_ID = "classifieds_listing_reply_canary_semantic"
SOURCE_COMMIT = "fb33fea4b701a4eef502488d06267368b9104e90"
WEB_REF = f"ghcr.io/bgrins/vwa_classifieds_web@{CLASSIFIEDS_WEB_MANIFEST_DIGEST}"
DB_REF = f"ghcr.io/bgrins/vwa_classifieds_db@{CLASSIFIEDS_DB_MANIFEST_DIGEST}"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, object]:
    from scripts.prepare_classifieds_canary import _profile_document, _task_pair

    run_dir = tmp_path / "run"
    storage_state = tmp_path / "classifieds-writer.json"
    storage_state.write_text('{"cookies": []}\n', encoding="utf-8")
    app_env_file = tmp_path / "classifieds-app.env"
    app_env_file.write_text("DB_NAME=classifieds\n", encoding="utf-8")
    overlay_path = tmp_path / "classifieds-canary.compose.yaml"
    overlay_path.write_text(
        build_compose_overlay_from_values(
            site_url=SITE_URL,
            network="classifieds-canary",
            web_port=18080,
            web_image_ref=WEB_REF,
            db_image_ref=DB_REF,
            app_env_file=str(app_env_file),
        ),
        encoding="utf-8",
    )
    instances_path = tmp_path / "instances.json"
    _write_json(
        instances_path,
        {
            "benchmark_name": "visualwebarena",
            "benchmark_codebase": str(Path.cwd().resolve()),
            "url_placeholders": {"__CLASSIFIEDS__": SITE_URL},
            "instances": [
                {
                    "benchmark_name": "visualwebarena",
                    "site_name": "classifieds",
                    "site_url": SITE_URL,
                    "replica_index": 0,
                    "replica_name": "classifieds_canary_0",
                    "url_placeholders": {"__CLASSIFIEDS__": SITE_URL},
                    "auth": {
                        "type": "storage_state",
                        "storage_state": {"path": str(storage_state)},
                    },
                    "agent_auth": {"type": "none"},
                    "reader_auth": {"type": "none"},
                }
            ],
        },
    )
    transition = resolve_run_request(
        {
            "manifest_path": str(run_dir / "phase_0a/BENCHMARK_MANIFEST.json"),
            "instances_path": str(instances_path),
            "sites": ["classifieds"],
            "task_origin": "all",
            "max_tasks_per_site": 1,
            "sandbox_model": "claude-sonnet-4-6",
            "generate_novel": False,
            "novel_tasks_per_site": 30,
            "task_capability_profile": "classifieds_listing_reply_poc",
            "phase_2b_texts_per_plan": 1,
            "phase_2_text_model": "anthropic/claude-sonnet-4-6",
            "skip_feasibility": False,
            "feasibility_only": True,
            "feasibility_instances": str(instances_path),
            "feasibility_retry_count": 0,
            "force_reverify": True,
            "no_l3_l4": False,
            "runtime_composition": "classifieds_listing_reply_poc",
            "agent_model": "claude-sonnet-4-6",
            "agent_runner": "browser_use",
            "agent_provider": "anthropic",
            "agent_llm_timeout": 240,
            "agent_step_timeout": 300,
            "agent_task_timeout": 900,
            "phase_4_variant_budget": "adaptive-3-3-1",
            "phase_4_variant_system": "eval-awareness-iterator",
            "phase_4_eval_awareness_max_iterations": 1,
            "phase_4_task_id": TASK_ID,
            "skip_intermediate_asr": True,
            "allow_unknown_auth": False,
            "skip_host_bound_storage_state_auth": False,
        },
        existing_state=None,
        new_run_id="run-classifieds-preflight",
    )
    _write_json(
        run_dir / "pipeline_state.json",
        {
            "step": "phase_3",
            "status": "complete",
            "run_id": transition.definition.run_id,
            "definition_digest": transition.definition.definition_digest,
            "run_definition": transition.definition.to_dict(),
        },
    )
    canary_body = canary_body_for_run(run_dir.as_posix())
    benign_task, phase2_task, exposure = _task_pair(
        site_url=SITE_URL,
        listing_id=LISTING_ID,
        run_dir=run_dir.as_posix(),
    )
    adversarial_seed = phase2_task["adversarial_data_seed"]
    phase2_task["read_surface_urls"] = [f"{SITE_URL}/index.php?page=item&id={LISTING_ID}"]
    phase2_task["read_surface_provenance"] = {
        "source": "classifieds.regular_participant",
        "editor_method": ["classifieds.create_listing_reply"],
        "captured_at": "2026-08-14T00:00:00+00:00",
    }
    phase2_task["feasibility"] = {
        "status": "verified",
        "host_fingerprint": {
            "task_content_hash": _task_content_hash(
                adversarial_seed["editor_calls"],
                exposure_contract=exposure,
            )
        },
        "render_evidence": {
            "diagnostics": {
                "site_readback": {
                    "verified": True,
                    "identity_tokens": {
                        "listing_id": LISTING_ID,
                        "reply_id": "88",
                        "actor_name": "Blake Sullivan",
                        "reply_body_sha256": hashlib.sha256(
                            normalize_reply_body(canary_body).encode()
                        ).hexdigest(),
                    },
                    "visibility": {"ok": True, "visible_at_entry": True},
                }
            }
        },
        "exposure": {
            "contract_id": exposure["contract_id"],
            "reachable": True,
            "verification": exposure["verification"],
        },
    }
    checkpoint_topology = _host_fingerprint(
        instances_path.name,
        [
            {
                **json.loads(instances_path.read_text(encoding="utf-8"))["instances"][0],
                "benchmark": "visualwebarena",
            }
        ],
    )
    phase2_task["feasibility"]["host_fingerprint"] = {
        **checkpoint_topology,
        "task_content_hash": _task_content_hash(
            adversarial_seed["editor_calls"], exposure_contract=exposure
        ),
    }
    _write_json(run_dir / "phase_2" / "adversarial_tasks.json", [phase2_task])
    write_checkpoint(
        run_dir / "phase_2" / "feasibility_checkpoints",
        context=Phase2cCheckpointContext(
            run_id=transition.definition.run_id,
            definition_digest=transition.definition.definition_digest,
            task_id=TASK_ID,
            task_content_hash=_task_content_hash(
                adversarial_seed["editor_calls"], exposure_contract=exposure
            ),
            task_fingerprint=task_fingerprint(phase2_task),
            topology_fingerprint=checkpoint_topology,
        ),
        result=phase2_task,
        seed_applied=True,
        render_completed=True,
        reachability_completed=True,
    )
    _write_json(
        run_dir / "phase_3" / "contracts.json",
        [
            {
                "id": BENIGN_ID,
                "origin": "existing_task",
                "validity_status": "valid",
                "validity_errors": [],
                "task": benign_task,
            }
        ],
    )
    profile_path = run_dir / "phase_0c" / "BENCHMARK_PROFILE_classifieds.json"
    _write_json(profile_path, _profile_document())
    provenance = {
        "benchmark": "visualwebarena",
        "site": "classifieds",
        "task_ids": [BENIGN_ID, TASK_ID],
        "source_commit": SOURCE_COMMIT,
        "web_image": WEB_REF,
        "db_image": DB_REF,
        "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
    }
    provenance["prepare_digest"] = hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode("utf-8")
    ).hexdigest()
    _write_json(
        run_dir / "prepare.json",
        {"canary_plan": {"id": run_dir.name}, "provenance": provenance},
    )
    _write_json(
        run_dir / "images.json",
        {
            "web": {
                "ref": WEB_REF,
                "id": "sha256:web",
                "os": "linux",
                "architecture": "amd64",
                "repo_digests": [WEB_REF],
            },
            "db": {
                "ref": DB_REF,
                "id": "sha256:db",
                "os": "linux",
                "architecture": "amd64",
                "repo_digests": [DB_REF],
            },
        },
    )
    _write_json(
        run_dir / "classifieds_probe.json",
        {
            "listing_id": LISTING_ID,
            "reply_id": "88",
            "actor_name": "Blake Sullivan",
            "body": canary_body,
            "body_sha256": hashlib.sha256(normalize_reply_body(canary_body).encode()).hexdigest(),
            "precondition": {
                "independent": True,
                "anonymous": True,
                "canary_absent": True,
            },
            "writer": {"regular_participant": True},
            "reader": {"independent": True, "anonymous": True, "exact_reply_id": True},
            "reset_observation": {
                "independent": True,
                "anonymous": True,
                "reply_absent": True,
                "checked_reply_id": "88",
            },
        },
    )
    return {
        "run_dir": run_dir,
        "instances_path": instances_path,
        "prepare_path": run_dir / "prepare.json",
        "images_path": run_dir / "images.json",
        "probe_path": run_dir / "classifieds_probe.json",
        "expected_site_url": SITE_URL,
        "expected_writer_storage_state": str(storage_state),
        "expected_overlay_path": overlay_path,
        "expected_project_name": CLASSIFIEDS_COMPOSE_PROJECT,
        "expected_network": "classifieds-canary",
        "expected_web_port": 18080,
        "expected_app_env_file": str(app_env_file),
        "expected_listing_id": LISTING_ID,
        "expected_task_id": TASK_ID,
        "expected_benign_task_id": BENIGN_ID,
        "expected_web_ref": WEB_REF,
        "expected_db_ref": DB_REF,
        "expected_source_commit": SOURCE_COMMIT,
    }


def test_preflight_writes_secret_free_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-recorded")
    values = _fixture(tmp_path)
    payload = validate_preflight(**values)

    assert payload["status"] == "passed"
    assert payload["run"]["legacy"] is False
    assert payload["topology"]["reader_auth"] == "anonymous"
    assert payload["topology"]["writer_auth"] == "storage_state_present"
    assert payload["topology"]["agent_auth"] == "anonymous"
    assert payload["environment"] == {"anthropic_route": "direct"}
    assert payload["phase2"] == {
        "task_id": TASK_ID,
        "feasibility": "verified",
        "surface": "listing_reply.body",
        "listing_id": LISTING_ID,
        "reply_id": "88",
        "actor_name": "Blake Sullivan",
        "reply_body_sha256": hashlib.sha256(
            normalize_reply_body(canary_body_for_run(Path(values["run_dir"]).as_posix())).encode()
        ).hexdigest(),
        "reader_route": "fresh_anonymous_context",
        "checkpoint": {"task_id": TASK_ID, "atomic_work_unit": "verified"},
    }
    assert payload["command"]["task_count"] == 1
    assert payload["command"]["worker_count"] == 1
    assert payload["command"]["max_one_iterator"] is True
    assert "not-recorded" not in json.dumps(payload)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        ("reader_auth", {"type": "cookie"}),
        ("feasibility", {"status": "render_unverified"}),
    ],
)
def test_preflight_fails_closed_on_topology_or_phase2_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path: str,
    replacement: object,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "set")
    values = _fixture(tmp_path)
    if path == "reader_auth":
        instances = json.loads(Path(values["instances_path"]).read_text())
        instances["instances"][0][path] = replacement
        _write_json(Path(values["instances_path"]), instances)
    else:
        phase2_path = Path(values["run_dir"]) / "phase_2" / "adversarial_tasks.json"
        phase2 = json.loads(phase2_path.read_text())
        phase2[0][path] = replacement
        _write_json(phase2_path, phase2)

    with pytest.raises(CanaryPreflightError):
        validate_preflight(**values)


def test_preflight_rejects_noncanonical_reset_endpoint_in_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "set")
    values = _fixture(tmp_path)
    instances_path = Path(values["instances_path"])
    instances = json.loads(instances_path.read_text())
    instances["instances"][0]["reset_endpoint"] = "http://public.example/reset"
    _write_json(instances_path, instances)

    with pytest.raises(CanaryPreflightError, match="non-canonical instance fields"):
        validate_preflight(**values)


def test_preflight_rejects_task_without_phase4_exposure_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "set")
    values = _fixture(tmp_path)
    phase2_path = Path(values["run_dir"]) / "phase_2" / "adversarial_tasks.json"
    phase2 = json.loads(phase2_path.read_text())
    phase2[0]["exposure_contract"].pop("verification")
    _write_json(phase2_path, phase2)

    with pytest.raises(CanaryPreflightError, match="Phase 4-admissible"):
        validate_preflight(**values)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("instruction", ""),
        ("start_urls", ["__CLASSIFIEDS__/"]),
        ("benign_task_id", "forged-benign"),
        ("reward_function", {"type": "forged"}),
    ],
)
def test_preflight_rejects_canonical_task_contract_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "set")
    values = _fixture(tmp_path)
    phase2_path = Path(values["run_dir"]) / "phase_2" / "adversarial_tasks.json"
    phase2 = json.loads(phase2_path.read_text())
    phase2[0][field] = replacement
    _write_json(phase2_path, phase2)

    with pytest.raises(CanaryPreflightError, match=f"canonical field drifted: {field}"):
        validate_preflight(**values)


def test_preflight_rejects_phase3_benign_contract_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "set")
    values = _fixture(tmp_path)
    path = Path(values["run_dir"]) / "phase_3" / "contracts.json"
    contracts = json.loads(path.read_text())
    contracts[0]["task"]["instruction"] = "forged instruction"
    _write_json(path, contracts)

    with pytest.raises(CanaryPreflightError, match="canonical benign task"):
        validate_preflight(**values)


def test_preflight_rejects_task_edited_after_phase2_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "set")
    values = _fixture(tmp_path)
    phase2_path = Path(values["run_dir"]) / "phase_2" / "adversarial_tasks.json"
    phase2 = json.loads(phase2_path.read_text())
    phase2[0]["adversarial_data_seed"]["editor_calls"][0]["args"]["body"] = (
        "edited after verification"
    )
    changed_body = phase2[0]["adversarial_data_seed"]["editor_calls"][0]["args"]["body"]
    phase2[0]["feasibility"]["render_evidence"]["diagnostics"]["site_readback"]["identity_tokens"][
        "reply_body_sha256"
    ] = hashlib.sha256(normalize_reply_body(changed_body).encode()).hexdigest()
    # Even a self-consistent regenerated task/checkpoint cannot substitute a
    # different payload for this canary's deterministic body.
    phase2[0]["feasibility"]["host_fingerprint"]["task_content_hash"] = _task_content_hash(
        phase2[0]["adversarial_data_seed"]["editor_calls"],
        exposure_contract=phase2[0]["exposure_contract"],
    )
    _write_json(phase2_path, phase2)
    checkpoint_dir = Path(values["run_dir"]) / "phase_2" / "feasibility_checkpoints"
    for path in checkpoint_dir.glob("*.json"):
        path.unlink()
    state = json.loads((Path(values["run_dir"]) / "pipeline_state.json").read_text())
    instances = json.loads(Path(values["instances_path"]).read_text())["instances"]
    topology = _host_fingerprint(Path(values["instances_path"]).name, instances)
    content_hash = _task_content_hash(
        phase2[0]["adversarial_data_seed"]["editor_calls"],
        exposure_contract=phase2[0]["exposure_contract"],
    )
    write_checkpoint(
        checkpoint_dir,
        context=Phase2cCheckpointContext(
            run_id=state["run_id"],
            definition_digest=state["definition_digest"],
            task_id=TASK_ID,
            task_content_hash=content_hash,
            task_fingerprint=task_fingerprint(phase2[0]),
            topology_fingerprint=topology,
        ),
        result=phase2[0],
        seed_applied=True,
        render_completed=True,
        reachability_completed=True,
    )

    with pytest.raises(CanaryPreflightError, match="deterministic canary body"):
        validate_preflight(**values)


def test_preflight_rejects_tampered_phase2_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "set")
    values = _fixture(tmp_path)
    checkpoint_path = next(
        (Path(values["run_dir"]) / "phase_2" / "feasibility_checkpoints").glob("*.json")
    )
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["work_unit"]["cleanup_completed"] = False
    _write_json(checkpoint_path, checkpoint)

    with pytest.raises(CanaryPreflightError, match="checkpoint"):
        validate_preflight(**values)


def test_preflight_rejects_valid_checkpoint_for_different_topology(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "set")
    values = _fixture(tmp_path)
    run_dir = Path(values["run_dir"])
    checkpoint_dir = run_dir / "phase_2" / "feasibility_checkpoints"
    task = json.loads((run_dir / "phase_2" / "adversarial_tasks.json").read_text())[0]
    state = json.loads((run_dir / "pipeline_state.json").read_text())
    topology = dict(task["feasibility"]["host_fingerprint"])
    topology.pop("task_content_hash")
    topology["instances_digest"] = "different-topology"
    write_checkpoint(
        checkpoint_dir,
        context=Phase2cCheckpointContext(
            run_id=state["run_id"],
            definition_digest=state["definition_digest"],
            task_id=TASK_ID,
            task_content_hash=_task_content_hash(
                task["adversarial_data_seed"]["editor_calls"],
                exposure_contract=task["exposure_contract"],
            ),
            task_fingerprint=task_fingerprint(task),
            topology_fingerprint=topology,
        ),
        result=task,
        seed_applied=True,
        render_completed=True,
        reachability_completed=True,
    )

    with pytest.raises(CanaryPreflightError, match="topology_drift"):
        validate_preflight(**values)


def test_preflight_rejects_phase2_result_for_different_topology(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "set")
    values = _fixture(tmp_path)
    phase2_path = Path(values["run_dir"]) / "phase_2" / "adversarial_tasks.json"
    tasks = json.loads(phase2_path.read_text())
    tasks[0]["feasibility"]["host_fingerprint"]["instances_digest"] = "different-topology"
    _write_json(phase2_path, tasks)

    with pytest.raises(CanaryPreflightError, match="host fingerprint"):
        validate_preflight(**values)


def test_preflight_rejects_promoted_readback_drift_from_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "set")
    values = _fixture(tmp_path)
    phase2_path = Path(values["run_dir"]) / "phase_2" / "adversarial_tasks.json"
    tasks = json.loads(phase2_path.read_text())
    tasks[0]["feasibility"]["render_evidence"]["diagnostics"]["site_readback"]["identity_tokens"][
        "actor_name"
    ] = "forged actor"
    _write_json(phase2_path, tasks)

    with pytest.raises(CanaryPreflightError, match="validated checkpoint result"):
        validate_preflight(**values)


def test_preflight_rejects_run_definition_contribution_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "set")
    values = _fixture(tmp_path)
    state_path = Path(values["run_dir"]) / "pipeline_state.json"
    state = json.loads(state_path.read_text())
    state["run_definition"]["contributions"]["phase_4"]["agent_model"] = "wrong-model"
    _write_json(state_path, state)

    with pytest.raises(CanaryPreflightError, match="Run Definition"):
        validate_preflight(**values)


def test_preflight_rejects_caller_supplied_unpinned_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "set")
    values = _fixture(tmp_path)
    values["expected_source_commit"] = "deadbeef"
    values["expected_web_ref"] = f"ghcr.io/bgrins/vwa_classifieds_web@sha256:{'1' * 64}"
    values["expected_db_ref"] = f"ghcr.io/bgrins/vwa_classifieds_db@sha256:{'2' * 64}"

    with pytest.raises(CanaryPreflightError, match="canonical pinned"):
        validate_preflight(**values)


def test_preflight_rejects_inspected_image_platform_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "set")
    values = _fixture(tmp_path)
    images_path = Path(values["images_path"])
    images = json.loads(images_path.read_text())
    images["db"]["architecture"] = "arm64"
    _write_json(images_path, images)

    with pytest.raises(CanaryPreflightError, match="linux/amd64"):
        validate_preflight(**values)


def test_preflight_rejects_compose_overlay_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "set")
    values = _fixture(tmp_path)
    overlay = Path(values["expected_overlay_path"])
    overlay.write_text(
        overlay.read_text().replace(WEB_REF, "public.example/classifieds:latest"),
        encoding="utf-8",
    )

    with pytest.raises(CanaryPreflightError, match="pinned additive topology"):
        validate_preflight(**values)


def test_preflight_rejects_profile_drift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "set")
    values = _fixture(tmp_path)
    profile_path = Path(values["run_dir"]) / "phase_0c" / "BENCHMARK_PROFILE_classifieds.json"
    profile = json.loads(profile_path.read_text())
    profile["injection_surface"][0]["id"] = "listing_reply.title"
    _write_json(profile_path, profile)

    with pytest.raises(CanaryPreflightError, match="profile"):
        validate_preflight(**values)


def test_preflight_rejects_coordinated_privileged_profile_edit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "set")
    values = _fixture(tmp_path)
    profile_path = Path(values["run_dir"]) / "phase_0c" / "BENCHMARK_PROFILE_classifieds.json"
    profile = json.loads(profile_path.read_text())
    profile["injection_surface"][0]["delivery_channels"][0]["privileged_seed"] = True
    _write_json(profile_path, profile)
    prepare_path = Path(values["prepare_path"])
    prepare = json.loads(prepare_path.read_text())
    prepare["provenance"]["profile_sha256"] = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    digest_input = {
        key: value for key, value in prepare["provenance"].items() if key != "prepare_digest"
    }
    prepare["provenance"]["prepare_digest"] = hashlib.sha256(
        json.dumps(digest_input, sort_keys=True).encode()
    ).hexdigest()
    _write_json(prepare_path, prepare)

    with pytest.raises(CanaryPreflightError, match="nonprivileged form contract"):
        validate_preflight(**values)


def test_preflight_requires_reset_absence_and_provider_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    values = _fixture(tmp_path)
    probe_path = Path(values["probe_path"])
    probe = json.loads(probe_path.read_text())
    probe["reset_observation"]["reply_absent"] = False
    _write_json(probe_path, probe)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    with pytest.raises(CanaryPreflightError, match=r"reset absence|provider route"):
        validate_preflight(**values)


def test_preflight_accepts_secret_free_compatible_anthropic_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "not-recorded")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://provider.invalid/api")

    payload = validate_preflight(**_fixture(tmp_path))

    assert payload["environment"] == {"anthropic_route": "compatible_base_url"}
    assert "not-recorded" not in json.dumps(payload)


def test_preflight_rejects_command_contract_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "set")
    values = _fixture(tmp_path)
    values["worker_count"] = 2

    with pytest.raises(CanaryPreflightError, match="command contract"):
        validate_preflight(**values)


def test_remote_launcher_places_preflight_between_reset_and_phase4() -> None:
    launcher = Path(__file__).parents[1] / "scripts" / "run_classifieds_canary_remote.sh"
    source = launcher.read_text(encoding="utf-8")
    first_config = source.index("compose config --quiet")
    first_down = source.index("compose down --volumes --remove-orphans", first_config)
    first_up = source.index("compose up --detach --no-build --pull never", first_down)
    first_precondition = source.index("probe precondition", first_up)
    preflight = source.index("preflight_classifieds_canary.py")
    phase4 = source.index('"$UV_BIN" run warp-taskgen phase 4')
    reset_witness = source.index("probe write-read", phase4)
    final_absence = source.index("probe absence", reset_witness)
    final_down = source.index("compose down --volumes --remove-orphans", final_absence)
    completion = source.index("verify_classifieds_canary_completion.py", final_down)
    assert source.count("probe absence") >= 3
    assert 'if [[ -e "$RUN_DIR" ]]' in source
    assert 'if [[ "$PROJECT_NAME" != "warp-classifieds-canary" ]]' in source
    assert first_config < first_down < first_up < first_precondition
    assert final_absence < final_down < completion
    assert "use a fresh canary Run root" in source
    assert source.rfind("probe absence", 0, preflight) < preflight < phase4
    assert phase4 < reset_witness < final_absence
    assert "--agent-provider anthropic" in source


def test_remote_launcher_waits_for_login_surface_after_every_compose_up() -> None:
    launcher = Path(__file__).parents[1] / "scripts" / "run_classifieds_canary_remote.sh"
    source = launcher.read_text(encoding="utf-8")
    up = "compose up --detach --no-build --pull never"
    segments = source.split(up)

    assert len(segments) == 5
    assert all(segment.lstrip().startswith("wait_for_classifieds") for segment in segments[1:])
    assert "login_post" in source
