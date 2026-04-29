# ruff: noqa
# Auto-split from tests/test_phase_4_adversarial.py; shared helpers live in tests/phase_4/_fixtures.py.
from ._fixtures import *  # noqa: F403,F401

def test_exposure_admission_requires_matching_evidence_contract_id():
    task = _phase4_exposure_fields()
    task["feasibility"]["exposure"]["contract_id"] = "gitlab:other:test"

    assert exposure_admission_error(task) == "exposure_contract_id_mismatch"

def test_exposure_admission_requires_phase4_capability():
    task = _phase4_exposure_fields()
    del task["exposure_contract"]["phase4_exposure"]

    assert exposure_admission_error(task) == "missing_phase4_exposure_capability"

def test_exposure_admission_rejects_phase4_inadmissible_topology():
    task = _phase4_exposure_fields()
    task["exposure_contract"]["phase4_exposure"] = {
        "schema_version": 1,
        "seed_timing": "pre_task",
        "admissible": False,
        "reason": "unforced_transitive_child_surface",
        "encounter_surface": "seed_created_child_detail",
        "requires_transition": True,
        "transition_forced_by_task": False,
        "requires_runtime_hook": False,
    }

    assert (
        exposure_admission_error(task)
        == "phase4_exposure_inadmissible:unforced_transitive_child_surface"
    )

def test_exposure_admission_visual_reachable_advisory():
    task = _phase4_exposure_fields()
    task["feasibility"]["exposure"]["visual_reachable"] = False

    assert exposure_admission_error(task) is None

def test_exposure_admission_allows_layout_visible_at_entry():
    task = _phase4_exposure_fields()
    task["feasibility"]["exposure"]["layout_visible_at_entry"] = True
    task["feasibility"]["exposure"]["scroll_to_visible_px"] = 0
    task["feasibility"]["exposure"]["requires_expand"] = False

    assert exposure_admission_error(task) is None

def test_exposure_admission_allows_far_below_fold_as_telemetry():
    task = _phase4_exposure_fields()
    task["feasibility"]["exposure"]["layout_visible_at_entry"] = False
    task["feasibility"]["exposure"]["scroll_to_visible_px"] = 3001
    task["feasibility"]["exposure"]["requires_expand"] = False

    assert exposure_admission_error(task) is None

def test_exposure_admission_allows_near_below_fold():
    task = _phase4_exposure_fields()
    task["feasibility"]["exposure"]["layout_visible_at_entry"] = False
    task["feasibility"]["exposure"]["scroll_to_visible_px"] = 3000
    task["feasibility"]["exposure"]["requires_expand"] = False

    assert exposure_admission_error(task) is None

def test_exposure_admission_allows_collapsed_ancestor_as_telemetry():
    task = _phase4_exposure_fields()
    task["feasibility"]["exposure"]["layout_visible_at_entry"] = False
    task["feasibility"]["exposure"]["scroll_to_visible_px"] = 10
    task["feasibility"]["exposure"]["requires_expand"] = True

    assert exposure_admission_error(task) is None

def test_exposure_admission_keeps_legacy_layout_fields_optional():
    task = _phase4_exposure_fields()

    assert "layout_visible_at_entry" not in task["feasibility"]["exposure"]
    assert exposure_admission_error(task) is None

def test_exposure_admission_keeps_present_but_missing_layout_probe_as_telemetry():
    task = _phase4_exposure_fields()
    task["feasibility"]["exposure"]["layout_visible_at_entry"] = None
    task["feasibility"]["exposure"]["scroll_to_visible_px"] = None
    task["feasibility"]["exposure"]["requires_expand"] = None

    assert exposure_admission_error(task) is None

def test_exposure_admission_requires_matching_verification_payload():
    task = _phase4_exposure_fields()
    task["feasibility"]["exposure"]["verification"] = {
        "url": "http://gitlab.test/different",
        "mode": "direct_detail",
        "witness_policy": "dom_text",
    }

    assert exposure_admission_error(task) == "exposure_verification_mismatch"

def test_rebase_adversarial_task_uses_validated_benign_contract():
    task, _ = _prepared_adv_task()
    validated_benign = json.loads(json.dumps(task))
    validated_benign["instruction"] = "Use the repaired benign instruction"
    validated_benign["reward_function"] = {
        "eval": [{"expected": {"retrieved_data": ["fixed"]}}],
    }
    validated_benign["agent_context"] = {"site_context": {"platform_name": "Shopping"}}

    stale_adversarial = json.loads(json.dumps(task))
    stale_adversarial["instruction"] = "stale phase 2 instruction"
    stale_adversarial["reward_function"]["benign_reward"] = {"type": "stale"}
    stale_adversarial["reward_function"]["adversarial_reward"] = {"type": "db_query_match"}
    stale_adversarial["agent_context"] = {"site_context": {"platform_name": "Wrong"}}

    rebuilt = phase_4_adversarial._rebase_adversarial_task(
        stale_adversarial,
        validated_benign,
    )

    assert rebuilt["instruction"] == "Use the repaired benign instruction"
    assert rebuilt["reward_function"]["benign_reward"] == validated_benign["reward_function"]
    assert rebuilt["reward_function"]["adversarial_reward"] == {"type": "db_query_match"}
    assert rebuilt["agent_context"] == validated_benign["agent_context"]

def test_rebase_adversarial_task_preserves_cross_site_delivery_sites():
    task, _ = _prepared_adv_task()
    validated_benign = json.loads(json.dumps(task))
    validated_benign["site"] = "shopping_admin"
    validated_benign["sites"] = ["shopping_admin"]

    stale_adversarial = json.loads(json.dumps(task))
    stale_adversarial["site"] = "shopping_admin"
    stale_adversarial["sites"] = ["shopping_admin", "shopping"]
    stale_adversarial["delivery_channel"] = {"delivery_site": "shopping"}
    stale_adversarial["reward_function"]["adversarial_reward"] = {"type": "db_query_match"}

    rebuilt = phase_4_adversarial._rebase_adversarial_task(
        stale_adversarial,
        validated_benign,
    )

    assert rebuilt["sites"] == ["shopping_admin", "shopping"]

def test_rebase_adversarial_task_rejects_non_self_contained_seed():
    task, _ = _prepared_adv_task()
    validated_benign = json.loads(json.dumps(task))
    validated_benign["data_seed"] = {
        "mechanism": "editor",
        "editor_calls": [
            {
                "benchmark": "webarena_verified",
                "site": "shopping",
                "method": "create_product_review",
                "args": {"entity_pk_value": 1, "detail": "baseline"},
            }
        ],
    }

    malformed_adversarial = json.loads(json.dumps(task))
    malformed_adversarial["adversarial_data_seed"] = {
        "mechanism": "editor",
        "editor_calls": [
            {
                "benchmark": "webarena_verified",
                "site": "shopping",
                "method": "create_product_review",
                "args": {"entity_pk_value": 1, "detail": "attack"},
            }
        ],
    }

    with pytest.raises(ValueError, match="preserve the benign data_seed"):
        phase_4_adversarial._rebase_adversarial_task(
            malformed_adversarial,
            validated_benign,
        )

@pytest.mark.asyncio
async def test_admission_excludes_invalid_contracts(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_2").mkdir(parents=True)
    (tmp_path / "phase_3").mkdir(parents=True)

    benign_valid = {
        "id": "benign-valid",
        "site": "shopping",
        "sites": ["shopping"],
        "instruction": "x",
        "start_urls": ["http://shopping.test/orders"],
        "data_seed": {"mechanism": "none"},
        "reward_function": {"eval": [{"expected": {"retrieved_data": ["ok"]}}]},
    }
    benign_invalid = dict(benign_valid)
    benign_invalid["id"] = "benign-invalid"

    (tmp_path / "phase_3" / "contracts.json").write_text(
        json.dumps(
            [
                {
                    "id": "benign-valid",
                    "origin": "existing_task",
                    "validity_status": "valid",
                    "validity_errors": [],
                    "task": benign_valid,
                },
                {
                    "id": "benign-invalid",
                    "origin": "new_task",
                    "validity_status": "invalid",
                    "validity_errors": ["reward_function must be a non-empty object"],
                    "task": benign_invalid,
                },
            ]
        )
    )

    adversarial_tasks = [
        {
            "id": "adv-valid",
            "benign_task_id": "benign-valid",
            "site": "shopping",
            "sites": ["shopping"],
            "reward_function": {
                "adversarial_reward": {"type": "noop"},
            },
            "adversarial_data_seed": {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "shopping",
                        "method": "create_product_review",
                        "args": {"entity_pk_value": 1, "detail": "payload"},
                    }
                ],
            },
            **_phase4_exposure_fields(
                benign_task_id="benign-valid",
                site="shopping",
                url="http://shopping.test/orders",
            ),
        },
        {
            "id": "adv-invalid-benign",
            "benign_task_id": "benign-invalid",
            "site": "shopping",
            "sites": ["shopping"],
            "reward_function": {
                "adversarial_reward": {"type": "noop"},
            },
            "adversarial_data_seed": {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "shopping",
                        "method": "create_product_review",
                        "args": {"entity_pk_value": 1, "detail": "payload"},
                    }
                ],
            },
            **_phase4_exposure_fields(
                benign_task_id="benign-invalid",
                site="shopping",
                url="http://shopping.test/orders",
            ),
        },
        {
            "id": "adv-orphan",
            "benign_task_id": "benign-missing",
            "site": "shopping",
            "sites": ["shopping"],
            "reward_function": {
                "adversarial_reward": {"type": "noop"},
            },
            "adversarial_data_seed": {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "shopping",
                        "method": "create_product_review",
                        "args": {"entity_pk_value": 1, "detail": "payload"},
                    }
                ],
            },
            **_phase4_exposure_fields(
                benign_task_id="benign-missing",
                site="shopping",
                url="http://shopping.test/orders",
            ),
        },
    ]
    (tmp_path / "phase_2" / "adversarial_tasks.json").write_text(json.dumps(adversarial_tasks))

    instances_path = tmp_path / "instances.json"
    instances_path.write_text(
        json.dumps(
            {
                "benchmark_name": "WebArena Verified",
                "benchmark_codebase": str(tmp_path),
                "instances": [
                    {
                        "site_name": "shopping",
                        "site_url": "http://shopping.test",
                        "pvpo_cdp_url": "http://127.0.0.1:9222",
                    }
                ],
            }
        )
    )

    called_with: dict = {}

    async def fake_run_tasks_by_site(**kwargs):
        called_with["tasks"] = kwargs["tasks"]
        return []

    monkeypatch.setattr(phase_4_adversarial, "preflight_auth_check", lambda: None)
    monkeypatch.setattr(phase_4_adversarial, "make_agent_factory", lambda **kw: lambda: None)
    monkeypatch.setattr(phase_4_adversarial, "_load_site_profiles", lambda *args, **kw: {})
    monkeypatch.setattr(
        phase_4_adversarial,
        "_collect_agent_auth_runtime_errors",
        lambda *args, **kw: [],
    )
    monkeypatch.setattr(
        phase_4_adversarial, "_probe_seed_base_state_for_task_targets", lambda *a, **kw: []
    )
    monkeypatch.setattr(phase_4_adversarial, "run_tasks_by_site", fake_run_tasks_by_site)

    rc = await phase_4_adversarial.run(
        Namespace(
            instances=instances_path,
            agent_model="demo-model",
            agent_provider=None,
            allow_unknown_auth=True,
        )
    )

    assert rc in (0, 1)
    admitted_ids = [t["id"] for t in called_with.get("tasks", [])]
    assert admitted_ids == ["adv-valid"]

@pytest.mark.asyncio
async def test_admission_rejects_non_list_contracts(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    instances_path = _prepare_malformed_contracts_fixture(tmp_path, {"not": "a list"})

    with caplog.at_level("ERROR"):
        rc = await phase_4_adversarial.run(
            Namespace(
                instances=instances_path,
                agent_model="demo-model",
                agent_provider=None,
                allow_unknown_auth=True,
            )
        )
    assert rc == 1
    assert any("must be a JSON array" in message for message in caplog.messages)

@pytest.mark.asyncio
async def test_admission_rejects_entry_missing_id(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    instances_path = _prepare_malformed_contracts_fixture(
        tmp_path,
        [{"origin": "existing_task", "validity_status": "valid", "task": {}}],
    )

    with caplog.at_level("ERROR"):
        rc = await phase_4_adversarial.run(
            Namespace(
                instances=instances_path,
                agent_model="demo-model",
                agent_provider=None,
                allow_unknown_auth=True,
            )
        )
    assert rc == 1
    assert any("missing or empty id" in message for message in caplog.messages)

@pytest.mark.asyncio
async def test_admission_rejects_unknown_validity_status(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    instances_path = _prepare_malformed_contracts_fixture(
        tmp_path,
        [{"id": "benign-1", "origin": "existing_task", "validity_status": "pending", "task": {}}],
    )

    with caplog.at_level("ERROR"):
        rc = await phase_4_adversarial.run(
            Namespace(
                instances=instances_path,
                agent_model="demo-model",
                agent_provider=None,
                allow_unknown_auth=True,
            )
        )
    assert rc == 1
    assert any("validity_status must be" in message for message in caplog.messages)

@pytest.mark.asyncio
async def test_admission_rejects_non_dict_entry(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    instances_path = _prepare_malformed_contracts_fixture(
        tmp_path,
        ["not-a-dict-entry"],
    )

    with caplog.at_level("ERROR"):
        rc = await phase_4_adversarial.run(
            Namespace(
                instances=instances_path,
                agent_model="demo-model",
                agent_provider=None,
                allow_unknown_auth=True,
            )
        )
    assert rc == 1
    assert any("not a JSON object" in message for message in caplog.messages)

@pytest.mark.asyncio
async def test_admission_rejects_valid_contract_missing_task_object(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    instances_path = _prepare_malformed_contracts_fixture(
        tmp_path,
        [{"id": "benign-1", "origin": "existing_task", "validity_status": "valid", "task": None}],
    )

    with caplog.at_level("ERROR"):
        rc = await phase_4_adversarial.run(
            Namespace(
                instances=instances_path,
                agent_model="demo-model",
                agent_provider=None,
                allow_unknown_auth=True,
            )
        )
    assert rc == 1
    assert any("missing task object" in message for message in caplog.messages)
