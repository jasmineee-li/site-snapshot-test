# ruff: noqa
# Auto-split from tests/test_phase_4_adversarial.py; shared helpers live in tests/phase_4/_fixtures.py.
from ._fixtures import *  # noqa: F403,F401


@pytest.mark.asyncio
async def test_phase_4_run_checks_existing_storage_state_freshness(monkeypatch, tmp_path):
    from worldsim import storage_state_preflight

    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_2").mkdir(parents=True)
    (tmp_path / "phase_3").mkdir(parents=True)
    adv_task = {
        "id": "adv-auth",
        "benign_task_id": "benign-1",
        "site": "gitlab",
        "sites": ["gitlab"],
        "instruction": "Open the issue",
        "start_urls": ["http://gitlab.test/issues"],
        "data_seed": {"mechanism": "none"},
        "reward_function": {"adversarial_reward": {"type": "noop"}},
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
    }
    adv_task.update(_phase4_exposure_fields())
    (tmp_path / "phase_2" / "adversarial_tasks.json").write_text(json.dumps([adv_task]))
    benign_task = dict(adv_task)
    benign_task["id"] = "benign-1"
    benign_task["reward_function"] = {"type": "noop"}
    (tmp_path / "phase_3" / "contracts.json").write_text(json.dumps(_as_contracts([benign_task])))
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(
        json.dumps(
            {
                "benchmark_name": "WebArena Verified",
                "benchmark_codebase": str(tmp_path),
                "instances": [
                    {
                        "site_name": "gitlab",
                        "site_url": "http://gitlab.test",
                        "reset_endpoint": "http://gitlab.test/init",
                        "pvpo_cdp_url": "http://127.0.0.1:9222",
                        "agent_auth": {
                            "type": "storage_state",
                            "storage_state": {"path": "auth/gitlab-state.json"},
                        },
                    }
                ],
            }
        )
    )

    checked_sites: list[str] = []

    async def fake_ensure_storage_state(instance, *, benchmark_root, benchmark_name):
        checked_sites.append(instance.site_name)
        return tmp_path / "auth" / "gitlab-state.json"

    async def fake_run_tasks_by_site(**kwargs):
        _ = kwargs
        return []

    monkeypatch.setattr(storage_state_preflight, "ensure_storage_state", fake_ensure_storage_state)
    monkeypatch.setattr(
        phase_4_adversarial,
        "inspect_storage_state_preflight",
        lambda *args, **kwargs: SimpleNamespace(errors=(), mismatches=()),
    )
    monkeypatch.setattr(phase_4_adversarial, "preflight_auth_check", lambda: None)
    monkeypatch.setattr(phase_4_adversarial, "make_agent_factory", lambda **kwargs: lambda: None)
    monkeypatch.setattr(phase_4_adversarial, "_load_site_profiles", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        phase_4_adversarial,
        "_collect_agent_auth_runtime_errors",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        phase_4_adversarial,
        "_probe_seed_base_state_for_task_targets",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(phase_4_adversarial, "run_tasks_by_site", fake_run_tasks_by_site)

    rc = await phase_4_adversarial.run(
        Namespace(
            instances=instances_path,
            benchmark=None,
            agent_model="demo-model",
            agent_provider=None,
            resume=False,
            skip_host_bound_storage_state_auth=False,
        )
    )

    assert rc == 0
    assert checked_sites == ["gitlab"]


@pytest.mark.asyncio
async def test_phase_4_run_agent_runtime_error_precedes_host_api_preflight(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_2").mkdir(parents=True)
    (tmp_path / "phase_3").mkdir(parents=True)
    (tmp_path / "phase_2" / "adversarial_tasks.json").write_text(
        json.dumps(
            [
                {
                    "id": "adv-auth-runtime",
                    "benign_task_id": "benign-1",
                    "site": "gitlab",
                    "sites": ["gitlab"],
                    "instruction": "Open the issue",
                    "start_urls": ["http://gitlab.test/issues"],
                    "data_seed": {"mechanism": "none"},
                    "reward_function": {"adversarial_reward": {"type": "noop"}},
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
                    **_phase4_exposure_fields(),
                }
            ]
        )
    )
    (tmp_path / "phase_3" / "contracts.json").write_text(
        json.dumps(
            _as_contracts(
                [
                    {
                        "id": "benign-1",
                        "site": "gitlab",
                        "sites": ["gitlab"],
                        "instruction": "Open the issue",
                        "start_urls": ["http://gitlab.test/issues"],
                        "data_seed": {"mechanism": "none"},
                        "reward_function": {"type": "noop"},
                    }
                ]
            )
        )
    )
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(
        json.dumps(
            {
                "benchmark_name": "WebArena Verified",
                "benchmark_codebase": str(tmp_path),
                "instances": [
                    {
                        "site_name": "gitlab",
                        "site_url": "http://gitlab.test",
                        "pvpo_cdp_url": "http://127.0.0.1:9222",
                    }
                ],
            }
        )
    )

    async def fail_if_called(*, sandbox_model: str):
        raise AssertionError("host API preflight should not run before agent runtime validation")

    monkeypatch.setattr(phase_4_adversarial, "preflight_auth_check", lambda: None)
    monkeypatch.setattr(phase_4_adversarial, "_preflight_host_messages_api", fail_if_called)
    monkeypatch.setattr(phase_4_adversarial, "_load_site_profiles", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        phase_4_adversarial,
        "_collect_agent_auth_runtime_errors",
        lambda *args, **kwargs: ["site 'gitlab': missing auth runtime config"],
    )

    rc = await phase_4_adversarial.run(
        Namespace(
            instances=instances_path,
            benchmark=tmp_path,
            agent_model="demo-model",
            agent_provider=None,
            allow_unknown_auth=True,
            resume=False,
        )
    )

    assert rc == 1
    state = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert state["reason"] == "agent_runtime_config_error"


@pytest.mark.asyncio
async def test_phase_4_run_skip_host_bound_storage_state_auth_rewrites_only_mismatched_instances(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_2").mkdir(parents=True)
    (tmp_path / "phase_3").mkdir(parents=True)
    (tmp_path / "phase_2" / "adversarial_tasks.json").write_text(
        json.dumps(
            [
                {
                    "id": "adv-auth",
                    "benign_task_id": "benign-1",
                    "site": "gitlab",
                    "sites": ["gitlab"],
                    "instruction": "Open the issue",
                    "start_urls": ["http://gitlab.test/issues"],
                    "data_seed": {"mechanism": "none"},
                    "reward_function": {"adversarial_reward": {"type": "noop"}},
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
                    **_phase4_exposure_fields(),
                }
            ]
        )
    )
    (tmp_path / "phase_3" / "contracts.json").write_text(
        json.dumps(
            _as_contracts(
                [
                    {
                        "id": "benign-1",
                        "site": "gitlab",
                        "sites": ["gitlab"],
                        "instruction": "Open the issue",
                        "start_urls": ["http://gitlab.test/issues"],
                        "data_seed": {"mechanism": "none"},
                        "reward_function": {"type": "noop"},
                    }
                ]
            )
        )
    )
    bad_state = tmp_path / "bad-state.json"
    bad_state.write_text(
        json.dumps({"cookies": [{"name": "session", "value": "x", "domain": "18.117.99.179"}]}),
        encoding="utf-8",
    )
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(
        json.dumps(
            {
                "benchmark_name": "WebArena Verified",
                "benchmark_codebase": str(tmp_path),
                "instances": [
                    {
                        "site_name": "gitlab",
                        "site_url": "http://gitlab.test",
                        "reset_endpoint": "http://gitlab.test/init",
                        "pvpo_cdp_url": "http://127.0.0.1:9222",
                        "agent_auth": {
                            "type": "storage_state",
                            "storage_state": {"path": str(bad_state)},
                        },
                    },
                    {
                        "site_name": "gitlab",
                        "site_url": "http://gitlab-alt.test",
                        "reset_endpoint": "http://gitlab-alt.test/init",
                        "pvpo_cdp_url": "http://127.0.0.1:9223",
                        "agent_auth": {
                            "type": "http_headers",
                            "http_headers": {"headers": {"X-Test": "1"}},
                        },
                    },
                ],
            }
        )
    )

    captured: dict[str, object] = {}

    async def fake_run_tasks_by_site(**kwargs):
        captured["instances"] = kwargs["instances"]
        return []

    async def fake_postprocess_one_task(**kwargs):
        return {}

    monkeypatch.setattr(phase_4_adversarial, "preflight_auth_check", lambda: None)
    monkeypatch.setattr(phase_4_adversarial, "make_agent_factory", lambda **kwargs: lambda: None)
    monkeypatch.setattr(phase_4_adversarial, "_load_site_profiles", lambda *args, **kwargs: {})
    monkeypatch.setattr(phase_4_adversarial, "run_tasks_by_site", fake_run_tasks_by_site)
    monkeypatch.setattr(phase_4_adversarial, "_postprocess_one_task", fake_postprocess_one_task)

    rc = await phase_4_adversarial.run(
        Namespace(
            instances=instances_path,
            benchmark=tmp_path,
            agent_model="demo-model",
            agent_provider=None,
            resume=False,
            skip_host_bound_storage_state_auth=True,
        )
    )

    assert rc == 0
    instances = captured["instances"]
    assert instances[0].agent_auth["type"] == "none"
    assert instances[1].agent_auth["type"] == "http_headers"


@pytest.mark.parametrize(
    ("selected_payload_index", "payload_factory", "match"),
    [
        (
            3,
            lambda metadata: [
                {
                    "rendered_payload": f"Visible note {metadata['url']}",
                    "raw_text": f"Visible note {metadata['url']}",
                    "framing_witnesses": ["Visible note"],
                    "concealment_witnesses": [metadata["url"]],
                }
            ],
            "out of range",
        ),
        (0, lambda metadata: [{}], "rendered_payload"),
    ],
)
def test_effective_adversarial_seed_fails_closed_on_malformed_v2_payload_metadata(
    selected_payload_index,
    payload_factory,
    match,
):
    metadata = _v2_payload_contract_fields()
    task = {
        "id": "adv-1",
        "concealment": "plaintext",
        "seed_template": {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "shopping",
                    "method": "create_product_review",
                    "args": {
                        "entity_pk_value": 1,
                        "detail": "{{PAYLOAD_TEXT}}",
                    },
                }
            ],
        },
        "required_tokens": metadata["required_tokens"],
        "length_budget": metadata["length_budget"],
        "payload_texts": payload_factory(metadata),
        "selected_payload_index": selected_payload_index,
        "adversarial_data_seed": {
            "mechanism": "form",
            "api_calls": [
                {
                    "method": "POST",
                    "path": "/api/seed",
                    "body_form": {"detail": "stale"},
                }
            ],
        },
    }

    with pytest.raises(ValueError, match=match):
        phase_4_execution_helpers._effective_adversarial_seed(task)


def test_effective_adversarial_seed_requires_selected_payload_index():
    metadata = _v2_payload_contract_fields()
    task = {
        "id": "adv-1",
        "concealment": "plaintext",
        "seed_template": {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "shopping",
                    "method": "create_product_review",
                    "args": {
                        "entity_pk_value": 1,
                        "detail": "{{PAYLOAD_TEXT}}",
                    },
                }
            ],
        },
        "required_tokens": metadata["required_tokens"],
        "length_budget": metadata["length_budget"],
        "payload_texts": [
            {
                "rendered_payload": f"Visible note {metadata['url']}",
                "raw_text": f"Visible note {metadata['url']}",
                "framing_witnesses": ["Visible note"],
                "concealment_witnesses": [metadata["url"]],
            }
        ],
        "adversarial_data_seed": {
            "mechanism": "form",
            "api_calls": [
                {
                    "method": "POST",
                    "path": "/api/seed",
                    "body_form": {"detail": "stale"},
                }
            ],
        },
    }

    with pytest.raises(ValueError, match="selected_payload_index must be present"):
        phase_4_execution_helpers._effective_adversarial_seed(task)


def test_effective_adversarial_seed_rejects_invalid_seed_template_contract():
    metadata = _v2_payload_contract_fields()
    task = {
        "id": "adv-1",
        "concealment": "plaintext",
        "seed_template": {
            "mechanism": "sql",
            "statements": ["INSERT INTO review_detail (detail) VALUES ('{{PAYLOAD_TEXT}}' || 'x')"],
        },
        "required_tokens": metadata["required_tokens"],
        "length_budget": metadata["length_budget"],
        "payload_texts": [
            {
                "rendered_payload": f"Visible note {metadata['url']}",
                "raw_text": f"Visible note {metadata['url']}",
                "framing_witnesses": ["Visible note"],
                "concealment_witnesses": [metadata["url"]],
            }
        ],
        "selected_payload_index": 0,
        "adversarial_data_seed": {
            "mechanism": "sql",
            "statements": ["INSERT INTO review_detail (detail) VALUES ('stale')"],
        },
    }

    with pytest.raises(ValueError, match="mechanism"):
        phase_4_execution_helpers._effective_adversarial_seed(task)


def test_effective_adversarial_seed_rejects_malformed_non_selected_payload_entry():
    metadata = _v2_payload_contract_fields()
    task = {
        "id": "adv-1",
        "concealment": "plaintext",
        "seed_template": {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "shopping",
                    "method": "create_product_review",
                    "args": {
                        "entity_pk_value": 1,
                        "detail": "{{PAYLOAD_TEXT}}",
                    },
                }
            ],
        },
        "required_tokens": metadata["required_tokens"],
        "length_budget": metadata["length_budget"],
        "payload_texts": [
            {
                "rendered_payload": f"Visible note {metadata['url']}",
                "raw_text": f"Visible note {metadata['url']}",
                "framing_witnesses": ["Visible note"],
                "concealment_witnesses": [metadata["url"]],
            },
            {},
        ],
        "selected_payload_index": 0,
        "adversarial_data_seed": {
            "mechanism": "form",
            "api_calls": [
                {
                    "method": "POST",
                    "path": "/api/seed",
                    "body_form": {"detail": "stale"},
                }
            ],
        },
    }

    with pytest.raises(ValueError, match=r"payload_texts\[1\].*rendered_payload"):
        phase_4_execution_helpers._effective_adversarial_seed(task)
