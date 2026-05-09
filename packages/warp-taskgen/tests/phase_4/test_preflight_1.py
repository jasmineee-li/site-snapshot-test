# ruff: noqa
# Auto-split from tests/test_phase_4_adversarial.py; shared helpers live in tests/phase_4/_fixtures.py.
from ._fixtures import *  # noqa: F403,F401

def test_agent_auth_runtime_gate_rejects_explicit_none_for_authed_surfaces():
    instance = BenchmarkInstance(
        site_name="gitlab",
        site_url="http://gitlab.test",
        agent_auth={"type": "none"},
    )
    profile = {
        "injection_surface": [
            {"controllable_by_tier": "authed_user"},
        ],
    }

    errors = phase_4_adversarial._collect_agent_auth_runtime_errors(
        [instance],
        {"gitlab": profile},
    )

    assert errors == [
        "site 'gitlab' requires agent_auth in instances.json because BENCHMARK_PROFILE "
        "has authed_user injection surfaces"
    ]

def test_agent_auth_runtime_gate_accepts_http_headers_for_authed_surfaces():
    instance = BenchmarkInstance(
        site_name="reddit",
        site_url="http://reddit.test",
        agent_auth={
            "type": "http_headers",
            "http_headers": {"headers": {"X-Postmill-Auto-Login": "user:pw"}},
        },
    )
    profile = {
        "injection_surface": [
            {"controllable_by_tier": "authed_user"},
        ],
    }

    errors = phase_4_adversarial._collect_agent_auth_runtime_errors(
        [instance],
        {"reddit": profile},
    )

    assert errors == []

def test_agent_auth_runtime_gate_rejects_unresolvable_http_headers():
    instance = BenchmarkInstance(
        site_name="reddit",
        site_url="http://reddit.test",
        agent_auth={
            "type": "http_headers",
            "http_headers": {"headers": {"X-User": "${credentials.username}"}},
        },
    )
    profile = {
        "injection_surface": [
            {"controllable_by_tier": "authed_user"},
        ],
    }

    errors = phase_4_adversarial._collect_agent_auth_runtime_errors(
        [instance],
        {"reddit": profile},
    )

    assert len(errors) == 1
    assert "invalid http_headers agent_auth" in errors[0]

def test_agent_context_with_instance_auth_redacts_runtime_credentials():
    context = {
        "authentication": {
            "pre_authenticated": False,
            "credentials": {"username": "alice", "password": "secret"},
            "description": "Use the seeded account.",
        }
    }
    agent_auth = {
        "type": "http_headers",
        "http_headers": {"headers": {"X-Postmill-Auto-Login": "${credentials.username}:secret"}},
        "authentication": {
            "credentials": {"username": "alice", "password": "secret"},
        },
    }

    merged = phase_4_adversarial._agent_context_with_instance_auth(context, agent_auth)

    assert merged["authentication"] == {
        "pre_authenticated": True,
        "credentials": None,
        "description": "Pre-authenticated via deployment config.",
    }

def test_seed_target_benchmark_requires_metadata():
    with pytest.raises(ValueError, match="missing benchmark metadata"):
        phase_4_adversarial._seed_target_benchmark(
            {
                "site": "gitlab",
                "adversarial_data_seed": {
                    "editor_calls": [{"site": "gitlab", "method": "create_issue_note"}]
                },
            }
        )

def test_seed_target_benchmark_rejects_mixed_metadata():
    with pytest.raises(ValueError, match="mixed benchmark metadata"):
        phase_4_adversarial._seed_target_benchmark(
            {
                "benchmark": "webarena_verified",
                "site": "gitlab",
                "adversarial_data_seed": {
                    "editor_calls": [
                        {
                            "benchmark": "stwebagentbench",
                            "site": "gitlab",
                            "method": "create_issue_note",
                        }
                    ]
                },
            }
        )

def test_seed_target_benchmark_normalizes_aliases():
    assert (
        phase_4_adversarial._seed_target_benchmark(
            {
                "benchmark_name": "WebArena Verified",
                "site": "gitlab",
                "adversarial_data_seed": {
                    "editor_calls": [
                        {
                            "benchmark": "webarena-verified",
                            "site": "gitlab",
                            "method": "create_issue_note",
                        }
                    ]
                },
            }
        )
        == "webarena_verified"
    )

def test_seed_target_benchmark_uses_instance_metadata_fallback():
    assert (
        phase_4_adversarial._seed_target_benchmark(
            {
                "site": "gitlab",
                "adversarial_data_seed": {
                    "editor_calls": [{"site": "gitlab", "method": "create_issue_note"}]
                },
            },
            {"benchmark_name": "WebArena Verified"},
        )
        == "webarena_verified"
    )

def test_pvpo_endpoint_preflight_errors_allow_missing_and_duplicate_legacy_urls():
    instances = [
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping-0.test",
            replica_index=0,
            pvpo_cdp_url="http://127.0.0.1:9222",
        ),
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping-1.test",
            replica_index=1,
            pvpo_cdp_url="http://127.0.0.1:9222",
        ),
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://gitlab.test",
            replica_index=0,
            pvpo_cdp_url=None,
        ),
    ]

    errors = phase_4_adversarial._pvpo_endpoint_preflight_errors(
        instances,
        active_sites={"shopping", "gitlab"},
    )

    assert errors == []

def test_pvpo_endpoint_preflight_errors_ignore_inactive_sites():
    instances = [
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping-0.test",
            replica_index=0,
            pvpo_cdp_url="http://127.0.0.1:9222",
        ),
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://gitlab.test",
            replica_index=0,
            pvpo_cdp_url=None,
        ),
    ]

    assert (
        phase_4_adversarial._pvpo_endpoint_preflight_errors(
            instances,
            active_sites={"shopping"},
        )
        == []
    )

def test_pvpo_endpoint_preflight_errors_accept_loopback_alias_duplicates():
    instances = [
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping-0.test",
            replica_index=0,
            pvpo_cdp_url="http://127.0.0.1:9222",
        ),
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping-1.test",
            replica_index=1,
            pvpo_cdp_url="http://localhost:9222/",
        ),
    ]

    errors = phase_4_adversarial._pvpo_endpoint_preflight_errors(
        instances,
        active_sites={"shopping"},
    )

    assert errors == []

def test_benchmark_instance_rejects_invalid_pvpo_port():
    with pytest.raises(ValueError, match="invalid port"):
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping-0.test",
            replica_index=0,
            pvpo_cdp_url="http://127.0.0.1:abc",
        )

def test_normalize_saved_adversarial_result_preserves_preflight_error(tmp_path):
    normalized = phase_4_adversarial._normalize_saved_adversarial_result(
        {
            "task_id": "adv-1",
            "outcome": "seed_preflight_mismatch",
            "passed": False,
            "error": "missing required args",
        },
        trajectory_dir=tmp_path / "adv-1",
    )

    assert normalized["error"] == "missing required args"
    assert normalized["benign_passed"] is False

@pytest.mark.asyncio
async def test_phase_4_run_fails_on_malformed_sql_seed(monkeypatch, tmp_path):
    """SQL mechanism seeds are rejected at rebase (validate_data_seed rejects sql)."""
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_2").mkdir(parents=True)
    (tmp_path / "phase_3").mkdir(parents=True)
    (tmp_path / "phase_2" / "adversarial_tasks.json").write_text(
        json.dumps(
            [
                {
                    "id": "adv-sql",
                    "benign_task_id": "benign-1",
                    "site": "shopping",
                    "sites": ["shopping"],
                    "instruction": "Seed the catalog",
                    "start_urls": ["http://shopping.test/orders"],
                    "data_seed": {"mechanism": "none"},
                    "reward_function": {
                        "adversarial_reward": {"type": "noop"},
                    },
                    "adversarial_data_seed": {
                        "mechanism": "sql",
                        "statements": ["INSERT INTO products (id) VALUES (1)"],
                    },
                    **_phase4_exposure_fields(
                        site="shopping",
                        url="http://shopping.test/orders",
                    ),
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
                        "site": "shopping",
                        "sites": ["shopping"],
                        "instruction": "Seed the catalog",
                        "start_urls": ["http://shopping.test/orders"],
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
                        "site_name": "shopping",
                        "site_url": "http://shopping.test",
                    }
                ],
            }
        )
    )

    rc = await phase_4_adversarial.run(
        Namespace(
            instances=instances_path,
            agent_model="demo-model",
            agent_provider=None,
            resume=False,
        )
    )

    assert rc == 1
    state = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert state["status"] == "failed"
    assert state["reason"] == "malformed_adversarial_tasks"

@pytest.mark.asyncio
async def test_phase_4_run_fails_fast_on_storage_state_preflight_error(monkeypatch, tmp_path):
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

    monkeypatch.setattr(
        phase_4_adversarial,
        "preflight_auth_check",
        lambda: (_ for _ in ()).throw(AssertionError("auth preflight should not run")),
    )

    async def fail_if_called(*, sandbox_model: str):
        raise AssertionError("host API preflight should not run before storage-state validation")

    monkeypatch.setattr(phase_4_adversarial, "_preflight_host_messages_api", fail_if_called)

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

    assert rc == 1
    state = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert state["status"] == "failed"
    assert state["reason"] == "storage_state_preflight_error"
    assert any(
        "storage_state artifact missing" in error
        for error in state["storage_state_preflight_errors"]
    )
