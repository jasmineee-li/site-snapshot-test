from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from worldsim.config import BenchmarkConfig, BenchmarkInstance
from worldsim.storage_state_preflight import (
    apply_skip_auth_for_host_bound_storage_states,
    ensure_storage_state,
    find_host_bound_storage_state_mismatches,
    inspect_storage_state_preflight,
    write_storage_state_meta,
)


def _write_storage_state(path: Path, *, domain: str) -> None:
    payload = {
        "cookies": [
            {
                "name": "session",
                "value": "abc",
                "domain": domain,
                "path": "/",
            }
        ],
        "origins": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_find_host_bound_storage_state_mismatches_detects_old_host(tmp_path: Path) -> None:
    state_path = tmp_path / "gitlab-state.json"
    _write_storage_state(state_path, domain="18.117.99.179")

    instances = [
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://3.12.221.9:8023",
            reset_endpoint="http://3.12.221.9:8024/init",
            agent_auth={
                "type": "storage_state",
                "storage_state": {"path": str(state_path)},
            },
        )
    ]

    mismatches = find_host_bound_storage_state_mismatches(instances)
    assert len(mismatches) == 1
    assert mismatches[0].site_name == "gitlab"
    assert mismatches[0].recorded_hosts == ("18.117.99.179",)
    assert mismatches[0].instance_hosts == ("3.12.221.9",)


def test_find_host_bound_storage_state_mismatches_allows_matching_parent_domain(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "gitlab-state.json"
    _write_storage_state(state_path, domain=".example.com")

    instances = [
        BenchmarkInstance(
            site_name="gitlab",
            site_url="https://gitlab.example.com:8023",
            reset_endpoint="https://gitlab.example.com:8024/init",
            agent_auth={
                "type": "storage_state",
                "storage_state": {"path": str(state_path)},
            },
        )
    ]

    assert find_host_bound_storage_state_mismatches(instances) == []


def test_apply_skip_auth_for_host_bound_storage_states_rewrites_only_mismatched_storage_state(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "gitlab-state.json"
    _write_storage_state(state_path, domain="18.117.99.179")
    healthy_state_path = tmp_path / "gitlab-healthy-state.json"
    _write_storage_state(healthy_state_path, domain="3.12.221.9")

    config = BenchmarkConfig.model_validate(
        {
            "benchmark_name": "WebArena Verified",
            "benchmark_codebase": str(tmp_path),
            "instances": [
                {
                    "site_name": "gitlab",
                    "site_url": "http://3.12.221.9:8023",
                    "reset_endpoint": "http://3.12.221.9:8024/init",
                    "agent_auth": {
                        "type": "storage_state",
                        "storage_state": {"path": str(state_path)},
                    },
                },
                {
                    "site_name": "gitlab",
                    "site_url": "http://3.12.221.9:8024",
                    "reset_endpoint": "http://3.12.221.9:8025/init",
                    "agent_auth": {
                        "type": "storage_state",
                        "storage_state": {"path": str(healthy_state_path)},
                    },
                },
                {
                    "site_name": "shopping",
                    "site_url": "http://3.12.221.9:7770",
                    "reset_endpoint": "http://3.12.221.9:7771/init",
                    "agent_auth": {"type": "http_headers", "http_headers": {"headers": {"X-Test": "1"}}},
                },
            ],
        }
    )

    mismatches = find_host_bound_storage_state_mismatches(config.instances)
    updated = apply_skip_auth_for_host_bound_storage_states(config, mismatches)

    gitlabs = [instance for instance in updated.instances if instance.site_name == "gitlab"]
    shopping = next(instance for instance in updated.instances if instance.site_name == "shopping")
    assert gitlabs[0].agent_auth["type"] == "none"
    assert gitlabs[1].agent_auth["type"] == "storage_state"
    assert shopping.agent_auth["type"] == "http_headers"


def test_inspect_storage_state_preflight_reports_missing_relative_path_without_benchmark_root(
    tmp_path: Path,
) -> None:
    instances = [
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://3.12.221.9:8023",
            reset_endpoint="http://3.12.221.9:8024/init",
            agent_auth={
                "type": "storage_state",
                "storage_state": {"path": "auth/gitlab-state.json"},
            },
        )
    ]

    report = inspect_storage_state_preflight(instances, benchmark_root=None)

    assert report.mismatches == ()
    assert len(report.errors) == 1
    assert "requires --benchmark" in report.errors[0].message


def test_inspect_storage_state_preflight_reports_invalid_json(tmp_path: Path) -> None:
    state_path = tmp_path / "gitlab-state.json"
    state_path.write_text("{not-json", encoding="utf-8")

    instances = [
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://3.12.221.9:8023",
            reset_endpoint="http://3.12.221.9:8024/init",
            agent_auth={
                "type": "storage_state",
                "storage_state": {"path": str(state_path)},
            },
        )
    ]

    report = inspect_storage_state_preflight(instances)

    assert report.mismatches == ()
    assert len(report.errors) == 1
    assert "invalid JSON" in report.errors[0].message


def test_inspect_storage_state_preflight_ignores_orphan_phase_0d_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    orphan_dir = tmp_path / "phase_0d" / "gitlab"
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "storage_state.json").write_text("{not-json", encoding="utf-8")

    instances = [
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://3.12.221.9:8023",
            reset_endpoint="http://3.12.221.9:8024/init",
            agent_auth={
                "type": "storage_state",
                "storage_state": {"path": "auth/gitlab-state.json"},
            },
        )
    ]

    report = inspect_storage_state_preflight(instances, benchmark_root=tmp_path)

    assert report.mismatches == ()
    assert len(report.errors) == 1
    assert "missing" in report.errors[0].message


def test_find_host_bound_storage_state_mismatches_is_instance_specific_for_multi_host_sites(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "gitlab-state.json"
    _write_storage_state(state_path, domain="host-a.example.com")

    instances = [
        BenchmarkInstance(
            site_name="gitlab",
            site_url="https://host-a.example.com:8023",
            reset_endpoint="https://host-a.example.com:8024/init",
            agent_auth={
                "type": "storage_state",
                "storage_state": {"path": str(state_path)},
            },
        ),
        BenchmarkInstance(
            site_name="gitlab",
            site_url="https://host-b.example.com:8023",
            reset_endpoint="https://host-b.example.com:8024/init",
            agent_auth={
                "type": "storage_state",
                "storage_state": {"path": str(state_path)},
            },
        ),
    ]

    mismatches = find_host_bound_storage_state_mismatches(instances)

    assert len(mismatches) == 1
    assert mismatches[0].instance_hosts == ("host-b.example.com",)


def test_inspect_storage_state_preflight_reports_empty_artifact(tmp_path: Path) -> None:
    state_path = tmp_path / "gitlab-state.json"
    state_path.write_text(json.dumps({"cookies": [], "origins": []}), encoding="utf-8")

    instances = [
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://3.12.221.9:8023",
            reset_endpoint="http://3.12.221.9:8024/init",
            agent_auth={
                "type": "storage_state",
                "storage_state": {"path": str(state_path)},
            },
        )
    ]

    report = inspect_storage_state_preflight(instances)

    assert report.mismatches == ()
    assert len(report.errors) == 1
    assert "storage_state_empty" in report.errors[0].message


def test_ensure_storage_state_raises_when_artifact_is_stale_and_auto_mint_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    state_path = tmp_path / "gitlab-state.json"
    _write_storage_state(state_path, domain="3.12.221.9")
    write_storage_state_meta(
        state_path,
        mechanism="test",
        now_fn=lambda: datetime(2000, 1, 1, tzinfo=UTC),
    )
    instance = BenchmarkInstance(
        site_name="gitlab",
        site_url="http://3.12.221.9:8023",
        reset_endpoint="http://3.12.221.9:8024/init",
        agent_auth={
            "type": "storage_state",
            "storage_state": {"path": str(state_path)},
        },
    )
    monkeypatch.setenv("WORLDSIM_AUTO_MINT_STORAGE_STATE", "false")

    import asyncio

    with pytest.raises(RuntimeError, match="storage_state_stale"):
        asyncio.run(
            ensure_storage_state(
                instance,
                benchmark_root=None,
                benchmark_name="WebArena Verified",
            )
        )


def test_ensure_storage_state_auto_mint_uses_current_sitespec_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from worldsim.phases import phase_0d_auth_bootstrap as bootstrap

    state_dir = tmp_path / "logs"
    state_dir.mkdir()
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(state_dir))
    monkeypatch.setenv("WORLDSIM_AUTO_MINT_STORAGE_STATE", "true")

    benchmark_root = tmp_path / "bench"
    stale_path = tmp_path / "stale-state.json"
    _write_storage_state(stale_path, domain="3.12.221.9")
    write_storage_state_meta(
        stale_path,
        mechanism="test",
        now_fn=lambda: datetime(2000, 1, 1, tzinfo=UTC),
    )
    instance = BenchmarkInstance(
        site_name="gitlab",
        site_url="http://3.12.221.9:8023",
        reset_endpoint="http://3.12.221.9:8024/init",
        agent_auth={
            "type": "storage_state",
            "storage_state": {
                "path": str(stale_path),
                "form_login": {
                    "login_url": "/users/sign_in",
                    "username_selector": "#user_login",
                    "password_selector": "#user_password",
                    "submit_selector": "input[type=submit]",
                    "success_url_substring": "/-/profile",
                },
            },
            "authentication": {
                "credentials": {"username": "root", "password": "admin1234"},
            },
        },
    )

    captured: dict[str, object] = {}

    async def fake_bootstrap_via_form_login(*, spec, site_url, output_path):
        captured["site_name"] = spec.site_name
        captured["mech_type"] = spec.mech_type
        captured["declared_path"] = spec.declared_path
        captured["per_task_refresh"] = spec.per_task_refresh
        captured["agent_context_source"] = spec.agent_context_source
        captured["site_url"] = site_url
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"cookies": [{"name": "session"}], "origins": []}), encoding="utf-8")

    monkeypatch.setattr(bootstrap, "_bootstrap_via_form_login", fake_bootstrap_via_form_login)

    result = asyncio.run(
        ensure_storage_state(
            instance,
            benchmark_root=benchmark_root,
            benchmark_name="WebArena Verified",
        )
    )

    assert result == state_dir / "phase_0d" / "gitlab" / "storage_state.json"
    assert captured == {
        "site_name": "gitlab",
        "mech_type": "form_login",
        "declared_path": "",
        "per_task_refresh": False,
        "agent_context_source": benchmark_root / "phase_0c" / "gitlab" / "AGENT_CONTEXT.json",
        "site_url": "http://3.12.221.9:8023",
    }
