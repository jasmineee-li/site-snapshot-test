from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from worldsim.config import BenchmarkConfig, BenchmarkInstance
from worldsim.storage_state_preflight import (
    _auto_mint_allowed,
    apply_skip_auth_for_host_bound_storage_states,
    ensure_storage_state,
    find_host_bound_storage_state_mismatches,
    inspect_storage_state_preflight,
    storage_state_is_fresh,
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


def test_auto_mint_allows_canonical_webarena_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLDSIM_AUTO_MINT_STORAGE_STATE", raising=False)

    assert _auto_mint_allowed("WebArena Verified")
    assert _auto_mint_allowed("webarena_verified")


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

    mismatches = find_host_bound_storage_state_mismatches(instances, benchmark_root=tmp_path)
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

    assert find_host_bound_storage_state_mismatches(instances, benchmark_root=tmp_path) == []


def test_find_host_bound_storage_state_mismatches_rejects_mixed_hosts(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "gitlab-state.json"
    payload = {
        "cookies": [
            {"name": "session", "value": "abc", "domain": "3.12.221.9", "path": "/"},
            {"name": "foreign", "value": "abc", "domain": "evil.test", "path": "/"},
        ],
        "origins": [{"origin": "http://3.12.221.9:8023", "localStorage": []}],
    }
    state_path.write_text(json.dumps(payload), encoding="utf-8")

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

    report = inspect_storage_state_preflight(instances, benchmark_root=tmp_path)
    assert report.mismatches == ()
    assert len(report.errors) == 1
    assert "storage_state_mixed_hosts" in report.errors[0].message
    assert "evil.test" in report.errors[0].message


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
                    "agent_auth": {
                        "type": "http_headers",
                        "http_headers": {"headers": {"X-Test": "1"}},
                    },
                },
            ],
        }
    )

    mismatches = find_host_bound_storage_state_mismatches(config.instances, benchmark_root=tmp_path)
    updated = apply_skip_auth_for_host_bound_storage_states(config, mismatches)

    gitlabs = [instance for instance in updated.instances if instance.site_name == "gitlab"]
    shopping = next(instance for instance in updated.instances if instance.site_name == "shopping")
    assert gitlabs[0].agent_auth["type"] == "none"
    assert gitlabs[1].agent_auth["type"] == "storage_state"
    assert shopping.agent_auth["type"] == "http_headers"


def test_inspect_storage_state_preflight_reports_missing_relative_path_without_benchmark_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path / "state"))
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
    assert "storage_state artifact missing" in report.errors[0].message


def test_inspect_storage_state_preflight_rejects_absolute_path_outside_allowed_roots(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "outside-state.json"
    _write_storage_state(state_path, domain="3.12.221.9")

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

    report = inspect_storage_state_preflight(instances, benchmark_root=tmp_path / "bench")

    assert report.mismatches == ()
    assert len(report.errors) == 1
    assert "outside allowed roots" in report.errors[0].message


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

    report = inspect_storage_state_preflight(instances, benchmark_root=tmp_path)

    assert report.mismatches == ()
    assert len(report.errors) == 1
    assert "invalid JSON" in report.errors[0].message


def test_inspect_storage_state_preflight_follows_symlinked_state_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mirrors hosts where the WorldSim state dir is reached through a symlink.
    # Relative storage_state paths anchor at the WorldSim state dir (where
    # Phase 0d writes), so the preflight must resolve through the symlink
    # without requiring the file to be copied into the physical tree.
    physical_state_dir = tmp_path / "physical" / "logs"
    physical_state_dir.mkdir(parents=True)
    auth_dir = physical_state_dir / "auth"
    auth_dir.mkdir()
    state_path = auth_dir / "gitlab-state.json"
    _write_storage_state(state_path, domain="3.12.221.9")

    symlink_state_dir = tmp_path / "logs-symlink"
    symlink_state_dir.symlink_to(physical_state_dir, target_is_directory=True)
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(symlink_state_dir))

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

    assert report.errors == ()
    assert report.mismatches == ()


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

    mismatches = find_host_bound_storage_state_mismatches(instances, benchmark_root=tmp_path)

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

    report = inspect_storage_state_preflight(instances, benchmark_root=tmp_path)

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
                benchmark_root=tmp_path,
                benchmark_name="WebArena Verified",
            )
        )


def test_storage_state_is_fresh_rejects_old_validator_version(tmp_path: Path) -> None:
    from worldsim.phases import phase_0d_auth_bootstrap as bootstrap

    state_path = tmp_path / "gitlab-state.json"
    _write_storage_state(state_path, domain="3.12.221.9")
    write_storage_state_meta(
        state_path,
        mechanism="test",
        validator_version=bootstrap.CURRENT_VALIDATOR_VERSION - 1,
    )

    assert storage_state_is_fresh(state_path) is False


def test_ensure_storage_state_auto_mint_reacquires_per_instance_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from worldsim.phases import phase_0d_auth_bootstrap as bootstrap

    state_dir = tmp_path / "logs"
    state_dir.mkdir()
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(state_dir))
    monkeypatch.setenv("WORLDSIM_AUTO_MINT_STORAGE_STATE", "true")

    benchmark_root = tmp_path / "bench"
    benchmark_root.mkdir()
    stale_path = benchmark_root / "stale-state.json"
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
    fresh_path = (
        state_dir
        / "phase_0d"
        / "gitlab"
        / "instances"
        / bootstrap.phase_0d_instance_id(instance.model_dump())
        / "storage_state.json"
    )

    async def fake_reacquire_storage_state(*, site_name, instance, benchmark_root):
        captured["site_name"] = site_name
        captured["site_url"] = instance["site_url"]
        captured["benchmark_root"] = benchmark_root
        fresh_path.parent.mkdir(parents=True, exist_ok=True)
        _write_storage_state(fresh_path, domain="3.12.221.9")
        return fresh_path

    monkeypatch.setattr(bootstrap, "reacquire_storage_state", fake_reacquire_storage_state)

    result = asyncio.run(
        ensure_storage_state(
            instance,
            benchmark_root=benchmark_root,
            benchmark_name="WebArena Verified",
        )
    )

    assert result == fresh_path
    assert captured == {
        "site_name": "gitlab",
        "site_url": "http://3.12.221.9:8023",
        "benchmark_root": benchmark_root,
    }


def test_ensure_storage_state_auto_mints_missing_webarena_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from worldsim.phases import phase_0d_auth_bootstrap as bootstrap

    state_dir = tmp_path / "logs"
    state_dir.mkdir()
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(state_dir))
    monkeypatch.setenv("WORLDSIM_AUTO_MINT_STORAGE_STATE", "true")

    missing_path = "auth/missing-state.json"
    instance = BenchmarkInstance(
        site_name="gitlab",
        site_url="http://3.12.221.9:8023",
        reset_endpoint="http://3.12.221.9:8024/init",
        agent_auth={
            "type": "storage_state",
            "storage_state": {
                "path": missing_path,
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

    fresh_path = (
        state_dir
        / "phase_0d"
        / "gitlab"
        / "instances"
        / bootstrap.phase_0d_instance_id(instance.model_dump())
        / "storage_state.json"
    )

    async def fake_reacquire_storage_state(*, site_name, instance, benchmark_root):
        _ = site_name, instance, benchmark_root
        fresh_path.parent.mkdir(parents=True, exist_ok=True)
        _write_storage_state(fresh_path, domain="3.12.221.9")
        return fresh_path

    monkeypatch.setattr(bootstrap, "reacquire_storage_state", fake_reacquire_storage_state)

    result = asyncio.run(
        ensure_storage_state(
            instance,
            benchmark_root=tmp_path / "bench",
            benchmark_name="WebArena Verified",
        )
    )

    assert result == fresh_path
    report = inspect_storage_state_preflight([instance], benchmark_root=tmp_path / "bench")
    assert report.errors == ()
    assert report.mismatches == ()
