"""Tests for Phase 0d's liveness-validated storage_state cache.

Covers the composition rule of :func:`_load_or_mint_storage_state`:

- Cache hit within SOFT_TTL skips the live probe and returns the cached state.
- Cache hit outside SOFT_TTL probes the storage_state browser session; alive -> reuse + bump
  ``last_validated_at``; dead -> re-mint via ``reacquire_storage_state``.
- ``WORLDSIM_STORAGE_STATE_FORCE_REMINT`` bypasses the cache entirely.
- A sidecar with a stale ``validator_version`` forces re-mint.
- Reddit's ``probe_authenticated`` remains True without any HTTP call (its
  per-request header auth cannot go stale).

The GitLab storage_state browser probe is exercised by a single focused test
that mocks ``session.get`` for ``/-/profile`` and asserts the 200 /
302-to-sign_in mapping.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import requests

from worldsim.editors.reddit import RedditEditor
from worldsim.phases import phase_0d_auth_bootstrap as bootstrap
from worldsim.storage_state_preflight import (
    read_storage_state_meta,
    write_storage_state_meta,
)

_INSTANCE: dict[str, Any] = {
    "site_name": "gitlab",
    "site_url": "http://gitlab.test",
    "agent_auth": {
        "type": "storage_state",
        "storage_state": {
            "path": "logs/phase_0d/gitlab/storage_state.json",
            "form_login": {
                "login_url": "/users/sign_in",
                "username_selector": "#user_login",
                "password_selector": "#user_password",
                "submit_selector": "input[type=submit]",
                "success_url_substring": "/-/profile",
            },
        },
        "authentication": {"credentials": {"username": "root", "password": "password"}},
    },
}


def _seed_artifact(state_dir: Path, *, last_validated_offset_seconds: float) -> Path:
    """Create a cached storage_state.json + sidecar pinned to a known timestamp."""
    artifact_path = bootstrap.phase_0d_artifact_path("gitlab", state_dir=state_dir)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "cookies": [
                    {
                        "name": "_gitlab_session",
                        "value": "cached",
                        "domain": "gitlab.test",
                        "path": "/",
                    }
                ],
                "origins": [],
            }
        ),
        encoding="utf-8",
    )
    last_validated = datetime.now(UTC) - timedelta(seconds=last_validated_offset_seconds)
    write_storage_state_meta(
        artifact_path,
        mechanism="form_login",
        last_validated_at=last_validated,
        validator_version=bootstrap.CURRENT_VALIDATOR_VERSION,
    )
    return artifact_path


def _setup_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    state_dir = tmp_path / "logs"
    state_dir.mkdir()
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(state_dir))
    monkeypatch.delenv(bootstrap.FORCE_REMINT_ENV, raising=False)
    return state_dir


def test_cache_hit_within_ttl_skips_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_dir = _setup_state_dir(tmp_path, monkeypatch)
    cached = _seed_artifact(state_dir, last_validated_offset_seconds=60)

    probe_calls: list[str] = []
    remint_calls: list[str] = []

    def fail_probe(self: Any) -> bool:
        probe_calls.append(self.site_name)
        raise AssertionError("probe must not run inside SOFT_TTL window")

    async def fail_remint(
        *, site_name: str, instance: dict[str, Any], benchmark_root: Any = None
    ) -> Path:
        remint_calls.append(site_name)
        raise AssertionError("re-mint must not run on a fresh cache hit")

    monkeypatch.setattr(bootstrap, "_storage_state_browser_liveness_check", fail_probe)
    monkeypatch.setattr(bootstrap, "reacquire_storage_state", fail_remint)

    result = asyncio.run(bootstrap._load_or_mint_storage_state(_INSTANCE))

    assert result == cached
    assert probe_calls == []
    assert remint_calls == []


def test_cache_hit_outside_ttl_runs_probe_and_returns_when_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = _setup_state_dir(tmp_path, monkeypatch)
    cached = _seed_artifact(
        state_dir,
        last_validated_offset_seconds=bootstrap.LIVENESS_SOFT_TTL_SECONDS + 120,
    )
    pre_meta = read_storage_state_meta(cached)
    assert pre_meta is not None
    pre_validated = pre_meta["last_validated_at"]

    probe_calls: list[str] = []
    remint_calls: list[str] = []

    def alive_probe(site_name: str, _instance: dict[str, Any], _session: Any) -> bool:
        probe_calls.append(site_name)
        return True

    async def fail_remint(
        *, site_name: str, instance: dict[str, Any], benchmark_root: Any = None
    ) -> Path:
        remint_calls.append(site_name)
        raise AssertionError("re-mint must not run when the live probe says alive")

    monkeypatch.setattr(bootstrap, "_storage_state_browser_liveness_check", alive_probe)
    monkeypatch.setattr(bootstrap, "reacquire_storage_state", fail_remint)

    result = asyncio.run(bootstrap._load_or_mint_storage_state(_INSTANCE))

    assert result == cached
    assert probe_calls == ["gitlab"]
    assert remint_calls == []
    refreshed_meta = read_storage_state_meta(cached)
    assert refreshed_meta is not None
    assert refreshed_meta["last_validated_at"] != pre_validated
    assert refreshed_meta["validator_version"] == bootstrap.CURRENT_VALIDATOR_VERSION


def test_cache_hit_outside_ttl_re_mints_when_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = _setup_state_dir(tmp_path, monkeypatch)
    _seed_artifact(
        state_dir,
        last_validated_offset_seconds=bootstrap.LIVENESS_SOFT_TTL_SECONDS + 120,
    )
    fresh_path = state_dir / "phase_0d" / "gitlab" / "instances" / "fresh" / "storage_state.json"
    fresh_path.parent.mkdir(parents=True, exist_ok=True)
    fresh_path.write_text(json.dumps({"cookies": [], "origins": []}), encoding="utf-8")

    probe_calls: list[str] = []
    remint_calls: list[str] = []

    def dead_probe(site_name: str, _instance: dict[str, Any], _session: Any) -> bool:
        probe_calls.append(site_name)
        return False

    async def fake_remint(
        *, site_name: str, instance: dict[str, Any], benchmark_root: Any = None
    ) -> Path:
        remint_calls.append(site_name)
        return fresh_path

    monkeypatch.setattr(bootstrap, "_storage_state_browser_liveness_check", dead_probe)
    monkeypatch.setattr(bootstrap, "reacquire_storage_state", fake_remint)

    result = asyncio.run(bootstrap._load_or_mint_storage_state(_INSTANCE))

    assert result == fresh_path
    assert probe_calls == ["gitlab"]
    assert remint_calls == ["gitlab"]


def test_force_remint_env_var_overrides_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = _setup_state_dir(tmp_path, monkeypatch)
    _seed_artifact(state_dir, last_validated_offset_seconds=60)
    fresh_path = state_dir / "phase_0d" / "gitlab" / "instances" / "fresh" / "storage_state.json"
    fresh_path.parent.mkdir(parents=True, exist_ok=True)
    fresh_path.write_text(json.dumps({"cookies": [], "origins": []}), encoding="utf-8")

    probe_calls: list[str] = []
    remint_calls: list[str] = []

    def fail_probe(self: Any) -> bool:
        probe_calls.append(self.site_name)
        raise AssertionError("probe must not run when the operator forced a re-mint")

    async def fake_remint(
        *, site_name: str, instance: dict[str, Any], benchmark_root: Any = None
    ) -> Path:
        remint_calls.append(site_name)
        return fresh_path

    monkeypatch.setattr(bootstrap, "_storage_state_browser_liveness_check", fail_probe)
    monkeypatch.setattr(bootstrap, "reacquire_storage_state", fake_remint)
    monkeypatch.setenv(bootstrap.FORCE_REMINT_ENV, "true")

    result = asyncio.run(bootstrap._load_or_mint_storage_state(_INSTANCE))

    assert result == fresh_path
    assert probe_calls == []
    assert remint_calls == ["gitlab"]


def test_validator_version_mismatch_re_mints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = _setup_state_dir(tmp_path, monkeypatch)
    cached = _seed_artifact(state_dir, last_validated_offset_seconds=60)
    # Downgrade the sidecar's validator_version below CURRENT_VALIDATOR_VERSION
    # so the cache must invalidate even though the TTL is fresh.
    sidecar = cached.with_name(cached.name.replace(".json", ".meta.json"))
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["validator_version"] = bootstrap.CURRENT_VALIDATOR_VERSION - 1
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    fresh_path = state_dir / "phase_0d" / "gitlab" / "instances" / "fresh" / "storage_state.json"
    fresh_path.parent.mkdir(parents=True, exist_ok=True)
    fresh_path.write_text(json.dumps({"cookies": [], "origins": []}), encoding="utf-8")

    probe_calls: list[str] = []
    remint_calls: list[str] = []

    def fail_probe(self: Any) -> bool:
        probe_calls.append(self.site_name)
        raise AssertionError("probe must not run when validator_version mismatches")

    async def fake_remint(
        *, site_name: str, instance: dict[str, Any], benchmark_root: Any = None
    ) -> Path:
        remint_calls.append(site_name)
        return fresh_path

    monkeypatch.setattr(bootstrap, "_storage_state_browser_liveness_check", fail_probe)
    monkeypatch.setattr(bootstrap, "reacquire_storage_state", fake_remint)

    result = asyncio.run(bootstrap._load_or_mint_storage_state(_INSTANCE))

    assert result == fresh_path
    assert probe_calls == []
    assert remint_calls == ["gitlab"]


def test_reddit_probe_always_true() -> None:
    """RedditEditor.probe_authenticated returns True without touching the network."""

    class _ExplodingSession:
        def get(self, *_a: Any, **_kw: Any) -> Any:
            raise AssertionError("reddit probe must not perform any HTTP request")

        def request(self, *_a: Any, **_kw: Any) -> Any:
            raise AssertionError("reddit probe must not perform any HTTP request")

    editor = RedditEditor({"site_name": "reddit"}, _ExplodingSession())  # type: ignore[arg-type]
    assert editor.probe_authenticated() is True


def test_gitlab_storage_state_browser_probe_maps_status_codes() -> None:
    """200 -> True; 302 to /users/sign_in -> False (no other plumbing tested)."""
    captured_calls: list[tuple[str, dict[str, Any]]] = []

    def make_session(response: requests.Response) -> Any:
        class _Session:
            def get(self, url: str, **kwargs: Any) -> requests.Response:
                captured_calls.append((url, kwargs))
                return response

        return _Session()

    instance = {"site_name": "gitlab", "site_url": "http://gitlab.test"}

    ok = SimpleNamespace(
        status_code=200,
        headers={},
        raise_for_status=lambda: None,
    )
    assert (
        bootstrap._storage_state_browser_liveness_check(  # type: ignore[arg-type]
            "gitlab", instance, make_session(ok)
        )
        is True
    )
    assert captured_calls[-1][0] == "http://gitlab.test/-/profile"

    redirect = SimpleNamespace(
        status_code=302,
        headers={"Location": "http://gitlab.test/users/sign_in"},
        raise_for_status=lambda: None,
    )
    assert (
        bootstrap._storage_state_browser_liveness_check(  # type: ignore[arg-type]
            "gitlab", instance, make_session(redirect)
        )
        is False
    )
