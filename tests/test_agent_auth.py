import json

from worldsim.agent_auth import resolve_agent_auth, resolve_storage_state_path


def _write_storage_state(path, *, domain="gitlab.test", same_site="no_restriction"):
    path.write_text(
        json.dumps(
            {
                "cookies": [
                    {
                        "name": "session",
                        "value": "abc",
                        "domain": domain,
                        "path": "/",
                        "sameSite": same_site,
                    }
                ],
                "origins": [{"origin": f"http://{domain}", "localStorage": []}],
            }
        ),
        encoding="utf-8",
    )


def test_storage_state_success_normalizes_same_site(tmp_path):
    path = tmp_path / "storage_state.json"
    _write_storage_state(path)

    resolved = resolve_agent_auth(
        {"type": "storage_state", "storage_state": {"path": str(path)}},
        site_name="gitlab",
        site_url="http://gitlab.test",
    )

    assert resolved.usable
    assert resolved.storage_state_path == path
    assert resolved.api_request_context_kwargs["storage_state"]["cookies"][0]["sameSite"] == "None"
    assert resolved.browser_context_kwargs == {
        "storage_state": resolved.api_request_context_kwargs["storage_state"]
    }


def test_storage_state_rejects_host_bound_cookie_for_other_host(tmp_path):
    path = tmp_path / "storage_state.json"
    _write_storage_state(path, domain="old-gitlab.test")

    resolved = resolve_agent_auth(
        {"type": "storage_state", "storage_state": {"path": str(path)}},
        site_name="gitlab",
        site_url="http://gitlab.test",
    )

    assert not resolved.usable
    assert "do not match live host" in (resolved.unusable_reason or "")


def test_http_headers_interpolate_credentials():
    resolved = resolve_agent_auth(
        {
            "type": "http_headers",
            "http_headers": {"headers": {"Authorization": "Basic ${credentials.username}:x"}},
            "authentication": {"credentials": {"username": "alice"}},
        },
        site_name="reddit",
        site_url="http://reddit.test",
    )

    assert resolved.usable
    assert resolved.api_request_context_kwargs == {
        "extra_http_headers": {"Authorization": "Basic alice:x"}
    }
    assert resolved.browser_context_kwargs == {
        "extra_http_headers": {"Authorization": "Basic alice:x"}
    }


def test_http_basic_maps_to_context_credentials():
    resolved = resolve_agent_auth(
        {
            "type": "http_basic",
            "http_basic": {"username": "alice", "password": "pw"},
        },
        site_name="site",
        site_url="http://site.test",
    )

    assert resolved.usable
    assert resolved.api_request_context_kwargs == {
        "http_credentials": {
            "username": "alice",
            "password": "pw",
            "origin": "http://site.test",
        }
    }
    assert resolved.browser_context_kwargs == resolved.api_request_context_kwargs


def test_http_basic_requires_valid_origin():
    resolved = resolve_agent_auth(
        {
            "type": "http_basic",
            "http_basic": {"username": "alice", "password": "pw"},
        },
        site_name="site",
        site_url="",
    )

    assert not resolved.usable
    assert "valid HTTP origin" in (resolved.unusable_reason or "")


def test_declared_storage_state_without_artifact_is_unusable(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path / "state"))

    resolved = resolve_agent_auth(
        {"type": "storage_state"},
        site_name="gitlab",
        site_url="http://gitlab.test",
    )

    assert not resolved.usable
    assert resolved.unusable_reason == "storage_state auth declared but no usable artifact was found"


def test_unknown_auth_is_unusable_for_preflight():
    resolved = resolve_agent_auth(
        {"type": "unknown"},
        site_name="gitlab",
        site_url="http://gitlab.test",
    )

    assert not resolved.usable
    assert "unknown" in (resolved.unusable_reason or "")


def test_relative_storage_state_path_cannot_escape_benchmark_root(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path / "state"))
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    outside = tmp_path / "outside" / "storage_state.json"
    outside.parent.mkdir()
    _write_storage_state(outside)

    resolved = resolve_storage_state_path(
        {"type": "storage_state", "storage_state": {"path": "../outside/storage_state.json"}},
        site_name="gitlab",
        benchmark_root=benchmark_root,
    )

    assert resolved is None


def test_absolute_declared_storage_state_cannot_escape_benchmark_root(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path / "state"))
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    outside = tmp_path / "outside" / "storage_state.json"
    outside.parent.mkdir()
    _write_storage_state(outside)

    resolved = resolve_storage_state_path(
        {"type": "storage_state", "storage_state": {"path": str(outside)}},
        site_name="gitlab",
        benchmark_root=benchmark_root,
    )

    assert resolved is None


def test_storage_state_override_cannot_escape_allowed_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path / "state"))
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    outside = tmp_path / "outside" / "storage_state.json"
    outside.parent.mkdir()
    _write_storage_state(outside)

    resolved = resolve_storage_state_path(
        {"type": "storage_state", "storage_state": {}},
        site_name="gitlab",
        storage_state_override=outside,
        benchmark_root=benchmark_root,
    )

    assert resolved is None


def test_storage_state_override_requires_benchmark_root(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path / "state"))
    override = tmp_path / "storage_state.json"
    _write_storage_state(override)

    resolved = resolve_storage_state_path(
        {"type": "storage_state", "storage_state": {}},
        site_name="gitlab",
        storage_state_override=override,
    )

    assert resolved is None


def test_storage_state_override_rejects_unsafe_site_name(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(state_dir))
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    phase_0d = state_dir / "phase_0d" / "gitlab" / "storage_state.json"
    phase_0d.parent.mkdir(parents=True)
    _write_storage_state(phase_0d)

    resolved = resolve_storage_state_path(
        {"type": "storage_state", "storage_state": {}},
        site_name="../gitlab",
        storage_state_override=phase_0d,
        benchmark_root=benchmark_root,
    )

    assert resolved is None


def test_storage_state_override_allows_phase_0d_root(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(state_dir))
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    phase_0d = state_dir / "phase_0d" / "gitlab" / "storage_state.json"
    phase_0d.parent.mkdir(parents=True)
    _write_storage_state(phase_0d)

    resolved = resolve_storage_state_path(
        {"type": "storage_state", "storage_state": {}},
        site_name="gitlab",
        storage_state_override=phase_0d,
        benchmark_root=benchmark_root,
    )

    assert resolved == phase_0d


def test_storage_state_fallback_rejects_unsafe_site_name(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(state_dir))
    escaped = state_dir / "escape" / "storage_state.json"
    escaped.parent.mkdir(parents=True)
    _write_storage_state(escaped)

    resolved = resolve_storage_state_path(
        {"type": "storage_state", "storage_state": {}},
        site_name="../escape",
    )

    assert resolved is None
