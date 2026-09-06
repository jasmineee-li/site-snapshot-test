"""Preflight auth resolution and request-context options."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from warp_taskgen.phase_2.phase_2c import (
    auth_preflight,
    probes,
)

from ._fixtures import (
    _bypass_preflight,  # noqa: F401
    _gitlab_instance,
    _stable_git_fingerprint,  # noqa: F401
)


def _write_storage_state(path: Path, *, domain: str = "gitlab.example") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cookies": [
                    {
                        "name": "session",
                        "value": "abc",
                        "domain": domain,
                        "path": "/",
                        "sameSite": "Lax",
                    }
                ],
                "origins": [{"origin": f"https://{domain}", "localStorage": []}],
            }
        )
    )


def test_preflight_auth_resolves_storage_state_relative_to_state_dir(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(state_dir))
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    storage_path = state_dir / "auth" / "storage_state.json"
    _write_storage_state(storage_path)
    instance = _gitlab_instance(
        agent_auth={"type": "storage_state", "storage_state": {"path": "auth/storage_state.json"}}
    )

    options, reason = auth_preflight._preflight_request_context_options(
        instance,
        benchmark_root=benchmark_root,
    )

    assert reason is None
    assert options["storage_state"]["cookies"][0]["name"] == "session"


def test_preflight_auth_rejects_storage_state_escape_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path / "state"))
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    outside = tmp_path / "outside" / "storage_state.json"
    _write_storage_state(outside)
    instance = _gitlab_instance(
        agent_auth={
            "type": "storage_state",
            "storage_state": {"path": "../outside/storage_state.json"},
        }
    )

    options, reason = auth_preflight._preflight_request_context_options(
        instance,
        benchmark_root=benchmark_root,
    )

    assert options == {}
    assert reason == "storage_state auth declared but no usable artifact was found"


def test_resolve_benign_storage_state_path_prefers_nested_agent_auth(tmp_path):
    state_path = tmp_path / "gitlab-state.json"
    state_path.write_text(json.dumps({"cookies": []}))

    resolved = auth_preflight._resolve_benign_storage_state_path(
        _gitlab_instance(
            benchmark_root=str(tmp_path),
            agent_auth={
                "type": "storage_state",
                "storage_state": {"path": str(state_path)},
            },
        )
    )

    assert resolved == str(state_path)


def test_resolve_benign_storage_state_path_falls_back_to_phase_0d(tmp_path, monkeypatch):
    fallback = tmp_path / "phase_0d" / "gitlab" / "storage_state.json"
    fallback.parent.mkdir(parents=True)
    fallback.write_text(json.dumps({"cookies": []}))
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))

    assert auth_preflight._resolve_benign_storage_state_path(
        _gitlab_instance(agent_auth={"type": "storage_state", "storage_state": {}})
    ) == str(fallback)


def test_resolve_benign_storage_state_path_requires_storage_state_auth_for_fallback(
    tmp_path, monkeypatch
):
    fallback = tmp_path / "phase_0d" / "gitlab" / "storage_state.json"
    fallback.parent.mkdir(parents=True)
    fallback.write_text(json.dumps({"cookies": []}))
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))

    assert auth_preflight._resolve_benign_storage_state_path(_gitlab_instance()) is None


def test_resolve_benign_storage_state_path_continues_past_missing_explicit_path(
    tmp_path, monkeypatch
):
    fallback = tmp_path / "phase_0d" / "gitlab" / "storage_state.json"
    fallback.parent.mkdir(parents=True)
    fallback.write_text(json.dumps({"cookies": []}))
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))

    resolved = auth_preflight._resolve_benign_storage_state_path(
        _gitlab_instance(
            storage_state_path=str(tmp_path / "missing.json"),
            agent_auth={"type": "storage_state", "storage_state": {}},
        )
    )

    assert resolved == str(fallback)


def test_resolve_benign_storage_state_path_ignores_nested_path_for_non_storage_auth(tmp_path):
    nested_path = tmp_path / "nested.json"
    nested_path.write_text(json.dumps({"cookies": []}))

    resolved = auth_preflight._resolve_benign_storage_state_path(
        _gitlab_instance(
            agent_auth={
                "type": "none",
                "storage_state": {"path": str(nested_path)},
            }
        )
    )

    assert resolved is None


def test_resolve_benign_browser_context_auth_supports_http_headers():
    context, reason = probes._resolve_benign_browser_context_auth(
        _gitlab_instance(
            agent_auth={
                "type": "http_headers",
                "http_headers": {"headers": {"X-User": "${credentials.username}"}},
                "authentication": {"credentials": {"username": "alice", "password": "pw"}},
            }
        )
    )

    assert reason is None
    assert context == {"extra_http_headers": {"X-User": "alice"}}


def test_preflight_request_context_uses_agent_http_headers():
    context, reason = auth_preflight._preflight_request_context_options(
        _gitlab_instance(
            agent_auth={
                "type": "http_headers",
                "http_headers": {
                    "headers": {
                        "X-User": "${credentials.username}",
                        "X-Static": "ok",
                    }
                },
                "authentication": {"credentials": {"username": "alice", "password": "pw"}},
            }
        )
    )

    assert reason is None
    assert context == {"extra_http_headers": {"X-User": "alice", "X-Static": "ok"}}


def test_preflight_request_context_rejects_host_bound_storage_state(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"cookies": [{"name": "s", "value": "1", "domain": "old.example"}]})
    )

    context, reason = auth_preflight._preflight_request_context_options(
        _gitlab_instance(
            benchmark_root=str(tmp_path),
            site_url="http://new.example:8023",
            storage_state_path=str(state_path),
            agent_auth={"type": "storage_state", "storage_state": {"path": str(state_path)}},
        ),
        benchmark_root=tmp_path,
    )

    assert context == {}
    assert reason is not None
    assert "do not match live host" in reason


def test_preflight_request_context_rejects_host_bound_cookies_even_with_matching_origin(
    tmp_path,
):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "cookies": [{"name": "s", "value": "1", "domain": "old.example"}],
                "origins": [{"origin": "https://gitlab.example"}],
            }
        )
    )

    context, reason = auth_preflight._preflight_request_context_options(
        _gitlab_instance(
            benchmark_root=str(tmp_path),
            storage_state_path=str(state_path),
            agent_auth={"type": "storage_state", "storage_state": {"path": str(state_path)}},
        ),
        benchmark_root=tmp_path,
    )

    assert context == {}
    assert reason is not None
    assert "mixes live host" in reason
    assert "old.example" in reason


def test_preflight_request_context_normalizes_storage_state_samesite(tmp_path):
    state_path = tmp_path / "state.json"
    original_payload = {
        "cookies": [
            {
                "name": "a",
                "value": "1",
                "domain": "gitlab.example",
                "sameSite": "no_restriction",
            },
            {
                "name": "b",
                "value": "2",
                "domain": "gitlab.example",
                "sameSite": "",
            },
            {
                "name": "c",
                "value": "3",
                "domain": "gitlab.example",
                "sameSite": "lax",
            },
            {
                "name": "d",
                "value": "4",
                "domain": "gitlab.example",
                "sameSite": None,
            },
            {
                "name": "e",
                "value": "5",
                "domain": "gitlab.example",
            },
            {
                "name": "f",
                "value": "6",
                "domain": "gitlab.example",
                "sameSite": "unspecified",
            },
        ]
    }
    state_path.write_text(json.dumps(original_payload))

    context, reason = auth_preflight._preflight_request_context_options(
        _gitlab_instance(
            benchmark_root=str(tmp_path),
            storage_state_path=str(state_path),
            agent_auth={"type": "storage_state", "storage_state": {"path": str(state_path)}},
        ),
        benchmark_root=tmp_path,
    )

    assert reason is None
    storage_state = context["storage_state"]
    assert isinstance(storage_state, dict)
    cookies = storage_state["cookies"]
    assert cookies[0]["sameSite"] == "None"
    assert cookies[1]["sameSite"] == "Lax"
    assert cookies[2]["sameSite"] == "Lax"
    assert cookies[3]["sameSite"] == "Lax"
    assert cookies[4]["sameSite"] == "Lax"
    assert cookies[5]["sameSite"] == "Lax"
    assert {cookie["sameSite"] for cookie in cookies} <= {"Strict", "Lax", "None"}
    assert json.loads(state_path.read_text()) == original_payload


def test_preflight_request_context_reads_storage_state_once(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "cookies": [
                    {
                        "name": "s",
                        "value": "1",
                        "domain": "gitlab.example",
                        "sameSite": "Lax",
                    }
                ]
            }
        )
    )
    calls: list[Path] = []
    original_read_text = Path.read_text

    def counted_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == state_path:
            calls.append(self)
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted_read_text)

    context, reason = auth_preflight._preflight_request_context_options(
        _gitlab_instance(
            benchmark_root=str(tmp_path),
            storage_state_path=str(state_path),
            agent_auth={"type": "storage_state", "storage_state": {"path": str(state_path)}},
        ),
        benchmark_root=tmp_path,
    )

    assert reason is None
    assert context["storage_state"]["cookies"][0]["sameSite"] == "Lax"
    assert calls == [state_path]


def test_preflight_request_context_skips_unsupported_storage_state_samesite(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "cookies": [
                    {
                        "name": "a",
                        "value": "1",
                        "domain": "gitlab.example",
                        "sameSite": "mystery",
                    }
                ]
            }
        )
    )

    context, reason = auth_preflight._preflight_request_context_options(
        _gitlab_instance(
            benchmark_root=str(tmp_path),
            storage_state_path=str(state_path),
            agent_auth={"type": "storage_state", "storage_state": {"path": str(state_path)}},
        ),
        benchmark_root=tmp_path,
    )

    assert context == {}
    assert reason is not None
    assert "unsupported sameSite" in reason
