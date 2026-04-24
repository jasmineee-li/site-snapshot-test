"""Tests for the ``auth_mechanism`` feature.

Covers:
- Validator acceptance and rejection paths for the new schema.
- ``_resolve_auth`` dispatcher mapping for the first batch
  (``storage_state``, ``http_basic``, ``none``).
- Stub types raise ``NotImplementedError``.
- CLI ``--allow-unknown-auth`` gate behavior.
"""

from __future__ import annotations

import asyncio
import json
from argparse import Namespace
from pathlib import Path

import pytest

from worldsim import _sandbox_validator as validator
from worldsim.browser_use_agent import (
    AuthArtifactMissingError,
    _phase_0d_fallback_path,
    _resolve_auth,
    _ScopedHeaderAuthInjector,
)
from worldsim.main import _dispatch_phase, _unknown_auth_sites, build_parser


def _valid_agent_context_with_auth(auth_mechanism: dict | None) -> dict:
    """Return an otherwise-valid AGENT_CONTEXT.json augmented with auth_mechanism."""
    data = {
        "response_format": {
            "requires_structured_output": False,
            "output_schema": None,
            "per_task_format_field": None,
            "description": "Evaluation inspects browser state directly.",
        },
        "authentication": {
            "pre_authenticated": True,
            "credentials": None,
            "description": "Session pre-authenticated via storage state.",
        },
        "agent_prompt_template": None,
        "site_context": {
            "platform_name": "Test Site",
            "description": "A site for tests.",
        },
    }
    if auth_mechanism is not None:
        data["auth_mechanism"] = auth_mechanism
    return data


# ---------------------------------------------------------------------------
# Validator acceptance / rejection
# ---------------------------------------------------------------------------


class TestValidateAuthMechanism:
    def test_accepts_good_storage_state(self):
        data = _valid_agent_context_with_auth(
            {
                "type": "storage_state",
                "storage_state": {
                    "path": "auth/shopping_state.json",
                    "generator_script": None,
                    "per_task_refresh": False,
                },
            }
        )
        errors = validator.validate_agent_context(data, site_name="shopping")
        assert errors == [], errors

    def test_rejects_storage_state_missing_path(self):
        data = _valid_agent_context_with_auth(
            {
                "type": "storage_state",
                "storage_state": {"generator_script": None, "per_task_refresh": False},
            }
        )
        errors = validator.validate_agent_context(data, site_name="shopping")
        assert any("storage_state.path" in e for e in errors), errors

    def test_rejects_storage_state_empty_path(self):
        data = _valid_agent_context_with_auth(
            {
                "type": "storage_state",
                "storage_state": {"path": "   "},
            }
        )
        errors = validator.validate_agent_context(data, site_name="shopping")
        assert any("storage_state.path" in e for e in errors), errors

    def test_rejects_unknown_type_value(self):
        data = _valid_agent_context_with_auth({"type": "magic_cookie", "notes": "nope"})
        errors = validator.validate_agent_context(data, site_name="shopping")
        assert any("type" in e and "allowed set" in e for e in errors), errors

    def test_rejects_type_missing(self):
        data = _valid_agent_context_with_auth({"notes": "no type here"})
        errors = validator.validate_agent_context(data, site_name="shopping")
        assert any("auth_mechanism missing 'type'" in e for e in errors), errors

    def test_accepts_http_basic(self):
        data = _valid_agent_context_with_auth(
            {"type": "http_basic", "http_basic": {"username": "u", "password": "p"}}
        )
        errors = validator.validate_agent_context(data, site_name="shopping")
        assert errors == [], errors

    def test_rejects_http_basic_missing_password(self):
        data = _valid_agent_context_with_auth(
            {"type": "http_basic", "http_basic": {"username": "u"}}
        )
        errors = validator.validate_agent_context(data, site_name="shopping")
        assert any("http_basic.password" in e for e in errors), errors

    def test_none_requires_notes(self):
        data = _valid_agent_context_with_auth({"type": "none"})
        errors = validator.validate_agent_context(data, site_name="public")
        assert any("notes" in e for e in errors), errors

    def test_accepts_none_with_notes(self):
        data = _valid_agent_context_with_auth({"type": "none", "notes": "Fully public site."})
        errors = validator.validate_agent_context(data, site_name="public")
        assert errors == [], errors

    def test_unknown_type_accepted_with_notes(self):
        data = _valid_agent_context_with_auth(
            {"type": "unknown", "notes": "Auth unclear; needs review."}
        )
        errors = validator.validate_agent_context(data, site_name="mystery")
        assert errors == [], errors

    def test_rejects_multiple_populated_sub_objects(self):
        data = _valid_agent_context_with_auth(
            {
                "type": "storage_state",
                "storage_state": {"path": "auth/s.json"},
                "http_basic": {"username": "u", "password": "p"},
            }
        )
        errors = validator.validate_agent_context(data, site_name="shopping")
        assert any("extra sub-objects" in e for e in errors), errors

    def test_rejects_none_with_populated_sub(self):
        data = _valid_agent_context_with_auth(
            {
                "type": "none",
                "notes": "public",
                "http_basic": {"username": "u", "password": "p"},
            }
        )
        errors = validator.validate_agent_context(data, site_name="public")
        assert any("must not populate any sub-object" in e for e in errors), errors

    def test_existing_authentication_only_schema_still_valid(self):
        # Legacy contexts (no auth_mechanism) must still validate — this is the
        # migration safety-net for the existing 246-test suite.
        data = _valid_agent_context_with_auth(None)
        errors = validator.validate_agent_context(data, site_name="legacy")
        assert errors == [], errors


# ---------------------------------------------------------------------------
# _resolve_auth dispatcher
# ---------------------------------------------------------------------------


class TestResolveAuth:
    def test_none_is_noop(self):
        kwargs, deferred = _resolve_auth(
            {"type": "none", "notes": "public"}, task=None, benchmark_root=None
        )
        assert kwargs == {}
        assert deferred == []

    def test_empty_auth_mechanism_is_noop(self):
        kwargs, deferred = _resolve_auth(None, task=None, benchmark_root=None)
        assert kwargs == {}
        assert deferred == []

    def test_unknown_is_noop(self):
        kwargs, deferred = _resolve_auth(
            {"type": "unknown", "notes": "needs review"},
            task=None,
            benchmark_root=None,
        )
        assert kwargs == {}
        assert deferred == []

    def test_storage_state_absolute_path(self, tmp_path: Path):
        bench = tmp_path / "bench"
        bench.mkdir()
        artifact = bench / "state.json"
        artifact.write_text("{}", encoding="utf-8")
        auth = {
            "type": "storage_state",
            "storage_state": {"path": str(artifact)},
        }
        kwargs, deferred = _resolve_auth(auth, task=None, benchmark_root=bench)
        assert kwargs == {"storage_state": {}}
        assert deferred == []

    def test_storage_state_absolute_path_requires_benchmark_root(self, tmp_path: Path):
        artifact = tmp_path / "state.json"
        artifact.write_text("{}", encoding="utf-8")
        auth = {
            "type": "storage_state",
            "storage_state": {"path": str(artifact)},
        }
        with pytest.raises(AuthArtifactMissingError, match="requires a benchmark root"):
            _resolve_auth(auth, task=None, benchmark_root=None)

    def test_storage_state_relative_path_resolves_against_benchmark_root(self, tmp_path: Path):
        bench = tmp_path / "bench"
        (bench / "auth").mkdir(parents=True)
        artifact = bench / "auth" / "state.json"
        artifact.write_text("{}", encoding="utf-8")
        auth = {
            "type": "storage_state",
            "storage_state": {"path": "auth/state.json"},
        }
        kwargs, _ = _resolve_auth(auth, task=None, benchmark_root=bench)
        assert kwargs == {"storage_state": {}}

    def test_storage_state_relative_path_requires_benchmark_root(self):
        auth = {
            "type": "storage_state",
            "storage_state": {"path": "auth/state.json"},
        }
        with pytest.raises(AuthArtifactMissingError, match="requires a benchmark root"):
            _resolve_auth(auth, task=None, benchmark_root=None)

    def test_storage_state_relative_path_cannot_escape_benchmark_root(self, tmp_path: Path):
        bench = tmp_path / "bench"
        bench.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "state.json").write_text("{}", encoding="utf-8")
        auth = {
            "type": "storage_state",
            "storage_state": {"path": "../outside/state.json"},
        }
        with pytest.raises(AuthArtifactMissingError, match="outside benchmark root"):
            _resolve_auth(auth, task=None, benchmark_root=bench)

    def test_storage_state_absolute_path_cannot_escape_benchmark_root(self, tmp_path: Path):
        bench = tmp_path / "bench"
        bench.mkdir()
        outside = tmp_path / "outside" / "state.json"
        outside.parent.mkdir()
        outside.write_text("{}", encoding="utf-8")
        auth = {
            "type": "storage_state",
            "storage_state": {"path": str(outside)},
        }
        with pytest.raises(AuthArtifactMissingError, match="outside benchmark root"):
            _resolve_auth(auth, task=None, benchmark_root=bench)

    def test_storage_state_rejects_host_bound_cookie_for_other_site(self, tmp_path: Path):
        bench = tmp_path / "bench"
        bench.mkdir()
        artifact = bench / "state.json"
        artifact.write_text(
            json.dumps(
                {
                    "cookies": [
                        {
                            "name": "session",
                            "value": "abc",
                            "domain": "old-gitlab.test",
                            "path": "/",
                        }
                    ],
                    "origins": [],
                }
            ),
            encoding="utf-8",
        )
        auth = {
            "type": "storage_state",
            "storage_state": {"path": str(artifact)},
        }
        with pytest.raises(AuthArtifactMissingError, match="do not match live host"):
            _resolve_auth(
                auth,
                task=None,
                benchmark_root=bench,
                site_url="http://gitlab.test",
            )

    def test_storage_state_missing_raises(self, tmp_path: Path):
        bench = tmp_path / "bench"
        bench.mkdir()
        auth = {
            "type": "storage_state",
            "storage_state": {"path": str(bench / "does-not-exist.json")},
        }
        with pytest.raises(AuthArtifactMissingError):
            _resolve_auth(auth, task=None, benchmark_root=bench)

    def test_storage_state_missing_with_generator_raises_informative(self, tmp_path: Path):
        bench = tmp_path / "bench"
        bench.mkdir()
        auth = {
            "type": "storage_state",
            "storage_state": {
                "path": str(bench / "nope.json"),
                "generator_script": "scripts/make_auth.py",
            },
        }
        with pytest.raises(AuthArtifactMissingError) as exc:
            _resolve_auth(auth, task=None, benchmark_root=bench)
        assert "scripts/make_auth.py" in str(exc.value)

    def test_http_basic_maps_to_http_credentials(self):
        auth = {
            "type": "http_basic",
            "http_basic": {"username": "admin", "password": "s3cret"},
        }
        kwargs, deferred = _resolve_auth(
            auth,
            task=None,
            benchmark_root=None,
            site_url="http://site.test/start",
        )
        assert kwargs == {
            "http_credentials": {
                "username": "admin",
                "password": "s3cret",
                "origin": "http://site.test",
            }
        }
        assert deferred == []

    def test_http_basic_requires_site_url_for_origin_scope(self):
        auth = {
            "type": "http_basic",
            "http_basic": {"username": "admin", "password": "s3cret"},
        }
        with pytest.raises(AuthArtifactMissingError, match="valid site_url"):
            _resolve_auth(auth, task=None, benchmark_root=None)

    @pytest.mark.parametrize(
        "mech_type",
        ["form_login", "pre_auth_script", "client_cert"],
    )
    def test_unimplemented_types_raise_not_implemented(self, mech_type: str):
        auth = {"type": mech_type, mech_type: {}}
        with pytest.raises(NotImplementedError) as exc:
            _resolve_auth(auth, task=None, benchmark_root=None)
        # Message should mention the type and point at the plan.
        assert mech_type in str(exc.value)

    def test_http_headers_resolves_with_credentials_interpolation(self):
        """http_headers pipes `extra_http_headers` and interpolates credentials."""
        auth = {
            "type": "http_headers",
            "http_headers": {
                "headers": {
                    "X-M2-Customer-Auto-Login": "${credentials.username}:${credentials.password}"
                }
            },
            "authentication": {
                "credentials": {"username": "emma.lopez@gmail.com", "password": "Password.123"}
            },
        }
        kwargs, deferred = _resolve_auth(
            auth,
            task=None,
            benchmark_root=None,
            site_url="http://reddit.test",
        )
        assert kwargs == {}
        assert len(deferred) == 1

    def test_http_headers_preserves_static_values_without_credentials(self):
        auth = {
            "type": "http_headers",
            "http_headers": {
                "headers": {
                    "X-Static-Token": "fixed-token",
                }
            },
        }
        kwargs, deferred = _resolve_auth(
            auth,
            task=None,
            benchmark_root=None,
            site_url="http://reddit.test",
        )
        assert kwargs == {}
        assert len(deferred) == 1

    def test_http_headers_accepts_username_only_placeholder(self):
        auth = {
            "type": "http_headers",
            "headers": {"X-User": "${credentials.username}"},
            "authentication": {"credentials": {"username": "alice"}},
        }
        kwargs, deferred = _resolve_auth(
            auth,
            task=None,
            benchmark_root=None,
            site_url="http://reddit.test",
        )
        assert kwargs == {}
        assert len(deferred) == 1

    def test_http_headers_accepts_password_only_placeholder(self):
        auth = {
            "type": "http_headers",
            "headers": {"X-Password": "${credentials.password}"},
            "authentication": {"credentials": {"password": "secret"}},
        }
        kwargs, deferred = _resolve_auth(
            auth,
            task=None,
            benchmark_root=None,
            site_url="http://reddit.test",
        )
        assert kwargs == {}
        assert len(deferred) == 1

    def test_http_headers_requires_site_url_for_origin_scope(self):
        auth = {
            "type": "http_headers",
            "http_headers": {"headers": {"X-Static-Token": "fixed-token"}},
        }
        with pytest.raises(AuthArtifactMissingError, match="valid site_url"):
            _resolve_auth(auth, task=None, benchmark_root=None)

    def test_http_headers_rejects_missing_credentials_for_placeholders(self):
        auth = {
            "type": "http_headers",
            "http_headers": {
                "headers": {
                    "X-M2-Customer-Auto-Login": "${credentials.username}:${credentials.password}"
                }
            },
        }
        with pytest.raises(
            AuthArtifactMissingError,
            match=r"references \$\{credentials.username\} without credentials",
        ):
            _resolve_auth(auth, task=None, benchmark_root=None)

    def test_http_headers_rejects_empty(self):
        from worldsim.browser_use_agent import AuthArtifactMissingError

        auth = {"type": "http_headers", "http_headers": {"headers": {}}}
        with pytest.raises(AuthArtifactMissingError):
            _resolve_auth(auth, task=None, benchmark_root=None)

    @pytest.mark.asyncio
    async def test_scoped_header_injector_adds_headers_only_same_origin(self):
        class _RegisterFetch:
            def __init__(self):
                self.handler = None

            def requestPaused(self, handler):
                self.handler = handler

        class _Register:
            def __init__(self):
                self.Fetch = _RegisterFetch()

        class _FetchSend:
            def __init__(self):
                self.enabled = []
                self.continued = []

            async def enable(self, params, *, session_id):
                self.enabled.append((params, session_id))

            async def continueRequest(self, params, *, session_id):
                self.continued.append((params, session_id))

        class _Send:
            def __init__(self):
                self.Fetch = _FetchSend()

        class _CDPClient:
            def __init__(self):
                self.register = _Register()
                self.send = _Send()

        class _Target:
            target_id = "target-1"

        class _SessionManager:
            def get_all_page_targets(self):
                return [_Target()]

        class _CDPSession:
            def __init__(self, client):
                self.cdp_client = client
                self.session_id = "session-1"

        class _BrowserSession:
            def __init__(self):
                self.cdp_client = _CDPClient()
                self.session_manager = _SessionManager()

            async def get_or_create_cdp_session(self, target_id, focus=False):
                return _CDPSession(self.cdp_client)

        session = _BrowserSession()
        injector = _ScopedHeaderAuthInjector(
            origin="http://reddit.test",
            headers={"X-Postmill-Auto-Login": "alice:pw"},
        )
        await injector.start(session)
        handler = session.cdp_client.register.Fetch.handler
        handler(
            {
                "requestId": "same",
                "request": {"url": "http://reddit.test/f/books", "headers": {"A": "B"}},
            },
            "session-1",
        )
        handler(
            {"requestId": "other", "request": {"url": "http://evil.test/", "headers": {}}},
            "session-1",
        )
        await asyncio.sleep(0)
        injector._running = False
        if injector._poll_task is not None:
            injector._poll_task.cancel()

        same, other = session.cdp_client.send.Fetch.continued
        assert same[0]["headers"] == [
            {"name": "A", "value": "B"},
            {"name": "X-Postmill-Auto-Login", "value": "alice:pw"},
        ]
        assert "headers" not in other[0]

    def test_per_task_refresh_true_raises_not_implemented(self, tmp_path: Path):
        """Schema-legal but runtime-unimplemented. Fail fast instead of silently stale."""
        artifact = tmp_path / "state.json"
        artifact.write_text("{}", encoding="utf-8")
        auth = {
            "type": "storage_state",
            "storage_state": {
                "path": str(artifact),
                "per_task_refresh": True,
            },
        }
        with pytest.raises(NotImplementedError) as exc:
            _resolve_auth(auth, task=None, benchmark_root=None)
        assert "per_task_refresh" in str(exc.value)

    def test_per_task_refresh_false_is_fine(self, tmp_path: Path):
        bench = tmp_path / "bench"
        bench.mkdir()
        artifact = bench / "state.json"
        artifact.write_text("{}", encoding="utf-8")
        auth = {
            "type": "storage_state",
            "storage_state": {
                "path": str(artifact),
                "per_task_refresh": False,
            },
        }
        kwargs, _ = _resolve_auth(auth, task=None, benchmark_root=bench)
        assert kwargs == {"storage_state": {}}

    def test_phase_0d_fallback_used_when_declared_path_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """When the declared path is missing but Phase 0d output exists, use it."""
        state_dir = tmp_path / "logs"
        state_dir.mkdir()
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(state_dir))

        bootstrap_dir = state_dir / "phase_0d" / "shopping"
        bootstrap_dir.mkdir(parents=True)
        bootstrap_artifact = bootstrap_dir / "storage_state.json"
        bootstrap_artifact.write_text("{}", encoding="utf-8")
        (bootstrap_dir / "completion.json").write_text("{}", encoding="utf-8")

        auth = {
            "type": "storage_state",
            "storage_state": {
                "path": "auth/shopping_state.json",
                "generator_script": "scripts/make_auth.py",
            },
        }
        task = {"site": "shopping"}
        kwargs, _ = _resolve_auth(auth, task=task, benchmark_root=tmp_path / "bench")
        assert kwargs == {"storage_state": {}}

    def test_phase_0d_fallback_ignores_orphan_artifact_without_completion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        state_dir = tmp_path / "logs"
        state_dir.mkdir()
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(state_dir))

        bootstrap_dir = state_dir / "phase_0d" / "shopping"
        bootstrap_dir.mkdir(parents=True)
        (bootstrap_dir / "storage_state.json").write_text("{}", encoding="utf-8")

        auth = {
            "type": "storage_state",
            "storage_state": {
                "path": "auth/shopping_state.json",
                "generator_script": "scripts/make_auth.py",
            },
        }
        with pytest.raises(AuthArtifactMissingError, match="storage_state artifact missing"):
            _resolve_auth(auth, task={"site": "shopping"}, benchmark_root=tmp_path / "bench")

    def test_phase_0d_fallback_rejects_unsafe_site_name(self, tmp_path: Path, monkeypatch):
        state_dir = tmp_path / "logs"
        state_dir.mkdir()
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(state_dir))
        assert _phase_0d_fallback_path({"site": "../escape"}) is None


# ---------------------------------------------------------------------------
# CLI gate: --allow-unknown-auth
# ---------------------------------------------------------------------------


class TestUnknownAuthGate:
    def _write_ctx(self, site_dir: Path, auth_mechanism: dict | None) -> None:
        site_dir.mkdir(parents=True, exist_ok=True)
        ctx = {
            "response_format": {
                "requires_structured_output": False,
                "output_schema": None,
                "per_task_format_field": None,
                "description": "Test",
            },
            "authentication": {
                "pre_authenticated": False,
                "credentials": None,
                "description": "Test",
            },
            "agent_prompt_template": None,
            "site_context": {"platform_name": "Test", "description": "Test."},
        }
        if auth_mechanism is not None:
            ctx["auth_mechanism"] = auth_mechanism
        (site_dir / "AGENT_CONTEXT.json").write_text(json.dumps(ctx), encoding="utf-8")

    def test_unknown_auth_sites_detected(self, tmp_path: Path):
        phase0c = tmp_path / "phase_0c"
        self._write_ctx(
            phase0c / "site_a",
            {"type": "unknown", "notes": "needs review"},
        )
        self._write_ctx(
            phase0c / "site_b",
            {"type": "none", "notes": "public"},
        )
        unknown = _unknown_auth_sites(tmp_path)
        assert unknown == ["site_a"]

    def test_unknown_auth_sites_empty_when_no_profile_dir(self, tmp_path: Path):
        assert _unknown_auth_sites(tmp_path) == []

    def test_unknown_auth_sites_ignores_unknown_profile_when_instances_supply_agent_auth(
        self, tmp_path: Path
    ):
        phase0c = tmp_path / "phase_0c"
        self._write_ctx(
            phase0c / "site_a",
            {"type": "unknown", "notes": "needs review"},
        )

        unknown = _unknown_auth_sites(
            tmp_path,
            instances=[
                {
                    "site_name": "site_a",
                    "agent_auth": {"type": "storage_state", "storage_state": {"path": "auth/s.json"}},
                }
            ],
        )

        assert unknown == []

    def test_unknown_auth_sites_rejects_malformed_context_artifacts(self, tmp_path: Path):
        phase0c = tmp_path / "phase_0c"
        phase0c.mkdir(parents=True)
        (phase0c / "AGENT_CONTEXT_broken.json").write_text("{not-json", encoding="utf-8")

        with pytest.raises(RuntimeError, match="Failed to read Phase 0c AGENT_CONTEXT"):
            _unknown_auth_sites(tmp_path)

    def test_cli_parses_allow_unknown_auth_flag(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "phase",
                "3",
                "--instances",
                "instances.json",
                "--allow-unknown-auth",
            ]
        )
        assert args.allow_unknown_auth is True

    def test_cli_defaults_allow_unknown_auth_false(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "phase",
                "3",
                "--instances",
                "instances.json",
            ]
        )
        assert args.allow_unknown_auth is False

    def test_dispatch_phase_refuses_phase_4_unknown_auth_without_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ):
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
        self._write_ctx(
            tmp_path / "phase_0c" / "site_a",
            {"type": "unknown", "notes": "needs review"},
        )

        called = False

        async def fake_phase_4_run(args):
            nonlocal called
            called = True
            return 0

        monkeypatch.setattr("worldsim.phases.phase_4_adversarial.run", fake_phase_4_run)

        rc = _dispatch_phase(
            Namespace(
                phase="4",
                benchmark=None,
                config=None,
                instances=None,
                agent_model=None,
                sandbox_model=None,
                agent_provider=None,
                allow_unknown_auth=False,
            )
        )

        assert rc == 2
        assert called is False
        assert "Phase 4 refused" in capsys.readouterr().err

    def test_dispatch_phase_allows_phase_4_unknown_auth_with_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
        self._write_ctx(
            tmp_path / "phase_0c" / "site_a",
            {"type": "unknown", "notes": "needs review"},
        )

        called = False

        async def fake_phase_4_run(args):
            nonlocal called
            called = True
            assert args.phase == "4"
            return 0

        monkeypatch.setattr("worldsim.phases.phase_4_adversarial.run", fake_phase_4_run)

        rc = _dispatch_phase(
            Namespace(
                phase="4",
                benchmark=None,
                config=None,
                instances=None,
                agent_model=None,
                sandbox_model=None,
                agent_provider=None,
                allow_unknown_auth=True,
            )
        )

        assert rc == 0
        assert called is True

    def test_dispatch_phase_allows_phase_4_when_instances_supply_agent_auth(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
        self._write_ctx(
            tmp_path / "phase_0c" / "site_a",
            {"type": "unknown", "notes": "needs review"},
        )
        instances_path = tmp_path / "instances.json"
        instances_path.write_text(
            json.dumps(
                {
                    "benchmark_name": "WebArena Verified",
                    "benchmark_codebase": str(tmp_path),
                    "instances": [
                        {
                            "site_name": "site_a",
                            "site_url": "http://site-a.test",
                            "agent_auth": {
                                "type": "storage_state",
                                "storage_state": {"path": "auth/state.json"},
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        called = False

        async def fake_phase_4_run(args):
            nonlocal called
            called = True
            return 0

        monkeypatch.setattr("worldsim.phases.phase_4_adversarial.run", fake_phase_4_run)

        rc = _dispatch_phase(
            Namespace(
                phase="4",
                benchmark=None,
                config=None,
                instances=instances_path,
                agent_model=None,
                sandbox_model=None,
                agent_provider=None,
                allow_unknown_auth=False,
            )
        )

        assert rc == 0
        assert called is True
