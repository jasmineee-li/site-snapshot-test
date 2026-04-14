"""Tests for the ``auth_mechanism`` feature.

Covers:
- Validator acceptance and rejection paths for the new schema.
- ``_resolve_auth`` dispatcher mapping for the first batch
  (``storage_state``, ``http_basic``, ``none``).
- Stub types raise ``NotImplementedError``.
- CLI ``--allow-unknown-auth`` gate behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldsim import _sandbox_validator as validator
from worldsim.browser_use_agent import (
    AuthArtifactMissingError,
    _resolve_auth,
)
from worldsim.main import _unknown_auth_sites, build_parser


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
        artifact = tmp_path / "state.json"
        artifact.write_text("{}", encoding="utf-8")
        auth = {
            "type": "storage_state",
            "storage_state": {"path": str(artifact)},
        }
        kwargs, deferred = _resolve_auth(auth, task=None, benchmark_root=None)
        assert kwargs == {"storage_state": str(artifact)}
        assert deferred == []

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
        assert kwargs == {"storage_state": str(artifact)}

    def test_storage_state_missing_raises(self, tmp_path: Path):
        auth = {
            "type": "storage_state",
            "storage_state": {"path": str(tmp_path / "does-not-exist.json")},
        }
        with pytest.raises(AuthArtifactMissingError):
            _resolve_auth(auth, task=None, benchmark_root=None)

    def test_storage_state_missing_with_generator_raises_informative(self, tmp_path: Path):
        auth = {
            "type": "storage_state",
            "storage_state": {
                "path": str(tmp_path / "nope.json"),
                "generator_script": "scripts/make_auth.py",
            },
        }
        with pytest.raises(AuthArtifactMissingError) as exc:
            _resolve_auth(auth, task=None, benchmark_root=None)
        assert "scripts/make_auth.py" in str(exc.value)

    def test_http_basic_maps_to_http_credentials(self):
        auth = {
            "type": "http_basic",
            "http_basic": {"username": "admin", "password": "s3cret"},
        }
        kwargs, deferred = _resolve_auth(auth, task=None, benchmark_root=None)
        assert kwargs == {"http_credentials": {"username": "admin", "password": "s3cret"}}
        assert deferred == []

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
        kwargs, deferred = _resolve_auth(auth, task=None, benchmark_root=None)
        assert kwargs == {
            "headers": {"X-M2-Customer-Auto-Login": "emma.lopez@gmail.com:Password.123"}
        }
        assert deferred == []

    def test_http_headers_rejects_empty(self):
        from worldsim.browser_use_agent import AuthArtifactMissingError

        auth = {"type": "http_headers", "http_headers": {"headers": {}}}
        with pytest.raises(AuthArtifactMissingError):
            _resolve_auth(auth, task=None, benchmark_root=None)

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
        artifact = tmp_path / "state.json"
        artifact.write_text("{}", encoding="utf-8")
        auth = {
            "type": "storage_state",
            "storage_state": {
                "path": str(artifact),
                "per_task_refresh": False,
            },
        }
        kwargs, _ = _resolve_auth(auth, task=None, benchmark_root=None)
        assert kwargs == {"storage_state": str(artifact)}

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

        auth = {
            "type": "storage_state",
            "storage_state": {
                "path": "auth/shopping_state.json",
                "generator_script": "scripts/make_auth.py",
            },
        }
        task = {"site": "shopping"}
        kwargs, _ = _resolve_auth(auth, task=task, benchmark_root=tmp_path / "bench")
        assert kwargs == {"storage_state": str(bootstrap_artifact)}


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
