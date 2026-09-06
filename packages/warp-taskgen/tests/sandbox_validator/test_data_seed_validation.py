# ruff: noqa: F403, F405
"""Sandbox validator data seed validation (``validate_data_seed``).

Split from tests/test_sandbox_validator.py; shared helpers live in
tests/sandbox_validator/_fixtures.py.
"""

from __future__ import annotations

from ._fixtures import *


class TestValidateDataSeed:
    def test_rejects_missing_mechanism(self):
        errors = validator.validate_data_seed({"foo": "bar"})
        assert any("mechanism" in e for e in errors)

    def test_rejects_api_mechanism_as_deprecated(self):
        errors = validator.validate_data_seed(
            {
                "mechanism": "api",
                "api_calls": [{"method": "POST", "path": "/x", "body": {}}],
            }
        )
        assert any("deprecated" in error for error in errors)

    def test_rejects_form_mechanism_as_deprecated(self):
        errors = validator.validate_data_seed(
            {
                "mechanism": "form",
                "api_calls": [{"method": "POST", "path": "/x", "body_form": {"k": "v"}}],
            }
        )
        assert any("deprecated" in error for error in errors)

    def test_rejects_state_push_mechanism_as_deprecated(self):
        errors = validator.validate_data_seed({"mechanism": "state_push", "state": {}})
        assert any("deprecated" in error for error in errors)

    def test_rejects_sql_mechanism(self):
        seed = {
            "mechanism": "sql",
            "statements": ["INSERT INTO reviews (body) VALUES ('hello')"],
        }
        errors = validator.validate_data_seed(seed)
        assert any("unknown data seed mechanism" in e for e in errors)


class TestEmptyAdversarialDataSeed:
    """adversarial_data_seed with mechanism present but empty payload."""

    def test_sql_mechanism_rejected(self):
        seed = {"mechanism": "sql", "statements": []}
        errors = validator.validate_data_seed(seed)
        assert any("unknown data seed mechanism" in e for e in errors)

    def test_api_mechanism_rejected_as_deprecated(self):
        seed = {"mechanism": "api", "api_calls": []}
        errors = validator.validate_data_seed(seed)
        assert any("deprecated" in e for e in errors)

    def test_none_mechanism_allowed_when_allow_none(self):
        seed = {"mechanism": "none"}
        errors = validator.validate_data_seed(seed, allow_none=True)
        assert errors == []

    def test_none_mechanism_rejected_by_default(self):
        seed = {"mechanism": "none"}
        errors = validator.validate_data_seed(seed, allow_none=False)
        assert any("non-empty mechanism" in e for e in errors)

    def test_none_mechanism_rejects_editor_calls(self):
        seed = {
            "mechanism": "none",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "gitlab",
                    "method": "create_issue_title",
                    "args": {"project_id": "1", "title": "Seeded title"},
                }
            ],
        }
        errors = validator.validate_data_seed(seed, allow_none=True)
        assert any("must not include editor_calls" in e for e in errors)
