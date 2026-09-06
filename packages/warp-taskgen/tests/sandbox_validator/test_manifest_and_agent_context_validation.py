# ruff: noqa: F403, F405
"""Sandbox validator manifest and agent context validation.

Split from tests/test_sandbox_validator.py; shared helpers live in
tests/sandbox_validator/_fixtures.py.
"""

from __future__ import annotations

from ._fixtures import *


class TestValidateManifest:
    def test_rejects_missing_sites(self):
        errors = validator.validate_manifest({"evaluation": {}})
        assert any("sites" in e for e in errors)

    def test_accepts_valid(self):
        data = {
            "sites": [{"name": "shopping"}],
            "evaluation": {"eval_types": ["AgentResponseEvaluator"]},
        }
        errors = validator.validate_manifest(data)
        assert errors == []


class TestValidateAgentContext:
    def test_rejects_non_object_output_schema_when_structured_output_required(self):
        data = {
            "response_format": {
                "requires_structured_output": True,
                "output_schema": ["task_type", "status"],
                "per_task_format_field": None,
                "description": "The evaluator parses JSON output.",
            },
            "authentication": {
                "pre_authenticated": False,
                "credentials": {"username": "demo", "password": "secret"},
                "description": "Use the seeded test account.",
            },
            "agent_prompt_template": None,
            "site_context": {
                "platform_name": "Shopping",
                "description": "Ecommerce storefront.",
            },
        }

        errors = validator.validate_agent_context(data, site_name="shopping")

        assert any("output_schema must be an object" in e for e in errors)
