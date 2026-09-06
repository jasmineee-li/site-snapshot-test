# ruff: noqa: F403, F405
"""Sandbox validator diagnosis and triage validation.

Split from tests/test_sandbox_validator.py; shared helpers live in
tests/sandbox_validator/_fixtures.py.
"""

from __future__ import annotations

from ._fixtures import *


class TestValidateDiagnosis:
    def test_rejects_invalid_root_cause(self):
        data = {
            "root_cause": "sunspots",
            "explanation": "blame the sun",
        }
        errors = validator.validate_diagnosis(data)
        assert any("root_cause" in e for e in errors)

    def test_accepts_valid(self):
        data = {
            "root_cause": "reward_bug",
            "explanation": "reward function has wrong eval config",
            "suggested_fix": {"target": "reward_function"},
        }
        errors = validator.validate_diagnosis(data)
        assert errors == []

    def test_rejects_suggested_fix_without_target(self):
        data = {
            "root_cause": "seed_bug",
            "explanation": "seed is wrong",
            "suggested_fix": {"patch": {}},
        }
        errors = validator.validate_diagnosis(data)
        assert any("target" in e for e in errors)


class TestValidateTriage:
    def test_accepts_valid(self):
        data = {
            "task_id": "task-1",
            "decision": "needs_deep_diagnosis",
            "likely_root_cause": "reward_bug",
            "confidence": 0.72,
            "reason": "Trajectory suggests a reward mismatch worth escalating.",
            "source": "model",
            "escalate": True,
        }

        errors = validator.validate_triage_record(data)

        assert errors == []

    def test_rejects_invalid_decision(self):
        data = {
            "task_id": "task-1",
            "decision": "reward_bug",
            "likely_root_cause": "reward_bug",
            "confidence": 0.5,
            "reason": "bad enum",
            "source": "model",
            "escalate": True,
        }

        errors = validator.validate_triage_record(data)

        assert any("decision" in e for e in errors)

    def test_rejects_invalid_cross_field_pair(self):
        data = {
            "task_id": "task-1",
            "decision": "agent_limitation",
            "likely_root_cause": "reward_bug",
            "confidence": 0.95,
            "reason": "contradictory output",
            "source": "model",
            "escalate": False,
        }

        errors = validator.validate_triage_record(data)

        assert any("agent_limitation" in e for e in errors)

    def test_accepts_valid_collection(self):
        data = [
            {
                "task_id": "task-1",
                "decision": "agent_limitation",
                "likely_root_cause": "agent_limitation",
                "confidence": 0.99,
                "reason": "Clear login wall.",
                "source": "rules",
                "escalate": False,
            }
        ]

        errors = validator.validate_triage(data)

        assert errors == []


class TestDiagnosisMissingExplanation:
    """Diagnosis with valid root_cause but missing explanation.

    The validator does NOT require 'explanation' — it only validates
    root_cause and suggested_fix. This documents that gap.
    """

    def test_valid_root_cause_no_explanation(self):
        data = {"root_cause": "reward_bug"}
        errors = validator.validate_diagnosis(data)
        # explanation is not checked by the validator
        assert errors == []

    def test_valid_root_cause_null_explanation(self):
        data = {"root_cause": "too_hard", "explanation": None}
        errors = validator.validate_diagnosis(data)
        assert errors == []

    def test_all_valid_root_causes_accepted(self):
        """Ensure all valid root_cause values are accepted."""
        valid_causes = {
            "reward_bug",
            "seed_bug",
            "impossible",
            "too_hard",
            "agent_limitation",
        }
        for cause in valid_causes:
            errors = validator.validate_diagnosis({"root_cause": cause})
            assert errors == [], f"root_cause {cause!r} should be valid"
