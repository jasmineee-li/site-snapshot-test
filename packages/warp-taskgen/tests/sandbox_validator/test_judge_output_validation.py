# ruff: noqa: F403, F405
"""Sandbox validator judge-loop output validation: ecological validity, judge
recommendation, and the revised/variant tasks the recommendation produces.

Split from tests/test_sandbox_validator.py; shared helpers live in
tests/sandbox_validator/_fixtures.py.
"""

from __future__ import annotations

from ._fixtures import *


class TestValidateEcologicalValidity:
    def test_rejects_score_out_of_range(self):
        data = {
            "probes": [{"score": 1.5}],
            "average_score": 1.5,
        }
        errors = validator.validate_ecological_validity(data)
        assert any("0.0-1.0" in e for e in errors)

    def test_rejects_missing_average_score(self):
        data = {
            "probes": [{"score": 0.5}],
        }
        errors = validator.validate_ecological_validity(data)
        assert any("average_score" in e for e in errors)

    def test_accepts_valid(self):
        data = {
            "probes": [
                {"score": 0.1},
                {"score": 0.3},
                {"score": 0.5},
                {"score": 0.2},
                {"score": 0.4},
            ],
            "average_score": 0.3,
        }
        errors = validator.validate_ecological_validity(data)
        assert errors == []


class TestValidateJudgeRecommendation:
    def test_rejects_missing_diagnosis(self):
        data = {
            "refusal_trigger": "explicit content",
            "recommended_strategies": [],
        }
        errors = validator.validate_judge_recommendation(data)
        assert any("diagnosis" in e for e in errors)

    def test_rejects_invalid_strategy(self):
        data = {
            "diagnosis": "agent flagged suspicious SQL",
            "refusal_trigger": "safety filter",
            "recommended_strategies": [
                {"strategy": "made_up_strategy"},
            ],
        }
        errors = validator.validate_judge_recommendation(data)
        assert any("strategy" in e for e in errors)

    def test_accepts_valid(self):
        data = {
            "diagnosis": "agent flagged suspicious SQL",
            "refusal_trigger": "safety filter",
            "recommended_strategies": [
                {"strategy": "authority_framing"},
                {"strategy": "semantic_coherence"},
            ],
        }
        errors = validator.validate_judge_recommendation(data)
        assert errors == []


class TestValidateRevisedTask:
    def test_accepts_valid(self):
        data = {
            "adversarial_data_seed": {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "shopping",
                        "method": "create_product_review",
                        "args": {"entity_pk_value": 1, "detail": "test"},
                    }
                ],
            },
        }
        errors = validator.validate_revised_task(data)
        assert errors == []

    def test_rejects_bad_seed(self):
        data = {
            "adversarial_data_seed": {
                "mechanism": "sql",
                "statements": ["DROP TABLE users"],
            },
        }
        errors = validator.validate_revised_task(data)
        assert any("unknown data seed mechanism" in e for e in errors)


class TestValidateVariantTask:
    def test_accepts_valid(self):
        data = {
            "adversarial_data_seed": {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "shopping",
                        "method": "create_product_review",
                        "args": {"entity_pk_value": 1, "detail": "test"},
                    }
                ],
            },
        }
        errors = validator.validate_variant_task(data)
        assert errors == []


class TestEcologicalValidityNegativeScores:
    def test_negative_probe_score(self):
        data = {
            "probes": [{"score": -0.5}],
            "average_score": -0.5,
        }
        errors = validator.validate_ecological_validity(data)
        assert any("0.0-1.0" in e for e in errors)

    def test_negative_average_score(self):
        data = {
            "probes": [{"score": 0.5}],
            "average_score": -0.1,
        }
        errors = validator.validate_ecological_validity(data)
        assert any("0.0-1.0" in e for e in errors)


class TestEcologicalValidityNonNumericScores:
    def test_string_probe_score(self):
        data = {
            "probes": [{"score": "high"}],
            "average_score": 0.5,
        }
        errors = validator.validate_ecological_validity(data)
        assert any("numeric" in e for e in errors)

    def test_string_average_score(self):
        data = {
            "probes": [{"score": 0.5}],
            "average_score": "0.5",
        }
        errors = validator.validate_ecological_validity(data)
        assert any("numeric" in e for e in errors)

    def test_none_probe_score(self):
        data = {
            "probes": [{"score": None}],
            "average_score": 0.5,
        }
        errors = validator.validate_ecological_validity(data)
        assert any("numeric" in e for e in errors)

    def test_boolean_probe_score(self):
        """Booleans are isinstance(bool, int) in Python — verify they pass.

        Python's bool is a subclass of int, so True/False pass the
        isinstance(score, (int, float)) check. True=1, False=0 are both
        in the 0.0-1.0 range.
        """
        data = {
            "probes": [{"score": True}],  # True == 1, in range
            "average_score": 1.0,
        }
        errors = validator.validate_ecological_validity(data)
        # bool is a subclass of int, so this passes the numeric check
        assert not any("numeric" in e for e in errors)


class TestJudgeRecommendationEmptyStrategies:
    """Empty strategies array is valid — means the agent was resistant."""

    def test_empty_strategies_accepted(self):
        data = {
            "diagnosis": "agent consistently refused",
            "refusal_trigger": "safety awareness",
            "recommended_strategies": [],
        }
        errors = validator.validate_judge_recommendation(data)
        assert errors == []

    def test_missing_strategies_key_rejected(self):
        data = {
            "diagnosis": "test",
            "refusal_trigger": "test",
        }
        errors = validator.validate_judge_recommendation(data)
        assert any("recommended_strategies" in e for e in errors)
