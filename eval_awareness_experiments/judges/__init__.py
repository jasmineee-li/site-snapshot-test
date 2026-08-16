"""Judges for detecting eval awareness in transcripts and website environments.

Copied from eval-awareness-testbed with adapted imports (no inspect_ai dependency).
"""

# Imported for their side effects: each module's @register_judge call
# populates the registry. Meta-eval judges (needham-style "is this an eval?"):
# binary_third_person, probability_third_person, purpose, purpose_continue_5q.
# Live-eval judge (spontaneous awareness during an eval): verbalized_awareness.
# Test Awareness Steering judge (awareness, recognition, performance):
# test_awareness_steering.
from eval_awareness_experiments.judges import (
    binary_third_person,  # noqa: F401
    probability_third_person,  # noqa: F401
    purpose,  # noqa: F401
    purpose_continue_5q,  # noqa: F401
    test_awareness_steering,  # noqa: F401
    verbalized_awareness,  # noqa: F401
)
from eval_awareness_experiments.judges.base import BaseJudge
from eval_awareness_experiments.judges.registry import get_judge, list_judges, register_judge

__all__ = [
    "BaseJudge",
    "get_judge",
    "list_judges",
    "register_judge",
]
