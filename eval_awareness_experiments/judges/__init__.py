"""Judges for detecting eval awareness in transcripts and website environments.

Copied from eval-awareness-testbed with adapted imports (no inspect_ai dependency).
"""

from eval_awareness_experiments.judges.base import BaseJudge
from eval_awareness_experiments.judges.registry import get_judge, list_judges, register_judge

# Meta-eval judges (for needham-style "is this an eval?" classification)
from eval_awareness_experiments.judges import binary_third_person  # noqa: F401
from eval_awareness_experiments.judges import probability_third_person  # noqa: F401
from eval_awareness_experiments.judges import purpose  # noqa: F401
from eval_awareness_experiments.judges import purpose_continue_5q  # noqa: F401

# Live-eval judges (for checking spontaneous awareness during actual evals)
from eval_awareness_experiments.judges import verbalized_awareness  # noqa: F401

# Test Awareness Steering judge (3-dimension: awareness, recognition, performance)
from eval_awareness_experiments.judges import test_awareness_steering  # noqa: F401

__all__ = [
    "BaseJudge",
    "get_judge",
    "list_judges",
    "register_judge",
]
