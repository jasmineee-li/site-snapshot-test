"""Reward function dispatcher.

Each benchmark's Phase 0c ``BENCHMARK_PROFILE_{site}.json`` enumerates the
eval types its evaluation harness supports (``url_exact_match``,
``html_match``, ``db_query_match``, ``string_match``, etc.). This module
dispatches a reward spec to the right checker at evaluation time.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` Phase 0a "Benchmark Discovery"
(enumerates eval types) and Phase 3 "Evaluation Infrastructure"
(``run_reward_function`` is called from ``run_task``).

The four checkers below are stubs; they are implemented as part of commit 9
(Phase 3 benign validation). They intentionally raise ``NotImplementedError``
so earlier-phase code that transitively imports this module does not crash,
while still failing loudly when anyone tries to actually evaluate.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def run_reward_function(
    reward: dict[str, Any],
    instance: dict[str, Any],
) -> tuple[bool, str]:
    """Run one reward spec against a benchmark instance.

    Args:
        reward: Reward spec with a ``type`` field matching one of the
            eval types discovered in the Phase 0c profile, plus
            type-specific parameters.
        instance: Running benchmark instance dict.

    Returns:
        ``(passed, message)`` tuple. ``message`` is a human-readable
        explanation suitable for the task result log.
    """
    eval_type = reward["type"]
    checker = _CHECKERS.get(eval_type)
    if checker is None:
        raise NotImplementedError(
            f"Reward type {eval_type!r} not registered in worldsim.rewards. "
            f"Extend _CHECKERS to add support."
        )
    return checker(reward, instance)


def _url_exact_match(
    reward: dict[str, Any], instance: dict[str, Any]
) -> tuple[bool, str]:
    raise NotImplementedError("Scheduled for commit 9 (Phase 3 implementation)")


def _html_match(
    reward: dict[str, Any], instance: dict[str, Any]
) -> tuple[bool, str]:
    raise NotImplementedError("Scheduled for commit 9 (Phase 3 implementation)")


def _db_query_match(
    reward: dict[str, Any], instance: dict[str, Any]
) -> tuple[bool, str]:
    raise NotImplementedError("Scheduled for commit 9 (Phase 3 implementation)")


def _string_match(
    reward: dict[str, Any], instance: dict[str, Any]
) -> tuple[bool, str]:
    raise NotImplementedError("Scheduled for commit 9 (Phase 3 implementation)")


#: Registry of eval_type -> checker callable.
#:
#: Populated at import time; extend this dict to add new eval types.
_CHECKERS: dict[str, Any] = {
    "url_exact_match": _url_exact_match,
    "html_match": _html_match,
    "db_query_match": _db_query_match,
    "string_match": _string_match,
}
