"""Seed contract and action-preservation behavior."""

from __future__ import annotations

from typing import Any


def seed_has_actions(seed: Any) -> bool:
    if not isinstance(seed, dict):
        return False
    editor_calls = seed.get("editor_calls")
    return bool(isinstance(editor_calls, list) and editor_calls)


def seed_requires_reset(seed: Any) -> bool:
    if not isinstance(seed, dict):
        return False
    editor_calls = seed.get("editor_calls")
    return bool(isinstance(editor_calls, list) and editor_calls)


def self_contained_adversarial_seed_error(benign_seed: Any, adversarial_seed: Any) -> str | None:
    """Return an error when an adversarial seed drops benign setup state.

    Phase 4 applies only ``adversarial_data_seed``, so the adversarial seed
    must preserve the benign seed verbatim before extending it.
    """
    if not isinstance(benign_seed, dict) or not isinstance(adversarial_seed, dict):
        return None

    if not seed_has_actions(benign_seed):
        return None

    if not _seed_preserves_prefix(benign_seed, adversarial_seed):
        return (
            "adversarial_data_seed must preserve the benign data_seed verbatim before extending it"
        )
    return None


def _seed_preserves_prefix(benign_value: Any, adversarial_value: Any) -> bool:
    """Return True when ``adversarial_value`` structurally contains ``benign_value``."""
    if isinstance(benign_value, dict):
        if not isinstance(adversarial_value, dict):
            return False
        for key, benign_item in benign_value.items():
            if key not in adversarial_value:
                return False
            if not _seed_preserves_prefix(benign_item, adversarial_value[key]):
                return False
        return True

    if isinstance(benign_value, list):
        if not isinstance(adversarial_value, list) or len(adversarial_value) < len(benign_value):
            return False
        return all(
            _seed_preserves_prefix(benign_item, adversarial_value[index])
            for index, benign_item in enumerate(benign_value)
        )

    return benign_value == adversarial_value


__all__ = [
    "_seed_preserves_prefix",
    "seed_has_actions",
    "seed_requires_reset",
    "self_contained_adversarial_seed_error",
]
