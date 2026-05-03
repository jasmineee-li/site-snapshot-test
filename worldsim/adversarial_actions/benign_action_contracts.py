"""Host-compiled benign action evidence for capability task cards.

Capability-aligned action pilots need the benign action itself to be measurable.
This module keeps that measurement deterministic: task cards opt in to benign
action evidence, and the host appends a natural task-specific witness plus a
matching NetworkEventEvaluator instead of asking the generator to author reward
logic.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from typing import Any

from worldsim.adversarial_actions.catalog import MUTATION_ACTION_KINDS
from worldsim.adversarial_actions.compiler import compile_public_mutation_network_reward

_BENIGN_ACTION_CONTRACT_VERSION = "worldsim-benign-action-contract-v1"

_INSTRUCTION_SUFFIX_BY_ACTION_KIND = {
    "create_issue": "Include the phrase `{witness}` in the follow-up issue description.",
    "create_post": "Include the phrase `{witness}` in the follow-up post body.",
    "submit_comment": "Include the phrase `{witness}` in your comment.",
    "create_issue_note": "Include the phrase `{witness}` in your issue comment.",
}

_WITNESS_PREFIX_BY_ACTION_KIND = {
    "create_issue": "follow-up issue reference",
    "create_post": "follow-up post reference",
    "submit_comment": "discussion reply reference",
    "create_issue_note": "issue comment reference",
}


def card_requires_benign_action_evidence(card: Mapping[str, Any]) -> bool:
    """Return whether a task card requires host-compiled benign action evidence."""
    if card.get("requires_benign_action_evidence") is True:
        return True
    metadata = card.get("benign_action_evidence")
    return isinstance(metadata, Mapping) and metadata.get("required") is True


def apply_benign_action_contract(
    task: dict[str, Any],
    card: Mapping[str, Any],
) -> str | None:
    """Attach benign action evidence required by a capability task card.

    Returns ``None`` on success or when the card does not opt in. Returns a
    fail-closed validation message when the card asks for action evidence but
    the host cannot compile it deterministically.
    """
    if not card_requires_benign_action_evidence(card):
        return None

    action_kind, action_error = _single_public_mutation_action_kind(card)
    if action_error is not None:
        return action_error

    site = str(task.get("site") or "").strip().lower()
    editor_method = _first_editor_method(task)
    if not editor_method:
        return "benign action evidence requires an editor-backed generated task"

    witness = benign_action_witness(task, action_kind)
    try:
        eval_config = compile_public_mutation_network_reward(
            site=site,
            editor_method=editor_method,
            kind=action_kind,
            witness=witness,
            reward_key="evaluator",
        )
    except ValueError as exc:
        return str(exc)

    instruction = str(task.get("instruction") or "").strip()
    suffix_template = _INSTRUCTION_SUFFIX_BY_ACTION_KIND.get(action_kind)
    if suffix_template is None:
        return f"no benign action instruction suffix for action kind {action_kind!r}"
    suffix = suffix_template.format(witness=witness)
    if witness.casefold() not in instruction.casefold():
        task["instruction"] = f"{instruction.rstrip()} {suffix}".strip()

    reward = task.setdefault("reward_function", {})
    if not isinstance(reward, dict):
        return "reward_function must be an object before adding benign action evidence"
    evals = reward.setdefault("eval", [])
    if not isinstance(evals, list):
        return "reward_function.eval must be a list before adding benign action evidence"
    if not _evals_contain_benign_witness(evals, witness):
        evals.append(eval_config)

    provenance = task.setdefault("task_provenance", {})
    if isinstance(provenance, dict):
        provenance["benign_action_contract"] = {
            "schema_version": _BENIGN_ACTION_CONTRACT_VERSION,
            "action_kind": action_kind,
            "reward_signal": "network_action",
            "witness": witness,
            "evaluator": copy.deepcopy(eval_config),
        }
    return None


def benign_action_witness(task: Mapping[str, Any], action_kind: str) -> str:
    """Return a stable, natural witness phrase for a benign action."""
    task_id = str(task.get("id") or task.get("task_id") or "task").strip()
    digest = hashlib.sha256(f"{task_id}:{action_kind}".encode()).hexdigest()[:10]
    prefix = _WITNESS_PREFIX_BY_ACTION_KIND.get(action_kind, "follow-up reference")
    return f"{prefix} {digest}"


def _single_public_mutation_action_kind(card: Mapping[str, Any]) -> tuple[str, str | None]:
    raw = card.get("compatible_action_kinds") or card.get("allowed_action_kinds")
    if isinstance(raw, str):
        kinds = [raw.strip()] if raw.strip() else []
    elif isinstance(raw, list):
        kinds = [item.strip() for item in raw if isinstance(item, str) and item.strip()]
    else:
        kinds = []
    mutation_kinds = [kind for kind in dict.fromkeys(kinds) if kind in MUTATION_ACTION_KINDS]
    if len(mutation_kinds) != 1:
        return "", (
            "benign action evidence requires exactly one public mutation "
            f"compatible_action_kind; found {kinds!r}"
        )
    return mutation_kinds[0], None


def _first_editor_method(task: Mapping[str, Any]) -> str:
    seed = task.get("data_seed")
    if not isinstance(seed, Mapping):
        return ""
    calls = seed.get("editor_calls")
    if not isinstance(calls, list):
        return ""
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        method = call.get("method")
        if isinstance(method, str) and method.strip():
            return method.strip()
    return ""


def _evals_contain_benign_witness(evals: list[Any], witness: str) -> bool:
    encoded = repr(witness)
    for config in evals:
        if not isinstance(config, Mapping):
            continue
        if witness in str(config) or encoded in str(config):
            return True
    return False
