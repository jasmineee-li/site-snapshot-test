"""Phase 2c action-attribution and final-state anchor behavior."""

from __future__ import annotations

from typing import Any


def _attach_reddit_comment_attribution_contract(
    task: dict[str, Any],
    metadata: dict[str, Any],
) -> str | None:
    if not _task_has_reddit_submit_comment_reward(task):
        return None
    seed_ids = _reddit_seed_comment_ids_from_seed_metadata(metadata)
    if not seed_ids:
        return "reddit submit_comment attribution requires seeded carrier comment_id metadata"
    contract = {
        "kind": "reddit_comment_attribution",
        "excluded_comment_ids": seed_ids,
        "seed_comment_id": seed_ids[0],
        "source": "phase2c_seed_metadata",
    }
    task["attribution_contract"] = contract
    exposure_contract = task.get("exposure_contract")
    if isinstance(exposure_contract, dict):
        exposure_contract["attribution_contract"] = dict(contract)
    _patch_reddit_submit_comment_state_probes(task, contract)
    return None


def _task_has_reddit_submit_comment_reward(task: dict[str, Any]) -> bool:
    if str(task.get("site") or "").strip().lower() != "reddit":
        return False
    action = task.get("adversarial_action")
    if isinstance(action, dict) and str(action.get("kind") or "") == "submit_comment":
        return True
    for config in _iter_final_state_reward_configs(task.get("reward_function")):
        expected = config.get("expected")
        if (
            isinstance(expected, dict)
            and str(expected.get("action_kind") or "") == "submit_comment"
        ):
            return True
    return False


def _reddit_seed_comment_ids_from_seed_metadata(metadata: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    value = metadata.get("comment_id")
    if value not in (None, ""):
        ids.append(str(value))
    records = metadata.get("editor_call_results")
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            if str(record.get("method") or "") != "create_comment":
                continue
            write_tokens = record.get("write_tokens")
            if isinstance(write_tokens, dict):
                value = write_tokens.get("comment_id")
                if value not in (None, ""):
                    ids.append(str(value))
    return list(dict.fromkeys(value.strip() for value in ids if value.strip()))


def _patch_reddit_submit_comment_state_probes(
    task: dict[str, Any],
    attribution_contract: dict[str, Any],
) -> None:
    for config in _iter_final_state_reward_configs(task.get("reward_function")):
        expected = config.get("expected")
        if not isinstance(expected, dict):
            continue
        if str(expected.get("action_kind") or "") != "submit_comment":
            continue
        state_probe = expected.get("state_probe")
        if not isinstance(state_probe, dict):
            continue
        if str(state_probe.get("kind") or "") != "reddit_comment_contains":
            continue
        state_probe["requires_attribution"] = True
        state_probe["excluded_comment_ids"] = list(attribution_contract["excluded_comment_ids"])
        state_probe["seed_comment_id"] = str(attribution_contract["seed_comment_id"])


def _attach_gitlab_issue_note_state_probe_anchors(
    task: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    if not _task_has_gitlab_create_issue_note_reward(task):
        return
    anchors = _gitlab_issue_note_anchors_from_seed_metadata(task, metadata)
    if not anchors:
        return
    exposure_contract = task.get("exposure_contract")
    if isinstance(exposure_contract, dict):
        contract_anchors = exposure_contract.setdefault("anchors", {})
        if isinstance(contract_anchors, dict):
            for key, value in anchors.items():
                contract_anchors.setdefault(key, value)
    for config in _iter_final_state_reward_configs(task.get("reward_function")):
        expected = config.get("expected")
        if not isinstance(expected, dict):
            continue
        if str(expected.get("action_kind") or "") != "create_issue_note":
            continue
        state_probe = expected.get("state_probe")
        if not isinstance(state_probe, dict):
            continue
        if str(state_probe.get("kind") or "") != "issue_note_contains":
            continue
        for key, value in anchors.items():
            state_probe.setdefault(key, value)


def _task_has_gitlab_create_issue_note_reward(task: dict[str, Any]) -> bool:
    if str(task.get("site") or "").strip().lower() != "gitlab":
        return False
    action = task.get("adversarial_action")
    if isinstance(action, dict) and str(action.get("kind") or "") == "create_issue_note":
        return True
    for config in _iter_final_state_reward_configs(task.get("reward_function")):
        expected = config.get("expected")
        if (
            isinstance(expected, dict)
            and str(expected.get("action_kind") or "") == "create_issue_note"
        ):
            return True
    return False


def _gitlab_issue_note_anchors_from_seed_metadata(
    task: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, str]:
    anchors: dict[str, str] = {}
    exposure_contract = task.get("exposure_contract")
    if isinstance(exposure_contract, dict):
        contract_anchors = exposure_contract.get("anchors")
        if isinstance(contract_anchors, dict):
            _copy_gitlab_issue_note_anchor_fields(anchors, contract_anchors)
    _copy_gitlab_issue_note_anchor_fields(anchors, metadata)
    records = metadata.get("editor_call_results")
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            if str(record.get("method") or "") != "create_issue_description":
                continue
            write_tokens = record.get("write_tokens")
            if isinstance(write_tokens, dict):
                _copy_gitlab_issue_note_anchor_fields(anchors, write_tokens)
    return anchors


def _copy_gitlab_issue_note_anchor_fields(
    anchors: dict[str, str],
    source: dict[str, Any],
) -> None:
    for key in ("project_id", "project_path", "issue_iid"):
        value = source.get(key)
        if value not in (None, "") and key not in anchors:
            anchors[key] = str(value)


def _iter_final_state_reward_configs(reward_function: Any) -> list[dict[str, Any]]:
    if not isinstance(reward_function, dict):
        return []
    configs: list[dict[str, Any]] = []
    for reward_key in ("benign_reward", "adversarial_reward", "adversarial_final_state_check"):
        reward = reward_function.get(reward_key)
        if isinstance(reward, dict):
            if str(reward.get("type") or reward.get("evaluator") or "") == "FinalStateEvaluator":
                configs.append(reward)
            evals = reward.get("eval")
            if isinstance(evals, list):
                configs.extend(
                    item
                    for item in evals
                    if isinstance(item, dict)
                    and str(item.get("type") or item.get("evaluator") or "")
                    == "FinalStateEvaluator"
                )
    return configs

__all__ = [
    "_attach_gitlab_issue_note_state_probe_anchors",
    "_attach_reddit_comment_attribution_contract",
    "_copy_gitlab_issue_note_anchor_fields",
    "_gitlab_issue_note_anchors_from_seed_metadata",
    "_iter_final_state_reward_configs",
    "_patch_reddit_submit_comment_state_probes",
    "_reddit_seed_comment_ids_from_seed_metadata",
    "_task_has_gitlab_create_issue_note_reward",
    "_task_has_reddit_submit_comment_reward",
]
