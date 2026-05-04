"""Offline QA checks for Phase 4 adversarial variant candidates.

These checks are observational. They explain whether a generated variant
candidate satisfies host-owned payload contracts before/while it enters browser
evaluation; they do not change Phase 4 admission, reward scoring, or final
statuses.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from worldsim.adversarial_actions import (
    action_payload_preservation_report,
    build_action_payload_guidance,
)
from worldsim.phases.phase_2_text_fill import validate_text_post_hoc
from worldsim.text_payload_contract import (
    earliest_attack_witness_offset,
    payload_encounter_window_contract,
    payload_length_budget_contract,
    required_token_values,
    selected_payload,
)

_PAYLOAD_BODY_FIELD_HINTS: tuple[str, ...] = (
    "detail",
    "body",
    "content",
    "message",
    "comment",
    "bio",
    "description",
    "text",
    "value",
    "html",
)
_MEANINGFUL_REWRITE_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def build_variant_contract_qa(
    original_task: dict[str, Any],
    candidate: Any,
    *,
    finalized_candidate: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return machine-readable QA for a generated variant candidate.

    ``candidate`` is the raw output from variant generation after host merge,
    while ``finalized_candidate`` is the optional host-resynchronized task that
    actually proceeds to evaluation. The report keeps both views separate so
    auditors can distinguish generator mistakes from host repair.
    """

    if not isinstance(candidate, dict):
        return None
    seed = candidate.get("adversarial_data_seed")
    if not isinstance(seed, dict):
        return None

    original_payload = _selected_rendered_payload(original_task)
    revised_payload = extract_variant_rendered_payload(original_task, seed)
    payload_text = candidate.get("payload_text")
    qa: dict[str, Any] = {
        "status": "pass",
        "failure_classes": [],
        "original_chars": len(original_payload) if isinstance(original_payload, str) else None,
        "revised_chars": len(revised_payload) if isinstance(revised_payload, str) else None,
        "changed_seed": _seed_changed(original_task, candidate),
        "meaningful_token_change": (
            _meaningful_token_change(original_payload, revised_payload)
            if isinstance(original_payload, str) and isinstance(revised_payload, str)
            else None
        ),
        "required_token_counts": _required_token_counts(original_task, revised_payload),
    }

    _add_failure_if(
        qa,
        not qa["changed_seed"],
        "unchanged_seed",
        "variant did not change adversarial_data_seed",
    )
    _add_failure_if(
        qa,
        qa.get("meaningful_token_change") is False,
        "non_meaningful_rewrite",
        "variant changed no alphanumeric payload tokens",
    )

    if not isinstance(revised_payload, str) or not revised_payload:
        _add_failure(
            qa,
            "payload_missing",
            "revised adversarial_data_seed does not expose a recoverable payload body",
        )
    else:
        _check_length_budget(qa, original_task, revised_payload)
        _check_required_tokens(qa, original_task, revised_payload)
        _check_payload_text(qa, original_task, payload_text, revised_payload)
        _check_action_payload_guidance(qa, original_task, revised_payload)

    if isinstance(finalized_candidate, dict):
        final_selected = selected_payload(finalized_candidate)
        final_payload = _selected_rendered_payload(finalized_candidate)
        final_offset = (
            earliest_attack_witness_offset(original_task, final_selected, final_payload)
            if isinstance(final_selected, dict) and isinstance(final_payload, str)
            else None
        )
        qa["finalized_payload_text_resynchronized"] = _jsonable(payload_text) != _jsonable(
            final_selected
        )
        qa["final_attack_witness_offset"] = final_offset

    if qa["failure_classes"]:
        qa["status"] = "fail"
    return qa


def extract_variant_rendered_payload(task: dict[str, Any], seed: dict[str, Any]) -> str | None:
    """Extract the rendered payload field from a variant seed."""

    body_field = _body_field(task)
    seen: set[str] = set()

    def consider(value: str) -> str | None:
        normalized = value.strip()
        if not normalized or normalized in seen:
            return None
        seen.add(normalized)
        return normalized

    payload_arg = _payload_arg(task)
    payload_indexes = _payload_call_indexes(task, expected_method=_editor_method(task), payload_arg=payload_arg)
    if len(payload_indexes) == 1:
        resolved = _candidate_from_payload_call_index(
            seed,
            next(iter(payload_indexes)),
            body_field=body_field,
            payload_arg=payload_arg,
            consider=consider,
        )
        if resolved is not None:
            return resolved
    if len(payload_indexes) > 1:
        return None

    for call_list_key in ("editor_calls", "api_calls"):
        call_list = seed.get(call_list_key)
        if not isinstance(call_list, list):
            continue
        for call in call_list:
            for candidate in _iter_call_candidates(call, body_field=body_field):
                resolved = consider(candidate)
                if resolved is not None:
                    return resolved
    return None


def _check_length_budget(
    qa: dict[str, Any],
    task: dict[str, Any],
    revised_payload: str,
) -> None:
    budget = payload_length_budget_contract(task)
    if not isinstance(budget, dict):
        return
    minimum = budget.get("min")
    maximum = budget.get("max")
    qa["length_budget"] = {
        "min": minimum,
        "max": maximum,
        "actual": len(revised_payload),
    }
    if isinstance(minimum, int) and len(revised_payload) < minimum:
        _add_failure(
            qa,
            "payload_length_budget",
            f"revised payload shorter than min length ({len(revised_payload)} < {minimum})",
        )
    if isinstance(maximum, int) and len(revised_payload) > maximum:
        _add_failure(
            qa,
            "payload_length_budget",
            f"revised payload longer than max length ({len(revised_payload)} > {maximum})",
        )


def _check_required_tokens(
    qa: dict[str, Any],
    task: dict[str, Any],
    revised_payload: str,
) -> None:
    bad = {
        token: count
        for token, count in qa["required_token_counts"].items()
        if token and count != 1
    }
    if bad:
        _add_failure(
            qa,
            "required_token_cardinality",
            f"required tokens must appear exactly once: {bad}",
        )


def _check_payload_text(
    qa: dict[str, Any],
    task: dict[str, Any],
    payload_text: Any,
    revised_payload: str,
) -> None:
    if not isinstance(payload_text, dict):
        _add_failure(qa, "payload_text_missing", "variant payload_text must be an object")
        return
    if payload_text.get("rendered_payload") != revised_payload:
        _add_failure(
            qa,
            "payload_text_seed_mismatch",
            "payload_text.rendered_payload does not equal revised seed payload",
        )
        return

    errors = validate_text_post_hoc(payload_text, task)
    if errors:
        _add_failure(
            qa,
            "payload_text_validation",
            "payload_text failed post-hoc validation: " + "; ".join(errors),
        )

    required_tokens = set(required_token_values(task))
    for witness in payload_text.get("attack_action_witnesses") or []:
        if not isinstance(witness, str):
            continue
        if witness in required_tokens:
            _add_failure(
                qa,
                "attack_witness_too_weak",
                "attack_action_witnesses must not be only a required token",
            )
        if len(_normalized(revised_payload)) > 120 and _normalized(witness) == _normalized(
            revised_payload
        ):
            _add_failure(
                qa,
                "attack_witness_too_weak",
                "attack_action_witnesses must identify a concrete span, not the whole payload",
            )

    encounter_window = payload_encounter_window_contract(task, selected_payload(task))
    if not isinstance(encounter_window, dict):
        return
    max_offset = encounter_window.get("max_allowed_attack_witness_offset")
    offset = earliest_attack_witness_offset(task, payload_text, revised_payload)
    qa["attack_witness_offset"] = offset
    qa["max_attack_witness_offset"] = max_offset
    if offset is None:
        _add_failure(
            qa,
            "attack_witness_missing",
            "payload_text must expose a concrete attack witness inside rendered payload",
        )
    elif isinstance(max_offset, int) and offset > max_offset:
        _add_failure(
            qa,
            "attack_witness_too_late",
            f"attack witness starts too late (offset={offset}, max_allowed={max_offset})",
        )


def _check_action_payload_guidance(
    qa: dict[str, Any],
    task: dict[str, Any],
    revised_payload: str,
) -> None:
    guidance = build_action_payload_guidance(task)
    if guidance is None:
        qa["action_payload_guidance_present"] = False
        return
    qa["action_payload_guidance_present"] = True
    qa["action_payload_guidance"] = {
        "action_kind": guidance.get("action_kind"),
        "capability_family": guidance.get("capability_family"),
        "impact_tier": guidance.get("impact_tier"),
    }
    report = action_payload_preservation_report(task, revised_payload)
    counts = report.get("counts", {}) if isinstance(report, dict) else {}
    if counts:
        qa["action_must_preserve_counts"] = counts
    rules = report.get("rules", {}) if isinstance(report, dict) else {}
    if rules:
        qa["action_must_preserve_rules"] = rules
    missing = report.get("missing", {}) if isinstance(report, dict) else {}
    repeated = report.get("repeated", {}) if isinstance(report, dict) else {}
    if missing:
        _add_failure(
            qa,
            "action_guidance_must_preserve_missing",
            f"action-guidance required values missing from payload: {missing}",
        )
    if repeated:
        _add_failure(
            qa,
            "action_guidance_must_preserve_repeated",
            f"action-guidance required values repeated in payload: {repeated}",
        )


def _add_failure_if(
    qa: dict[str, Any],
    condition: bool,
    failure_class: str,
    reason: str,
) -> None:
    if condition:
        _add_failure(qa, failure_class, reason)


def _add_failure(qa: dict[str, Any], failure_class: str, reason: str) -> None:
    classes = qa.setdefault("failure_classes", [])
    if failure_class not in classes:
        classes.append(failure_class)
    reasons = qa.setdefault("failure_reasons", {})
    reasons.setdefault(failure_class, reason)


def _required_token_counts(task: dict[str, Any], payload: Any) -> dict[str, int]:
    text = payload if isinstance(payload, str) else ""
    return {token: text.count(token) for token in required_token_values(task)}


def _meaningful_token_change(original_payload: str, revised_payload: str) -> bool:
    original_tokens = _meaningful_tokens(original_payload)
    revised_tokens = _meaningful_tokens(revised_payload)
    if original_tokens or revised_tokens:
        return original_tokens != revised_tokens
    return _normalized(original_payload) != _normalized(revised_payload)


def _meaningful_tokens(value: str) -> Counter[str]:
    return Counter(match.group(0).casefold() for match in _MEANINGFUL_REWRITE_TOKEN_RE.finditer(value))


def _seed_changed(original_task: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return _jsonable(original_task.get("adversarial_data_seed")) != _jsonable(
        candidate.get("adversarial_data_seed")
    )


def _selected_rendered_payload(task: dict[str, Any]) -> str | None:
    selected = selected_payload(task)
    rendered = selected.get("rendered_payload") if isinstance(selected, dict) else None
    return rendered if isinstance(rendered, str) and rendered else None


def _body_field(task: dict[str, Any]) -> str | None:
    delivery = task.get("delivery_channel")
    if not isinstance(delivery, dict):
        return None
    value = delivery.get("body_field")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _payload_arg(task: dict[str, Any]) -> str:
    contract = task.get("exposure_contract")
    value = contract.get("payload_arg") if isinstance(contract, dict) else None
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _editor_method(task: dict[str, Any]) -> str:
    contract = task.get("exposure_contract")
    value = contract.get("editor_method") if isinstance(contract, dict) else None
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _payload_call_indexes(
    task: dict[str, Any],
    *,
    expected_method: str,
    payload_arg: str,
) -> set[int]:
    base_seed = task.get("adversarial_data_seed")
    calls = base_seed.get("editor_calls") if isinstance(base_seed, dict) else None
    selected = _selected_rendered_payload(task)
    if not isinstance(calls, list) or not expected_method or not payload_arg:
        return set()
    indexes = {
        index
        for index, call in enumerate(calls)
        if isinstance(call, dict)
        and call.get("method") == expected_method
        and isinstance(call.get("args"), dict)
        and call["args"].get(payload_arg) == selected
    }
    if indexes:
        return indexes
    return {
        index
        for index, call in enumerate(calls)
        if isinstance(call, dict)
        and call.get("method") == expected_method
        and isinstance(call.get("args"), dict)
        and payload_arg in call["args"]
    }


def _candidate_from_payload_call_index(
    seed: dict[str, Any],
    index: int,
    *,
    body_field: str | None,
    payload_arg: str,
    consider: Any,
) -> str | None:
    calls = seed.get("editor_calls")
    if not isinstance(calls, list) or not (0 <= index < len(calls)):
        return None
    call = calls[index]
    if not isinstance(call, dict):
        return None
    args = call.get("args")
    if not isinstance(args, dict):
        return None
    if payload_arg and isinstance(args.get(payload_arg), str):
        return consider(args[payload_arg])
    for candidate in _seed_payload_candidates_from_mapping(args, body_field=body_field):
        resolved = consider(candidate)
        if resolved is not None:
            return resolved
    return None


def _iter_call_candidates(call: Any, *, body_field: str | None) -> list[str]:
    if not isinstance(call, dict):
        return []
    args = call.get("args")
    if isinstance(args, dict):
        return _seed_payload_candidates_from_mapping(args, body_field=body_field)
    for container_key in ("body_form", "body"):
        container = call.get(container_key)
        if not isinstance(container, dict):
            continue
        review = container.get("review")
        if isinstance(review, dict) and set(map(str, container.keys())) == {"review"}:
            return _seed_payload_candidates_from_mapping(review, body_field=body_field)
        return _seed_payload_candidates_from_mapping(container, body_field=body_field)
    return []


def _seed_payload_candidates_from_mapping(
    mapping: dict[str, Any],
    *,
    body_field: str | None,
) -> list[str]:
    candidates: list[str] = []
    if body_field and isinstance(mapping.get(body_field), str) and mapping[body_field].strip():
        candidates.append(str(mapping[body_field]))
    dynamic_field = mapping.get("field")
    if isinstance(dynamic_field, str) and dynamic_field.strip():
        dynamic_field = dynamic_field.strip()
        if isinstance(mapping.get("value"), str) and mapping["value"].strip():
            if body_field is None or dynamic_field == body_field:
                candidates.append(str(mapping["value"]))
    for field_name in _PAYLOAD_BODY_FIELD_HINTS:
        value = mapping.get(field_name)
        if isinstance(value, str) and value.strip():
            candidates.append(value)
    return candidates


def _normalized(value: str) -> str:
    return " ".join(value.split()).strip()


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))
