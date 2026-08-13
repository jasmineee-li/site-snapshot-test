from __future__ import annotations

from typing import Any

from warp_taskgen.phase_2.text_fill.constants import PAYLOAD_PLACEHOLDER
from warp_taskgen.seeding import validate_data_seed


def materialize_adversarial_seed(
    seed_template: dict[str, Any], rendered_payload: str
) -> dict[str, Any]:
    validate_seed_template_contract(seed_template)
    # PVPO locates the rendered payload on the delivery page by substring-
    # matching ``rendered_payload`` against every text node in the live
    # DOM — no DOM attribute or HTML wrapper is required. See
    # ``warp_taskgen/phase_4/pvpo_query.js`` and the Implementation Status
    # section of
    # ``docs/handoffs/codex-handoff-paint-verified-oracle.md`` for the
    # content-match anchor strategy. An earlier draft wrapped the payload
    # in ``<span data-worldsim-payload>`` but Magento/reddit/gitlab all
    # sanitize HTML in UGC fields to varying degrees (empirically verified
    # against live r5 2026-04-19), and the literal span would either be
    # stripped (breaking the attribute lookup) or rendered as visible text
    # (a giant eval artifact for P(eval)). Content-match is oblivious to
    # both.
    return _replace_payload_placeholder(seed_template, rendered_payload)


def _replace_payload_placeholder(value: Any, payload: str) -> Any:
    if isinstance(value, str):
        return value.replace(PAYLOAD_PLACEHOLDER, payload)
    if isinstance(value, list):
        return [_replace_payload_placeholder(item, payload) for item in value]
    if isinstance(value, dict):
        return {key: _replace_payload_placeholder(item, payload) for key, item in value.items()}
    return value


def validate_seed_template_contract(seed_template: dict[str, Any]) -> None:
    if not isinstance(seed_template, dict):
        raise ValueError("seed_template must be an object")

    mechanism = seed_template.get("mechanism")
    editor_calls = seed_template.get("editor_calls")
    if mechanism == "none" and isinstance(editor_calls, list) and editor_calls:
        validation_seed = dict(seed_template)
        validation_seed["mechanism"] = "editor"
    else:
        validation_seed = seed_template
    validate_data_seed(validation_seed, allow_none=False)

    total_placeholders = _count_placeholder_occurrences(seed_template)
    if total_placeholders != 1:
        raise ValueError("seed_template must contain exactly one {{PAYLOAD_TEXT}} placeholder")

    if isinstance(editor_calls, list) and editor_calls:
        placeholder_count = 0
        for call in editor_calls:
            if not isinstance(call, dict):
                raise ValueError("seed_template editor_calls entries must be objects")
            args = call.get("args")
            if not isinstance(args, dict):
                raise ValueError("seed_template editor_calls entries must include args")
            placeholder_count += _count_placeholder_occurrences(args)
        if placeholder_count != 1:
            raise ValueError(
                "seed_template must place {{PAYLOAD_TEXT}} in exactly one editor_calls[*].args field"
            )
        return

    if mechanism not in {"api", "form"}:
        raise ValueError("seed_template mechanism must be one of {'api', 'form'}")

    api_calls = seed_template.get("api_calls")
    if not isinstance(api_calls, list):
        raise ValueError(f"{mechanism} seed_template must include api_calls")
    expected_body_key = "body_form" if mechanism == "form" else "body"
    placeholder_count = 0
    for call in api_calls:
        if not isinstance(call, dict):
            raise ValueError(f"{mechanism} seed_template api_calls entries must be objects")
        if "target" in call:
            raise ValueError(
                "seed_template api_calls must not use target-based calls; use editor_calls"
            )
        body = call.get(expected_body_key)
        if isinstance(body, dict):
            placeholder_count += _count_placeholder_occurrences(body)
    if placeholder_count != 1:
        raise ValueError(
            f"{mechanism} seed_template must place {{PAYLOAD_TEXT}} in exactly one {expected_body_key} field"
        )


def _count_placeholder_occurrences(value: Any) -> int:
    if isinstance(value, str):
        return value.count(PAYLOAD_PLACEHOLDER)
    if isinstance(value, list):
        return sum(_count_placeholder_occurrences(item) for item in value)
    if isinstance(value, dict):
        return sum(_count_placeholder_occurrences(item) for item in value.values())
    return 0
