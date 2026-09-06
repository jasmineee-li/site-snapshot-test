"""Editor seed argument naming, filtering, and benchmark inference."""

from __future__ import annotations

import inspect
import logging
import re
from typing import Any

from warp_taskgen.benchmark_capabilities import infer_benchmark_name, normalize_benchmark_name
from warp_taskgen.editors import EditorError
from warp_taskgen.seeding.validation import _validate_pre_call_delay

logger = logging.getLogger(__name__)


_REDDIT_COMMENT_BODY_FIELD_PATTERN = re.compile(
    r"^reply_to_submission_(?:\{[^}\]]+\}|[^[]+)\[comment\]$"
)


def _infer_editor_call_benchmark(call: dict[str, Any], instance: dict[str, Any]) -> str:
    try:
        benchmark = infer_benchmark_name(
            (
                call.get("benchmark"),
                call.get("benchmark_name"),
                call.get("benchmark_adapter"),
                instance.get("benchmark"),
                instance.get("benchmark_name"),
                instance.get("benchmark_adapter"),
            )
        )
    except ValueError as exc:
        raise EditorError("benchmark_mismatch", str(exc)) from exc
    if benchmark is not None:
        return benchmark
    return normalize_benchmark_name("webarena_verified")


def _infer_task_benchmark(task: dict[str, Any]) -> str:
    values: list[Any] = [
        task.get("benchmark"),
        task.get("benchmark_name"),
        task.get("benchmark_adapter"),
    ]
    seed = task.get("adversarial_data_seed")
    calls = seed.get("editor_calls") if isinstance(seed, dict) else None
    if isinstance(calls, list):
        for call in calls:
            if isinstance(call, dict):
                values.extend(
                    (
                        call.get("benchmark"),
                        call.get("benchmark_name"),
                        call.get("benchmark_adapter"),
                    )
                )
    benchmark = infer_benchmark_name(values)
    return benchmark or normalize_benchmark_name("webarena_verified")


def _editor_call_pre_delay_s(call: dict[str, Any]) -> float:
    value = call.get("pre_call_delay_s")
    _validate_pre_call_delay(value)
    if value is None:
        return 0.0
    return float(value)


def _filter_editor_method_args(
    editor_method: Any,
    args: dict[str, Any],
    *,
    editor_site_name: str,
    method_name: str,
) -> dict[str, Any]:
    """Drop kwargs not in the editor method's signature.

    Phase 4 placement_fix and variant_api can hallucinate extra editor args
    (e.g. ``position``, ``score``, ``author``) that the editor method does
    not declare. Calling ``editor_method(**args)`` with such args raises
    ``TypeError`` and aborts the whole post-processing pass. Filter unknown
    kwargs here so a single API hallucination does not cascade. Required
    args were already checked by ``editor.validate_args`` upstream.
    """
    try:
        sig = inspect.signature(editor_method)
    except (TypeError, ValueError):
        return args
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return args
    accepted = {
        name
        for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    }
    unknown = sorted(set(args) - accepted)
    if not unknown:
        return args
    logger.warning(
        "editor %s.%s received %d unknown arg(s) %s; dropping before invocation",
        editor_site_name,
        method_name,
        len(unknown),
        unknown,
    )
    return {k: v for k, v in args.items() if k in accepted}


def _editor_delivery_key(call: Any) -> tuple[str, str] | None:
    if not isinstance(call, dict):
        return None
    site_name = str(call.get("site") or "").strip().lower()
    method_name = str(call.get("method") or "").strip()
    if not site_name or not method_name or not isinstance(call.get("args"), dict):
        return None
    return (site_name, method_name)


def _editor_arg_name(call: Any, canonical_name: str) -> str | None:
    editor_key = _editor_delivery_key(call)
    if editor_key == ("reddit", "create_comment") and _REDDIT_COMMENT_BODY_FIELD_PATTERN.match(
        canonical_name
    ):
        return "body"
    aliases = {
        ("gitlab", "create_issue"): {"body": "body_template", "description": "body_template"},
        ("gitlab", "create_issue_note"): {"body": "note_body"},
        ("gitlab", "create_mr"): {"body": "body_template", "description": "body_template"},
        ("gitlab", "create_mr_note"): {"body": "note_body"},
        ("gitlab", "update_user_status"): {"message": "message"},
        ("gitlab", "update_user_profile"): {
            "bio": "bio",
            "name": "name",
            "user[bio]": "bio",
            "user[name]": "name",
        },
        ("reddit", "create_forum"): {
            "name": "name_template",
            "description": "description_template",
            "forum[name]": "name_template",
            "forum[description]": "description_template",
        },
        ("reddit", "create_submission"): {
            "title": "title_template",
            "body": "body_template",
            "submission[title]": "title_template",
            "submission[body]": "body_template",
        },
        ("reddit", "create_submission_title"): {
            "title": "title",
            "body": "body",
            "submission[title]": "title",
            "submission[body]": "body",
        },
        ("reddit", "create_comment"): {"body": "body"},
        ("reddit", "update_user_bio"): {"bio": "bio_text"},
        ("shopping", "create_product_review"): {"detail": "detail", "title": "title"},
        ("shopping", "update_customer_profile"): {"value": "value"},
        ("shopping_admin", "update_admin_profile"): {"value": "value"},
    }.get(editor_key, {})
    arg_name = aliases.get(canonical_name)
    return str(arg_name) if isinstance(arg_name, str) else None


__all__ = [
    "_editor_arg_name",
    "_editor_call_pre_delay_s",
    "_editor_delivery_key",
    "_filter_editor_method_args",
    "_infer_editor_call_benchmark",
    "_infer_task_benchmark",
]
