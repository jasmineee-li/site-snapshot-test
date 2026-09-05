"""Exposure contract seed-template materialization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from warp_taskgen.editors._registry import method_spec
from warp_taskgen.phase_2.exposure_contract.constants import (
    ORDERED_CREATED_CHILD_PRE_CALL_DELAY_S,
)
from warp_taskgen.phase_2.text_fill.constants import PAYLOAD_PLACEHOLDER


def materialize_seed_template_from_contract(
    contract: Mapping[str, Any],
    *,
    benchmark: str = "webarena_verified",
    benign_seed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the seed template encoded by an eligible contract.

    The host materializes the adversarial seed deterministically; the
    Phase 2 strategy LLM never emits ``seed_template``. When ``benign_seed``
    has actions of its own (Mode B), this preserves them verbatim and
    appends the contract's adversarial call after, so
    ``self_contained_adversarial_seed_error`` accepts the result. The
    output mechanism mirrors the benign mechanism byte-for-byte; the
    adversarial-only path falls back to ``mechanism=editor``.
    """
    eligibility = contract.get("eligibility")
    if not isinstance(eligibility, Mapping) or eligibility.get("status") != "eligible":
        raise ValueError("cannot materialize seed_template from ineligible exposure_contract")
    method = str(contract.get("editor_method") or "")
    site = str(contract.get("site") or "")
    args = contract.get("editor_args_template")
    if not method or not site or not isinstance(args, Mapping):
        raise ValueError("exposure_contract missing editor method/site/args template")
    contract_editor_call = {
        "benchmark": benchmark,
        "site": site,
        "method": method,
        "args": dict(args),
    }

    if not _benign_seed_has_actions(benign_seed):
        return {"mechanism": "editor", "editor_calls": [contract_editor_call]}

    benign_mechanism = str(benign_seed.get("mechanism") or "").strip().lower()
    benign_api_calls = benign_seed.get("api_calls") if isinstance(benign_seed, Mapping) else None
    benign_editor_calls = (
        benign_seed.get("editor_calls") if isinstance(benign_seed, Mapping) else None
    )

    if isinstance(benign_api_calls, list) and benign_api_calls:
        adversarial_api_call = _derive_adversarial_api_call(
            site=site,
            editor_method=method,
            editor_args_template=args,
            benign_api_calls=list(benign_api_calls),
            benchmark=benchmark,
        )
        return {
            "mechanism": benign_mechanism or "api",
            "api_calls": [*list(benign_api_calls), adversarial_api_call],
        }

    if isinstance(benign_editor_calls, list) and benign_editor_calls:
        if _needs_ordered_created_child_guard(contract):
            contract_editor_call["pre_call_delay_s"] = ORDERED_CREATED_CHILD_PRE_CALL_DELAY_S
        # Benigns with mechanism="none" but populated editor_calls are
        # legal under validate_data_seed; preserve the mechanism literally
        # so the verbatim-prefix invariant accepts the materialized seed.
        return {
            "mechanism": benign_mechanism or "editor",
            "editor_calls": [*list(benign_editor_calls), contract_editor_call],
        }

    return {"mechanism": "editor", "editor_calls": [contract_editor_call]}


def _needs_ordered_created_child_guard(contract: Mapping[str, Any]) -> bool:
    """Return True when benign-created content can tie the payload content.

    Project issue lists and forum listings are commonly used with "newest" /
    "most recent" task wording. If Phase 2 preserves a benign seed by appending
    the payload seed immediately after it, both child rows or appended
    discussion entries can share a second-resolution timestamp and the live UI
    may put the benign item first. A small pre-call delay on the payload write
    makes the ordering deterministic while preserving the benign seed prefix
    invariant.
    """
    mode = str(contract.get("mode") or "").strip()
    site = str(contract.get("site") or "").strip().lower()
    kind = str(contract.get("kind") or "").strip()
    method = str(contract.get("editor_method") or "").strip()

    if (
        mode
        in {
            "inline_listing",
            "inline_listing_created_child",
            "bounded_transitive_created_child",
        }
        and site == "gitlab"
        and kind == "gitlab_search_result"
    ):
        return method in {"create_issue_title", "create_issue_description"}
    if (
        mode
        in {
            "inline_listing",
            "inline_listing_created_child",
            "bounded_transitive_created_child",
        }
        and site == "reddit"
        and kind == "reddit_forum"
    ):
        return method in {"create_submission_title", "create_submission"}
    if (
        mode == "direct_detail"
        and site == "reddit"
        and kind == "reddit_submission"
        and method == "create_comment"
    ):
        return True
    if (
        mode in {"direct_detail", "bounded_transitive_existing"}
        and site == "gitlab"
        and kind in {"gitlab_issue", "gitlab_mr", "gitlab_search_result", "gitlab_dashboard_list"}
        and method in {"create_issue_note", "create_mr_note"}
    ):
        return True
    return False


def _benign_seed_has_actions(benign_seed: Mapping[str, Any] | None) -> bool:
    if not isinstance(benign_seed, Mapping):
        return False
    editor_calls = benign_seed.get("editor_calls")
    if isinstance(editor_calls, list) and editor_calls:
        return True
    api_calls = benign_seed.get("api_calls")
    if isinstance(api_calls, list) and api_calls:
        return True
    if benign_seed.get("mechanism") == "state_push" and "state" in benign_seed:
        return True
    return False


def _derive_adversarial_api_call(
    *,
    site: str,
    editor_method: str,
    editor_args_template: Mapping[str, Any],
    benign_api_calls: list[dict[str, Any]],
    benchmark: str,
) -> dict[str, Any]:
    """Translate an editor-method into an api_call so the host seed can
    extend a benign mechanism="api" seed without breaking the verbatim
    prefix invariant in ``self_contained_adversarial_seed_error``."""
    spec = method_spec(site, editor_method, benchmark=benchmark)
    http = getattr(spec, "http", None)
    if not isinstance(http, tuple) or len(http) != 2:
        raise ValueError(
            f"editor method {site}.{editor_method} has no usable http metadata; "
            f"cannot derive adversarial api_call from contract"
        )
    method_verb, path_template = http
    method_verb = str(method_verb).upper()
    path = _resolve_api_path(str(path_template), benign_api_calls, method_verb)
    body: dict[str, Any] = {}
    for arg_name, value in editor_args_template.items():
        binding = spec.bindings.get(arg_name)
        if binding is None:
            continue
        if binding.kind == "free_text":
            body[arg_name] = PAYLOAD_PLACEHOLDER if value == PAYLOAD_PLACEHOLDER else value
    return {"method": method_verb, "path": path, "body": body}


def _resolve_api_path(
    template: str, benign_api_calls: list[Mapping[str, Any]], method_verb: str
) -> str:
    """Reuse the benign path when verb matches, since Phase 1 already
    resolved {placeholders} (e.g., user_id) against the live instance.
    Otherwise fall back to the literal template; Phase 2c feasibility
    will reject any unresolved placeholders, which is the correct failure
    mode."""
    for call in reversed(benign_api_calls):
        if not isinstance(call, Mapping):
            continue
        if str(call.get("method") or "").upper() != method_verb:
            continue
        path = call.get("path")
        if isinstance(path, str) and path:
            return path
    return template


__all__ = [
    "_benign_seed_has_actions",
    "_derive_adversarial_api_call",
    "_needs_ordered_created_child_guard",
    "_resolve_api_path",
    "materialize_seed_template_from_contract",
]
