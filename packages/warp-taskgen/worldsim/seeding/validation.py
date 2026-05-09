"""Seed schema validation behavior."""

from __future__ import annotations

from typing import Any

from worldsim.benchmark_capabilities import normalize_benchmark_name
from worldsim.editors import EDITOR_REGISTRY


def validate_data_seed(seed: dict[str, Any], *, allow_none: bool = False) -> None:
    """Validate a seed payload before it is persisted or executed."""
    if not isinstance(seed, dict):
        raise ValueError("data seed must be an object")

    mechanism = seed.get("mechanism")
    editor_calls = seed.get("editor_calls")
    has_editor_calls = isinstance(editor_calls, list) and bool(editor_calls)
    if mechanism is None:
        if has_editor_calls:
            _validate_editor_calls(editor_calls)
            return
        if allow_none:
            return
        raise ValueError("data seed must declare a non-empty mechanism")

    if mechanism == "none":
        if has_editor_calls:
            raise ValueError(
                "data_seed.mechanism='none' must not include editor_calls; "
                "use mechanism='editor'"
            )
        if allow_none:
            return
        raise ValueError("data seed must declare a non-empty mechanism")

    if mechanism == "editor":
        if not has_editor_calls:
            raise ValueError("editor data seed must include a non-empty editor_calls list")
        if seed.get("api_calls") is not None:
            raise ValueError("editor data seed must not include api_calls")
        _validate_editor_calls(editor_calls)
        return

    if mechanism in {"api", "form", "state_push"}:
        raise ValueError(
            f"data_seed.mechanism={mechanism!r} is deprecated; use mechanism='editor' "
            "with editor_calls referencing site editor methods. The api/form/state_push "
            "paths were removed in the editor migration; see "
            "docs/handoffs/researcher-handoff-project-status.md."
        )

    raise ValueError(f"unknown data seed mechanism: {mechanism!r}")


def _validate_editor_calls(editor_calls: Any) -> None:
    if not isinstance(editor_calls, list) or not editor_calls:
        raise ValueError("editor data seed must include a non-empty editor_calls list")
    for call in editor_calls:
        if not isinstance(call, dict):
            raise ValueError("editor_calls entries must be objects")
        benchmark = call.get("benchmark")
        if benchmark is not None and (not isinstance(benchmark, str) or not benchmark.strip()):
            raise ValueError("editor_calls benchmark must be a non-empty string when provided")
        site = call.get("site")
        method = call.get("method")
        args = call.get("args")
        _validate_pre_call_delay(call.get("pre_call_delay_s"))
        if not isinstance(site, str) or not site.strip():
            raise ValueError("editor_calls entries must include site")
        if not isinstance(method, str) or not method.strip():
            raise ValueError("editor_calls entries must include method")
        if not isinstance(args, dict):
            raise ValueError("editor_calls entries must include args as an object")
        method_name = method.strip()
        if method_name.startswith("_"):
            raise ValueError("editor_calls method must not be private")
        benchmark_key = normalize_benchmark_name(benchmark or "webarena_verified")
        editor_cls = EDITOR_REGISTRY.get((benchmark_key, site.strip().lower()))
        if editor_cls is not None and method_name not in editor_cls.supported_methods:
            raise ValueError(
                f"editor_calls method {method_name!r} is not supported for {(benchmark_key, site.strip().lower())!r}"
            )
        _validate_untrusted_selector_args(site.strip().lower(), args)


def _validate_pre_call_delay(value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("editor_calls pre_call_delay_s must be a number when provided")
    if value < 0 or value > 5:
        raise ValueError("editor_calls pre_call_delay_s must be between 0 and 5 seconds")


def _validate_untrusted_selector_args(site_name: str, args: dict[str, Any]) -> None:
    if site_name != "gitlab":
        return
    for selector in ("project_id", "issue_iid", "mr_iid"):
        value = args.get(selector)
        if value in (None, ""):
            continue
        if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            continue
        raise ValueError(
            f"editor_calls gitlab selector {selector!r} must come from prior seed context, not a literal value"
        )

__all__ = [
    "_validate_editor_calls",
    "_validate_pre_call_delay",
    "_validate_untrusted_selector_args",
    "validate_data_seed",
]
