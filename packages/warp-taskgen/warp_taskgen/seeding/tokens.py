"""Benign seed token binding contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from warp_taskgen.editors import EditorError
from warp_taskgen.seeding.context import _FORMAT_TOKEN_PATTERN
from warp_taskgen.seeding.editor_args import _infer_editor_call_benchmark, _infer_task_benchmark
from warp_taskgen.seeding.site_contracts import SeedSiteRegistry


class UnboundTokenError(ValueError):
    """Raised at seed-apply time when a ``{benign_<anchor_key>}`` token
    is referenced but not reachable via the task's
    ``benign_target_resource.anchors`` — the "silently renders empty"
    failure mode commit 4's registry validator now catches at Phase 2a,
    and which this check guarantees at Phase 2c.

    Phase 2c categorizes this as ``error.kind = "contract_violation"``,
    distinct from ``schema_mismatch`` (shape/type violations).
    """

    def __init__(
        self,
        *,
        task_id: str,
        kind: str,
        token: str,
        available_tokens: frozenset[str],
        anchors: dict[str, Any],
    ) -> None:
        msg = (
            f"Phantom token {token!r} in seed for task {task_id!r} "
            f"(kind={kind!r}): not reachable via anchors. "
            f"Available: {sorted(available_tokens)}; anchors: {dict(anchors)}"
        )
        super().__init__(msg)
        self.task_id = task_id
        self.kind = kind
        self.token = token
        self.available_tokens = frozenset(available_tokens)
        self.anchors = dict(anchors)


def _collect_benign_tokens(value: Any) -> set[str]:
    """Walk ``value`` and collect every ``{benign_<x>}`` token it
    references. Used by :func:`_assert_benign_tokens_bound` to pre-check
    seed calls before substitution."""
    tokens: set[str] = set()
    if isinstance(value, str):
        for match in _FORMAT_TOKEN_PATTERN.finditer(value):
            key = match.group(1)
            if key.startswith("benign_"):
                tokens.add(f"{{{key}}}")
    elif isinstance(value, dict):
        for k, v in value.items():
            tokens |= _collect_benign_tokens(k)
            tokens |= _collect_benign_tokens(v)
    elif isinstance(value, list):
        for item in value:
            tokens |= _collect_benign_tokens(item)
    return tokens


def _declared_tokens_for_seed_call(
    value: Any,
    task: Any,
    *,
    seed_registry: SeedSiteRegistry,
) -> frozenset[str]:
    """Read declared token bindings from an explicit per-run editor registry."""

    if not isinstance(value, Mapping) or not isinstance(task, dict):
        return frozenset()
    resource = task.get("benign_target_resource")
    if not isinstance(resource, Mapping):
        return frozenset()
    kind = resource.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        return frozenset()
    site = str(value.get("site") or task.get("site") or "").strip().lower()
    method = str(value.get("method") or "").strip()
    if not site or not method:
        return frozenset()
    try:
        benchmark = _infer_editor_call_benchmark(value, task)
    except EditorError:
        return frozenset()
    registration = seed_registry.get(benchmark, site)
    if registration is None:
        return frozenset()
    method_spec = getattr(
        getattr(registration.editor_factory, method, None),
        "_editor_method_spec",
        None,
    )
    if not isinstance(method_spec, Mapping) or kind not in method_spec.get("kinds", ()):
        return frozenset()
    bindings = method_spec.get("bindings")
    if not isinstance(bindings, Mapping):
        return frozenset()
    declared: set[str] = set()
    for binding in bindings.values():
        tokens = getattr(binding, "tokens", ())
        if isinstance(tokens, (set, frozenset, tuple, list)):
            declared.update(token for token in tokens if isinstance(token, str))
    return frozenset(declared)


def _assert_benign_tokens_bound(
    value: Any,
    task: Any,
    *,
    seed_registry: SeedSiteRegistry | None = None,
) -> None:
    """Raise :class:`UnboundTokenError` if ``value`` references any
    ``{benign_<x>}`` token not in the contract's
    :func:`available_tokens_for_kind` for the task's kind + anchors.

    No-op when the task lacks a ``benign_target_resource`` with a
    non-null kind (legacy tasks, pending-L3 records). The Option A
    validator rejects null-kind tasks at Phase 2a; feasibility still
    benefits from the check because it catches cases where Phase 2a
    regeneration hasn't happened yet.
    """
    if not isinstance(task, dict):
        return
    resource = task.get("benign_target_resource")
    if not isinstance(resource, dict):
        return
    kind = resource.get("kind")
    if not isinstance(kind, str) or not kind:
        return
    anchors_raw = resource.get("anchors")
    anchors = anchors_raw if isinstance(anchors_raw, dict) else {}

    tokens_referenced = _collect_benign_tokens(value)
    if not tokens_referenced:
        return

    site = str(task.get("site") or "").strip().lower() or None
    try:
        benchmark = _infer_task_benchmark(task)
    except ValueError as exc:
        raise ValueError(f"seed token contract benchmark metadata is invalid: {exc}") from exc
    # An explicit Site registry is intentionally isolated from the historical
    # process-wide editor contract registry.  Read its method declaration when
    # available; the legacy path retains the existing global lookup and shapes.
    if seed_registry is not None:
        declared = _declared_tokens_for_seed_call(
            value,
            task,
            seed_registry=seed_registry,
        )
        available = declared & frozenset(f"{{benign_{key}}}" for key in anchors)
    else:
        # Lazy import — defer the global registry import until a seed call
        # actually needs the contract check.
        from warp_taskgen.editors._registry import available_tokens_for_kind

        available = available_tokens_for_kind(kind, anchors, benchmark=benchmark, site=site)
    for token in sorted(tokens_referenced):
        if token not in available:
            task_id = task.get("id") or task.get("benign_task_id") or "unknown"
            raise UnboundTokenError(
                task_id=str(task_id),
                kind=kind,
                token=token,
                available_tokens=available,
                anchors=anchors,
            )


__all__ = [
    "UnboundTokenError",
    "_assert_benign_tokens_bound",
    "_collect_benign_tokens",
]
