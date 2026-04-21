"""Editor-method contract registry.

Populated at package import time by :func:`register_editor`, which walks
a registered editor class's ``supported_methods`` frozenset and reads the
``_editor_method_spec`` attribute that :func:`worldsim.editors._method_spec.editor_method`
attached to each method.

Five host-side consumers derive their behavior from the populated
registry:

1. The resolver's ``attach_surfaces_for_kind`` (phase_2_target_resolver).
2. The Option A validator in phase_2_injections.
3. The seed substituter in :mod:`worldsim.seeding` (fails loud on
   phantom tokens).
4. The prompt renderer that fills the ``<!-- EDITOR_CONTRACT_TABLE -->``
   sentinel in ``generate-injections.md``.
5. The pre-shard feasibility filter in phase_2_injections.

A sixth consumer, the stdlib-only sandbox validator, reads a JSON
serialization of this registry written at sandbox-payload build time (see
:func:`serialize_registry`).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

from worldsim.editors._method_spec import BindingSpec


class RegistryError(RuntimeError):
    """Raised when :func:`register_editor` finds drift between an editor
    class's ``supported_methods`` and the decorators on its attributes."""


@dataclass(frozen=True)
class EditorMethodSpec:
    site: str
    method: str
    kinds: frozenset[str]
    http: tuple[str, str]
    bindings: Mapping[str, BindingSpec]
    surface_id_per_kind: Mapping[str, str]
    required_editor_args: tuple[str, ...]


@dataclass(frozen=True)
class KindContract:
    kind: str
    valid_methods: frozenset[str]
    required_anchor_keys: frozenset[str]
    available_tokens: frozenset[str]


_REGISTRY: dict[tuple[str, str], EditorMethodSpec] = {}

# Tokens that don't derive from anchors (come from agent-context identity).
IDENTITY_TOKENS: frozenset[str] = frozenset({"{benign_user_handle}"})


def register_editor(cls: type, site: str) -> None:
    """Register every method in ``cls.supported_methods`` into ``_REGISTRY``.

    Raises :class:`RegistryError` if any ``supported_methods`` member lacks
    the ``@editor_method`` decorator, or if ``(site, method)`` is already
    registered. The opt-out is to remove the method from
    ``supported_methods`` — explicit and grep-visible.
    """
    supported = getattr(cls, "supported_methods", None)
    if not isinstance(supported, (frozenset, set, tuple, list)) or not supported:
        raise RegistryError(
            f"{cls.__name__}.supported_methods is missing or empty; cannot register"
        )
    for method_name in sorted(supported):
        attr = getattr(cls, method_name, None)
        spec_meta = getattr(attr, "_editor_method_spec", None)
        if spec_meta is None:
            raise RegistryError(
                f"{cls.__name__}.{method_name} is in supported_methods but has no "
                f"@editor_method decorator — either decorate it or remove it from "
                f"supported_methods"
            )
        key = (site, method_name)
        if key in _REGISTRY:
            raise RegistryError(f"duplicate registration: {key!r}")
        _REGISTRY[key] = EditorMethodSpec(
            site=site,
            method=method_name,
            kinds=spec_meta["kinds"],
            http=spec_meta["http"],
            bindings=dict(spec_meta["bindings"]),
            surface_id_per_kind=dict(spec_meta["surface_id_per_kind"]),
            required_editor_args=spec_meta["required_editor_args"],
        )
    _clear_caches()


def _clear_caches() -> None:
    """Reset ``@cache``'d helpers. Call after mutating ``_REGISTRY`` in tests."""
    method_spec.cache_clear()
    kind_contract.cache_clear()
    attach_surfaces_for_kind.cache_clear()


@cache
def method_spec(site: str, method: str) -> EditorMethodSpec:
    """Return the :class:`EditorMethodSpec` for ``(site, method)``.

    Raises :class:`KeyError` if unknown.
    """
    return _REGISTRY[(site, method)]


@cache
def kind_contract(kind: str) -> KindContract:
    """Build the per-kind contract by unioning all methods that address it.

    ``valid_methods`` is the set of method names (across all sites) that
    declare ``kind`` in their ``kinds`` set. ``available_tokens`` is the
    union of all tokens any such method's bindings can accept, plus the
    identity tokens (``{benign_user_handle}``). ``required_anchor_keys``
    is the union of anchor keys implied by those tokens (e.g.
    ``{benign_project_id}`` → ``project_id``).
    """
    valid_methods: set[str] = set()
    available: set[str] = set(IDENTITY_TOKENS)
    for spec in _REGISTRY.values():
        if kind not in spec.kinds:
            continue
        valid_methods.add(spec.method)
        for binding in spec.bindings.values():
            available.update(binding.tokens)
    required_anchor_keys = frozenset(
        _anchor_key_from_token(tok) for tok in available if tok not in IDENTITY_TOKENS
    ) - frozenset({""})
    return KindContract(
        kind=kind,
        valid_methods=frozenset(valid_methods),
        required_anchor_keys=required_anchor_keys,
        available_tokens=frozenset(available),
    )


def _anchor_key_from_token(token: str) -> str:
    """Extract the anchor key from a ``{benign_<anchor_key>}`` token string."""
    inner = token.strip("{}")
    if inner.startswith("benign_"):
        return inner[len("benign_") :]
    return ""


@cache
def attach_surfaces_for_kind(kind: str) -> tuple[dict[str, Any], ...]:
    """Return the attach-surface list the resolver emits for ``kind``.

    Shape matches the pre-refactor ``_ATTACH_SURFACES`` output: a tuple of
    dicts, one per ``(site, method)`` addressing ``kind``, each with
    ``surface_id``, ``attach_method``, ``required_editor_args``.
    """
    out: list[dict[str, Any]] = []
    for (_, method), spec in _REGISTRY.items():
        if kind not in spec.kinds:
            continue
        surface_id = spec.surface_id_per_kind.get(kind, method)
        out.append(
            {
                "surface_id": surface_id,
                "attach_method": method,
                "required_editor_args": list(spec.required_editor_args),
            }
        )
    return tuple(out)


def available_tokens_for_kind(kind: str, anchors: Mapping[str, Any]) -> frozenset[str]:
    """Tokens declared valid for ``kind`` AND reachable via ``anchors``.

    Intersects the contract's declared tokens with the set of
    ``{benign_<anchor_key>}`` tokens the resolver actually emitted for
    this task, plus the identity tokens. The substituter raises
    :class:`worldsim.seeding.UnboundTokenError` on any seed token outside
    this set.
    """
    declared = kind_contract(kind).available_tokens
    reachable = frozenset(f"{{benign_{k}}}" for k in anchors) | IDENTITY_TOKENS
    return declared & reachable


def iter_specs(
    site: str | None = None,
    kinds: frozenset[str] | None = None,
) -> Iterator[EditorMethodSpec]:
    """Iterate registered specs, optionally filtered by site and/or kinds."""
    for spec in _REGISTRY.values():
        if site is not None and spec.site != site:
            continue
        if kinds is not None and not (spec.kinds & kinds):
            continue
        yield spec


def serialize_registry() -> dict[str, Any]:
    """Export ``_REGISTRY`` as a plain JSON-compatible dict.

    Consumed by the stdlib-only sandbox validator after
    :func:`worldsim.modal_sandbox.run_claude_in_sandbox` writes the
    result to ``/workspace/_editor_registry.json`` in the sandbox payload.
    The ``version`` field is reserved for forward-compat; readers should
    tolerate additive schema changes.
    """
    return {
        "version": 1,
        "specs": [
            {
                "site": spec.site,
                "method": spec.method,
                "kinds": sorted(spec.kinds),
                "http_verb": spec.http[0],
                "http_path": spec.http[1],
                "bindings": {
                    arg: {
                        "kind": binding.kind,
                        "tokens": sorted(binding.tokens),
                        "selector_group": binding.selector_group,
                        "required": binding.required,
                    }
                    for arg, binding in spec.bindings.items()
                },
                "surface_id_per_kind": dict(spec.surface_id_per_kind),
                "required_editor_args": list(spec.required_editor_args),
            }
            for spec in sorted(_REGISTRY.values(), key=lambda s: (s.site, s.method))
        ],
    }


__all__ = [
    "IDENTITY_TOKENS",
    "_REGISTRY",
    "EditorMethodSpec",
    "KindContract",
    "RegistryError",
    "_clear_caches",
    "attach_surfaces_for_kind",
    "available_tokens_for_kind",
    "iter_specs",
    "kind_contract",
    "method_spec",
    "register_editor",
    "serialize_registry",
]
