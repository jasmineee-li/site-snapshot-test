"""Compatibility facade for deterministic Phase 2 exposure contracts."""

from __future__ import annotations

from typing import Any

# ruff: noqa: F403
from worldsim.phase_2.exposure_contract import *
from worldsim.phase_2.exposure_contract import _impl as _legacy_impl

globals().update(
    {
        name: value
        for name, value in vars(_legacy_impl).items()
        if not name.startswith("__")
    }
)

_PATCHABLE_GLOBALS = (
    "available_tokens_for_kind",
    "iter_specs",
    "method_spec",
    "canonical_core_surface",
    "is_active_carrier_surface",
    "is_core_surface",
    "active_carrier_ineligible_reason",
    "PAYLOAD_PLACEHOLDER",
    "PREFERRED_PAYLOAD_ARGS",
    "PREFERRED_TOKEN_ORDER",
)


def _sync_legacy_patches() -> None:
    for name in _PATCHABLE_GLOBALS:
        if name in globals():
            setattr(_legacy_impl, name, globals()[name])


def exposure_contract_signature() -> dict[str, Any]:
    _sync_legacy_patches()
    return _legacy_impl.exposure_contract_signature()


def signature_hash() -> str:
    _sync_legacy_patches()
    return _legacy_impl.signature_hash()


def build_exposure_contract(*args: Any, **kwargs: Any) -> dict[str, Any]:
    _sync_legacy_patches()
    return _legacy_impl.build_exposure_contract(*args, **kwargs)


def materialize_seed_template_from_contract(*args: Any, **kwargs: Any) -> dict[str, Any]:
    _sync_legacy_patches()
    return _legacy_impl.materialize_seed_template_from_contract(*args, **kwargs)
