"""Public package surface for deterministic Phase 2 exposure contracts."""

from __future__ import annotations

# ruff: noqa: F401
from worldsim.phase_2.exposure_contract import _impl as _legacy_impl
from worldsim.phase_2.exposure_contract.builder import build_exposure_contract
from worldsim.phase_2.exposure_contract.seed_template import (
    materialize_seed_template_from_contract,
)


def exposure_contract_signature() -> dict[str, object]:
    return _legacy_impl.exposure_contract_signature()


def signature_hash() -> str:
    return _legacy_impl.signature_hash()
