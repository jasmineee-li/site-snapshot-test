"""Public package surface for data seeding."""

from __future__ import annotations

from typing import Any

# ruff: noqa: F401, F403
from warp_taskgen.seeding import _impl as _legacy_impl
from warp_taskgen.seeding import validation as _validation
from warp_taskgen.seeding._impl import *
from warp_taskgen.seeding.site_contracts import (
    CreatedResourceFact,
    EditorSeedResult,
    ReadSurfaceFact,
    SeedSiteRegistration,
    SeedSiteRegistry,
)

globals().update(
    {name: value for name, value in vars(_legacy_impl).items() if not name.startswith("__")}
)

_PATCHABLE_GLOBALS = (
    "EDITOR_REGISTRY",
    "EditorError",
    "requests",
    "time",
    "_REDDIT_TABLE_NAME_CACHE",
    "_connect_db",
    "_render_editor_seed_call",
    "_get_editor_for_seed_call",
    "_apply_editor_seed_call",
    "_derive_reddit_seed_context",
    "_derive_map_seed_context",
)


def _sync_legacy_patches() -> None:
    for name in _PATCHABLE_GLOBALS:
        if name in globals():
            setattr(_legacy_impl, name, globals()[name])
            if name == "EDITOR_REGISTRY":
                setattr(_validation, name, globals()[name])


def validate_data_seed(
    seed: dict[str, Any],
    *,
    allow_none: bool = False,
    seed_registry: SeedSiteRegistry | None = None,
) -> None:
    _sync_legacy_patches()
    return _legacy_impl.validate_data_seed(
        seed,
        allow_none=allow_none,
        seed_registry=seed_registry,
    )


def apply_data_seed(
    seed: dict[str, Any],
    instance: dict[str, Any],
    *,
    seed_registry: SeedSiteRegistry | None = None,
) -> tuple[Any, dict[str, Any]]:
    _sync_legacy_patches()
    return _legacy_impl.apply_data_seed(seed, instance, seed_registry=seed_registry)


async def apply_data_seed_async(
    seed: dict[str, Any],
    instance: dict[str, Any],
    *,
    seed_registry: SeedSiteRegistry | None = None,
) -> tuple[Any, dict[str, Any]]:
    _sync_legacy_patches()
    return await _legacy_impl.apply_data_seed_async(
        seed,
        instance,
        seed_registry=seed_registry,
    )


def preflight_editor_seed_calls(
    seed: dict[str, Any],
    instance: dict[str, Any],
    *,
    seed_registry: SeedSiteRegistry | None = None,
) -> list[dict[str, Any]]:
    _sync_legacy_patches()
    return _legacy_impl.preflight_editor_seed_calls(
        seed,
        instance,
        seed_registry=seed_registry,
    )


def collect_seed_runtime_errors(*args: Any, **kwargs: Any) -> list[str]:
    _sync_legacy_patches()
    return _legacy_impl.collect_seed_runtime_errors(*args, **kwargs)


def _build_seed_context(seed: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any]:
    _sync_legacy_patches()
    return _legacy_impl._build_seed_context(seed, instance)


def _resolve_reddit_forum(task: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any] | None:
    _sync_legacy_patches()
    return _legacy_impl._resolve_reddit_forum(task, instance)


def _resolve_reddit_submission_id(
    task: dict[str, Any], instance: dict[str, Any]
) -> dict[str, Any] | None:
    _sync_legacy_patches()
    return _legacy_impl._resolve_reddit_submission_id(task, instance)
