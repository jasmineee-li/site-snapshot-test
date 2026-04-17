from __future__ import annotations

from typing import Any

from .types import ResolvedCall, ResolverError


def create(
    target: dict[str, Any],
    instance: dict[str, Any],
    seed_context: dict[str, Any],
) -> ResolvedCall:
    raise ResolverError(
        "deferred_site",
        "map target-based seeding is deferred; quarantine map tasks instead of migrating them",
    )


def update(
    target: dict[str, Any],
    instance: dict[str, Any],
    seed_context: dict[str, Any],
) -> ResolvedCall:
    raise ResolverError(
        "deferred_site",
        "map target-based seeding is deferred; quarantine map tasks instead of migrating them",
    )
