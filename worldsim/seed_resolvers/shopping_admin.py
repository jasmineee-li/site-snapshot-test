from __future__ import annotations

from typing import Any

from . import shopping
from .types import ResolvedCall, ResolverError


def create(
    target: dict[str, Any],
    instance: dict[str, Any],
    seed_context: dict[str, Any],
) -> ResolvedCall:
    return shopping.create(target, instance, seed_context)


def update(
    target: dict[str, Any],
    instance: dict[str, Any],
    seed_context: dict[str, Any],
) -> ResolvedCall:
    resource_type = str(target.get("resource_type", "")).strip().lower()
    if resource_type != "admin_profile":
        raise ResolverError(
            "unsupported_resource_type",
            f"shopping_admin resolver does not support update resource_type={resource_type!r}",
        )
    site_url = str(instance.get("site_url", "")).rstrip("/")
    if not site_url:
        raise ResolverError(
            "missing_site_url",
            "shopping_admin resolver requires instance['site_url']",
        )
    return ResolvedCall(
        method="PUT",
        url=f"{site_url}/rest/V1/users/me",
        headers={},
    )
