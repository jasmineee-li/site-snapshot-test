from __future__ import annotations

from types import ModuleType

from .types import ResolverError


def get_resolver(benchmark: str, site_name: str) -> ModuleType:
    normalized_benchmark = str(benchmark or "webarena_verified").strip().lower()
    if normalized_benchmark not in {"", "webarena_verified"}:
        raise ResolverError(
            "unsupported_benchmark",
            f"no seed target resolver registry for benchmark {benchmark!r}",
        )

    normalized_site = str(site_name).strip().lower()
    if normalized_site == "gitlab":
        from . import gitlab as resolver_module

        return resolver_module
    if normalized_site == "shopping":
        from . import shopping as resolver_module

        return resolver_module
    if normalized_site == "shopping_admin":
        from . import shopping_admin as resolver_module

        return resolver_module
    if normalized_site == "reddit":
        from . import reddit as resolver_module

        return resolver_module
    if normalized_site == "map":
        from . import map as resolver_module

        return resolver_module
    raise ResolverError("unsupported_site", f"no seed target resolver for site {site_name!r}")
