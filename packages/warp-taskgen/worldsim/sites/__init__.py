"""Explicit Site Targeting seam.

The Site Targeting module is deliberately small.  Callers bind a benchmark
profile and an immutable targeting context once, then ask the bound Site for
route descriptors or a deterministic target.  Authentication, browser
reachability, editors, exposure, and scoring remain outside this package.
"""

from worldsim.sites.catalog import (
    BoundSite,
    CanonicalRoute,
    ResolvedTarget,
    SiteCatalog,
    SiteTargetingDefinitionError,
    TargetingContext,
    TargetingFailure,
    default_catalog,
)
from worldsim.sites.gitlab import GitLabSite
from worldsim.sites.reddit import RedditSite

__all__ = [
    "BoundSite",
    "CanonicalRoute",
    "GitLabSite",
    "RedditSite",
    "ResolvedTarget",
    "SiteCatalog",
    "SiteTargetingDefinitionError",
    "TargetingContext",
    "TargetingFailure",
    "default_catalog",
]
