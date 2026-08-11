"""Explicit Site Targeting seam.

The Site Targeting module is deliberately small.  Callers bind a benchmark
profile and an immutable targeting context once, then ask the bound Site for
route descriptors or a deterministic target.  Authentication, browser
reachability, editors, exposure, and scoring remain outside this package.
"""

from worldsim.sites.candidate_resolution import SourceListing, TargetCandidate
from worldsim.sites.catalog import BoundSite, SiteCatalog, default_catalog
from worldsim.sites.contracts import (
    CanonicalRoute,
    ResolvedTarget,
    SiteAdapter,
    SiteTargetingDefinitionError,
    TargetingContext,
    TargetingFailure,
)
from worldsim.sites.gitlab import GitLabSite
from worldsim.sites.listing_resolution import ListingItemCandidate, ListingSiteAdapter
from worldsim.sites.reddit import RedditSite

__all__ = [
    "BoundSite",
    "CanonicalRoute",
    "GitLabSite",
    "ListingItemCandidate",
    "ListingSiteAdapter",
    "RedditSite",
    "ResolvedTarget",
    "SiteAdapter",
    "SiteCatalog",
    "SiteTargetingDefinitionError",
    "SourceListing",
    "TargetCandidate",
    "TargetingContext",
    "TargetingFailure",
    "default_catalog",
]
