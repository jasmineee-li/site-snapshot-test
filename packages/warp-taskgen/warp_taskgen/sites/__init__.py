"""Explicit Site Targeting seam.

The Site Targeting module is deliberately small.  Callers bind a benchmark
profile and an immutable targeting context once, then ask the bound Site for
route descriptors or a deterministic target.  Authentication, browser
reachability, editors, exposure, and scoring remain outside this package.
"""

from warp_taskgen.sites.bound_site import BoundSite
from warp_taskgen.sites.candidate_resolution import SourceListing, TargetCandidate
from warp_taskgen.sites.catalog import SiteCatalog, default_catalog
from warp_taskgen.sites.classifieds_reader import (
    CLASSIFIEDS_READER_AUTH_TYPE,
    CLASSIFIEDS_READER_CONTRACT_SCHEMA_VERSION,
    ClassifiedsReaderPreflight,
    preflight_classifieds_reader,
)
from warp_taskgen.sites.contracts import (
    CanonicalRoute,
    ResolvedTarget,
    SiteAdapter,
    SiteCarrierPolicy,
    SiteCarrierPolicyCapability,
    SiteProfileRouteCapability,
    SiteRouteContractFacts,
    SiteTargetingDefinitionError,
    SurfaceResolution,
    TargetingContext,
    TargetingFailure,
)
from warp_taskgen.sites.gitlab import GitLabSite
from warp_taskgen.sites.listing_resolution import ListingItemCandidate, ListingSiteAdapter
from warp_taskgen.sites.read_surface import (
    ReadSurfacePlanFailure,
    ReadSurfaceVerificationPlan,
    SiteReadSurfaceCapability,
)
from warp_taskgen.sites.readback import (
    ReadbackDecision,
    ReadbackFailure,
    ReadbackObservation,
    ReadbackObservationCapability,
)
from warp_taskgen.sites.reddit import RedditSite
from warp_taskgen.sites.rocketchat import RocketChatSite

__all__ = [
    "CLASSIFIEDS_READER_AUTH_TYPE",
    "CLASSIFIEDS_READER_CONTRACT_SCHEMA_VERSION",
    "BoundSite",
    "CanonicalRoute",
    "ClassifiedsReaderPreflight",
    "GitLabSite",
    "ListingItemCandidate",
    "ListingSiteAdapter",
    "ReadSurfacePlanFailure",
    "ReadSurfaceVerificationPlan",
    "ReadbackDecision",
    "ReadbackFailure",
    "ReadbackObservation",
    "ReadbackObservationCapability",
    "RedditSite",
    "ResolvedTarget",
    "RocketChatSite",
    "SiteAdapter",
    "SiteCarrierPolicy",
    "SiteCarrierPolicyCapability",
    "SiteCatalog",
    "SiteProfileRouteCapability",
    "SiteReadSurfaceCapability",
    "SiteRouteContractFacts",
    "SiteTargetingDefinitionError",
    "SourceListing",
    "SurfaceResolution",
    "TargetCandidate",
    "TargetingContext",
    "TargetingFailure",
    "default_catalog",
    "preflight_classifieds_reader",
]
