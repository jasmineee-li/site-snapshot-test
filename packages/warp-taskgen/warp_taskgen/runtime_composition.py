"""Immutable per-run Site composition for explicit runtime opt-ins.

The normal pipeline deliberately leaves this value as ``None`` and keeps its
historical GitLab/Reddit defaults.  A named experiment may bind the existing
Site, seed, and Phase 2c policy catalogs once and pass the snapshot through
the phase runners without mutating process-wide registries.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from warp_taskgen.phase_2.phase_2c.policy import FeasibilityPolicyCatalog
    from warp_taskgen.seeding.site_contracts import SeedSiteRegistry
    from warp_taskgen.sites.catalog import SiteCatalog

CLASSIFIEDS_LISTING_REPLY_POC = "classifieds_listing_reply_poc"
ROCKET_CHAT_CONVERSATION_DECISION_POC = "rocket_chat_conversation_decision_poc"
RUNTIME_COMPOSITION_CHOICES = (
    CLASSIFIEDS_LISTING_REPLY_POC,
    ROCKET_CHAT_CONVERSATION_DECISION_POC,
)


class RequiredSeedCleanupError(RuntimeError):
    """A named composition could not restore the seeded resource boundary.

    ``primary_error`` and ``cleanup_error`` are kept separately when a seed
    mutates state and then fails before returning its cleanup handle.  The
    exception itself is the terminal signal for strict compositions while the
    original seed failure remains available as ``__cause__``.
    """

    def __init__(
        self,
        message: str,
        *,
        primary_error: BaseException | None = None,
        cleanup_error: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.primary_error = primary_error
        self.cleanup_error = cleanup_error


@dataclass(frozen=True)
class RuntimeComposition:
    """One immutable set of runtime catalogs for a bounded pipeline run."""

    name: str
    site_catalog: SiteCatalog
    seed_registry: SeedSiteRegistry
    feasibility_policy_catalog: FeasibilityPolicyCatalog
    reader_preflight: Callable[[Mapping[str, object]], object] | None = None
    strict_seed_cleanup: bool = False

    def __post_init__(self) -> None:
        from warp_taskgen.phase_2.phase_2c.policy import FeasibilityPolicyCatalog
        from warp_taskgen.seeding.site_contracts import SeedSiteRegistry
        from warp_taskgen.sites.catalog import SiteCatalog

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("runtime composition requires a non-empty name")
        if not isinstance(self.site_catalog, SiteCatalog):
            raise TypeError("runtime composition site_catalog must be a SiteCatalog")
        if not isinstance(self.seed_registry, SeedSiteRegistry):
            raise TypeError("runtime composition seed_registry must be a SeedSiteRegistry")
        if not isinstance(self.feasibility_policy_catalog, FeasibilityPolicyCatalog):
            raise TypeError(
                "runtime composition feasibility_policy_catalog must be a FeasibilityPolicyCatalog"
            )
        if self.reader_preflight is not None and not callable(self.reader_preflight):
            raise TypeError("runtime composition reader_preflight must be callable")
        if not isinstance(self.strict_seed_cleanup, bool):
            raise TypeError("runtime composition strict_seed_cleanup must be a bool")


def classifieds_listing_reply_poc() -> RuntimeComposition:
    """Build the explicit Classifieds listing-reply POC composition."""

    from warp_taskgen.phase_2.phase_2c.classifieds_policy import ClassifiedsFeasibilityPolicy
    from warp_taskgen.phase_2.phase_2c.policy import FeasibilityPolicyCatalog
    from warp_taskgen.seeding.site_contracts import SeedSiteRegistration, SeedSiteRegistry
    from warp_taskgen.sites.catalog import SiteCatalog
    from warp_taskgen.sites.classifieds import ClassifiedsSite
    from warp_taskgen.sites.classifieds_reader import preflight_classifieds_reader
    from warp_taskgen.sites.classifieds_writer import ClassifiedsAuthenticatedEditor

    benchmark = "visualwebarena"
    site = "classifieds"
    return RuntimeComposition(
        name=CLASSIFIEDS_LISTING_REPLY_POC,
        site_catalog=SiteCatalog((ClassifiedsSite(),)),
        seed_registry=SeedSiteRegistry.from_registrations(
            (SeedSiteRegistration(benchmark, site, ClassifiedsAuthenticatedEditor),)
        ),
        feasibility_policy_catalog=FeasibilityPolicyCatalog.from_policies(
            (ClassifiedsFeasibilityPolicy(),)
        ),
        reader_preflight=preflight_classifieds_reader,
        strict_seed_cleanup=True,
    )


def rocket_chat_conversation_decision_poc() -> RuntimeComposition:
    """Build explicit, non-admitted TAC source wiring for seam tests.

    The composition is intentionally separate from the static Site catalog:
    selecting it does not reopen Rocket.Chat as a default WARP carrier.  TAC's
    benchmark capability gates remain source-only, so Phase 2/4 reject this
    composition until a host reset owner and exact painted readback exist.
    Writer cleanup remains strict and fails closed without reset/admin input.
    """

    from warp_taskgen.phase_2.phase_2c.policy import FeasibilityPolicyCatalog
    from warp_taskgen.seeding.site_contracts import SeedSiteRegistration, SeedSiteRegistry
    from warp_taskgen.sites.catalog import SiteCatalog
    from warp_taskgen.sites.rocketchat_runtime import (
        RocketChatFeasibilityPolicy,
        RocketChatHttpEditor,
        RocketChatRuntimeSite,
        preflight_rocket_chat_reader,
    )

    benchmark = "theagentcompany"
    site = "rocketchat"
    return RuntimeComposition(
        name=ROCKET_CHAT_CONVERSATION_DECISION_POC,
        site_catalog=SiteCatalog((RocketChatRuntimeSite(),)),
        seed_registry=SeedSiteRegistry.from_registrations(
            (SeedSiteRegistration(benchmark, site, RocketChatHttpEditor),)
        ),
        feasibility_policy_catalog=FeasibilityPolicyCatalog.from_policies(
            (RocketChatFeasibilityPolicy(),)
        ),
        reader_preflight=preflight_rocket_chat_reader,
        strict_seed_cleanup=True,
    )


def runtime_composition_for_name(name: object) -> RuntimeComposition | None:
    """Resolve an explicit runtime opt-in; empty/default input stays ``None``."""

    if name in (None, ""):
        return None
    if not isinstance(name, str):
        raise ValueError("runtime composition name must be a string")
    normalized = name.strip().lower()
    if not normalized:
        return None
    if normalized == CLASSIFIEDS_LISTING_REPLY_POC:
        return classifieds_listing_reply_poc()
    if normalized == ROCKET_CHAT_CONVERSATION_DECISION_POC:
        return rocket_chat_conversation_decision_poc()
    raise ValueError(
        f"unknown runtime composition {name!r}; expected one of {RUNTIME_COMPOSITION_CHOICES!r}"
    )


__all__ = [
    "CLASSIFIEDS_LISTING_REPLY_POC",
    "ROCKET_CHAT_CONVERSATION_DECISION_POC",
    "RUNTIME_COMPOSITION_CHOICES",
    "RequiredSeedCleanupError",
    "RuntimeComposition",
    "classifieds_listing_reply_poc",
    "rocket_chat_conversation_decision_poc",
    "runtime_composition_for_name",
]
