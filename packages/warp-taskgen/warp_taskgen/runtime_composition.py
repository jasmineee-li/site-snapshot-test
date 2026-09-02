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
    from warp_taskgen.benchmark_contracts import BenchmarkCapabilities
    from warp_taskgen.phase_2.phase_2c.policy import FeasibilityPolicyCatalog
    from warp_taskgen.seeding.site_contracts import SeedSiteRegistry
    from warp_taskgen.sites.catalog import SiteCatalog

CLASSIFIEDS_LISTING_REPLY_POC = "classifieds_listing_reply_poc"
ROCKET_CHAT_CONVERSATION_DECISION_POC = "rocket_chat_conversation_decision_poc"
RUNTIME_COMPOSITION_CHOICES = (
    CLASSIFIEDS_LISTING_REPLY_POC,
    ROCKET_CHAT_CONVERSATION_DECISION_POC,
)
_DEFAULT_ROCKET_CHAT_READBACK = object()


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
class Phase2RuntimeAdmission:
    """Narrow, typed decision returned by an explicit Phase 2 composition."""

    admitted: bool
    reason: str
    checks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.admitted, bool):
            raise TypeError("Phase 2 runtime admission admitted must be boolean")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("Phase 2 runtime admission requires a reason")
        if any(not isinstance(check, str) or not check.strip() for check in self.checks):
            raise ValueError("Phase 2 runtime admission checks must be bounded text")
        object.__setattr__(self, "checks", tuple(self.checks))

    @property
    def ok(self) -> bool:
        return self.admitted

    def as_dict(self) -> dict[str, object]:
        return {
            "admitted": self.admitted,
            "reason": self.reason,
            "checks": list(self.checks),
        }


@dataclass(frozen=True)
class RuntimeComposition:
    """One immutable set of runtime catalogs for a bounded pipeline run."""

    name: str
    site_catalog: SiteCatalog
    seed_registry: SeedSiteRegistry
    feasibility_policy_catalog: FeasibilityPolicyCatalog
    benchmark_capabilities: BenchmarkCapabilities | None = None
    reader_preflight: Callable[[Mapping[str, object]], object] | None = None
    phase_2_admission: Callable[[object, object], Phase2RuntimeAdmission] | None = None
    strict_seed_cleanup: bool = False

    def __post_init__(self) -> None:
        from warp_taskgen.benchmark_contracts import BenchmarkCapabilities
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
        if self.benchmark_capabilities is not None and not isinstance(
            self.benchmark_capabilities, BenchmarkCapabilities
        ):
            raise TypeError("runtime composition benchmark_capabilities must be BenchmarkCapabilities")
        if self.reader_preflight is not None and not callable(self.reader_preflight):
            raise TypeError("runtime composition reader_preflight must be callable")
        if self.phase_2_admission is not None and not callable(self.phase_2_admission):
            raise TypeError("runtime composition phase_2_admission must be callable")
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


def rocket_chat_conversation_decision_poc(
    *, readback_adapter: object = _DEFAULT_ROCKET_CHAT_READBACK
) -> RuntimeComposition:
    """Build explicit TAC wiring for source and opt-in seam tests.

    The composition is intentionally separate from the static Site catalog:
    selecting it does not reopen Rocket.Chat as a default WARP carrier.  TAC's
    benchmark capability gates remain source-only globally.  The named
    composition binds a pinned Rocket.Chat 5.3 thread-panel adapter by
    default; callers may pass ``None`` to exercise the fail-closed, unbound
    seam or inject a deployment-specific adapter.
    Writer cleanup remains strict and fails closed without reset/admin input.
    Phase 2 admission additionally requires the caller's instance to provide
    an explicit reset endpoint and independent reader browser auth.  The
    default adapter only establishes exact DOM identity; the render executor
    supplies the separate geometry witness needed for Painted Visibility.
    """

    from warp_taskgen.benchmark_contracts import BenchmarkCapabilities
    from warp_taskgen.phase_2.phase_2c.policy import FeasibilityPolicyCatalog
    from warp_taskgen.seeding.site_contracts import SeedSiteRegistration, SeedSiteRegistry
    from warp_taskgen.sites.catalog import SiteCatalog
    from warp_taskgen.sites.rocketchat_admission import rocket_chat_phase2_admission
    from warp_taskgen.sites.rocketchat_readback import RocketChatThreadPanelReadbackAdapter
    from warp_taskgen.sites.rocketchat_runtime import (
        RocketChatFeasibilityPolicy,
        RocketChatHttpEditor,
        RocketChatRuntimeSite,
        preflight_rocket_chat_reader,
    )

    benchmark = "theagentcompany"
    site = "rocketchat"
    if readback_adapter is _DEFAULT_ROCKET_CHAT_READBACK:
        readback_adapter = RocketChatThreadPanelReadbackAdapter()
    site_catalog = SiteCatalog((RocketChatRuntimeSite(readback_adapter=readback_adapter),))
    return RuntimeComposition(
        name=ROCKET_CHAT_CONVERSATION_DECISION_POC,
        site_catalog=site_catalog,
        seed_registry=SeedSiteRegistry.from_registrations(
            (SeedSiteRegistration(benchmark, site, RocketChatHttpEditor),)
        ),
        feasibility_policy_catalog=FeasibilityPolicyCatalog.from_policies(
            (RocketChatFeasibilityPolicy(),)
        ),
        benchmark_capabilities=BenchmarkCapabilities(
            canonical_name=benchmark,
            default_runner="browser_use",
            supported_runners=("browser_use",),
            capabilities=frozenset(
                {
                    "phase_1_generation",
                    "phase_2_generation",
                    "phase_2_feasibility",
                    "phase_4_execution",
                    "warp_evaluation",
                }
            ),
            phase_4_mode="worldsim_v5",
            evaluator_authorities=("warp_local_task_idless",),
        ),
        reader_preflight=preflight_rocket_chat_reader,
        phase_2_admission=lambda tasks, instances: rocket_chat_phase2_admission(
            tasks,
            instances,
            site_catalog=site_catalog,
            reader_preflight=preflight_rocket_chat_reader,
        ),
        strict_seed_cleanup=True,
    )


def benchmark_capabilities_for_runtime(
    benchmark: object,
    runtime_composition: RuntimeComposition | None = None,
):
    """Resolve a composition-scoped capability override without mutating the catalog."""

    from warp_taskgen.benchmark_capabilities import (
        get_benchmark_capabilities,
        normalize_benchmark_name,
    )

    canonical = normalize_benchmark_name(benchmark)
    override = (
        runtime_composition.benchmark_capabilities
        if runtime_composition is not None
        else None
    )
    if override is not None and override.canonical_name == canonical:
        return override
    return get_benchmark_capabilities(benchmark)


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
    "Phase2RuntimeAdmission",
    "RequiredSeedCleanupError",
    "RuntimeComposition",
    "benchmark_capabilities_for_runtime",
    "classifieds_listing_reply_poc",
    "rocket_chat_conversation_decision_poc",
    "runtime_composition_for_name",
]
