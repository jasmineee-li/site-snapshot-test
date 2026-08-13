from __future__ import annotations

from collections.abc import Mapping

import pytest

from warp_taskgen.phases.phase_1_route_contracts import build_task_route_contracts
from warp_taskgen.sites import (
    CanonicalRoute,
    GitLabSite,
    RedditSite,
    SiteCatalog,
    SiteProfileRouteCapability,
    SiteRouteContractFacts,
    SiteTargetingDefinitionError,
    SurfaceResolution,
    TargetingContext,
)
from warp_taskgen.surface_identity import has_surface_mapping


def _profile(site: str, surface_id: str) -> dict[str, object]:
    return {
        "site_name": site,
        "injection_surface": [{"id": surface_id, "source_field": "message.body"}],
        "data_model": [],
    }


@pytest.mark.parametrize(
    ("site", "adapter", "surface_id", "canonical"),
    [
        ("gitlab", GitLabSite(), "gitlab_issue_description", "issue.description"),
        ("reddit", RedditSite(), "submission_body_detail", "submission.body"),
    ],
)
def test_active_sites_conform_to_profile_route_capability(
    site: str,
    adapter: object,
    surface_id: str,
    canonical: str,
) -> None:
    assert isinstance(adapter, SiteProfileRouteCapability)
    catalog = SiteCatalog([adapter])  # type: ignore[list-item]
    bound = catalog.bind(site=site, profile=_profile(site, surface_id))
    assert bound.supports_profile_routes()
    assert bound.canonicalize_surface_id(surface_id) == canonical
    resolution = bound.resolve_profile_surface(canonical)
    assert isinstance(resolution, SurfaceResolution)
    assert resolution.profile_surface_id == surface_id


class FakeProfileRouteSite:
    """Injected feature double proving profile/route facts are Site-owned."""

    site = "fake"
    supported_benchmarks = frozenset({"webarena_verified"})

    def validate(self) -> None:
        return None

    def validate_task(self, task: Mapping[str, object]) -> tuple[str, str] | None:
        del task
        return None

    def routes(self, context: TargetingContext) -> tuple[CanonicalRoute, ...]:
        return (
            CanonicalRoute(
                id="fake.message",
                site=context.site,
                kind="message",
                allowed_start_url_patterns=("/messages/{message_id}",),
                anchor_examples=({"message_id": "1"},),
            ),
        )

    def match(
        self,
        url: str,
        task: Mapping[str, object],
        context: TargetingContext,
    ) -> tuple[str, dict[str, object]] | None:
        del task, context
        return ("message", {"message_id": "1"}) if url.endswith("/messages/1") else None

    def reconstruct(
        self,
        kind: str,
        anchors: Mapping[str, object],
        context: TargetingContext,
    ) -> str | None:
        if kind != "message" or not anchors.get("message_id"):
            return None
        origin = context.site_origin()
        return f"{origin}/messages/{anchors['message_id']}" if origin else None

    def is_listing(self, kind: str) -> bool:
        del kind
        return False

    def listing_start_url(
        self,
        kind: str,
        resolved_url: str,
        fallback_url: str | None,
    ) -> str | None:
        del kind, resolved_url
        return fallback_url

    def canonicalize_surface_id(
        self,
        *,
        benchmark: str,
        raw_surface_id: str | None,
    ) -> str | None:
        return (
            "message.body"
            if benchmark == "webarena_verified" and raw_surface_id == "body"
            else None
        )

    def resolve_profile_surface(
        self,
        *,
        benchmark: str,
        profile: Mapping[str, object],
        target_surface_id: str,
        kind: str | None = None,
        method: str | None = None,
        editor_surface_id: str | None = None,
    ) -> SurfaceResolution | None:
        del kind, method, editor_surface_id
        if benchmark != "webarena_verified" or target_surface_id != "message.body":
            return None
        surfaces = profile.get("injection_surface")
        if not isinstance(surfaces, list) or len(surfaces) != 1:
            return None
        surface = surfaces[0]
        if not isinstance(surface, Mapping):
            return None
        return SurfaceResolution(
            benchmark=benchmark,
            site=self.site,
            canonical_surface_id=target_surface_id,
            profile_surface_id=str(surface.get("id") or ""),
            profile_surface=surface,
            evidence="fake_profile_alias",
        )

    def route_contract_facts(
        self,
        *,
        benchmark: str,
        profile: Mapping[str, object],
        kind: str,
    ) -> SiteRouteContractFacts:
        del profile
        if benchmark != "webarena_verified" or kind != "message":
            return SiteRouteContractFacts()
        return SiteRouteContractFacts(
            allowed_start_url_patterns=("__FAKE__/{message_id}",),
            anchor_examples=({"message_id": "7", "start_url": "__FAKE__/messages/7"},),
            requires_inventory_backed_start_url=True,
            route_variant="detail",
        )


def test_injected_fake_profile_route_capability_is_bound_without_phase_policy() -> None:
    adapter = FakeProfileRouteSite()
    assert isinstance(adapter, SiteProfileRouteCapability)
    bound = SiteCatalog([adapter]).bind(
        site="fake",
        profile=_profile("fake", "body"),
    )
    facts = bound.route_contract_facts("message")
    assert facts.allowed_start_url_patterns == ("__FAKE__/{message_id}",)
    assert facts.anchor_examples[0]["message_id"] == "7"
    assert facts.requires_inventory_backed_start_url is True
    assert facts.route_variant == "detail"


def test_phase1_route_builder_fails_closed_for_mismatched_profile_site() -> None:
    contracts = build_task_route_contracts(
        site_name="gitlab",
        profile={
            "site_name": "reddit",
            "injection_surface": [{"id": "gitlab_issue_description"}],
            "data_model": [],
        },
    )

    assert contracts["route_families"] == []


@pytest.mark.parametrize(
    "profile",
    [
        None,
        [],
        "invalid",
        {"site_name": None},
        {"site_name": ""},
        {"site_name": "gitlab", "injection_surface": None},
        {"site_name": "gitlab", "data_model": None},
        {"site_name": "gitlab", "existing_task_coverage": []},
    ],
)
def test_phase1_route_builder_fails_closed_for_malformed_profile(profile: object) -> None:
    contracts = build_task_route_contracts(
        site_name="gitlab",
        profile=profile,  # type: ignore[arg-type]
    )

    assert contracts["route_families"] == []


@pytest.mark.parametrize(
    ("site", "kind", "placeholder"),
    [
        ("gitlab", "gitlab_dashboard_list", "__GITLAB__"),
        ("reddit", "reddit_dashboard_list", "__REDDIT__"),
    ],
)
def test_route_contract_patterns_derive_from_canonical_routes(
    site: str,
    kind: str,
    placeholder: str,
) -> None:
    bound = SiteCatalog().bind(site=site, profile={"site_name": site})
    route = next(candidate for candidate in bound.routes() if candidate.compatibility_kind == kind)

    assert bound.route_contract_facts(kind).allowed_start_url_patterns == tuple(
        f"{placeholder}{pattern}" for pattern in route.allowed_start_url_patterns
    )


def test_profile_identity_aliases_must_agree() -> None:
    with pytest.raises(SiteTargetingDefinitionError, match="site fields disagree"):
        SiteCatalog().bind(
            site="gitlab",
            profile={"site_name": "gitlab", "site": "reddit"},
        )


def test_surface_mapping_is_benchmark_specific() -> None:
    assert has_surface_mapping(benchmark="webarena_verified", site="gitlab")
    assert not has_surface_mapping(benchmark="unknown", site="gitlab")
