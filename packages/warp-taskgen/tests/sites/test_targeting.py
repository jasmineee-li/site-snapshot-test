from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from worldsim.sites import (
    CanonicalRoute,
    SiteCatalog,
    SiteTargetingDefinitionError,
    SourceListing,
    TargetCandidate,
    TargetingContext,
    TargetingFailure,
)
from worldsim.sites.gitlab import GitLabSite
from worldsim.sites.reddit import RedditSite

PLACEHOLDERS = {
    "__GITLAB__": "https://gitlab.local",
    "__REDDIT__": "https://reddit.local",
}


def _task(site: str, expected_url: str, *, start_url: str | None = None) -> dict[str, Any]:
    return {
        "sites": [site],
        "start_urls": [start_url or f"__{site.upper()}__"],
        "reward_function": {
            "eval": [
                {
                    "evaluator": "NetworkEventEvaluator",
                    "expected": {"url": expected_url},
                }
            ]
        },
    }


def test_default_catalog_exposes_routes_without_a_deployment_origin():
    catalog = SiteCatalog()

    assert catalog.sites == ("gitlab", "reddit")
    assert catalog.bind(site="gitlab").routes()
    assert catalog.bind(site="reddit").routes()


def test_gitlab_issue_resolves_to_a_local_kind_and_canonical_route():
    target = (
        SiteCatalog()
        .bind(site="gitlab", placeholders=PLACEHOLDERS)
        .resolve(_task("gitlab", "__GITLAB__/namespace/project/-/issues/7"))
    )

    assert target.kind == "issue"
    assert target.anchors == {"project_path": "namespace/project", "issue_iid": "7"}
    assert target.start_url_resolved == "https://gitlab.local/namespace/project/-/issues/7"
    assert target.canonical_route is not None
    assert target.canonical_route.id == "gitlab.issue"


def test_reddit_submission_resolves_to_a_local_kind_and_canonical_route():
    target = (
        SiteCatalog()
        .bind(site="reddit", placeholders=PLACEHOLDERS)
        .resolve(_task("reddit", "__REDDIT__/f/books/12/-/comment"))
    )

    assert target.kind == "submission"
    assert target.anchors == {"forum_name": "books", "submission_id": "12"}
    assert target.start_url_resolved == "https://reddit.local/f/books/12"
    assert target.canonical_route is not None
    assert target.canonical_route.id == "reddit.submission"


def test_resolution_requires_an_explicit_origin_or_placeholder():
    failure = (
        SiteCatalog()
        .bind(site="gitlab")
        .resolve(_task("gitlab", "__GITLAB__/namespace/project/-/issues/7"))
    )

    assert isinstance(failure, TargetingFailure)
    assert failure.reason == "missing_origin"


def test_resolution_rejects_an_absolute_url_from_another_origin():
    failure = (
        SiteCatalog()
        .bind(site="gitlab", placeholders=PLACEHOLDERS)
        .resolve(_task("gitlab", "https://attacker.invalid/namespace/project/-/issues/7"))
    )

    assert isinstance(failure, TargetingFailure)
    assert failure.reason == "unresolved_evidence"


def test_binding_rejects_profile_and_origin_disagreement():
    try:
        SiteCatalog().bind(site="gitlab", profile={"site_name": "reddit"})
    except SiteTargetingDefinitionError as exc:
        assert "profile site" in str(exc)
    else:
        raise AssertionError("mismatched profile site must fail closed")

    failure = (
        SiteCatalog()
        .bind(
            site="gitlab",
            origin="https://another-gitlab.local",
            placeholders=PLACEHOLDERS,
        )
        .resolve(_task("gitlab", "__GITLAB__/namespace/project/-/issues/7"))
    )
    assert isinstance(failure, TargetingFailure)
    assert failure.reason == "missing_origin"


def test_malformed_nested_task_metadata_returns_a_failure():
    malformed_reward = _task("gitlab", "__GITLAB__/namespace/project/-/issues/7")
    malformed_reward["reward_function"] = "not-a-mapping"
    failure = SiteCatalog().bind(site="gitlab", placeholders=PLACEHOLDERS).resolve(
        malformed_reward
    )
    assert isinstance(failure, TargetingFailure)

    malformed_eval = _task("gitlab", "__GITLAB__/namespace/project/-/issues/7")
    malformed_eval["reward_function"]["eval"] = {
        "expected": {"url": "__GITLAB__/namespace/project/-/issues/7"}
    }
    failure = SiteCatalog().bind(site="gitlab", placeholders=PLACEHOLDERS).resolve(
        malformed_eval
    )
    assert isinstance(failure, TargetingFailure)
    assert failure.reason == "malformed_metadata"

    malformed_start_urls = _task("gitlab", "__GITLAB__/namespace/project/-/issues/7")
    malformed_start_urls["start_urls"] = [
        {"url": "__GITLAB__/namespace/project/-/issues/7"}
    ]
    failure = SiteCatalog().bind(site="gitlab", placeholders=PLACEHOLDERS).resolve(
        malformed_start_urls
    )
    assert isinstance(failure, TargetingFailure)
    assert failure.reason == "malformed_metadata"

    malformed_delivery = _task("gitlab", "__GITLAB__/namespace/project/-/issues/7")
    malformed_delivery["delivery_channel"] = ["gitlab"]
    failure = SiteCatalog().bind(site="gitlab", placeholders=PLACEHOLDERS).resolve(
        malformed_delivery
    )
    assert isinstance(failure, TargetingFailure)
    assert failure.reason == "malformed_metadata"

    malformed_context = _task("gitlab", "__GITLAB__/overlap")
    malformed_context["agent_context"] = "not-a-mapping"
    failure = SiteCatalog().bind(site="gitlab", placeholders=PLACEHOLDERS).resolve(
        malformed_context
    )
    assert isinstance(failure, TargetingFailure)


def test_unknown_benchmark_and_kind_fail_closed():
    unsupported_benchmark = (
        SiteCatalog()
        .bind(benchmark="comparison_only", site="gitlab", placeholders=PLACEHOLDERS)
        .resolve(_task("gitlab", "__GITLAB__/namespace/project/-/issues/7"))
    )
    assert isinstance(unsupported_benchmark, TargetingFailure)
    assert unsupported_benchmark.reason == "unsupported_benchmark"

    class UnknownKindSite(GitLabSite):
        def match(self, url, task, context):  # type: ignore[no-untyped-def]
            return "not_a_route", {}

    failure = (
        SiteCatalog([UnknownKindSite()])
        .bind(site="gitlab", placeholders=PLACEHOLDERS)
        .resolve(_task("gitlab", "__GITLAB__/namespace/project/-/issues/7"))
    )
    assert isinstance(failure, TargetingFailure)
    assert failure.reason == "unknown_route"


def test_benchmark_aliases_normalize_before_profile_and_adapter_validation():
    target = (
        SiteCatalog()
        .bind(
            benchmark="WebArena Verified",
            site="gitlab",
            profile={"benchmark_name": "webarena-verified", "site_name": "gitlab"},
            placeholders=PLACEHOLDERS,
        )
        .resolve(_task("gitlab", "__GITLAB__/namespace/project/-/issues/7"))
    )

    assert not isinstance(target, TargetingFailure)
    assert target.kind == "issue"


def test_bound_site_rejects_layers_outside_deterministic_targeting():
    failure = SiteCatalog().bind(site="gitlab", placeholders=PLACEHOLDERS).resolve(
        _task("gitlab", "__GITLAB__/namespace/project/-/issues/7"),
        allow_layers=("L1", "L3"),
    )

    assert isinstance(failure, TargetingFailure)
    assert failure.reason == "unsupported_resolution_layer"


def test_ambiguous_root_profile_does_not_guess_user_or_group():
    task = _task("gitlab", "__GITLAB__/overlap")
    task["agent_context"] = {"gitlab": {"user_handles": ["overlap"], "group_handles": ["overlap"]}}

    failure = SiteCatalog().bind(site="gitlab", placeholders=PLACEHOLDERS).resolve(task)

    assert isinstance(failure, TargetingFailure)
    assert failure.reason == "unresolved_evidence"


def test_multi_site_metadata_fails_closed_instead_of_using_the_first_site():
    task = _task("gitlab", "__GITLAB__/namespace/project/-/issues/7")
    task.pop("site", None)
    task["sites"] = ["gitlab", "reddit"]

    failure = SiteCatalog().bind(site="gitlab", placeholders=PLACEHOLDERS).resolve(task)

    assert isinstance(failure, TargetingFailure)
    assert failure.reason == "ambiguous_site_metadata"


def test_task_site_and_sites_conflict_fails_closed():
    task = _task("gitlab", "__GITLAB__/namespace/project/-/issues/7")
    task["site"] = "gitlab"
    task["sites"] = ["reddit"]

    failure = SiteCatalog().bind(site="gitlab", placeholders=PLACEHOLDERS).resolve(task)

    assert isinstance(failure, TargetingFailure)
    assert failure.reason == "conflicting_site_metadata"


def test_delivery_site_explains_secondary_site_without_overriding_task_site():
    task = _task("gitlab", "__GITLAB__/namespace/project/-/issues/7")
    task["site"] = "gitlab"
    task["sites"] = ["gitlab", "reddit"]
    task["delivery_channel"] = {"delivery_site": "reddit"}

    target = SiteCatalog().bind(site="gitlab", placeholders=PLACEHOLDERS).resolve(task)

    assert not isinstance(target, TargetingFailure)
    assert target.site == "gitlab"
    assert target.kind == "issue"


def test_delivery_site_without_a_page_site_does_not_select_a_task_site():
    task = _task("gitlab", "__GITLAB__/namespace/project/-/issues/7")
    task.pop("site", None)
    task["sites"] = ["gitlab", "reddit"]
    task["delivery_channel"] = {"delivery_site": "reddit"}

    failure = SiteCatalog().bind(site="gitlab", placeholders=PLACEHOLDERS).resolve(task)

    assert isinstance(failure, TargetingFailure)
    assert failure.reason == "ambiguous_site_metadata"


@pytest.mark.parametrize("site", ["gitlab", "reddit"])
def test_every_declared_route_example_round_trips_through_its_site(site: str):
    adapters = {"gitlab": GitLabSite(), "reddit": RedditSite()}
    adapter = adapters[site]
    context = TargetingContext(
        benchmark="webarena_verified", site=site, origin=f"https://{site}.local"
    )
    bound = SiteCatalog().bind(site=site, origin=f"https://{site}.local")

    routes = bound.routes()
    assert routes
    for route in routes:
        assert len(route.anchor_examples) >= len(route.allowed_start_url_patterns)
        for example in route.anchor_examples:
            reconstructed = adapter.reconstruct(route.kind, example, context)
            assert reconstructed, f"{route.id} example did not reconstruct: {example!r}"
            task = _task(site, reconstructed, start_url=reconstructed)
            if route.id == "gitlab.user_profile":
                task["agent_context"] = {"gitlab": {"user_handles": [example["username"]]}}
            elif route.id == "gitlab.group":
                task["agent_context"] = {"gitlab": {"group_handles": [example["group_path"]]}}

            hit = adapter.match(reconstructed, task, context)
            assert hit is not None, f"{route.id} example did not match: {reconstructed}"
            assert hit[0] == route.kind
            target = bound.resolve(task)
            assert not isinstance(target, TargetingFailure), target
            assert target.canonical_route == route
            assert target.kind == route.kind


class FakeSite:
    site = "fake"
    supported_benchmarks = frozenset({"webarena_verified"})

    def validate(self) -> None:
        return None

    def validate_task(self, task: Mapping[str, Any]) -> tuple[str, str] | None:
        del task
        return None

    def routes(self, context: TargetingContext):
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
        self, url: str, task: Mapping[str, Any], context: TargetingContext
    ) -> tuple[str, dict[str, Any]] | None:
        del task, context
        if url.endswith("/messages/1"):
            return "message", {"message_id": "1"}
        return None

    def reconstruct(
        self,
        kind: str,
        anchors: Mapping[str, Any],
        context: TargetingContext,
    ) -> str | None:
        if kind != "message" or not anchors.get("message_id"):
            return None
        origin = context.site_origin()
        return f"{origin}/messages/{anchors['message_id']}" if origin else None

    def is_listing(self, kind: str) -> bool:
        return False

    def listing_start_url(
        self, kind: str, resolved_url: str, fallback_url: str | None
    ) -> str | None:
        del kind, resolved_url
        return fallback_url


def test_injected_fake_site_proves_catalog_seam_without_production_registration():
    catalog = SiteCatalog([GitLabSite(), RedditSite(), FakeSite()])
    target = catalog.bind(site="fake", origin="https://fake.local").resolve(
        _task("fake", "https://fake.local/messages/1", start_url="https://fake.local")
    )

    assert target.kind == "message"
    assert target.start_url_resolved == "https://fake.local/messages/1"
    assert SiteCatalog().sites == ("gitlab", "reddit")


def test_gitlab_issue_listing_is_classified_by_the_site_grammar():
    target = SiteCatalog().bind(site="gitlab", placeholders=PLACEHOLDERS).resolve(
        _task(
            "gitlab",
            "__GITLAB__/namespace/project/-/issues",
            start_url="__GITLAB__/namespace/project/-/issues",
        )
    )

    assert target.kind == "search_result"
    assert target.anchors == {"project_path": "namespace/project"}
    assert target.start_url_resolved == "https://gitlab.local/namespace/project/-/issues"


def test_legacy_matchers_preserve_prefixed_resource_kinds():
    from worldsim.phase_2.target_resolution.url_matching import _match_gitlab, _match_reddit

    assert _match_gitlab("https://gitlab.local/a/b/-/issues/7") == (
        "gitlab_issue",
        {"project_path": "a/b", "issue_iid": "7"},
    )
    assert _match_reddit("https://reddit.local/f/books/12/-/comment") == (
        "reddit_submission",
        {"forum_name": "books", "submission_id": "12"},
    )


def test_l3_candidate_accepts_legacy_kind_and_preserves_local_route():
    bound = SiteCatalog().bind(site="gitlab", placeholders=PLACEHOLDERS)
    target = bound.materialize(
        TargetCandidate(
            kind="gitlab_issue",
            anchors={"project_path": "namespace/project", "issue_iid": "7"},
            probe_query={"api": "search_user_issues"},
        )
    )

    assert target.kind == "issue"
    assert target.canonical_route is not None
    assert target.canonical_route.compatibility_kind == "gitlab_issue"
    assert target.start_url_resolved == "https://gitlab.local/namespace/project/-/issues/7"


def test_l3_candidate_fails_closed_instead_of_using_fallback_url():
    bound = SiteCatalog().bind(site="gitlab", placeholders=PLACEHOLDERS)
    failure = bound.materialize(
        TargetCandidate(
            kind="gitlab_issue",
            anchors={"project_path": "namespace/project"},
            probe_query={"api": "search_user_issues"},
            fallback_url="https://gitlab.local/namespace/project",
        )
    )

    assert isinstance(failure, TargetingFailure)
    assert failure.reason == "missing_anchor"
    assert failure.as_record()["start_url_resolved"] is None


def test_l3_candidate_mappings_are_immutable_snapshots():
    anchors = {"project_path": "namespace/project", "issue_iid": "7"}
    probe_query = {"api": "search_user_issues"}
    candidate = TargetCandidate(
        kind="gitlab_issue",
        anchors=anchors,
        probe_query=probe_query,
    )

    anchors["issue_iid"] = "99"
    probe_query["query"] = "changed"
    assert candidate.anchors["issue_iid"] == "7"
    assert "query" not in candidate.probe_query
    with pytest.raises(TypeError):
        candidate.anchors["issue_iid"] = "8"  # type: ignore[index]
    with pytest.raises(TypeError):
        candidate.probe_query["api"] = "changed"  # type: ignore[index]


def test_l3_candidate_rejects_foreign_evidence_url():
    bound = SiteCatalog().bind(site="gitlab", placeholders=PLACEHOLDERS)
    failure = bound.materialize(
        TargetCandidate(
            kind="gitlab_issue",
            anchors={"project_path": "namespace/project", "issue_iid": "7"},
            probe_query={"api": "search_user_issues"},
            evidence_url="https://attacker.invalid/issue/7",
        )
    )

    assert isinstance(failure, TargetingFailure)
    assert failure.reason == "foreign_origin"
    assert failure.evidence_url is None


def test_l3_candidate_requires_absolute_reconstructed_url():
    class RelativeGitLab(GitLabSite):
        def reconstruct(self, kind, anchors, context):  # type: ignore[no-untyped-def]
            del kind, anchors, context
            return "/relative/issues/7"

    bound = SiteCatalog([RelativeGitLab(), RedditSite()]).bind(
        site="gitlab", placeholders=PLACEHOLDERS
    )
    failure = bound.materialize(
        TargetCandidate(
            kind="gitlab_issue",
            anchors={"project_path": "namespace/project", "issue_iid": "7"},
            probe_query={"api": "search_user_issues"},
        )
    )

    assert isinstance(failure, TargetingFailure)
    assert failure.reason == "invalid_target_url"


def test_l3_adapter_hook_exceptions_return_structured_failures():
    class RaisingValidationGitLab(GitLabSite):
        def validate_candidate(self, kind, probe_query, anchors, context):  # type: ignore[no-untyped-def]
            raise RuntimeError("validation boom")

    validation_bound = SiteCatalog([RaisingValidationGitLab(), RedditSite()]).bind(
        site="gitlab", placeholders=PLACEHOLDERS
    )
    validation_failure = validation_bound.materialize(
        TargetCandidate(
            kind="gitlab_issue",
            anchors={"project_path": "namespace/project", "issue_iid": "7"},
            probe_query={"api": "search_user_issues"},
        )
    )
    assert isinstance(validation_failure, TargetingFailure)
    assert validation_failure.reason == "adapter_error"

    class RaisingReconstructionGitLab(GitLabSite):
        def reconstruct(self, kind, anchors, context):  # type: ignore[no-untyped-def]
            raise RuntimeError("reconstruction boom")

    reconstruction_bound = SiteCatalog([RaisingReconstructionGitLab(), RedditSite()]).bind(
        site="gitlab", placeholders=PLACEHOLDERS
    )
    reconstruction_failure = reconstruction_bound.materialize(
        TargetCandidate(
            kind="gitlab_issue",
            anchors={"project_path": "namespace/project", "issue_iid": "7"},
            probe_query={"api": "search_user_issues"},
        )
    )
    assert isinstance(reconstruction_failure, TargetingFailure)
    assert reconstruction_failure.reason == "adapter_error"

    class RaisingSourceListingGitLab(GitLabSite):
        def source_listing(self, kind, probe_query, anchors, context):  # type: ignore[no-untyped-def]
            raise RuntimeError("listing boom")

    listing_bound = SiteCatalog([RaisingSourceListingGitLab(), RedditSite()]).bind(
        site="gitlab", placeholders=PLACEHOLDERS
    )
    listing_failure = listing_bound.source_listing(
        TargetCandidate(
            kind="gitlab_issue",
            anchors={"project_path": "namespace/project", "issue_iid": "7"},
            probe_query={"api": "search_user_issues"},
        )
    )
    assert isinstance(listing_failure, TargetingFailure)
    assert listing_failure.reason == "adapter_error"

    class RelativeSourceGitLab(GitLabSite):
        def source_listing(self, kind, probe_query, anchors, context):  # type: ignore[no-untyped-def]
            del kind, probe_query, anchors, context
            return "search_result", "/relative/issues"

    relative_bound = SiteCatalog([RelativeSourceGitLab(), RedditSite()]).bind(
        site="gitlab", placeholders=PLACEHOLDERS
    )
    relative_failure = relative_bound.source_listing(
        TargetCandidate(
            kind="gitlab_issue",
            anchors={"project_path": "namespace/project", "issue_iid": "7"},
            probe_query={"api": "search_user_issues"},
        )
    )
    assert isinstance(relative_failure, TargetingFailure)
    assert relative_failure.reason == "invalid_source_listing"


def test_l3_candidate_api_kind_mismatch_is_site_owned():
    bound = SiteCatalog().bind(site="gitlab", placeholders=PLACEHOLDERS)
    failure = bound.materialize(
        TargetCandidate(
            kind="gitlab_dashboard_list",
            anchors={"dashboard": "issues"},
            probe_query={"api": "list_project_issues_recent"},
        )
    )

    assert isinstance(failure, TargetingFailure)
    assert failure.reason == "probe_kind_mismatch"


def test_l3_candidate_exposes_site_owned_source_listing_facts():
    bound = SiteCatalog().bind(site="gitlab", placeholders=PLACEHOLDERS)
    listing = bound.source_listing(
        TargetCandidate(
            kind="gitlab_issue",
            anchors={"project_path": "namespace/project", "issue_iid": "7"},
            probe_query={"api": "search_user_issues", "project_path": "namespace/project"},
        )
    )

    assert isinstance(listing, SourceListing)
    assert listing.kind == "gitlab_search_result"
    assert listing.start_url == "https://gitlab.local/namespace/project/-/issues"


def test_injected_fake_site_proves_candidate_seam_without_editor_contract():
    class FakeCandidateSite(FakeSite):
        pass

    catalog = SiteCatalog([GitLabSite(), RedditSite(), FakeCandidateSite()])
    target = catalog.bind(site="fake", origin="https://fake.local").materialize(
        TargetCandidate(kind="message", anchors={"message_id": "9"})
    )

    assert target.kind == "message"
    assert target.start_url_resolved == "https://fake.local/messages/9"
