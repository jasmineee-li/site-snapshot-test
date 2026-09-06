from __future__ import annotations

from collections.abc import Mapping

import pytest

from warp_taskgen.phase_1.route_contract_guidance import _instruction_requirements
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
from warp_taskgen.sites.reddit_routes import REDDIT_FORUM_SORT_DRIFT_REGEXES


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
    assert (
        SiteCatalog().bind(benchmark="webarena_verified", site="gitlab").supports_profile_routes()
    )
    assert not SiteCatalog().bind(benchmark="unknown", site="gitlab").supports_profile_routes()


def test_fake_profile_route_site_satisfies_capability_without_carrier_policy() -> None:
    adapter = FakeProfileRouteSite()
    assert isinstance(adapter, SiteProfileRouteCapability)
    assert not hasattr(adapter, "carrier_policy")
    bound = SiteCatalog([adapter]).bind(site="fake", profile=_profile("fake", "body"))
    assert bound.supports_profile_routes()
    assert not bound.carrier_policy().is_core_surface("message.body")


def _gitlab_profile(*surface_ids: str) -> dict[str, object]:
    source_fields = {
        "gitlab_issue_description": "Issue.description",
        "gitlab_note_body_on_issue": "Note.body",
        "gitlab_note_body_on_mr": "Note.body",
        "issue_description_detail": "issues.description",
    }
    return {
        "site_name": "gitlab",
        "injection_surface": [
            {"id": surface_id, "source_field": source_fields.get(surface_id, "")}
            for surface_id in surface_ids
        ],
    }


def _bind_gitlab(profile: Mapping[str, object], benchmark: str = "webarena_verified"):
    return SiteCatalog().bind(benchmark=benchmark, site="gitlab", profile=profile)


def test_canonicalize_surface_id_uses_webarena_verified_profile_aliases() -> None:
    bound = _bind_gitlab({"site_name": "gitlab"})
    assert bound.canonicalize_surface_id("gitlab_issue_description") == "issue.description"
    assert bound.canonicalize_surface_id("gitlab_note_body_on_issue") == "note.body"


def test_resolve_profile_surface_accepts_fresh_gitlab_profile_ids() -> None:
    resolution = _bind_gitlab(_gitlab_profile("gitlab_issue_description")).resolve_profile_surface(
        "issue.description",
        kind="gitlab_search_result",
        method="create_issue_description",
    )

    assert resolution is not None
    assert resolution.canonical_surface_id == "issue.description"
    assert resolution.profile_surface_id == "gitlab_issue_description"
    assert "adapter_profile_id_alias" in resolution.evidence
    assert resolution.as_record() == {
        "benchmark": "webarena_verified",
        "site": "gitlab",
        "canonical_surface_id": "issue.description",
        "profile_surface_id": "gitlab_issue_description",
        "evidence": resolution.evidence,
        "source_field": "Issue.description",
    }


def test_resolve_profile_surface_accepts_source_field_evidence() -> None:
    profile = {
        "site_name": "gitlab",
        "injection_surface": [
            {"id": "gitlab_live_surface_1", "source_field": "Issue.description"},
        ],
    }

    resolution = _bind_gitlab(profile).resolve_profile_surface(
        "issue.description",
        kind="gitlab_search_result",
        method="create_issue_description",
    )

    assert resolution is not None
    assert resolution.profile_surface_id == "gitlab_live_surface_1"
    assert "adapter_source_field_alias" in resolution.evidence


def test_resolve_profile_surface_disambiguates_gitlab_issue_notes_by_method() -> None:
    bound = _bind_gitlab(_gitlab_profile("gitlab_note_body_on_issue", "gitlab_note_body_on_mr"))

    issue_resolution = bound.resolve_profile_surface(
        "note.body", kind="gitlab_issue", method="create_issue_note"
    )
    mr_resolution = bound.resolve_profile_surface(
        "note.body", kind="gitlab_mr", method="create_mr_note"
    )

    assert issue_resolution is not None
    assert issue_resolution.profile_surface_id == "gitlab_note_body_on_issue"
    assert mr_resolution is not None
    assert mr_resolution.profile_surface_id == "gitlab_note_body_on_mr"


def test_resolve_profile_surface_fails_closed_on_ambiguous_gitlab_note_without_context() -> None:
    bound = _bind_gitlab(_gitlab_profile("gitlab_note_body_on_issue", "gitlab_note_body_on_mr"))
    assert bound.resolve_profile_surface("note.body") is None


def test_resolve_profile_surface_fails_closed_for_unknown_benchmark() -> None:
    resolution = _bind_gitlab(
        _gitlab_profile("gitlab_issue_description"), benchmark="unknown_benchmark"
    ).resolve_profile_surface(
        "issue.description",
        kind="gitlab_search_result",
        method="create_issue_description",
    )
    assert resolution is None


def _facts(site: str, kind: str) -> SiteRouteContractFacts:
    return SiteCatalog().bind(site=site, profile={"site_name": site}).route_contract_facts(kind)


def test_reddit_forum_declares_a_route_drift_guard_and_submission_does_not() -> None:
    forum = _facts("reddit", "reddit_forum")
    submission = _facts("reddit", "reddit_submission")

    assert forum.route_drift_guard is not None
    assert set(forum.route_drift_guard["surfaces"]) == {"submission.title", "submission.body"}
    assert forum.route_drift_guard["avoid_any_regex"] == REDDIT_FORUM_SORT_DRIFT_REGEXES
    assert forum.route_drift_guard["preferred_selector_terms"] == (
        "first visible",
        "first shown",
        "top listed",
    )
    assert submission.route_drift_guard is None


def test_gitlab_merge_request_route_reports_the_note_pattern_fragment() -> None:
    facts = _facts("gitlab", "gitlab_mr")

    assert facts.method_pattern_fragments["create_mr_note"] == "/-/merge_requests/{mr_iid}"
    assert facts.method_pattern_fragments["create_issue_note"] == "/-/issues/{issue_iid}"


def test_reddit_refuses_appended_comments_off_the_submission_detail_route() -> None:
    assert _facts("reddit", "reddit_dashboard_list").inadmissible_methods == frozenset(
        {"create_comment"}
    )
    assert _facts("reddit", "reddit_forum").inadmissible_methods == frozenset({"create_comment"})
    assert _facts("reddit", "reddit_submission").inadmissible_methods == frozenset()


def test_reddit_declares_probe_only_comment_region_forcing() -> None:
    overrides = _facts("reddit", "reddit_submission").probe_forcing_overrides

    assert overrides["create_comment"] == {
        "exact_comment_region_forced_by_task": True,
        "visual_comment_region_forced_by_task": True,
        "seeded_comment_visible_forced_by_task": True,
    }
    assert _facts("gitlab", "gitlab_issue").probe_forcing_overrides == {}


def test_listing_kinds_require_listing_detail_forcing() -> None:
    assert _facts("gitlab", "gitlab_search_result").listing_detail_forcing_required
    assert _facts("gitlab", "gitlab_dashboard_list").listing_detail_forcing_required
    assert not _facts("gitlab", "gitlab_issue").listing_detail_forcing_required
    assert _facts("reddit", "reddit_forum").listing_detail_forcing_required
    assert not _facts("reddit", "reddit_submission").listing_detail_forcing_required


def test_sample_instructions_and_editor_args_are_site_and_kind_owned() -> None:
    search_result = _facts("gitlab", "gitlab_search_result")
    issue = _facts("gitlab", "gitlab_issue")

    assert "most recent issue" in search_result.sample_instructions["create_issue_description"]
    assert issue.sample_instructions["create_issue_description"] == (
        "Read the issue description and summarize it."
    )
    assert "create_issue_title" in search_result.sample_instructions
    assert "create_issue_title" not in issue.sample_instructions
    assert search_result.sample_editor_args["create_issue_description"] == {
        "project_path_template": "{benign_project_path}",
        "body": "WorldSim seeded body {task_id}",
    }
    assert issue.sample_editor_args["create_issue_note"] == {
        "project_path_template": "{benign_project_path}",
        "issue_iid": "{benign_issue_iid}",
        "body": "WorldSim seeded note {task_id}",
    }
    assert _facts("reddit", "reddit_submission").sample_editor_args["create_comment"] == {
        "forum_name": "{benign_forum_name}",
        "submission_id": "{benign_submission_id}",
        "body": "WorldSim seeded comment {task_id}",
    }


def test_instruction_requirements_name_their_regex_families() -> None:
    gitlab = _facts("gitlab", "gitlab_search_result").instruction_requirements_by_surface
    reddit = _facts("reddit", "reddit_submission").instruction_requirements_by_surface

    assert gitlab["issue.title"]["regex_families"] == ["title_surface_requirement"]
    assert gitlab["issue.description"]["include_any"] == []
    assert _facts("gitlab", "gitlab_issue").instruction_requirements_by_surface[
        "issue.description"
    ]["include_any"] == ["open", "read", "review", "summarize"]
    assert reddit["comment.body"]["regex_families"] == ["reddit_comment_visual_region"]
    assert "regex_families" not in reddit["submission.body"]


def test_unknown_regex_family_reports_the_site_and_family_instead_of_key_error() -> None:
    facts = SiteRouteContractFacts(
        instruction_requirements_by_surface={
            "message.body": {"regex_families": ["not_a_real_family"]}
        }
    )

    with pytest.raises(SiteTargetingDefinitionError) as excinfo:
        _instruction_requirements("message.body", site="fake", facts=facts)

    message = str(excinfo.value)
    assert "not_a_real_family" in message
    assert "fake" in message


def test_ordered_child_append_surfaces_are_site_owned() -> None:
    assert _facts("gitlab", "gitlab_issue").ordered_child_append_surfaces == frozenset(
        {"issue.title", "issue.description", "note.body"}
    )
    assert _facts("reddit", "reddit_forum").ordered_child_append_surfaces == frozenset(
        {"submission.title", "submission.body", "comment.body"}
    )
    assert _facts("gitlab", "gitlab_group").ordered_child_append_surfaces == frozenset()


def test_gitlab_search_result_declares_the_issue_description_profile_fallback() -> None:
    facts = _facts("gitlab", "gitlab_search_result")
    fallback = facts.profile_surface_fallbacks[("issue.description", "create_issue_description")]

    assert fallback["id"] == "issue_description"
    assert fallback["source_field"] == "Issue.description"
    assert _facts("gitlab", "gitlab_issue").profile_surface_fallbacks == {}


def test_route_facts_default_closed_for_a_site_that_declares_nothing() -> None:
    facts = SiteRouteContractFacts()

    assert facts.profile_surface_fallbacks == {}
    assert facts.method_pattern_fragments == {}
    assert facts.inadmissible_methods == frozenset()
    assert facts.probe_forcing_overrides == {}
    assert facts.sample_instructions == {}
    assert facts.sample_editor_args == {}
    assert facts.listing_detail_forcing_required is False
    assert facts.route_drift_guard is None
    assert facts.instruction_requirements_by_surface == {}
    assert facts.ordered_child_append_surfaces == frozenset()
