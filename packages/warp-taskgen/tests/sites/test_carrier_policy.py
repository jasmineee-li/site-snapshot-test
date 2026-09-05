from __future__ import annotations

import pytest

from tests.sites.synthetic_discussion_forum.site import SyntheticDiscussionForumSite
from warp_taskgen.sites import (
    GitLabSite,
    RedditSite,
    SiteCarrierPolicy,
    SiteCarrierPolicyCapability,
    SiteCatalog,
    SiteTargetingDefinitionError,
    default_catalog,
    gitlab_profile,
    reddit_profile,
)

BENCHMARK = "webarena_verified"


def _policy(site: str) -> SiteCarrierPolicy:
    adapter = {"gitlab": GitLabSite(), "reddit": RedditSite()}[site]
    policy = adapter.carrier_policy(benchmark=BENCHMARK)
    assert policy is not None
    return policy


def test_active_sites_own_a_carrier_policy_for_webarena_verified() -> None:
    for adapter in (GitLabSite(), RedditSite()):
        assert isinstance(adapter, SiteCarrierPolicyCapability)
        assert adapter.carrier_policy(benchmark=BENCHMARK) is not None
        assert adapter.carrier_policy(benchmark="unknown_benchmark") is None


def test_core_surfaces_list_only_user_controllable() -> None:
    """Path A core surfaces must stay inside non-admin user-writable UGC."""
    user_controllable_non_admin = {
        "gitlab": {
            "issue.title",
            "issue.description",
            "mr.title",
            "mr.description",
            "note.body",
        },
        "reddit": {
            "submission.title",
            "submission.body",
            "comment.body",
        },
    }

    assert set(default_catalog().sites) == set(user_controllable_non_admin)
    for site, surfaces in user_controllable_non_admin.items():
        assert _policy(site).core_surfaces <= surfaces


def test_legacy_editor_surface_ids_canonicalize_to_core_surfaces() -> None:
    gitlab = _policy("gitlab")
    reddit = _policy("reddit")
    assert gitlab.canonical_surface("issue_list_title") == "issue.title"
    assert gitlab.canonical_surface("issue_title_in_list") == "issue.title"
    assert gitlab.canonical_surface("issue_description_detail") == "issue.description"
    assert gitlab.canonical_surface("issue_detail_description") == "issue.description"
    assert gitlab.canonical_surface("mr_list_title") == "mr.title"
    assert gitlab.canonical_surface("mr_title_in_list") == "mr.title"
    assert gitlab.canonical_surface("mr_description_detail") == "mr.description"
    assert gitlab.canonical_surface("mr_detail_description") == "mr.description"
    assert gitlab.canonical_surface("note_on_issue") == "note.body"
    assert gitlab.canonical_surface("note_body_on_issue") == "note.body"
    assert gitlab.canonical_surface("note_on_mr") == "note.body"
    assert gitlab.canonical_surface("note_body_on_mr") == "note.body"
    assert reddit.canonical_surface("submission_title") == "submission.title"
    assert reddit.canonical_surface("submission_title_listing") == "submission.title"
    assert reddit.canonical_surface("submission_title_detail") == "submission.title"
    assert reddit.canonical_surface("submission_body") == "submission.body"
    assert reddit.canonical_surface("submission_body_detail") == "submission.body"
    assert reddit.canonical_surface("comment_body") == "comment.body"
    assert reddit.canonical_surface("comment_body_thread") == "comment.body"
    assert reddit.canonical_surface("comment_body_detail") == "comment.body"


def test_non_core_and_unknown_sites_fail_closed() -> None:
    assert not _policy("gitlab").is_core_surface("label.description")
    assert not _policy("gitlab").is_core_surface("milestone.description")
    assert not _policy("reddit").is_core_surface("comment_form_placeholder")
    assert not _policy("gitlab").is_core_surface(None)
    with pytest.raises(SiteTargetingDefinitionError):
        SiteCatalog().bind(site="shopping")
    closed = SiteCarrierPolicy.closed(BENCHMARK)
    assert not closed.is_core_surface("review.body")
    assert closed.canonical_surface("review.body") == "review.body"
    assert closed.canonical_surface("") is None


def test_title_surfaces_are_core_metadata_but_retired_as_active_carriers() -> None:
    gitlab = _policy("gitlab")
    reddit = _policy("reddit")
    assert gitlab.is_core_surface("issue.title")
    assert gitlab.is_core_surface("mr.title")
    assert reddit.is_core_surface("submission.title")

    assert gitlab.retired_reason_for("issue.title") == "retired_title_carrier_surface"
    assert gitlab.retired_reason_for("mr.title") == "retired_title_carrier_surface"
    assert reddit.retired_reason_for("submission.title") == "retired_title_carrier_surface"

    assert not gitlab.is_active_carrier("issue.title")
    assert not gitlab.is_active_carrier("mr.title")
    assert not reddit.is_active_carrier("submission.title")


def test_body_surfaces_remain_active_carriers() -> None:
    assert _policy("gitlab").is_active_carrier("issue.description")
    assert _policy("gitlab").is_active_carrier("note.body")
    assert _policy("reddit").is_active_carrier("submission.body")
    assert _policy("reddit").is_active_carrier("comment.body")


def test_gitlab_merge_request_routes_are_not_active_mainline_carriers() -> None:
    gitlab = _policy("gitlab")
    assert gitlab.ineligible_reason("mr.description") == "unsupported_merge_request_carrier_surface"
    assert (
        gitlab.ineligible_reason("note.body", kind="gitlab_mr")
        == "unsupported_merge_request_carrier_surface"
    )
    assert (
        gitlab.ineligible_reason("note.body", method="create_mr_note")
        == "unsupported_merge_request_carrier_surface"
    )
    assert not gitlab.is_active_carrier("mr.description")
    assert not gitlab.is_active_carrier("note.body", kind="gitlab_mr")
    assert not gitlab.is_active_carrier("note.body", method="create_mr_note")

    assert gitlab.is_active_carrier("note.body", kind="gitlab_issue")
    assert gitlab.is_active_carrier("note.body", method="create_issue_note")


# Results of the retired ``phases.phase_2_core_surfaces`` functions
# (canonical_core_surface, is_core_surface, retired_carrier_reason,
# active_carrier_ineligible_reason, is_active_carrier_surface) for every
# legacy alias key and every core/retired surface of both Sites.
_LEGACY_RESULTS: tuple[tuple[str, str, str | None, bool, str | None, str | None, bool], ...] = (
    (
        "gitlab",
        "issue_title",
        "issue.title",
        True,
        "retired_title_carrier_surface",
        "retired_title_carrier_surface",
        False,
    ),
    (
        "gitlab",
        "issue_title_list",
        "issue.title",
        True,
        "retired_title_carrier_surface",
        "retired_title_carrier_surface",
        False,
    ),
    (
        "gitlab",
        "issue_list_title",
        "issue.title",
        True,
        "retired_title_carrier_surface",
        "retired_title_carrier_surface",
        False,
    ),
    (
        "gitlab",
        "issue_title_in_list",
        "issue.title",
        True,
        "retired_title_carrier_surface",
        "retired_title_carrier_surface",
        False,
    ),
    ("gitlab", "issue_description", "issue.description", True, None, None, True),
    ("gitlab", "issue_description_detail", "issue.description", True, None, None, True),
    ("gitlab", "issue_detail_description", "issue.description", True, None, None, True),
    (
        "gitlab",
        "mr_title",
        "mr.title",
        True,
        "retired_title_carrier_surface",
        "retired_title_carrier_surface",
        False,
    ),
    (
        "gitlab",
        "mr_title_list",
        "mr.title",
        True,
        "retired_title_carrier_surface",
        "retired_title_carrier_surface",
        False,
    ),
    (
        "gitlab",
        "mr_list_title",
        "mr.title",
        True,
        "retired_title_carrier_surface",
        "retired_title_carrier_surface",
        False,
    ),
    (
        "gitlab",
        "mr_title_in_list",
        "mr.title",
        True,
        "retired_title_carrier_surface",
        "retired_title_carrier_surface",
        False,
    ),
    (
        "gitlab",
        "mr_description",
        "mr.description",
        True,
        None,
        "unsupported_merge_request_carrier_surface",
        False,
    ),
    (
        "gitlab",
        "mr_description_detail",
        "mr.description",
        True,
        None,
        "unsupported_merge_request_carrier_surface",
        False,
    ),
    (
        "gitlab",
        "mr_detail_description",
        "mr.description",
        True,
        None,
        "unsupported_merge_request_carrier_surface",
        False,
    ),
    ("gitlab", "issue_note", "note.body", True, None, None, True),
    ("gitlab", "issue_note_body", "note.body", True, None, None, True),
    ("gitlab", "note_on_issue", "note.body", True, None, None, True),
    ("gitlab", "note_body_on_issue", "note.body", True, None, None, True),
    ("gitlab", "mr_note", "note.body", True, None, None, True),
    ("gitlab", "mr_note_body", "note.body", True, None, None, True),
    ("gitlab", "note_on_mr", "note.body", True, None, None, True),
    ("gitlab", "note_body_on_mr", "note.body", True, None, None, True),
    ("gitlab", "issue.description", "issue.description", True, None, None, True),
    (
        "gitlab",
        "issue.title",
        "issue.title",
        True,
        "retired_title_carrier_surface",
        "retired_title_carrier_surface",
        False,
    ),
    (
        "gitlab",
        "mr.description",
        "mr.description",
        True,
        None,
        "unsupported_merge_request_carrier_surface",
        False,
    ),
    (
        "gitlab",
        "mr.title",
        "mr.title",
        True,
        "retired_title_carrier_surface",
        "retired_title_carrier_surface",
        False,
    ),
    ("gitlab", "note.body", "note.body", True, None, None, True),
    (
        "reddit",
        "submission_title",
        "submission.title",
        True,
        "retired_title_carrier_surface",
        "retired_title_carrier_surface",
        False,
    ),
    (
        "reddit",
        "submission_title_forum_listing",
        "submission.title",
        True,
        "retired_title_carrier_surface",
        "retired_title_carrier_surface",
        False,
    ),
    (
        "reddit",
        "submission_title_listing",
        "submission.title",
        True,
        "retired_title_carrier_surface",
        "retired_title_carrier_surface",
        False,
    ),
    (
        "reddit",
        "submission_title_feed",
        "submission.title",
        True,
        "retired_title_carrier_surface",
        "retired_title_carrier_surface",
        False,
    ),
    (
        "reddit",
        "submission_title_detail",
        "submission.title",
        True,
        "retired_title_carrier_surface",
        "retired_title_carrier_surface",
        False,
    ),
    ("reddit", "submission_body", "submission.body", True, None, None, True),
    ("reddit", "submission_body_post_detail", "submission.body", True, None, None, True),
    ("reddit", "submission_body_detail", "submission.body", True, None, None, True),
    ("reddit", "comment_body", "comment.body", True, None, None, True),
    ("reddit", "comment_body_post_detail", "comment.body", True, None, None, True),
    ("reddit", "comment_body_thread", "comment.body", True, None, None, True),
    ("reddit", "comment_body_detail", "comment.body", True, None, None, True),
    ("reddit", "comment.body", "comment.body", True, None, None, True),
    ("reddit", "submission.body", "submission.body", True, None, None, True),
    (
        "reddit",
        "submission.title",
        "submission.title",
        True,
        "retired_title_carrier_surface",
        "retired_title_carrier_surface",
        False,
    ),
)


@pytest.mark.parametrize(
    ("site", "raw", "canonical", "core", "retired", "ineligible", "active"),
    _LEGACY_RESULTS,
)
def test_policy_matches_retired_module_functions(
    site: str,
    raw: str,
    canonical: str | None,
    core: bool,
    retired: str | None,
    ineligible: str | None,
    active: bool,
) -> None:
    policy = default_catalog().bind(benchmark=BENCHMARK, site=site).carrier_policy()
    assert policy.canonical_surface(raw) == canonical
    assert policy.is_core_surface(raw) is core
    assert policy.retired_reason_for(raw) == retired
    assert policy.ineligible_reason(raw) == ineligible
    assert policy.is_active_carrier(raw) is active


@pytest.mark.parametrize(
    ("site", "module"),
    [("gitlab", gitlab_profile), ("reddit", reddit_profile)],
)
def test_policy_aliases_are_the_profile_mapping_aliases(site: str, module: object) -> None:
    policy = default_catalog().bind(benchmark=BENCHMARK, site=site).carrier_policy()
    mapping = module.mapping_for(BENCHMARK)  # type: ignore[attr-defined]
    assert mapping is not None
    assert policy.surface_aliases is mapping.profile_id_aliases


def test_bound_site_without_carrier_capability_binds_a_closed_policy() -> None:
    adapter = SyntheticDiscussionForumSite()
    assert not isinstance(adapter, SiteCarrierPolicyCapability)
    bound = SiteCatalog([adapter]).bind(site=adapter.site)  # type: ignore[list-item]
    policy = bound.carrier_policy()
    assert policy == SiteCarrierPolicy.closed(BENCHMARK)
    assert not policy.is_core_surface("comment.body")
    assert policy.canonical_surface("thread_reply") == "thread_reply"
    assert not policy.is_active_carrier("comment.body")


def test_bound_site_binds_closed_policy_for_other_benchmarks() -> None:
    bound = SiteCatalog().bind(benchmark="visualwebarena", site="gitlab")
    assert bound.carrier_policy() == SiteCarrierPolicy.closed("visualwebarena")
    assert not bound.carrier_policy().is_core_surface("issue.description")
