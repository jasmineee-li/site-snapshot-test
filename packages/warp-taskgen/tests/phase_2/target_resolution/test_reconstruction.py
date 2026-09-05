# ruff: noqa
# Auto-split from tests/test_phase_2_target_resolver.py; shared helpers live in tests/phase_2/target_resolution/_fixtures.py.
from ._fixtures import *  # noqa: F403,F401


def test_l1_issue_reconstructs_start_url_from_anchors_over_stale_template():
    task = _gitlab_task(
        eval_url="__GITLAB__/byteblaze/a11y-webring.club/-/issues/30",
        start_urls=["__GITLAB__/primer/design"],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["layer"] == "L1"
    assert result["kind"] == "gitlab_issue"
    assert (
        result["start_url_resolved"]
        == "https://gitlab.local/byteblaze/a11y-webring.club/-/issues/30"
    )


def test_l1_issue_reconstructs_from_bare_host_start_url():
    task = _gitlab_task(
        eval_url="__GITLAB__/a11yproject/a11yproject.com/-/issues/1064",
        start_urls=["__GITLAB__"],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert (
        result["start_url_resolved"]
        == "https://gitlab.local/a11yproject/a11yproject.com/-/issues/1064"
    )


def test_l1_mr_reconstructs_start_url():
    task = _gitlab_task(
        eval_url="__GITLAB__/org/repo/-/merge_requests/42",
        start_urls=["__GITLAB__"],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "gitlab_mr"
    assert result["start_url_resolved"] == "https://gitlab.local/org/repo/-/merge_requests/42"


def test_l1_search_result_reconstructs_with_url_encoded_query():
    task = _gitlab_task(
        eval_url="__GITLAB__/search?search=theme+editor&scope=issues",
        start_urls=["__GITLAB__"],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "gitlab_search_result"
    # `+` is preserved as a space-substitute (matches raw eval URL shape
    # and GitLab's accepted form).
    assert (
        result["start_url_resolved"]
        == "https://gitlab.local/search?search=theme+editor&scope=issues"
    )


def test_l1_dashboard_list_reconstructs_start_url():
    task = _gitlab_task(
        eval_url="__GITLAB__/dashboard/todos",
        start_urls=["__GITLAB__"],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "gitlab_dashboard_list"
    assert result["start_url_resolved"] == "https://gitlab.local/dashboard/todos"


def test_l2_fallback_reconstructs_when_start_url_matches():
    # L2 picks the benign task's start_urls; if they're already a concrete
    # URL, reconstruction is idempotent for issue kinds.
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__/byteblaze/dotfiles/-/issues/7"],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["layer"] == "L2"
    assert result["start_url_resolved"] == "https://gitlab.local/byteblaze/dotfiles/-/issues/7"


def test_l1_reddit_submission_reconstructs():
    task = _reddit_task(
        eval_url="__REDDIT__/f/pittsburgh/89821/some-title",
        start_urls=["__REDDIT__"],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "reddit_submission"
    assert result["start_url_resolved"] == "https://reddit.local/f/pittsburgh/89821"


def test_l1_reddit_forum_reconstructs():
    task = _reddit_task(
        eval_url="__REDDIT__/f/books",
        start_urls=["__REDDIT__"],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "reddit_forum"
    assert result["start_url_resolved"] == "https://reddit.local/f/books"


def test_l1_reddit_dashboard_list_reconstructs():
    task = _reddit_task(
        eval_url="__REDDIT__/user/MarvelsGrantMan136/submitted",
        start_urls=["__REDDIT__"],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "reddit_dashboard_list"
    assert result["start_url_resolved"] == "https://reddit.local/user/MarvelsGrantMan136/submitted"


def test_l4_item_record_reconstructs_start_url_strips_localhost_prefix():
    # L4 probe returns items whose project_path comes from `web_url`
    # parsing at :line 1234 — that produces bare "byteblaze/dotfiles".
    # But resource.anchors.project_path can also carry an authority
    # prefix like "localhost:8023/byteblaze/a11y-webring.club" (observed
    # in the 0/107 feasibility report). _clean_project_path strips it.
    from warp_taskgen.phase_2.target_resolution.reconstruction import _project_item_to_record

    base = {
        "kind": "gitlab_search_result",
        "anchors": {"query": "theme", "scope": "issues"},
        "start_url_resolved": "https://gitlab.local",
        "attach_surfaces": [],
        "encounter_requirements": {},
        "layer": "L2",
    }
    item = {
        "_item_kind": "gitlab_issue",
        "project_id": 179,
        "iid": 30,
        "web_url": "http://localhost:8023/byteblaze/a11y-webring.club/-/issues/30",
        "title": "theme editor",
    }
    record = _project_item_to_record(base, item, PLACEHOLDERS)
    assert record is not None
    # _ISSUE_RE against web_url captures "byteblaze/a11y-webring.club"
    # without the host prefix; reconstruction then joins against the
    # placeholder origin. No double-host leak.
    assert (
        record["start_url_resolved"]
        == "https://gitlab.local/byteblaze/a11y-webring.club/-/issues/30"
    )


def test_reconstruction_helper_handles_unknown_kind():
    from warp_taskgen.phase_2.target_resolution.reconstruction import (
        _reconstruct_start_url_from_anchors,
    )

    assert (
        _reconstruct_start_url_from_anchors("gitlab", "not_a_kind", {"foo": "bar"}, PLACEHOLDERS)
        is None
    )


def test_reconstruction_helper_returns_none_when_anchors_insufficient():
    from warp_taskgen.phase_2.target_resolution.reconstruction import (
        _reconstruct_start_url_from_anchors,
    )

    assert (
        _reconstruct_start_url_from_anchors(
            "gitlab", "gitlab_issue", {"project_path": "a/b"}, PLACEHOLDERS
        )
        is None
    )


def test_canonicalize_strips_localhost_authority():
    from warp_taskgen.phase_2.target_resolution.url_matching import _canonicalize_project_path

    assert (
        _canonicalize_project_path("localhost:8023/a11yproject/a11yproject.com")
        == "a11yproject/a11yproject.com"
    )


def test_canonicalize_strips_full_https_url():
    from warp_taskgen.phase_2.target_resolution.url_matching import _canonicalize_project_path

    assert _canonicalize_project_path("https://gitlab.local/primer/design") == "primer/design"


def test_canonicalize_strips_bare_gitlab_local():
    from warp_taskgen.phase_2.target_resolution.url_matching import _canonicalize_project_path

    assert _canonicalize_project_path("gitlab.local/foo/bar") == "foo/bar"


def test_canonicalize_preserves_subgroup_paths():
    from warp_taskgen.phase_2.target_resolution.url_matching import _canonicalize_project_path

    assert _canonicalize_project_path("namespace/subgroup/project") == "namespace/subgroup/project"


def test_canonicalize_idempotent_on_clean_path():
    from warp_taskgen.phase_2.target_resolution.url_matching import _canonicalize_project_path

    canonical = "primer/design"
    assert _canonicalize_project_path(canonical) == canonical
    assert _canonicalize_project_path(_canonicalize_project_path(canonical)) == canonical


def test_canonicalize_handles_empty_input():
    from warp_taskgen.phase_2.target_resolution.url_matching import _canonicalize_project_path

    assert _canonicalize_project_path("") == ""
    assert _canonicalize_project_path("   ") == ""


def test_canonicalize_strips_leading_and_trailing_slashes():
    from warp_taskgen.phase_2.target_resolution.url_matching import _canonicalize_project_path

    assert _canonicalize_project_path("/primer/design/") == "primer/design"


def test_patch_benign_target_resource_urls_rewrites_and_is_idempotent(tmp_path):
    import json

    import scripts.patch_benign_target_resource_urls as patch_script

    path = tmp_path / "adversarial_tasks.json"
    tasks = [
        {
            "sites": ["gitlab"],
            "benign_target_resource": {
                "kind": "gitlab_issue",
                "anchors": {
                    "project_path": "byteblaze/a11y-webring.club",
                    "issue_iid": "7",
                },
                "start_url_resolved": "https://gitlab.local/dashboard/issues",
            },
        }
    ]
    path.write_text(json.dumps(tasks))

    assert patch_script.main(["patch", str(path)]) == 0
    patched = json.loads(path.read_text())
    assert (
        patched[0]["benign_target_resource"]["start_url_resolved"]
        == "https://gitlab.local/byteblaze/a11y-webring.club/-/issues/7"
    )

    assert patch_script.main(["patch", str(path)]) == 0
    assert json.loads(path.read_text()) == patched


def test_bound_site_reconstructs_through_the_adapter_hook():
    from .test_l4 import _fake_l4_catalog

    bound = _fake_l4_catalog().bind(site="fake", origin="https://fake.local")

    assert bound.reconstruct("list", {}) == "https://fake.local/list"
    assert bound.reconstruct("fake_list", {}) == "https://fake.local/list"
    assert bound.reconstruct("message", {"message_id": "7"}) == "https://fake.local/messages/7"
    assert bound.reconstruct("fake_message", {}) is None
    assert bound.reconstruct("not_a_kind", {"message_id": "7"}) is None
    # No origin: the adapter has nothing to bind the route to.
    assert _fake_l4_catalog().bind(site="fake").reconstruct("list", {}) is None


def test_reconstruction_helper_resolves_a_fourth_site_through_the_catalog(monkeypatch):
    from warp_taskgen.phase_2.target_resolution import reconstruction
    from warp_taskgen.phase_2.target_resolution.listing_records import compose_listing_record
    from warp_taskgen.sites import ListingItemCandidate

    from .test_l4 import _fake_l4_catalog

    catalog = _fake_l4_catalog()
    monkeypatch.setattr(reconstruction, "default_catalog", lambda: catalog)
    placeholders = {"__FAKE__": "https://fake.local"}

    assert (
        reconstruction._reconstruct_start_url_from_anchors(
            "fake", "fake_message", {"message_id": "7"}, placeholders
        )
        == "https://fake.local/messages/7"
    )
    assert reconstruction._reconstruct_start_url_from_anchors("fake", "list", {}, placeholders) == (
        "https://fake.local/list"
    )
    # A Site the catalog does not know fails closed instead of raising.
    assert (
        reconstruction._reconstruct_start_url_from_anchors("gitlab", "list", {}, placeholders)
        is None
    )

    base = {"kind": "fake_list", "anchors": {}, "start_url_resolved": "https://fake.local/list"}
    item = {"_item_kind": "fake_message", "id": "7", "title": "hello"}
    record = reconstruction._project_item_to_record(base, item, placeholders)
    bound = catalog.bind(site="fake", origin="https://fake.local", placeholders=placeholders)
    candidate = ListingItemCandidate(
        source_kind="fake_list",
        item_kind="fake_message",
        payload=item,
        evidence_url=base["start_url_resolved"],
    )
    expected = compose_listing_record(base, candidate, bound.materialize_listing_entry(candidate))
    assert record is not None
    assert record == expected
    assert record["start_url_resolved"] == "https://fake.local/messages/7"


def test_bound_site_probe_item_anchors_wraps_the_active_site_hooks():
    from warp_taskgen.sites import default_catalog
    from warp_taskgen.sites.gitlab import GitLabSite
    from warp_taskgen.sites.reddit import RedditSite

    item = {"project_id": 5, "iid": 30, "web_url": "https://gitlab.local/a/b/-/issues/30"}
    gitlab = default_catalog().bind(site="gitlab", placeholders=PLACEHOLDERS)
    assert gitlab.probe_item_anchors(
        item, kind_hint="gitlab_issue"
    ) == GitLabSite.anchors_from_item(item, kind_hint="gitlab_issue")
    # The hook wraps ``anchors_from_item`` verbatim: the iid key follows the
    # kind hint and ``project_path`` still carries the web_url authority that
    # ``_canonicalize_project_path`` strips downstream.
    assert gitlab.probe_item_anchors(item, kind_hint="search_user_mrs") == {
        "project_id": "5",
        "mr_iid": "30",
        "project_path": "gitlab.local/a/b",
    }

    entry = {"id": "12", "title": "t"}
    reddit = default_catalog().bind(site="reddit", placeholders=PLACEHOLDERS)
    assert reddit.probe_item_anchors(
        entry, kind_hint="reddit_submission", forum_name="books"
    ) == RedditSite.anchors_from_submission(entry, "books")
    # Reddit anchors need a forum; without one there is nothing to attribute.
    assert reddit.probe_item_anchors(entry, kind_hint="reddit_submission") is None


def test_bound_site_probe_item_anchors_is_none_without_the_adapter_hook():
    from tests.sites.synthetic_discussion_forum.site import ORIGIN, SyntheticDiscussionForumSite
    from warp_taskgen.sites import SiteCatalog

    bound = SiteCatalog([SyntheticDiscussionForumSite()]).bind(
        site="synthetic_discussion_forum", origin=ORIGIN
    )

    assert bound.probe_item_anchors({"id": "1"}, kind_hint="thread") is None
    assert bound.probe_item_anchors({"id": "1"}, kind_hint="thread", forum_name="x") is None
    assert bound.project_path_from_listing_task("open a/b issues", resolved_start=None) is None


def test_target_resolution_holds_no_module_level_site_instances():
    from warp_taskgen.phase_2.target_resolution import listing_intent, reconstruction, url_matching

    for module in (url_matching, reconstruction, listing_intent):
        assert not hasattr(module, "_GITLAB_SITE"), module.__name__
        assert not hasattr(module, "_REDDIT_SITE"), module.__name__
