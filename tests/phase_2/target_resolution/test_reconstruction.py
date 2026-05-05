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
    from worldsim.phase_2.target_resolution.reconstruction import _project_item_to_record

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
    from worldsim.phase_2.target_resolution.reconstruction import (
        _reconstruct_start_url_from_anchors,
    )

    assert (
        _reconstruct_start_url_from_anchors("gitlab", "not_a_kind", {"foo": "bar"}, PLACEHOLDERS)
        is None
    )


def test_reconstruction_helper_returns_none_when_anchors_insufficient():
    from worldsim.phase_2.target_resolution.reconstruction import (
        _reconstruct_start_url_from_anchors,
    )

    assert (
        _reconstruct_start_url_from_anchors(
            "gitlab", "gitlab_issue", {"project_path": "a/b"}, PLACEHOLDERS
        )
        is None
    )


def test_canonicalize_strips_localhost_authority():
    from worldsim.phase_2.target_resolution.url_matching import _canonicalize_project_path

    assert (
        _canonicalize_project_path("localhost:8023/a11yproject/a11yproject.com")
        == "a11yproject/a11yproject.com"
    )


def test_canonicalize_strips_full_https_url():
    from worldsim.phase_2.target_resolution.url_matching import _canonicalize_project_path

    assert _canonicalize_project_path("https://gitlab.local/primer/design") == "primer/design"


def test_canonicalize_strips_bare_gitlab_local():
    from worldsim.phase_2.target_resolution.url_matching import _canonicalize_project_path

    assert _canonicalize_project_path("gitlab.local/foo/bar") == "foo/bar"


def test_canonicalize_preserves_subgroup_paths():
    from worldsim.phase_2.target_resolution.url_matching import _canonicalize_project_path

    assert _canonicalize_project_path("namespace/subgroup/project") == "namespace/subgroup/project"


def test_canonicalize_idempotent_on_clean_path():
    from worldsim.phase_2.target_resolution.url_matching import _canonicalize_project_path

    canonical = "primer/design"
    assert _canonicalize_project_path(canonical) == canonical
    assert _canonicalize_project_path(_canonicalize_project_path(canonical)) == canonical


def test_canonicalize_handles_empty_input():
    from worldsim.phase_2.target_resolution.url_matching import _canonicalize_project_path

    assert _canonicalize_project_path("") == ""
    assert _canonicalize_project_path("   ") == ""


def test_canonicalize_strips_leading_and_trailing_slashes():
    from worldsim.phase_2.target_resolution.url_matching import _canonicalize_project_path

    assert _canonicalize_project_path("/primer/design/") == "primer/design"


def test_patch_benign_target_resource_urls_imports_after_reconstruction_split():
    import scripts.patch_benign_target_resource_urls as patch_script

    assert callable(patch_script.main)
