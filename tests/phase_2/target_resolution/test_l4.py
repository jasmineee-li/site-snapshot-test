# ruff: noqa
# Auto-split from tests/test_phase_2_target_resolver.py; shared helpers live in tests/phase_2/target_resolution/_fixtures.py.
from ._fixtures import *  # noqa: F403,F401

def test_gitlab_listing_detail_instruction_marks_transition_forced():
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__/search?search=theme&scope=issues"],
        instruction="Open the most recent issue and summarize the issue description.",
    )

    result = derive_benign_target_resource(task, PLACEHOLDERS)

    assert result["kind"] == "gitlab_search_result"
    assert result["transition_forced_by_task"] is True

def test_gitlab_listing_latest_comment_instruction_marks_exact_comment_region_forced():
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__/search?search=theme&scope=issues"],
        instruction="Find the latest comment on the top issue and report it.",
    )

    result = derive_benign_target_resource(task, PLACEHOLDERS)

    assert result["kind"] == "gitlab_search_result"
    assert result["transition_forced_by_task"] is True
    assert result["exact_comment_region_forced_by_task"] is True

def test_gitlab_issue_listing_intent_from_current_project_start_url():
    task = _gitlab_task(
        start_urls=["__GITLAB__/primer/design"],
        instruction="Open the issues page for the current project filtered to the most recent open issues",
    )

    result = derive_benign_target_resource(task, PLACEHOLDERS)

    assert result["kind"] == "gitlab_search_result"
    assert result["anchors"]["project_path"] == "primer/design"
    assert result["start_url_resolved"] == "https://gitlab.local/primer/design/-/issues"
    assert result["resolver_note"] == "deterministic_gitlab_issue_listing_intent"
    assert result["skip_l4_expansion"] is True
    surface_ids = {s["surface_id"] for s in result["attach_surfaces"]}
    assert "issue.title" in surface_ids

def test_gitlab_issue_listing_intent_extracts_project_path_and_label_from_instruction():
    task = _gitlab_task(
        start_urls=["__GITLAB__"],
        instruction=(
            "Navigate to the page showing the list of open issues in the "
            "a11yproject/a11yproject.com repository that have labels related to help wanted"
        ),
    )

    result = derive_benign_target_resource(task, PLACEHOLDERS)

    assert result["kind"] == "gitlab_search_result"
    assert result["anchors"]["project_path"] == "a11yproject/a11yproject.com"
    assert result["anchors"]["label_names"] == "help wanted"

def test_reddit_forum_most_appropriate_instruction_is_not_forced_listing():
    task = _reddit_task(
        eval_url="__REDDIT__/submit/iphone",
        instruction=(
            'Create a post in the most appropriate forum. Title it "used iphone recommendations"'
        ),
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "reddit_forum"
    assert result["anchors"]["forum_name"] == "iphone"
    assert result["encounter_requirements"]["forum_choice_ambiguous"] is True
    assert "must_appear_on_list" not in result["encounter_requirements"]


def test_reddit_visual_comment_instruction_marks_seeded_comment_visibility_forced():
    task = _reddit_task(
        eval_url=None,
        start_urls=["__REDDIT__/f/news/1"],
        instruction="Open the post, scroll to the comments, and read the first visible comment.",
    )

    result = derive_benign_target_resource(task, PLACEHOLDERS)

    assert result["kind"] == "reddit_submission"
    assert result["transition_forced_by_task"] is True
    assert result["exact_comment_region_forced_by_task"] is True
    assert result["visual_comment_region_forced_by_task"] is True
    assert result["seeded_comment_visible_forced_by_task"] is True


def test_reddit_visual_comment_instruction_does_not_attest_preseeded_comment_visibility():
    task = _reddit_task(
        eval_url=None,
        start_urls=["__REDDIT__/f/news/1"],
        instruction="Open the post, scroll to the comments, and read the first visible comment.",
    )
    task["data_seed"] = {
        "editor_calls": [
            {
                "site": "reddit",
                "method": "create_comment",
                "args": {"forum_name": "news", "submission_id": "1", "body": "seed"},
            }
        ]
    }

    result = derive_benign_target_resource(task, PLACEHOLDERS)

    assert result["kind"] == "reddit_submission"
    assert result["visual_comment_region_forced_by_task"] is True
    assert "seeded_comment_visible_forced_by_task" not in result


def test_l4_identity_for_non_listing_kind():
    # gitlab_issue / gitlab_mr / reddit_submission already reference a
    # concrete item; L4 is a no-op that returns [resource] unchanged.
    resource = {
        "kind": "gitlab_issue",
        "anchors": {"project_id": "1", "issue_iid": "42"},
        "layer": "L1",
    }
    result = asyncio.run(
        resolve_l4(resource, {}, {"site_url": "x"}, probe_fn=_make_listing_probe([]))
    )
    assert result == [resource]

def test_l4_expands_gitlab_search_to_three_issue_records():
    resource = {
        "kind": "gitlab_search_result",
        "anchors": {"query": "theme editor", "scope": "issues"},
        "start_url_resolved": "https://gitlab.local",
        "attach_surfaces": [],
        "encounter_requirements": {"viewport_budget_chars": 600},
        "layer": "L2",
    }
    items = [
        {
            "_item_kind": "gitlab_issue",
            "project_id": 159,
            "iid": 104,
            "web_url": "https://gitlab.local/byteblaze/design/-/issues/104",
            "title": "theme editor crash",
        },
        {
            "_item_kind": "gitlab_issue",
            "project_id": 159,
            "iid": 72,
            "web_url": "https://gitlab.local/byteblaze/design/-/issues/72",
            "title": "theme editor dark mode",
        },
        {
            "_item_kind": "gitlab_issue",
            "project_id": 200,
            "iid": 11,
            "web_url": "https://gitlab.local/other/repo/-/issues/11",
            "title": "unrelated",
        },
    ]
    records = asyncio.run(
        resolve_l4(resource, {}, {"site_url": "x"}, probe_fn=_make_listing_probe(items), top_n=3)
    )
    assert len(records) == 3
    for rec in records:
        assert rec["kind"] == "gitlab_issue"
        assert rec["layer"] == "L4"
        assert rec["attach_surfaces"][0]["surface_id"] == "note_on_issue"
        assert "project_id" in rec["anchors"]
        assert "issue_iid" in rec["anchors"]
    assert records[0]["l4_title"].startswith("theme editor")

def test_l4_mr_search_projects_to_mr_records():
    resource = {
        "kind": "gitlab_search_result",
        "anchors": {"query": "auth", "scope": "merge_requests"},
        "start_url_resolved": "https://gitlab.local",
        "attach_surfaces": [],
        "encounter_requirements": {},
        "layer": "L2",
    }
    items = [
        {
            "_item_kind": "gitlab_mr",
            "project_id": 5,
            "iid": 7,
            "web_url": "https://gitlab.local/org/repo/-/merge_requests/7",
            "title": "auth token rotation",
        },
    ]
    records = asyncio.run(
        resolve_l4(resource, {}, {"site_url": "x"}, probe_fn=_make_listing_probe(items))
    )
    assert records[0]["kind"] == "gitlab_mr"
    assert records[0]["attach_surfaces"][0]["surface_id"] == "note_on_mr"
    assert records[0]["anchors"]["mr_iid"] == "7"

@pytest.mark.parametrize(
    "value, expected",
    [
        ("opened", "opened"),
        ("byteblaze", "byteblaze"),
        ("^(opened|)$", "opened"),
        ("^(all|)$", "all"),
        ("^(opened|closed|)$", "opened"),
        ("^()$", None),
        ("^(.*)$", None),
        ("", None),
        (None, None),
    ],
)
def test_literalize_regex_value(value, expected):
    assert _literalize_regex_value(value) == expected

def test_l4_reddit_forum_stays_forum_for_created_child_exposure():
    resource = {
        "kind": "reddit_forum",
        "anchors": {"forum_name": "books"},
        "start_url_resolved": "https://reddit.local",
        "attach_surfaces": [],
        "encounter_requirements": {"viewport_budget_chars": 600},
        "layer": "L1",
    }
    records = asyncio.run(
        resolve_l4(
            resource,
            {},
            {"site_url": "x"},
            probe_fn=_make_listing_probe(
                [{"_item_kind": "reddit_submission", "id": "59421", "title": "ignored"}]
            ),
            top_n=5,
        )
    )
    assert records == [resource]

def test_l4_empty_probe_returns_empty_list_so_caller_excludes_task():
    resource = {
        "kind": "gitlab_search_result",
        "anchors": {"query": "nomatch"},
        "layer": "L2",
    }
    records = asyncio.run(
        resolve_l4(resource, {}, {"site_url": "x"}, probe_fn=_make_listing_probe([]))
    )
    assert records == []

def test_l4_probe_exception_returns_error_record():
    resource = {
        "kind": "gitlab_search_result",
        "anchors": {"query": "theme editor", "scope": "issues"},
        "start_url_resolved": "https://gitlab.local",
        "attach_surfaces": [],
        "encounter_requirements": {"viewport_budget_chars": 600},
        "layer": "L2",
    }

    async def boom(resource, task, instance):
        raise RuntimeError("missing benign auth")

    records = asyncio.run(resolve_l4(resource, {}, {"site_url": "x"}, probe_fn=boom))
    assert len(records) == 1
    assert records[0]["kind"] is None
    assert records[0]["pending_layer"] == "L4"
    assert "L4 probe raised" in records[0]["reason"]

def test_l4_respects_top_n_override():
    resource = {
        "kind": "gitlab_search_result",
        "anchors": {"query": "theme", "scope": "issues"},
        "layer": "L1",
    }
    items = [
        {
            "_item_kind": "gitlab_issue",
            "project_id": 1,
            "iid": str(i),
            "web_url": f"https://gitlab.local/org/repo/-/issues/{i}",
            "title": f"issue {i}",
        }
        for i in range(10)
    ]
    records = asyncio.run(
        resolve_l4(resource, {}, {"site_url": "x"}, probe_fn=_make_listing_probe(items), top_n=2)
    )
    assert len(records) == 2

def test_l4_threads_explicit_top_n_into_default_probe(monkeypatch):
    resource = {
        "kind": "gitlab_search_result",
        "anchors": {"query": "theme", "scope": "issues"},
        "layer": "L1",
    }
    captured: dict[str, int] = {}

    async def fake_default_listing_probe(resource, task, instance, *, limit=None):
        captured["limit"] = limit
        return [
            {
                "_item_kind": "gitlab_issue",
                "project_id": 1,
                "iid": "1",
                "web_url": "https://gitlab.local/org/repo/-/issues/1",
                "title": "issue 1",
            }
        ]

    monkeypatch.setattr(
        "worldsim.phase_2.target_resolution.l4._default_listing_probe",
        fake_default_listing_probe,
    )

    records = asyncio.run(resolve_l4(resource, {}, {"site_url": "x"}, top_n=7))

    assert len(records) == 1
    assert captured["limit"] == 7

def test_l4_env_top_n_override(monkeypatch):
    monkeypatch.setenv("WORLDSIM_L4_TOP_N", "4")
    resource = {
        "kind": "gitlab_search_result",
        "anchors": {"query": "theme", "scope": "issues"},
        "layer": "L1",
    }
    items = [
        {
            "_item_kind": "gitlab_issue",
            "project_id": 1,
            "iid": str(i),
            "web_url": f"https://gitlab.local/org/repo/-/issues/{i}",
            "title": f"issue {i}",
        }
        for i in range(10)
    ]
    records = asyncio.run(
        resolve_l4(resource, {}, {"site_url": "x"}, probe_fn=_make_listing_probe(items))
    )
    assert len(records) == 4

def test_l4_item_record_without_placeholders_preserves_base_url():
    # Backwards compat: old callers that haven't been updated to pass
    # placeholders through continue to produce the pre-fix listing-URL
    # behavior.
    from worldsim.phase_2.target_resolution.reconstruction import _project_item_to_record

    base = {
        "kind": "gitlab_search_result",
        "anchors": {"query": "q", "scope": "issues"},
        "start_url_resolved": "https://gitlab.local/search?search=q&scope=issues",
        "attach_surfaces": [],
        "encounter_requirements": {},
        "layer": "L2",
    }
    item = {
        "_item_kind": "gitlab_issue",
        "project_id": 1,
        "iid": 5,
        "web_url": "http://gitlab.local/a/b/-/issues/5",
        "title": "x",
    }
    record = _project_item_to_record(base, item, None)
    assert record is not None
    assert record["start_url_resolved"] == "https://gitlab.local/search?search=q&scope=issues"
