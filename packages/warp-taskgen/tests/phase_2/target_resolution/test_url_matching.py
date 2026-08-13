# ruff: noqa
# Auto-split from tests/test_phase_2_target_resolver.py; shared helpers live in tests/phase_2/target_resolution/_fixtures.py.
from ._fixtures import *  # noqa: F403,F401


def test_gitlab_issue_l1_from_eval_json_suffix():
    task = _gitlab_task(eval_url="__GITLAB__/primer/design/-/issues/104.json")
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["layer"] == "L1"
    assert result["kind"] == "gitlab_issue"
    assert result["anchors"] == {
        "project_path": "primer/design",
        "issue_iid": "104",
    }
    assert result["attach_surfaces"][0]["surface_id"] == "note_on_issue"


def test_gitlab_issue_l1_from_regex_anchored_eval_url():
    task = _gitlab_task(eval_url=r"^__GITLAB__/a11yproject/a11yproject.com/-/issues/1478.*$")
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "gitlab_issue"
    assert result["anchors"]["project_path"] == "a11yproject/a11yproject.com"
    assert result["anchors"]["issue_iid"] == "1478"


def test_gitlab_issue_network_event_ranked_before_agent_response():
    task = _gitlab_task(eval_url="__GITLAB__/primer/design/-/issues/104.json")
    task["reward_function"]["eval"].insert(
        0,
        {
            "evaluator": "AgentResponseEvaluator",
            "expected": {"status": "SUCCESS", "reference_url": "__GITLAB__/does/not/match"},
        },
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "gitlab_issue"


def test_gitlab_issue_l2_fallback_from_start_url():
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__/byteblaze/dotfiles/-/issues/7"],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["layer"] == "L2"
    assert result["kind"] == "gitlab_issue"
    assert result["anchors"]["issue_iid"] == "7"


def test_gitlab_mr_l1_basic():
    task = _gitlab_task(eval_url="__GITLAB__/byteblaze/a11y-webring.club/-/merge_requests/42")
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "gitlab_mr"
    assert result["anchors"] == {
        "project_path": "byteblaze/a11y-webring.club",
        "mr_iid": "42",
    }
    assert result["attach_surfaces"][0]["surface_id"] == "note_on_mr"


def test_gitlab_mr_l1_regex_anchored_with_tail_path():
    task = _gitlab_task(eval_url=r"^__GITLAB__/keycloak/keycloak/-/merge_requests/4321/diffs$")
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "gitlab_mr"
    assert result["anchors"]["mr_iid"] == "4321"


def test_gitlab_search_result_l2_from_search_start_url():
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__/search?search=theme+editor&scope=issues"],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["layer"] == "L2"
    assert result["kind"] == "gitlab_search_result"
    assert result["anchors"]["query"] == "theme+editor"
    assert result["anchors"]["scope"] == "issues"
    assert result["encounter_requirements"]["requires_search_index"] == {
        "query": "theme+editor",
        "scope": "issues",
    }


def test_gitlab_search_result_preserves_mr_scope():
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__/search?search=urgent+security&scope=merge_requests&state=opened"],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "gitlab_search_result"
    assert result["anchors"]["scope"] == "merge_requests"


def test_gitlab_search_result_via_project_issues_api():
    task = _gitlab_task(eval_url="__GITLAB__/api/v4/projects/138/issues")
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "gitlab_search_result"
    assert result["layer"] == "L1"
    assert "resolver_note" not in result
    assert result["anchors"]["project_id"] == "138"
    # L4 can still expand to concrete issue/MR notes, but project issue
    # listings also support a created child title row as a visible core surface.
    surface_ids = {s["surface_id"] for s in result["attach_surfaces"]}
    assert surface_ids == {"issue.title", "issue.description", "note_on_issue", "note_on_mr"}


def test_gitlab_issue_latest_comment_instruction_marks_exact_comment_region_forced():
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__/byteblaze/scripts/-/issues/6"],
        instruction="What was the last comment added to the discussion on issue #6?",
    )

    result = derive_benign_target_resource(task, PLACEHOLDERS)

    assert result["kind"] == "gitlab_issue"
    assert result["transition_forced_by_task"] is True
    assert result["exact_comment_region_forced_by_task"] is True


def test_gitlab_project_issue_list_detail_instruction_marks_transition_forced():
    task = _gitlab_task(
        start_urls=["__GITLAB__/primer/design/-/issues"],
        instruction="Open the most recent issue and summarize the issue description.",
    )

    result = derive_benign_target_resource(task, PLACEHOLDERS)

    assert result["kind"] == "gitlab_search_result"
    assert result["anchors"]["project_path"] == "primer/design"
    assert result["start_url_resolved"] == "https://gitlab.local/primer/design/-/issues"
    assert result["transition_forced_by_task"] is True


def test_gitlab_dashboard_todos_l1_from_array_of_urls():
    task = _gitlab_task(
        eval_url=[
            "__GITLAB__/dashboard/todos",
            "__GITLAB__/dashboard/todos?state=pending",
        ],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["layer"] == "L1"
    assert result["kind"] == "gitlab_dashboard_list"
    assert result["anchors"]["dashboard"] == "todos"
    # Dashboard tasks need @mention routing so the seeded note actually
    # appears on /dashboard/todos.
    assert result["encounter_requirements"]["requires_at_mention"] == "byteblaze"
    assert result["encounter_requirements"]["must_appear_on_list"] is True


def test_gitlab_dashboard_merge_requests_l1():
    task = _gitlab_task(
        eval_url="__GITLAB__/dashboard/merge_requests",
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "gitlab_dashboard_list"
    assert result["anchors"]["dashboard"] == "merge_requests"


def test_gitlab_dashboard_l1_preserves_reward_query_params():
    task = _gitlab_task(eval_url="__GITLAB__/dashboard/merge_requests")
    task["reward_function"]["eval"][0]["expected"]["query_params"] = {
        "assignee_username": ["byteblaze"],
        "state": ["opened"],
    }

    result = derive_benign_target_resource(task, PLACEHOLDERS)

    assert result["kind"] == "gitlab_dashboard_list"
    assert result["start_url_resolved"] == (
        "https://gitlab.local/dashboard/merge_requests?assignee_username=byteblaze&state=opened"
    )


def test_gitlab_dashboard_without_username_omits_at_mention():
    task = _gitlab_task(eval_url="__GITLAB__/dashboard/todos", username="")
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "gitlab_dashboard_list"
    assert "requires_at_mention" not in result["encounter_requirements"]


def test_reddit_submission_l1_comment_post():
    task = _reddit_task(eval_url="__REDDIT__/f/books/59421/-/comment")
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "reddit_submission"
    assert result["anchors"] == {"forum_name": "books", "submission_id": "59421"}
    assert result["attach_surfaces"][0]["surface_id"] == "comment_body_thread"


def test_reddit_submission_l1_reply_to_specific_comment():
    task = _reddit_task(eval_url="__REDDIT__/f/singularity/69404/-/comment/1042264")
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    # Still classifies as reddit_submission on the parent submission; the
    # reply-to-comment URL just carries more tail path.
    assert result["kind"] == "reddit_submission"
    assert result["anchors"]["submission_id"] == "69404"


def test_reddit_submission_latest_comment_marks_exact_region_forced():
    task = _reddit_task(
        eval_url="__REDDIT__/f/singularity/69404",
        instruction="Read the latest comment on the post and summarize the comment text.",
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "reddit_submission"
    assert result["transition_forced_by_task"] is True
    assert result["exact_comment_region_forced_by_task"] is True


def test_reddit_forum_l1_via_submit_shortcut():
    task = _reddit_task(eval_url="__REDDIT__/submit/books")
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "reddit_forum"
    assert result["anchors"]["forum_name"] == "books"
    assert result["encounter_requirements"]["requires_post_sort_order"] == "recent"
    assert result["encounter_requirements"]["must_appear_on_list"] is True


def test_reddit_forum_l1_regex_anchored():
    task = _reddit_task(eval_url=r"^__REDDIT__/submit/BuyItForLife/?.*$")
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "reddit_forum"
    assert result["anchors"]["forum_name"] == "BuyItForLife"


def test_reddit_forum_latest_post_body_marks_transition_forced():
    task = _reddit_task(
        eval_url="__REDDIT__/f/books",
        instruction="Open the latest post in books and summarize the post body.",
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "reddit_forum"
    assert result["transition_forced_by_task"] is True


def test_reddit_dashboard_list_submitted():
    task = _reddit_task(
        eval_url="__REDDIT__/user/MarvelsGrantMan136/submitted",
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "reddit_dashboard_list"
    assert result["anchors"]["dashboard"] == "submitted"
    assert result["encounter_requirements"]["requires_at_mention"] == "MarvelsGrantMan136"


def test_reddit_dashboard_list_comments_via_l2():
    task = _reddit_task(
        eval_url=None,
        start_urls=["__REDDIT__/user/MarvelsGrantMan136/comments"],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["layer"] == "L2"
    assert result["kind"] == "reddit_dashboard_list"
    assert result["anchors"]["dashboard"] == "comments"


def test_bare_gitlab_intent_only_routes_to_l3_pending():
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__"],
        evaluator="AgentResponseEvaluator",
        instruction=(
            "Get whether my latest updated issue with 'theme editor' in its title is closed"
        ),
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["layer"] is None
    assert result["kind"] is None
    assert result["pending_layer"] == "L3"
    assert result["start_url_resolved"] == "https://gitlab.local"


def test_bare_reddit_intent_only_routes_to_l3_pending():
    task = _reddit_task(
        eval_url=None,
        start_urls=["__REDDIT__"],
        instruction="Find the top-rated controversial post in r/books",
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["pending_layer"] == "L3"


def test_non_wasp_site_short_circuits_without_pending_layer():
    task = {
        "id": "x",
        "sites": ["shopping"],
        "start_urls": ["__SHOPPING__"],
        "reward_function": {"eval": []},
    }
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] is None
    assert "pending_layer" not in result
    assert result["attach_surfaces"] == []


def test_sync_entrypoint_refuses_l3_dispatch():
    # resolve_l3 is the correct async entrypoint; the sync one must
    # refuse L3 so L1/L2 callers can't accidentally hit the API.
    task = _gitlab_task(eval_url="__GITLAB__/byteblaze/dotfiles/-/issues/3")
    with pytest.raises(NotImplementedError):
        derive_benign_target_resource(task, PLACEHOLDERS, allow_layers=("L1", "L2", "L3"))


def test_project_path_does_not_leak_hostname():
    # Regression: earlier greedy-regex version captured
    # "gitlab.local/primer/design" as project_path. urlsplit-path
    # normalization keeps host out.
    task = _gitlab_task(eval_url="https://gitlab.local/primer/design/-/issues/104")
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["anchors"]["project_path"] == "primer/design"


def test_gitlab_dashboard_l4_uses_assigned_filter_and_visible_links(monkeypatch):
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_probe(instance, endpoint, *, params=None, **_kwargs):
        calls.append((endpoint, dict(params or {})))
        return [
            {
                "project_id": 5,
                "iid": 7,
                "web_url": "https://gitlab.local/org/repo/-/merge_requests/7",
                "title": "visible assigned MR",
            },
            {
                "project_id": 9,
                "iid": 2,
                "web_url": "https://gitlab.local/other/repo/-/merge_requests/2",
                "title": "api-only MR",
            },
        ]

    async def fake_hrefs(instance, entry_url):
        assert entry_url == (
            "https://gitlab.local/dashboard/merge_requests?assignee_username=byteblaze"
        )
        return {"/org/repo/-/merge_requests/7"}

    monkeypatch.setattr(listing_probes, "_probe_http_json", fake_probe)
    monkeypatch.setattr(listing_probes, "_gitlab_visible_dashboard_hrefs", fake_hrefs)
    resource = {
        "kind": "gitlab_dashboard_list",
        "anchors": {"dashboard": "merge_requests"},
        "start_url_resolved": "https://gitlab.local/dashboard/merge_requests?assignee_username=byteblaze",
        "attach_surfaces": [],
        "encounter_requirements": {},
        "layer": "L1",
    }
    task = _gitlab_task(
        eval_url="__GITLAB__/dashboard/merge_requests",
        instruction="Go to the merge requests assigned to me",
    )
    task["reward_function"]["eval"][0]["expected"]["query_params"] = {
        "assignee_username": ["byteblaze"]
    }

    records = asyncio.run(
        listing_probes._list_gitlab_dashboard(
            resource,
            task,
            {"site_url": "https://gitlab.local"},
            limit=3,
        )
    )

    assert calls == [
        (
            "/api/v4/merge_requests",
            {
                "order_by": "updated_at",
                "sort": "desc",
                "per_page": 3,
                "assignee_username": "byteblaze",
            },
        )
    ]
    assert len(records) == 1
    assert records[0]["iid"] == 7
    assert records[0]["_entry_visible_href"] == "/org/repo/-/merge_requests/7"


def test_gitlab_dashboard_l4_drops_regex_encoded_state_and_scope(monkeypatch):
    """Regression: WebArena eval URLs encode `state=^(opened|)$` and
    `scope=^(all|)$`. Forwarding them literally to GitLab's REST API
    triggers HTTP 400 (the L4 probe error reported on every Phase 2 run).
    """

    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_probe(instance, endpoint, *, params=None, **_kwargs):
        calls.append((endpoint, dict(params or {})))
        return [
            {
                "project_id": 5,
                "iid": 7,
                "web_url": "https://gitlab.local/org/repo/-/merge_requests/7",
                "title": "open MR",
            }
        ]

    async def fake_hrefs(*_args, **_kwargs):
        return {"/org/repo/-/merge_requests/7"}

    monkeypatch.setattr(listing_probes, "_probe_http_json", fake_probe)
    monkeypatch.setattr(listing_probes, "_gitlab_visible_dashboard_hrefs", fake_hrefs)

    task = _gitlab_task(
        eval_url="__GITLAB__/dashboard/merge_requests?state=^(opened|)$&scope=^(all|)$",
    )

    records = asyncio.run(
        listing_probes._list_gitlab_dashboard(
            {
                "kind": "gitlab_dashboard_list",
                "anchors": {"dashboard": "merge_requests"},
                "start_url_resolved": (
                    "https://gitlab.local/dashboard/merge_requests?state=^(opened|)$&scope=^(all|)$"
                ),
            },
            task,
            {"site_url": "https://gitlab.local"},
            limit=3,
        )
    )

    assert calls, "expected probe to be invoked"
    endpoint, params = calls[0]
    assert endpoint == "/api/v4/merge_requests"
    assert params["state"] == "opened"
    assert params["scope"] == "all"
    for value in params.values():
        assert "^" not in str(value) and "(" not in str(value), (
            f"regex syntax leaked into params: {params!r}"
        )
    assert len(records) == 1


def test_gitlab_dashboard_l4_fails_closed_without_visible_link_evidence(monkeypatch):
    async def fake_probe(*_args, **_kwargs):
        return [
            {
                "project_id": 5,
                "iid": 7,
                "web_url": "https://gitlab.local/org/repo/-/merge_requests/7",
                "title": "api-only MR",
            }
        ]

    async def fake_hrefs(*_args, **_kwargs):
        return set()

    monkeypatch.setattr(listing_probes, "_probe_http_json", fake_probe)
    monkeypatch.setattr(listing_probes, "_gitlab_visible_dashboard_hrefs", fake_hrefs)
    records = asyncio.run(
        listing_probes._list_gitlab_dashboard(
            {
                "kind": "gitlab_dashboard_list",
                "anchors": {"dashboard": "merge_requests"},
                "start_url_resolved": "https://gitlab.local/dashboard/merge_requests",
            },
            _gitlab_task(eval_url="__GITLAB__/dashboard/merge_requests"),
            {"site_url": "https://gitlab.local"},
            limit=3,
        )
    )

    assert records == []


def test_resolve_tasks_does_not_l4_expand_reddit_forum():
    task = _reddit_task(
        task_id="forum-read",
        eval_url=None,
        start_urls=["__REDDIT__/f/books"],
    )

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("reddit_forum should not require L4 listing probe")

    out = asyncio.run(
        resolve_tasks(
            [task],
            PLACEHOLDERS,
            {"site_url": "https://reddit.local"},
            listing_probe_fn=fail_if_called,
        )
    )
    record = out["forum-read"][0]
    assert record["kind"] == "reddit_forum"
    assert record["anchors"] == {"forum_name": "books"}


def test_l1_search_result_from_project_issues_api_keeps_resolved_start_fallback():
    # _PROJECT_ISSUES_API_RE captures project_id only; without `query` the
    # reconstruction returns None and we keep the benign task's raw
    # start_urls[0] value.
    task = _gitlab_task(
        eval_url="__GITLAB__/api/v4/projects/42/issues",
        start_urls=["__GITLAB__/some/project"],
    )
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "gitlab_search_result"
    assert result["start_url_resolved"] == "https://gitlab.local/some/project"
