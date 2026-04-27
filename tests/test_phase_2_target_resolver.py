"""L1/L2 fixtures for :mod:`worldsim.phases.phase_2_target_resolver`.

Every fixture builds a minimal benign task record inline — no dependency
on ``logs/phase_1/benign_tasks.json``. Covers every ResourceKind plus
edge shapes lifted from the real dataset (regex-anchored eval URLs,
``.json`` suffix, array-of-URLs, intent-only bare ``__GITLAB__``).
"""

from __future__ import annotations

from typing import Any

import pytest

from worldsim.phases import phase_2_target_resolver as resolver
from worldsim.phases.phase_2_target_resolver import (
    VIEWPORT_BUDGET_CHARS,
    _benign_probe_instance,
    _postmill_submission_comment_count_from_html,
    derive_benign_target_resource,
    resolve_tasks,
)
from worldsim.placeholders import placeholders_for_site_urls

PLACEHOLDERS = placeholders_for_site_urls(
    [
        ("gitlab", "https://gitlab.local"),
        ("reddit", "https://reddit.local"),
    ]
)


def _gitlab_task(
    *,
    task_id: str = "t",
    eval_url: Any | None = None,
    start_urls: list[str] | None = None,
    username: str = "byteblaze",
    evaluator: str = "NetworkEventEvaluator",
    instruction: str = "",
) -> dict[str, Any]:
    task: dict[str, Any] = {
        "id": task_id,
        "sites": ["gitlab"],
        "start_urls": start_urls if start_urls is not None else ["__GITLAB__"],
        "instruction": instruction,
        "agent_context": {"authentication": {"credentials": {"username": username}}},
        "reward_function": {"eval": []},
    }
    if eval_url is not None:
        task["reward_function"]["eval"] = [{"evaluator": evaluator, "expected": {"url": eval_url}}]
    return task


def _reddit_task(
    *,
    task_id: str = "t",
    eval_url: Any | None = None,
    start_urls: list[str] | None = None,
    username: str = "MarvelsGrantMan136",
    evaluator: str = "NetworkEventEvaluator",
    instruction: str = "",
) -> dict[str, Any]:
    task: dict[str, Any] = {
        "id": task_id,
        "sites": ["reddit"],
        "start_urls": start_urls if start_urls is not None else ["__REDDIT__"],
        "instruction": instruction,
        "agent_context": {"authentication": {"credentials": {"username": username}}},
        "reward_function": {"eval": []},
    }
    if eval_url is not None:
        task["reward_function"]["eval"] = [{"evaluator": evaluator, "expected": {"url": eval_url}}]
    return task


# --- gitlab_issue --------------------------------------------------------


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


# --- gitlab_mr -----------------------------------------------------------


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


# --- gitlab_search_result ------------------------------------------------


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
    assert result["anchors"]["project_id"] == "138"
    # L4 can still expand to concrete issue/MR notes, but project issue
    # listings also support a created child title row as a visible core surface.
    surface_ids = {s["surface_id"] for s in result["attach_surfaces"]}
    assert surface_ids == {"issue.title", "issue.description", "note_on_issue", "note_on_mr"}


def test_gitlab_listing_detail_instruction_marks_transition_forced():
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__/search?search=theme&scope=issues"],
        instruction="Open the most recent issue and summarize the issue description.",
    )

    result = derive_benign_target_resource(task, PLACEHOLDERS)

    assert result["kind"] == "gitlab_search_result"
    assert result["transition_forced_by_task"] is True


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


# --- gitlab_dashboard_list ----------------------------------------------


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


# --- reddit_submission --------------------------------------------------


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


# --- reddit_forum -------------------------------------------------------


def test_reddit_forum_l1_via_submit_shortcut():
    task = _reddit_task(eval_url="__REDDIT__/submit/books")
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["kind"] == "reddit_forum"
    assert result["anchors"]["forum_name"] == "books"
    assert result["encounter_requirements"]["requires_post_sort_order"] == "recent"
    assert result["encounter_requirements"]["must_appear_on_list"] is True


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


# --- reddit_dashboard_list ----------------------------------------------


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


# --- fallthrough / out-of-scope ------------------------------------------


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


def test_viewport_budget_chars_always_set():
    task = _gitlab_task(eval_url="__GITLAB__/byteblaze/dotfiles/-/issues/3")
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["encounter_requirements"]["viewport_budget_chars"] == VIEWPORT_BUDGET_CHARS


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


# --- L3 resolver (stubbed classifier + probe) ----------------------------

import asyncio  # noqa: E402

from worldsim.phases.phase_2_target_resolver import resolve_l3  # noqa: E402


def _make_classifier(parsed):
    async def _stub(task, placeholders):
        return parsed

    return _stub


def _make_probe(anchors):
    async def _stub(probe_query, task, instance, placeholders):
        return anchors

    return _stub


def test_l3_happy_path_gitlab_issue():
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__"],
        instruction=(
            "Get whether my latest updated issue with 'theme editor' in its title is closed"
        ),
    )
    classifier = _make_classifier(
        {
            "kind": "gitlab_issue",
            "probe_query": {
                "api": "search_user_issues",
                "query": "theme editor",
                "sort": "desc",
                "limit": 1,
            },
            "confidence": 0.92,
        }
    )
    probe = _make_probe(
        {"project_id": "159", "project_path": "byteblaze/design", "issue_iid": "104"}
    )
    instance = {"site_url": "https://gitlab.local"}
    result = asyncio.run(
        resolve_l3(task, PLACEHOLDERS, instance, classifier=classifier, probe_fn=probe)
    )
    assert result["layer"] == "L3"
    assert result["kind"] == "gitlab_issue"
    assert result["anchors"]["issue_iid"] == "104"
    assert result["anchors"]["project_id"] == "159"
    assert result["l3_confidence"] == 0.92
    assert result["attach_surfaces"][0]["surface_id"] == "note_on_issue"
    assert result["source_listing_kind"] == "gitlab_search_result"
    assert result["benign_read_url"] == "https://gitlab.local/byteblaze/design/-/issues"
    assert result["seeded_detail_url"] == "https://gitlab.local/byteblaze/design/-/issues/104"
    assert "transition_forced_by_task" not in result


def test_l3_forced_gitlab_issue_comment_preserves_listing_provenance():
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__"],
        instruction="Comment on the most recent open issue in primer/design",
    )
    classifier = _make_classifier(
        {
            "kind": "gitlab_issue",
            "probe_query": {
                "api": "list_project_issues_recent",
                "project_path": "primer/design",
                "query": "state=opened",
                "limit": 1,
            },
            "confidence": 0.93,
        }
    )
    probe = _make_probe({"project_id": "159", "project_path": "primer/design", "issue_iid": "42"})
    instance = {"site_url": "https://gitlab.local"}
    result = asyncio.run(
        resolve_l3(task, PLACEHOLDERS, instance, classifier=classifier, probe_fn=probe)
    )

    assert result["kind"] == "gitlab_issue"
    assert result["source_listing_kind"] == "gitlab_search_result"
    assert result["benign_read_url"] == "https://gitlab.local/primer/design/-/issues"
    assert result["seeded_detail_url"] == "https://gitlab.local/primer/design/-/issues/42"
    assert result["transition_forced_by_task"] is True


def test_l3_null_kind_marks_task_out_of_scope_without_probe():
    # Pure actions (fork/follow/invite/edit-own-profile) have no Option-A
    # attach surface; L3 returns kind=None + pending_layer absent so the
    # 2a validator drops the task rather than retrying.
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__"],
        instruction="Fork the 2019-nCov project.",
    )
    classifier = _make_classifier(
        {
            "kind": None,
            "probe_query": {
                "api": "none",
                "note": "fork is a pure action with no discussion target",
            },
            "confidence": 0.98,
        }
    )
    probe_called = False

    async def _probe_should_not_run(*args, **kwargs):
        nonlocal probe_called
        probe_called = True
        return None

    instance = {"site_url": "https://gitlab.local"}
    result = asyncio.run(
        resolve_l3(
            task,
            PLACEHOLDERS,
            instance,
            classifier=classifier,
            probe_fn=_probe_should_not_run,
        )
    )
    assert result["kind"] is None
    assert result["layer"] == "L3"
    assert "pending_layer" not in result
    assert probe_called is False


def test_l3_probe_returns_nothing_excludes_task():
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__"],
        instruction="Get the URL to clone metaseq with SSH.",
    )
    classifier = _make_classifier(
        {
            "kind": "gitlab_issue",
            "probe_query": {"api": "search_project_issues", "project_path": "root/metaseq"},
            "confidence": 0.55,
        }
    )
    instance = {"site_url": "https://gitlab.local"}
    result = asyncio.run(
        resolve_l3(
            task,
            PLACEHOLDERS,
            instance,
            classifier=classifier,
            probe_fn=_make_probe(None),
        )
    )
    assert result["kind"] is None
    assert result["layer"] == "L3"
    assert "pending_layer" not in result
    assert "no anchors" in result["reason"]


def test_l3_classifier_failure_leaves_task_pending_for_retry():
    # classifier returns None (API 5xx / parse failure). Transient; the
    # task stays pending_layer="L3" so the caller may retry.
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__"],
        instruction="Anything.",
    )
    classifier = _make_classifier(None)
    instance = {"site_url": "https://gitlab.local"}
    result = asyncio.run(
        resolve_l3(
            task,
            PLACEHOLDERS,
            instance,
            classifier=classifier,
            probe_fn=_make_probe({}),
        )
    )
    assert result["kind"] is None
    assert result.get("pending_layer") == "L3"


def test_l3_unknown_kind_string_triggers_pending_for_retry():
    task = _gitlab_task(eval_url=None, start_urls=["__GITLAB__"])
    classifier = _make_classifier(
        {"kind": "not_a_real_kind", "probe_query": {"api": "none"}, "confidence": 1.0}
    )
    instance = {"site_url": "https://gitlab.local"}
    result = asyncio.run(
        resolve_l3(
            task,
            PLACEHOLDERS,
            instance,
            classifier=classifier,
            probe_fn=_make_probe({}),
        )
    )
    assert result["kind"] is None
    assert result.get("pending_layer") == "L3"


def test_l3_reddit_submission_via_forum_title_match():
    task = _reddit_task(
        eval_url=None,
        start_urls=["__REDDIT__"],
        instruction=("Find the top-rated post about lithography in the singularity forum."),
    )
    classifier = _make_classifier(
        {
            "kind": "reddit_submission",
            "probe_query": {
                "api": "find_submission_by_title",
                "forum_name": "singularity",
                "query": "lithography",
            },
            "confidence": 0.87,
        }
    )
    probe = _make_probe({"forum_name": "singularity", "submission_id": "69404"})
    instance = {"site_url": "https://reddit.local"}
    result = asyncio.run(
        resolve_l3(task, PLACEHOLDERS, instance, classifier=classifier, probe_fn=probe)
    )
    assert result["kind"] == "reddit_submission"
    assert result["anchors"]["submission_id"] == "69404"
    assert result["layer"] == "L3"
    assert result["source_listing_kind"] == "reddit_dashboard_list"
    assert result["benign_read_url"] == "https://reddit.local/f/singularity"
    assert result["seeded_detail_url"] == "https://reddit.local/f/singularity/69404"
    assert "transition_forced_by_task" not in result


def test_l3_reddit_reply_task_marks_submission_transition_forced():
    task = _reddit_task(
        eval_url=None,
        start_urls=["__REDDIT__"],
        instruction='Reply to the newest post in books forum with "thanks"',
    )
    classifier = _make_classifier(
        {
            "kind": "reddit_submission",
            "probe_query": {
                "api": "list_forum_submissions_recent",
                "forum_name": "books",
                "limit": 1,
            },
            "confidence": 0.91,
        }
    )
    probe = _make_probe({"forum_name": "books", "submission_id": "59421"})
    instance = {"site_url": "https://reddit.local"}
    result = asyncio.run(
        resolve_l3(task, PLACEHOLDERS, instance, classifier=classifier, probe_fn=probe)
    )

    assert result["kind"] == "reddit_submission"
    assert result["source_listing_kind"] == "reddit_forum"
    assert result["benign_read_url"] == "https://reddit.local/f/books"
    assert result["seeded_detail_url"] == "https://reddit.local/f/books/59421"
    assert result["transition_forced_by_task"] is True


def test_l3_non_wasp_site_short_circuits_before_classifier():
    # classifier must not be called when the site is out of WASP scope.
    called = False

    async def _classifier_counter(task, placeholders):
        nonlocal called
        called = True
        return None

    task = {
        "id": "x",
        "sites": ["shopping"],
        "start_urls": ["__SHOPPING__"],
        "reward_function": {"eval": []},
    }
    result = asyncio.run(
        resolve_l3(
            task,
            PLACEHOLDERS,
            {"site_url": "https://whatever"},
            classifier=_classifier_counter,
            probe_fn=_make_probe({}),
        )
    )
    assert result["kind"] is None
    assert called is False


def test_l3_dashboard_probe_api_maps_to_gitlab_dashboard_anchor():
    task = _gitlab_task(
        task_id="t_dash",
        eval_url=None,
        start_urls=["__GITLAB__"],
        instruction="Show my todos.",
    )
    classifier = _make_classifier(
        {
            "kind": "gitlab_dashboard_list",
            "probe_query": {"api": "list_user_todos"},
            "confidence": 0.91,
        }
    )
    result = asyncio.run(
        resolve_l3(
            task,
            PLACEHOLDERS,
            {"site_url": "https://gitlab.local", "auth": {"type": "http_headers", "headers": {}}},
            classifier=classifier,
            probe_fn=None,
        )
    )
    assert result["kind"] == "gitlab_dashboard_list"
    assert result["anchors"]["dashboard"] == "todos"
    assert result["layer"] == "L3"


def test_l3_reddit_dashboard_probe_api_maps_to_dashboard_anchor():
    task = _reddit_task(
        task_id="t_dash",
        eval_url=None,
        start_urls=["__REDDIT__"],
        instruction="Check my submitted posts.",
    )
    classifier = _make_classifier(
        {
            "kind": "reddit_dashboard_list",
            "probe_query": {"api": "list_user_submitted"},
            "confidence": 0.88,
        }
    )
    result = asyncio.run(
        resolve_l3(
            task,
            PLACEHOLDERS,
            {"site_url": "https://reddit.local", "auth": {"type": "http_headers", "headers": {}}},
            classifier=classifier,
            probe_fn=None,
        )
    )
    assert result["kind"] == "reddit_dashboard_list"
    assert result["anchors"]["dashboard"] == "submitted"
    assert result["layer"] == "L3"


def test_resolve_l3_returns_stub_when_probe_raises():
    task = _gitlab_task(
        task_id="t1",
        eval_url=None,
        start_urls=["__GITLAB__"],
        instruction="Show my todos.",
    )
    classifier = _make_classifier(
        {
            "kind": "gitlab_dashboard_list",
            "probe_query": {"api": "list_user_todos"},
            "confidence": 0.9,
        }
    )

    async def boom(*args, **kwargs):
        raise RuntimeError("missing benign auth")

    result = asyncio.run(
        resolve_l3(
            task,
            PLACEHOLDERS,
            {"site_url": "https://gitlab.local"},
            classifier=classifier,
            probe_fn=boom,
        )
    )
    assert result["kind"] is None
    assert result["pending_layer"] == "L3"
    assert "L3 probe raised" in result["reason"]


def test_benign_probe_instance_drops_api_auth_when_benign_auth_present():
    instance = {
        "site_url": "https://gitlab.local",
        "auth": {"type": "http_headers", "headers": {"X-Test": "benign"}},
        "api_auth": {"type": "bearer_token", "token": "privileged"},
    }

    probe_instance = _benign_probe_instance(instance)

    assert "api_auth" not in probe_instance
    assert probe_instance["auth"]["headers"]["X-Test"] == "benign"


def test_benign_probe_instance_rejects_api_auth_only():
    instance = {
        "site_url": "https://gitlab.local",
        "api_auth": {"type": "bearer_token", "token": "privileged"},
    }

    with pytest.raises(RuntimeError, match="no benign auth"):
        _benign_probe_instance(instance)


def test_benign_probe_instance_preserves_benign_bearer_token():
    instance = {
        "site_url": "https://gitlab.local",
        "auth": {"type": "bearer_token", "token": "benign"},
        "api_auth": {"type": "bearer_token", "token": "privileged"},
    }

    probe_instance = _benign_probe_instance(instance)

    assert probe_instance["auth"]["token"] == "benign"
    assert "api_auth" not in probe_instance


# --- L4 dynamic listing expansion ----------------------------------------

from worldsim.phases.phase_2_target_resolver import resolve_l4  # noqa: E402


def _make_listing_probe(items):
    async def _stub(resource, task, instance):
        return items

    return _stub


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

    monkeypatch.setattr(resolver, "_probe_http_json", fake_probe)
    monkeypatch.setattr(resolver, "_gitlab_visible_dashboard_hrefs", fake_hrefs)
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
        resolver._list_gitlab_dashboard(
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
    assert resolver._literalize_regex_value(value) == expected


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

    monkeypatch.setattr(resolver, "_probe_http_json", fake_probe)
    monkeypatch.setattr(resolver, "_gitlab_visible_dashboard_hrefs", fake_hrefs)

    task = _gitlab_task(
        eval_url="__GITLAB__/dashboard/merge_requests?state=^(opened|)$&scope=^(all|)$",
    )

    records = asyncio.run(
        resolver._list_gitlab_dashboard(
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

    monkeypatch.setattr(resolver, "_probe_http_json", fake_probe)
    monkeypatch.setattr(resolver, "_gitlab_visible_dashboard_hrefs", fake_hrefs)
    records = asyncio.run(
        resolver._list_gitlab_dashboard(
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
        "worldsim.phases.phase_2_target_resolver._default_listing_probe",
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


# Real live tests live in tests/integration/test_phase_2_target_resolver_live.py
# behind pytest.mark.live_l3 (skipped by default; run with -m live_l3).


# ---------------------------------------------------------------------
# Anchor / contract conformance self-check (commit 3 of the registry work)
# ---------------------------------------------------------------------

from worldsim.phases.phase_2_target_resolver import (  # noqa: E402
    ResolverContractDriftError,
    _assert_anchor_contract_conformance,
)


class TestAnchorContractConformance:
    def test_none_kind_passes(self) -> None:
        _assert_anchor_contract_conformance({"kind": None, "anchors": {}})

    def test_missing_kind_passes(self) -> None:
        # Pending/empty records without a kind field trivially pass.
        _assert_anchor_contract_conformance({"anchors": {}})

    def test_known_kind_passes(self) -> None:
        _assert_anchor_contract_conformance(
            {
                "kind": "gitlab_issue",
                "anchors": {"project_path": "foo/bar", "issue_iid": "42"},
            }
        )

    def test_all_real_resolver_kinds_pass(self) -> None:
        # Every ResourceKind emitted by the resolver must be addressable by
        # at least one registered editor method. If this fails, the
        # resolver and the editor contracts have drifted.
        for kind in (
            "gitlab_issue",
            "gitlab_mr",
            "gitlab_search_result",
            "gitlab_dashboard_list",
            "reddit_submission",
            "reddit_forum",
            "reddit_dashboard_list",
        ):
            _assert_anchor_contract_conformance({"kind": kind, "anchors": {}})

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ResolverContractDriftError, match="no editor method"):
            _assert_anchor_contract_conformance(
                {"kind": "synthetic_never_registered_kind", "anchors": {}}
            )

    def test_derive_benign_target_resource_honors_conformance(self) -> None:
        # Running the real L1 path on a well-formed task must not raise.
        task = _gitlab_task(eval_url="__GITLAB__/byteblaze/a11yproject/-/issues/17")
        record = derive_benign_target_resource(task, PLACEHOLDERS)
        assert record["kind"] == "gitlab_issue"


class TestPostmillCommentCountFromHtml:
    def test_counts_underscore_ids_like_live_postmill(self) -> None:
        html = "\n".join(f'<article id="comment_{i}"></article>' for i in range(1, 202))
        assert _postmill_submission_comment_count_from_html(html) == 201

    def test_counts_hyphen_ids_backward_compatible(self) -> None:
        html = '<div id="comment-1"></div><div id="comment-2"></div>'
        assert _postmill_submission_comment_count_from_html(html) == 2

    def test_strong_header_fallback_when_no_ids(self) -> None:
        html = (
            "<header><h1>Thread</h1><strong>201 comments</strong></header>"
            "<p>No comment nodes in this fixture.</p>"
        )
        assert _postmill_submission_comment_count_from_html(html) == 201

    def test_strong_singular_comment(self) -> None:
        html = "<div><strong>1 comment</strong></div>"
        assert _postmill_submission_comment_count_from_html(html) == 1

    def test_prefers_id_count_over_strong_when_both_present(self) -> None:
        html = '<article id="comment_1"></article><strong>201 comments</strong>'
        assert _postmill_submission_comment_count_from_html(html) == 1


class TestResolveTasks:
    def test_l1_l2_only_path_is_offline(self) -> None:
        """``allow_layers=('L1','L2')`` runs the sync resolver over the
        batch and returns a single-record list per task. No instance
        required; classifier/probe never touched."""
        tasks = [
            _gitlab_task(
                task_id="t1",
                eval_url="__GITLAB__/byteblaze/a11yproject/-/issues/17",
            ),
            _gitlab_task(
                task_id="t2",
                eval_url="__GITLAB__/root/metaseq/-/merge_requests/3",
            ),
        ]
        out = asyncio.run(resolve_tasks(tasks, PLACEHOLDERS, None, allow_layers=("L1", "L2")))
        assert set(out) == {"t1", "t2"}
        assert out["t1"][0]["kind"] == "gitlab_issue"
        assert out["t2"][0]["kind"] == "gitlab_mr"
        for records in out.values():
            assert len(records) == 1

    def test_instance_required_for_l3_or_l4(self) -> None:
        tasks = [_gitlab_task(task_id="t1", eval_url="__GITLAB__/a/b/-/issues/1")]
        with pytest.raises(ValueError, match="instance is required"):
            asyncio.run(resolve_tasks(tasks, PLACEHOLDERS, None))
        with pytest.raises(ValueError, match="instance is required"):
            asyncio.run(resolve_tasks(tasks, PLACEHOLDERS, None, allow_layers=("L1", "L2", "L3")))

    def test_l3_fallback_runs_only_for_pending_layer_tasks(self) -> None:
        """Tasks that L1/L2 resolved keep their record; intent-only tasks
        (L1/L2 emit ``pending_layer='L3'``) get routed to resolve_l3."""
        classifier_calls: list[str] = []

        async def classifier(task, placeholders):
            classifier_calls.append(str(task.get("id")))
            return {
                "kind": "gitlab_issue",
                "probe_query": {"api": "search_user_issues", "query": "x"},
                "confidence": 0.9,
            }

        probe = _make_probe(
            {"project_id": "159", "project_path": "byteblaze/design", "issue_iid": "104"}
        )

        tasks = [
            _gitlab_task(
                task_id="t_concrete",
                eval_url="__GITLAB__/a/b/-/issues/5",
            ),
            _gitlab_task(
                task_id="t_intent",
                eval_url=None,
                start_urls=["__GITLAB__"],
                instruction="Find latest issue with theme editor in title.",
            ),
        ]
        instance = {"site_url": "https://gitlab.local"}
        out = asyncio.run(
            resolve_tasks(
                tasks,
                PLACEHOLDERS,
                instance,
                allow_layers=("L1", "L2", "L3"),
                classifier=classifier,
                probe_fn=probe,
            )
        )
        assert classifier_calls == ["t_intent"]
        assert out["t_concrete"][0]["layer"] in ("L1", "L2")
        assert out["t_intent"][0]["layer"] == "L3"
        assert out["t_intent"][0]["anchors"]["issue_iid"] == "104"

    def test_reddit_submission_busy_thread_is_filtered_with_live_instance(self) -> None:
        async def comment_count(instance, forum_name, submission_id):
            assert forum_name == "books"
            assert submission_id == "59421"
            return 402

        tasks = [
            _reddit_task(
                task_id="busy",
                eval_url="__REDDIT__/f/books/59421/-/comment",
            )
        ]
        out = asyncio.run(
            resolve_tasks(
                tasks,
                PLACEHOLDERS,
                {"site_url": "https://reddit.local"},
                allow_layers=("L1", "L2"),
                reddit_comment_count_fn=comment_count,
            )
        )

        record = out["busy"][0]
        assert record["kind"] is None
        assert record["reason"] == "reddit_submission_too_many_existing_comments_for_visible_seed"
        assert record["reddit_existing_comment_count"] == 402

    def test_reddit_submission_short_thread_records_visibility_budget(self) -> None:
        async def comment_count(instance, forum_name, submission_id):
            return 14

        tasks = [
            _reddit_task(
                task_id="short",
                eval_url="__REDDIT__/f/television/70886/-/comment",
            )
        ]
        out = asyncio.run(
            resolve_tasks(
                tasks,
                PLACEHOLDERS,
                {"site_url": "https://reddit.local"},
                allow_layers=("L1", "L2"),
                reddit_comment_count_fn=comment_count,
            )
        )

        requirements = out["short"][0]["encounter_requirements"]
        assert out["short"][0]["kind"] == "reddit_submission"
        assert requirements["existing_comment_count"] == 14
        assert requirements["max_existing_comments_for_comment_seed"] == 20

    def test_l3_stub_record_on_classifier_failure_keeps_task_in_output(self) -> None:
        """Classifier returning ``None`` yields a stub record with
        ``pending_layer='L3'``; the task stays in the output so the
        downstream eligibility filter can drop it on kind=None (the
        dispatcher must not swallow failures silently)."""
        tasks = [
            _gitlab_task(
                task_id="t1",
                eval_url=None,
                start_urls=["__GITLAB__"],
                instruction="Anything intent-only.",
            )
        ]
        instance = {"site_url": "https://gitlab.local"}
        out = asyncio.run(
            resolve_tasks(
                tasks,
                PLACEHOLDERS,
                instance,
                allow_layers=("L1", "L2", "L3"),
                classifier=_make_classifier(None),
                probe_fn=_make_probe({}),
            )
        )
        assert "t1" in out
        assert out["t1"][0]["kind"] is None
        assert out["t1"][0].get("pending_layer") == "L3"

    def test_l3_raised_exception_is_isolated_to_task(self) -> None:
        """One task's L3 exception must not take down the batch."""

        async def _boom(task, placeholders):
            raise RuntimeError("classifier broke")

        tasks = [
            _gitlab_task(
                task_id="t_ok",
                eval_url="__GITLAB__/a/b/-/issues/1",
            ),
            _gitlab_task(
                task_id="t_boom",
                eval_url=None,
                start_urls=["__GITLAB__"],
                instruction="Intent-only.",
            ),
        ]
        instance = {"site_url": "https://gitlab.local"}
        out = asyncio.run(
            resolve_tasks(
                tasks,
                PLACEHOLDERS,
                instance,
                allow_layers=("L1", "L2", "L3"),
                classifier=_boom,
                probe_fn=_make_probe({}),
            )
        )
        assert out["t_ok"][0]["kind"] == "gitlab_issue"
        assert out["t_boom"][0]["kind"] is None
        assert "L3 raised" in out["t_boom"][0]["reason"]

    def test_l4_expansion_multiplies_records_per_task(self) -> None:
        """L3 classifies a dashboard listing; L4 fans it out to N."""

        async def classifier(task, placeholders):
            return {
                "kind": "gitlab_dashboard_list",
                "probe_query": {"api": "list_user_todos"},
                "confidence": 0.9,
            }

        probe = _make_probe({"dashboard": "todos"})
        items = [
            {
                "_item_kind": "gitlab_issue",
                "project_id": i,
                "iid": i * 10,
                "web_url": f"https://gitlab.local/a/b/-/issues/{i * 10}",
                "title": f"item {i}",
            }
            for i in range(1, 4)
        ]

        tasks = [
            _gitlab_task(
                task_id="t_dash",
                eval_url=None,
                start_urls=["__GITLAB__"],
                instruction="Show my todos.",
            )
        ]
        instance = {"site_url": "https://gitlab.local"}
        out = asyncio.run(
            resolve_tasks(
                tasks,
                PLACEHOLDERS,
                instance,
                allow_layers=("L1", "L2", "L3", "L4"),
                classifier=classifier,
                probe_fn=probe,
                listing_probe_fn=_make_listing_probe(items),
            )
        )
        assert len(out["t_dash"]) == 3
        assert all(r["layer"] == "L4" for r in out["t_dash"])
        assert all(r["kind"] == "gitlab_issue" for r in out["t_dash"])

    def test_l4_empty_omits_task_from_output(self) -> None:
        """Listing with zero items → task excluded entirely (no stub)."""

        async def classifier(task, placeholders):
            return {
                "kind": "gitlab_search_result",
                "probe_query": {"api": "search_project_issues"},
                "confidence": 0.8,
            }

        tasks = [
            _gitlab_task(
                task_id="t_search",
                eval_url=None,
                start_urls=["__GITLAB__"],
                instruction="Search for nothing.",
            )
        ]
        instance = {"site_url": "https://gitlab.local"}
        out = asyncio.run(
            resolve_tasks(
                tasks,
                PLACEHOLDERS,
                instance,
                allow_layers=("L1", "L2", "L3", "L4"),
                classifier=classifier,
                probe_fn=_make_probe({"query": "nothing"}),
                listing_probe_fn=_make_listing_probe([]),
            )
        )
        assert "t_search" not in out

    def test_concurrency_bound_is_respected(self) -> None:
        """Both L3 and L4 use asyncio.Semaphore; verify the dispatcher
        does not launch more than ``l3_concurrency`` classifier calls at
        once even with many pending-L3 tasks."""
        peak = 0
        in_flight = 0
        lock = asyncio.Lock()

        async def classifier(task, placeholders):
            nonlocal in_flight, peak
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            async with lock:
                in_flight -= 1
            return {
                "kind": "gitlab_issue",
                "probe_query": {"api": "search_user_issues", "query": "x"},
                "confidence": 0.9,
            }

        tasks = [
            _gitlab_task(
                task_id=f"t{i}",
                eval_url=None,
                start_urls=["__GITLAB__"],
                instruction=f"intent {i}",
            )
            for i in range(10)
        ]
        instance = {"site_url": "https://gitlab.local"}
        asyncio.run(
            resolve_tasks(
                tasks,
                PLACEHOLDERS,
                instance,
                allow_layers=("L1", "L2", "L3"),
                l3_concurrency=2,
                classifier=classifier,
                probe_fn=_make_probe(
                    {
                        "project_id": "1",
                        "project_path": "a/b",
                        "issue_iid": "1",
                    }
                ),
            )
        )
        assert peak <= 2, f"L3 concurrency bound violated: peak={peak}"

    def test_preserves_input_task_order(self) -> None:
        tasks = [
            _gitlab_task(
                task_id=f"t{i}",
                eval_url=f"__GITLAB__/a/b/-/issues/{i}",
            )
            for i in range(5)
        ]
        out = asyncio.run(resolve_tasks(tasks, PLACEHOLDERS, None, allow_layers=("L1", "L2")))
        assert list(out.keys()) == ["t0", "t1", "t2", "t3", "t4"]

    def test_empty_task_id_is_dropped(self) -> None:
        tasks = [
            _gitlab_task(task_id="", eval_url="__GITLAB__/a/b/-/issues/1"),
            _gitlab_task(task_id="t1", eval_url="__GITLAB__/a/b/-/issues/2"),
        ]
        out = asyncio.run(resolve_tasks(tasks, PLACEHOLDERS, None, allow_layers=("L1", "L2")))
        assert list(out.keys()) == ["t1"]


# -----------------------------------------------------------------------
# start_url_resolved reconstruction (Bug A) — Phase 2c anchor-vs-probe
# alignment. The probe must navigate to the concrete entity where the
# seed lives, not whatever project root the benign task's raw
# start_urls[0] happens to carry.
# -----------------------------------------------------------------------


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


def test_l3_pending_record_keeps_raw_start_url():
    # Bare __GITLAB__ with no eval URL → L3-pending; reconstruction
    # cannot run (no anchors yet) and resolved_start is preserved.
    task = _gitlab_task(eval_url=None, start_urls=["__GITLAB__"])
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result.get("pending_layer") == "L3"
    assert result["start_url_resolved"] == "https://gitlab.local"


def test_l4_item_record_reconstructs_start_url_strips_localhost_prefix():
    # L4 probe returns items whose project_path comes from `web_url`
    # parsing at :line 1234 — that produces bare "byteblaze/dotfiles".
    # But resource.anchors.project_path can also carry an authority
    # prefix like "localhost:8023/byteblaze/a11y-webring.club" (observed
    # in the 0/107 feasibility report). _clean_project_path strips it.
    from worldsim.phases.phase_2_target_resolver import _project_item_to_record

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


def test_l4_item_record_without_placeholders_preserves_base_url():
    # Backwards compat: old callers that haven't been updated to pass
    # placeholders through continue to produce the pre-fix listing-URL
    # behavior.
    from worldsim.phases.phase_2_target_resolver import _project_item_to_record

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


def test_reconstruction_helper_handles_unknown_kind():
    from worldsim.phases.phase_2_target_resolver import _reconstruct_start_url_from_anchors

    assert (
        _reconstruct_start_url_from_anchors("gitlab", "not_a_kind", {"foo": "bar"}, PLACEHOLDERS)
        is None
    )


def test_reconstruction_helper_returns_none_when_anchors_insufficient():
    from worldsim.phases.phase_2_target_resolver import _reconstruct_start_url_from_anchors

    assert (
        _reconstruct_start_url_from_anchors(
            "gitlab", "gitlab_issue", {"project_path": "a/b"}, PLACEHOLDERS
        )
        is None
    )


# --- _canonicalize_project_path -----------------------------------------
#
# The L3 LLM sometimes emits project_path values like
# 'localhost:8023/a11yproject/a11yproject.com' because the API probe's
# `web_url` carries the localhost authority. The GitLab project-by-path
# endpoint requires a host-stripped path that we then percent-encode.
# These tests pin the host-stripping behavior so encoding bugs surface
# at unit-test time, not at probe time.


def test_canonicalize_strips_localhost_authority():
    from worldsim.phases.phase_2_target_resolver import _canonicalize_project_path

    assert (
        _canonicalize_project_path("localhost:8023/a11yproject/a11yproject.com")
        == "a11yproject/a11yproject.com"
    )


def test_canonicalize_strips_full_https_url():
    from worldsim.phases.phase_2_target_resolver import _canonicalize_project_path

    assert _canonicalize_project_path("https://gitlab.local/primer/design") == "primer/design"


def test_canonicalize_strips_bare_gitlab_local():
    from worldsim.phases.phase_2_target_resolver import _canonicalize_project_path

    assert _canonicalize_project_path("gitlab.local/foo/bar") == "foo/bar"


def test_canonicalize_preserves_subgroup_paths():
    from worldsim.phases.phase_2_target_resolver import _canonicalize_project_path

    assert _canonicalize_project_path("namespace/subgroup/project") == "namespace/subgroup/project"


def test_canonicalize_idempotent_on_clean_path():
    from worldsim.phases.phase_2_target_resolver import _canonicalize_project_path

    canonical = "primer/design"
    assert _canonicalize_project_path(canonical) == canonical
    assert _canonicalize_project_path(_canonicalize_project_path(canonical)) == canonical


def test_canonicalize_handles_empty_input():
    from worldsim.phases.phase_2_target_resolver import _canonicalize_project_path

    assert _canonicalize_project_path("") == ""
    assert _canonicalize_project_path("   ") == ""


def test_canonicalize_strips_leading_and_trailing_slashes():
    from worldsim.phases.phase_2_target_resolver import _canonicalize_project_path

    assert _canonicalize_project_path("/primer/design/") == "primer/design"


# --- L3 out_of_scope_for_option_a kind -----------------------------------
#
# Bucket C of the GitLab attrition (18 unique tasks) was the L3 LLM
# being forced by `tool_choice: "tool"` to pick a kind even on commit-
# count / blob-view / fork-action tasks. Adding `out_of_scope_for_option_a`
# to the kind enum gives the LLM a clean abstain branch. The resolver
# must treat it as terminal: kind=None, layer=L3, no L4 retry.


def test_l3_out_of_scope_kind_is_clean_terminal():
    task = _gitlab_task(
        eval_url=None,
        start_urls=["__GITLAB__"],
        instruction="How many commits did kilian make to a11yproject on 2023-03-05?",
    )
    classifier = _make_classifier(
        {
            "kind": "out_of_scope_for_option_a",
            "probe_query": {
                "api": "none",
                "note": "commit-history task; rendered surface is /-/commits",
            },
            "confidence": 0.95,
        }
    )
    probe_called = {"n": 0}

    async def _probe_should_not_run(*args, **kwargs):
        probe_called["n"] += 1
        return None

    instance = {"site_url": "https://gitlab.local"}
    result = asyncio.run(
        resolve_l3(
            task, PLACEHOLDERS, instance, classifier=classifier, probe_fn=_probe_should_not_run
        )
    )
    assert result["kind"] is None
    assert result["layer"] == "L3"
    assert "pending_layer" not in result, "out_of_scope must not retry at L4"
    assert result.get("l3_out_of_scope") is True
    assert probe_called["n"] == 0
    assert "out_of_scope_for_option_a" in result["reason"]


# --- L3 probe-kind coherence check --------------------------------------
#
# The L3 LLM sometimes emits a (kind, probe_query.api) pair where the
# probe's result shape can't fill the kind's anchor schema. We catch the
# mismatch before running the probe so the failure is diagnostic rather
# than a silent "no anchors" log line.


def test_l3_coherence_blocks_dashboard_kind_with_project_probe():
    # gitlab_dashboard_list anchors only allow {dashboard: ...}; running
    # list_project_issues_recent would silently produce zero anchors.
    task = _gitlab_task(eval_url=None, instruction="Show my open MRs")
    classifier = _make_classifier(
        {
            "kind": "gitlab_dashboard_list",
            "probe_query": {
                "api": "list_project_issues_recent",
                "project_path": "primer/design",
            },
            "confidence": 0.7,
        }
    )
    probe_called = {"n": 0}

    async def _probe_should_not_run(*args, **kwargs):
        probe_called["n"] += 1
        return None

    instance = {"site_url": "https://gitlab.local"}
    result = asyncio.run(
        resolve_l3(
            task, PLACEHOLDERS, instance, classifier=classifier, probe_fn=_probe_should_not_run
        )
    )
    assert result["kind"] is None
    assert result.get("pending_layer") == "L3"
    assert "probe-kind mismatch" in result["reason"]
    assert probe_called["n"] == 0


def test_l3_coherence_admits_user_dashboard_probe_for_dashboard_kind():
    # list_user_todos is the canonical probe for gitlab_dashboard_list.
    task = _gitlab_task(eval_url=None, instruction="Show my todos")
    classifier = _make_classifier(
        {
            "kind": "gitlab_dashboard_list",
            "probe_query": {"api": "list_user_todos"},
            "confidence": 0.9,
        }
    )

    async def _probe(probe_query, task, instance, placeholders):
        return {"dashboard": "todos"}

    instance = {"site_url": "https://gitlab.local"}
    result = asyncio.run(
        resolve_l3(task, PLACEHOLDERS, instance, classifier=classifier, probe_fn=_probe)
    )
    assert result["layer"] == "L3"
    assert result["kind"] == "gitlab_dashboard_list"
    assert result["anchors"]["dashboard"] == "todos"


def test_l3_coherence_blocks_none_api_with_concrete_kind():
    # api='none' is the abstain sentinel; pairing it with a concrete kind
    # is itself a mismatch the LLM shouldn't emit.
    task = _gitlab_task(eval_url=None, instruction="Find my issue")
    classifier = _make_classifier(
        {
            "kind": "gitlab_issue",
            "probe_query": {"api": "none", "note": "no idea"},
            "confidence": 0.4,
        }
    )

    async def _probe_should_not_run(*args, **kwargs):
        raise AssertionError("probe must not run on api=none coherence mismatch")

    instance = {"site_url": "https://gitlab.local"}
    result = asyncio.run(
        resolve_l3(
            task, PLACEHOLDERS, instance, classifier=classifier, probe_fn=_probe_should_not_run
        )
    )
    assert result["kind"] is None
    assert "probe-kind mismatch" in result["reason"]


# --- L3 classifier failure includes class name --------------------------


def test_l3_classifier_failure_records_exception_class_name():
    """When the classifier returns None, resolve_l3 reads the contextvar
    and includes the exception class name in the reason for triage."""
    from worldsim.phases.phase_2_target_resolver import _l3_failure_class_var

    task = _gitlab_task(eval_url=None, start_urls=["__GITLAB__"], instruction="anything")

    async def _classifier_with_failure(task, placeholders):
        # Simulate what _call_anthropic_classifier does on a non-retryable
        # exception (e.g., BadRequestError). It stashes the class name on
        # the contextvar and returns None.
        _l3_failure_class_var.set("BadRequestError")
        return None

    instance = {"site_url": "https://gitlab.local"}
    result = asyncio.run(
        resolve_l3(
            task,
            PLACEHOLDERS,
            instance,
            classifier=_classifier_with_failure,
            probe_fn=_make_probe({}),
        )
    )
    assert "BadRequestError" in result["reason"]
    assert result.get("l3_failure_class") == "BadRequestError"
    assert result.get("pending_layer") == "L3"
