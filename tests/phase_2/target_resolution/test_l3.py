# ruff: noqa
# Auto-split from tests/test_phase_2_target_resolver.py; shared helpers live in tests/phase_2/target_resolution/_fixtures.py.
from ._fixtures import *  # noqa: F403,F401

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

def test_l3_pending_record_keeps_raw_start_url():
    # Bare __GITLAB__ with no eval URL → L3-pending; reconstruction
    # cannot run (no anchors yet) and resolved_start is preserved.
    task = _gitlab_task(eval_url=None, start_urls=["__GITLAB__"])
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result.get("pending_layer") == "L3"
    assert result["start_url_resolved"] == "https://gitlab.local"

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

def test_l3_classifier_failure_records_exception_class_name():
    """When the classifier returns None, resolve_l3 reads the contextvar
    and includes the exception class name in the reason for triage."""
    from worldsim.phase_2.target_resolution.l3 import _l3_failure_class_var

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
