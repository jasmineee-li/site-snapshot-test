# ruff: noqa
# Auto-split from tests/test_phase_2_target_resolver.py; shared helpers live in tests/phase_2/target_resolution/_fixtures.py.
from ._fixtures import *  # noqa: F403,F401

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
