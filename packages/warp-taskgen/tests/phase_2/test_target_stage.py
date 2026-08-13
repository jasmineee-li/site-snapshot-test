# ruff: noqa
# Auto-split from tests/test_phase_2_injections.py; shared helpers live in tests/phase_2/_fixtures.py.
from ._fixtures import *  # noqa: F403,F401
from worldsim.phase_2 import target_stage


class TestResolveBenignTargetResourcesForShard:
    """``_resolve_benign_target_resources_for_shard`` is the shim between
    the async resolver dispatcher and the existing dict-shaped
    ``benign_target_resources`` map Phase 2a expects. Covers the no-instance
    fallback, the live-instance happy path, token-failure fallback,
    resolver-exception fallback, and L4 suffixed-ID fan-out."""

    def _gitlab_site_task(self, task_id: str, eval_url: str | None) -> dict:
        task = {
            "id": task_id,
            "site": "gitlab",
            "sites": ["gitlab"],
            "start_urls": ["__GITLAB__"],
            "instruction": "anything",
            "reward_function": {"eval": []},
        }
        if eval_url is not None:
            task["reward_function"]["eval"] = [{"expected": {"url": eval_url}}]
        return task

    def test_no_instance_returns_l1_l2_offline(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
        tasks = [
            self._gitlab_site_task("t1", "__GITLAB__/a/b/-/issues/5"),
            self._gitlab_site_task("t2", "__GITLAB__/a/b/-/merge_requests/9"),
        ]
        expanded, resources = asyncio.run(
            target_stage._resolve_benign_target_resources_for_shard(
                site_tasks=tasks,
                instance=None,
                site_name="gitlab",
                label="test",
            )
        )
        assert expanded == tasks
        assert resources["t1"]["kind"] == "gitlab_issue"
        assert resources["t2"]["kind"] == "gitlab_mr"

    def test_l4_fanout_produces_suffixed_clones(self, tmp_path, monkeypatch):
        """When resolve_tasks returns N > 1 records for a task, the helper
        must clone the benign task N times with suffixed IDs and preserve
        ``source_task_id`` on each clone."""
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))

        tasks = [self._gitlab_site_task("t_dash", None)]

        async def fake_resolve_tasks(*args, **kwargs):
            assert kwargs["allow_layers"] == ("L1", "L2", "L3", "L4")
            return {
                "t_dash": [
                    {
                        "kind": "gitlab_issue",
                        "anchors": {
                            "project_id": str(i),
                            "issue_iid": str(i * 10),
                            "project_path": f"a/b{i}",
                        },
                        "layer": "L4",
                        "attach_surfaces": [],
                        "encounter_requirements": {},
                    }
                    for i in range(1, 4)
                ]
            }

        def fake_acquire(*_, **__):
            return []

        monkeypatch.setattr(target_stage, "resolve_tasks", fake_resolve_tasks)
        monkeypatch.setattr(target_stage, "acquire_tokens_for_instances", fake_acquire)

        expanded, resources = asyncio.run(
            target_stage._resolve_benign_target_resources_for_shard(
                site_tasks=tasks,
                instance={"site_name": "gitlab", "site_url": "https://x"},
                site_name="gitlab",
                label="test",
            )
        )
        assert [t["id"] for t in expanded] == [
            "t_dash_l4_0",
            "t_dash_l4_1",
            "t_dash_l4_2",
        ]
        for clone in expanded:
            assert clone["source_task_id"] == "t_dash"
        assert set(resources) == {
            "t_dash_l4_0",
            "t_dash_l4_1",
            "t_dash_l4_2",
        }
        assert resources["t_dash_l4_0"]["anchors"]["issue_iid"] == "10"

    def test_route_contracted_new_tasks_preserve_l1_l2_resolution(self, tmp_path, monkeypatch):
        """Generated tasks already carry route contracts, so L4 must not
        rewrite a search-route comment task into concrete issue-detail clones."""

        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
        task = self._gitlab_site_task("novel_gitlab_1", None)
        task.update(
            {
                "origin": "new_task",
                "route_id": "gitlab.note_body.gitlab_search_result.create_issue_note",
                "start_urls": ["__GITLAB__/search?search=auth&scope=issues"],
                "instruction": "Find the latest comment on the top issue and report it.",
                "data_seed": {
                    "mechanism": "editor",
                    "editor_calls": [
                        {
                            "benchmark": "webarena_verified",
                            "site": "gitlab",
                            "method": "create_issue_note",
                            "args": {
                                "project_id": "{benign_project_id}",
                                "issue_iid": "{benign_issue_iid}",
                                "body": "Seeded note",
                            },
                        }
                    ],
                },
            }
        )

        async def fake_resolve_tasks(*args, **kwargs):
            return {
                "novel_gitlab_1": [
                    {
                        "kind": "gitlab_issue",
                        "anchors": {
                            "project_id": "1",
                            "issue_iid": str(i),
                            "project_path": "a/b",
                        },
                        "layer": "L4",
                    }
                    for i in range(1, 4)
                ]
            }

        monkeypatch.setattr(target_stage, "resolve_tasks", fake_resolve_tasks)
        monkeypatch.setattr(target_stage, "acquire_tokens_for_instances", lambda *_: [])

        expanded, resources = asyncio.run(
            target_stage._resolve_benign_target_resources_for_shard(
                site_tasks=[task],
                instance={"site_name": "gitlab", "site_url": "https://x"},
                site_name="gitlab",
                label="test",
            )
        )

        assert [item["id"] for item in expanded] == ["novel_gitlab_1"]
        assert resources["novel_gitlab_1"]["kind"] == "gitlab_search_result"
        assert resources["novel_gitlab_1"]["allowed_editor_methods"] == ["create_issue_note"]
        assert resources["novel_gitlab_1"].get("exact_comment_region_forced_by_task") is True

    def test_l4_empty_omits_task_from_shard(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))

        async def fake_resolve_tasks(*args, **kwargs):
            return {}

        def fake_acquire(*_, **__):
            return []

        monkeypatch.setattr(target_stage, "resolve_tasks", fake_resolve_tasks)
        monkeypatch.setattr(target_stage, "acquire_tokens_for_instances", fake_acquire)

        expanded, resources = asyncio.run(
            target_stage._resolve_benign_target_resources_for_shard(
                site_tasks=[self._gitlab_site_task("t_dash", None)],
                instance={"site_name": "gitlab", "site_url": "https://x"},
                site_name="gitlab",
                label="test",
            )
        )

        assert expanded == []
        assert resources == {}

    def test_resolver_exception_falls_back_to_l1_l2(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))

        async def boom(*args, **kwargs):
            raise RuntimeError("classifier API outage")

        def fake_acquire(*_, **__):
            return []

        monkeypatch.setattr(target_stage, "resolve_tasks", boom)
        monkeypatch.setattr(target_stage, "acquire_tokens_for_instances", fake_acquire)

        tasks = [self._gitlab_site_task("t1", "__GITLAB__/a/b/-/issues/5")]
        expanded, resources = asyncio.run(
            target_stage._resolve_benign_target_resources_for_shard(
                site_tasks=tasks,
                instance={"site_name": "gitlab", "site_url": "https://x"},
                site_name="gitlab",
                label="test",
            )
        )
        # Fall back to L1 — same task count, kind resolved offline.
        assert expanded == tasks
        assert resources["t1"]["kind"] == "gitlab_issue"

    def test_token_failure_falls_back_to_l1_l2(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
        monkeypatch.setattr(
            target_stage,
            "acquire_tokens_for_instances",
            lambda *_: ["bad credentials"],
        )
        tasks = [self._gitlab_site_task("t1", "__GITLAB__/a/b/-/issues/5")]
        expanded, resources = asyncio.run(
            target_stage._resolve_benign_target_resources_for_shard(
                site_tasks=tasks,
                instance={"site_name": "gitlab", "site_url": "https://x"},
                site_name="gitlab",
                label="test",
            )
        )
        assert expanded == tasks
        assert resources["t1"]["kind"] == "gitlab_issue"

    def test_token_failure_drops_probe_dependent_listing_kind(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
        monkeypatch.setattr(
            target_stage,
            "acquire_tokens_for_instances",
            lambda *_args, **_kwargs: ["bad credentials"],
        )
        tasks = [
            self._gitlab_site_task(
                "t_search",
                "__GITLAB__/groups/gitlab-org/-/issues?search=theme&scope=all",
            )
        ]
        expanded, resources = asyncio.run(
            target_stage._resolve_benign_target_resources_for_shard(
                site_tasks=tasks,
                instance={
                    "site_name": "gitlab",
                    "site_url": "https://x",
                    "auth": {"type": "bearer_token", "token": ""},
                },
                site_name="gitlab",
                label="test",
            )
        )
        assert expanded == tasks
        assert resources["t_search"]["kind"] is None
        assert "token acquisition failure" in resources["t_search"]["reason"]

    def test_api_auth_without_benign_auth_falls_back_to_l1_l2(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
        tasks = [self._gitlab_site_task("t1", "__GITLAB__/a/b/-/issues/5")]
        expanded, resources = asyncio.run(
            target_stage._resolve_benign_target_resources_for_shard(
                site_tasks=tasks,
                instance={
                    "site_name": "gitlab",
                    "site_url": "https://x",
                    "api_auth": {"type": "bearer_token", "token": "privileged"},
                },
                site_name="gitlab",
                label="test",
            )
        )
        assert expanded == tasks
        assert resources["t1"]["kind"] == "gitlab_issue"

    def test_api_auth_without_benign_auth_drops_probe_dependent_listing_kind(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
        tasks = [
            self._gitlab_site_task(
                "t_search",
                "__GITLAB__/groups/gitlab-org/-/issues?search=theme&scope=all",
            )
        ]
        expanded, resources = asyncio.run(
            target_stage._resolve_benign_target_resources_for_shard(
                site_tasks=tasks,
                instance={
                    "site_name": "gitlab",
                    "site_url": "https://x",
                    "api_auth": {"type": "bearer_token", "token": "privileged"},
                },
                site_name="gitlab",
                label="test",
            )
        )
        assert expanded == tasks
        assert resources["t_search"]["kind"] is None
        assert resources["t_search"]["pending_layer"] == "L3"
        assert "missing benign auth" in resources["t_search"]["reason"]

    def test_api_auth_without_benign_auth_keeps_reddit_dashboard_kind(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
        tasks = [
            {
                "id": "t_dash",
                "site": "reddit",
                "sites": ["reddit"],
                "start_urls": ["__REDDIT__/user/MarvelsGrantMan136/comments"],
                "instruction": "anything",
                "reward_function": {"eval": []},
            }
        ]
        expanded, resources = asyncio.run(
            target_stage._resolve_benign_target_resources_for_shard(
                site_tasks=tasks,
                instance={
                    "site_name": "reddit",
                    "site_url": "https://x",
                    "api_auth": {"type": "bearer_token", "token": "privileged"},
                },
                site_name="reddit",
                label="test",
            )
        )
        assert expanded == tasks
        assert resources["t_dash"]["kind"] == "reddit_dashboard_list"
        assert resources["t_dash"]["anchors"]["dashboard"] == "comments"

    def test_api_auth_without_benign_auth_keeps_reddit_forum_kind(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
        tasks = [
            {
                "id": "t_forum",
                "site": "reddit",
                "sites": ["reddit"],
                "start_urls": ["__REDDIT__/f/deeplearning"],
                "instruction": "Review recent posts in the deeplearning forum.",
                "reward_function": {"eval": []},
            }
        ]
        expanded, resources = asyncio.run(
            target_stage._resolve_benign_target_resources_for_shard(
                site_tasks=tasks,
                instance={
                    "site_name": "reddit",
                    "site_url": "https://x",
                    "api_auth": {"type": "bearer_token", "token": "privileged"},
                },
                site_name="reddit",
                label="test",
            )
        )
        assert expanded == tasks
        assert resources["t_forum"]["kind"] == "reddit_forum"
        assert resources["t_forum"]["anchors"] == {"forum_name": "deeplearning"}

    def test_persists_target_resolution_to_logs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))

        async def fake_resolve_tasks(*args, **kwargs):
            return {
                "t1": [
                    {
                        "kind": "gitlab_issue",
                        "anchors": {
                            "project_id": "1",
                            "issue_iid": "5",
                            "project_path": "a/b",
                        },
                        "layer": "L3",
                    }
                ]
            }

        def fake_acquire(*_, **__):
            return []

        monkeypatch.setattr(target_stage, "resolve_tasks", fake_resolve_tasks)
        monkeypatch.setattr(target_stage, "acquire_tokens_for_instances", fake_acquire)

        asyncio.run(
            target_stage._resolve_benign_target_resources_for_shard(
                site_tasks=[self._gitlab_site_task("t1", None)],
                instance={"site_name": "gitlab", "site_url": "https://x"},
                site_name="gitlab",
                label="test",
            )
        )
        out_file = tmp_path / "phase_2" / "target_resolution" / "gitlab.json"
        assert out_file.exists()
        payload = json.loads(out_file.read_text())
        assert payload["t1"]["layer"] == "L3"

    def test_target_resolution_persistence_merges_existing_shards(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))

        target_stage._persist_target_resolution(
            site_name="gitlab",
            resources={"t1": {"kind": "gitlab_issue", "layer": "L3"}},
        )
        target_stage._persist_target_resolution(
            site_name="gitlab",
            resources={"t2": {"kind": "gitlab_mr", "layer": "L4"}},
        )

        out_file = tmp_path / "phase_2" / "target_resolution" / "gitlab.json"
        payload = json.loads(out_file.read_text())
        assert payload["t1"]["kind"] == "gitlab_issue"
        assert payload["t2"]["kind"] == "gitlab_mr"
