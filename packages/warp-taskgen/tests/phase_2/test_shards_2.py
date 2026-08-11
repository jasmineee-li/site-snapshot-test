# ruff: noqa
# Auto-split from tests/test_phase_2_injections.py; shared helpers live in tests/phase_2/_fixtures.py.
from ._fixtures import *  # noqa: F403,F401


class TestRecoverOrphanedShards:
    """Regression tests for the orphan-shard recovery folded into the
    Phase 2 aggregator — prevents repeat of the 49-orphan drop on the
    current 107-task dataset where one shard re-ran in isolation and
    the earlier persisted sidecars were silently discarded."""

    @staticmethod
    def _plan(task_id: str, site: str = "gitlab") -> dict:
        # Build a placement-valid skeleton so every plan survives the
        # Option A re-validation that orphan recovery now applies. Uses
        # the same {benign_*} token shape that the registry validator
        # requires.
        if site == "gitlab":
            return {
                "id": task_id,
                "site": site,
                "sites": [site],
                "benign_target_resource": {
                    "kind": "gitlab_issue",
                    "anchors": {
                        "project_id": "1",
                        "issue_iid": "1",
                        "project_path": "fixture/project",
                    },
                    "start_url_resolved": "https://gitlab.local/fixture/project/-/issues/1",
                    "layer": "L3",
                },
                "seed_template": {
                    "mechanism": "editor",
                    "editor_calls": [
                        {
                            "benchmark": "webarena_verified",
                            "site": "gitlab",
                            "method": "create_issue_note",
                            "args": {
                                "project_id": "{project_id}",
                                "issue_iid": "{benign_issue_iid}",
                                "body": "{{PAYLOAD_TEXT}}",
                            },
                        }
                    ],
                },
            }
        if site == "reddit":
            return {
                "id": task_id,
                "site": site,
                "sites": [site],
                "benign_target_resource": {
                    "kind": "reddit_submission",
                    "anchors": {"forum_name": "books", "submission_id": "1"},
                    "start_url_resolved": "https://reddit.local/f/books/1",
                    "layer": "L3",
                },
                "seed_template": {
                    "mechanism": "editor",
                    "editor_calls": [
                        {
                            "benchmark": "webarena_verified",
                            "site": "reddit",
                            "method": "create_comment",
                            "args": {
                                "forum_name": "{benign_forum_name}",
                                "submission_id": "{benign_submission_id}",
                                "body": "{{PAYLOAD_TEXT}}",
                            },
                        }
                    ],
                },
            }
        # Out-of-scope sites pass through untouched (recovery filter
        # short-circuits before placement validation).
        return {"id": task_id, "site": site, "sites": [site]}

    def test_merges_disjoint_shards(self, tmp_path: Path):
        shards_dir = tmp_path / "shards"
        shards_dir.mkdir()
        (shards_dir / "gitlab-shard-0.json").write_text(
            json.dumps([self._plan("adv-100"), self._plan("adv-101")])
        )
        (shards_dir / "reddit-shard-0.json").write_text(
            json.dumps([self._plan("adv-200", site="reddit")])
        )
        in_memory: list[dict] = []
        merged, recovered = phase_2_injections._recover_orphaned_shards(
            shards_dir, in_memory, allowed_sites={"gitlab", "reddit"}
        )
        assert {plan["id"] for plan in merged} == {"adv-100", "adv-101", "adv-200"}
        assert recovered == sorted(["adv-100", "adv-101", "adv-200"])

    def test_paused_recovery_requires_exact_run_bound_manifest(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from worldsim.phase_2.pause_control import write_planning_shard_checkpoint
        from worldsim.run_transition import resolve_run_request

        monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
        definition = resolve_run_request(
            {"sandbox_model": "model-a"},
            existing_state=None,
            new_run_id="run-phase2-recovery",
        ).definition
        state = {
            "step": "phase_2",
            "status": "paused",
            "logs_dir": str(tmp_path),
            "phase_2_stage": "planning",
            "run_definition": definition.to_dict(),
        }
        (tmp_path / "pipeline_state.json").write_text(json.dumps(state))
        shards_dir = tmp_path / "phase_2" / "shards"
        shard_path = shards_dir / "shopping-shard-0.json"
        payload = [self._plan("adv-bound", site="shopping")]
        write_planning_shard_checkpoint(
            shard_path,
            payload,
            label="shopping-shard-0",
            input_task_ids=["benign-bound"],
        )

        merged, recovered = phase_2_injections._recover_orphaned_shards(
            shards_dir,
            [],
            allowed_sites={"shopping"},
            required_checkpoint_definition=definition,
        )
        assert [task["id"] for task in merged] == ["adv-bound"]
        assert recovered == ["adv-bound"]

        manifest_path = shard_path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        manifest["definition_digest"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))

        merged, recovered = phase_2_injections._recover_orphaned_shards(
            shards_dir,
            [],
            allowed_sites={"shopping"},
            required_checkpoint_definition=definition,
        )
        assert merged == []
        assert recovered == []

    def test_exact_paused_option_a_shard_rejects_unknown_benign_parent(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from worldsim.phase_2.pause_control import write_planning_shard_checkpoint
        from worldsim.phase_2.shards import _load_reusable_planning_shard
        from worldsim.run_transition import resolve_run_request

        monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
        definition = resolve_run_request(
            {"sandbox_model": "model-a"},
            existing_state=None,
            new_run_id="run-phase2-parent-check",
        ).definition
        state = {
            "step": "phase_2",
            "status": "paused",
            "logs_dir": str(tmp_path),
            "phase_2_stage": "planning",
            "run_definition": definition.to_dict(),
        }
        (tmp_path / "pipeline_state.json").write_text(json.dumps(state))
        shard_path = tmp_path / "phase_2" / "shards" / "gitlab.json"
        payload = [{**self._plan("adv-unknown-parent"), "benign_task_id": "missing"}]
        write_planning_shard_checkpoint(
            shard_path,
            payload,
            label="gitlab",
            input_task_ids=["benign-current"],
        )

        assert (
            _load_reusable_planning_shard(
                shard_path,
                expected_site="gitlab",
                expected_input_task_ids=["benign-current"],
                definition=definition,
                benign_by_id={},
                site_profiles={"gitlab": {}},
            )
            is None
        )

    def test_existing_inmemory_plan_wins_over_shard_copy(self, tmp_path: Path):
        shards_dir = tmp_path / "shards"
        shards_dir.mkdir()
        (shards_dir / "gitlab-shard-0.json").write_text(
            json.dumps(
                [
                    {**self._plan("adv-100"), "marker": "from-shard"},
                    self._plan("adv-101"),
                ]
            )
        )
        in_memory = [{**self._plan("adv-100"), "marker": "from-memory"}]
        merged, recovered = phase_2_injections._recover_orphaned_shards(
            shards_dir, in_memory, allowed_sites={"gitlab"}
        )
        # adv-100 already in memory → shard copy is ignored.
        # adv-101 is the only orphan.
        assert recovered == ["adv-101"]
        adv_100 = next(plan for plan in merged if plan["id"] == "adv-100")
        assert adv_100["marker"] == "from-memory"

    def test_newest_shard_wins_on_cross_shard_collision(self, tmp_path: Path):
        import os
        import time

        shards_dir = tmp_path / "shards"
        shards_dir.mkdir()
        older = shards_dir / "gitlab-shard-0.json"
        older.write_text(json.dumps([{**self._plan("adv-100"), "gen": "old"}]))
        old_mtime = time.time() - 120
        os.utime(older, (old_mtime, old_mtime))

        newer = shards_dir / "gitlab-shard-1.json"
        newer.write_text(json.dumps([{**self._plan("adv-100"), "gen": "new"}]))
        # newer keeps default mtime (now), which exceeds old_mtime.

        merged, recovered = phase_2_injections._recover_orphaned_shards(
            shards_dir, [], allowed_sites={"gitlab"}
        )
        assert recovered == ["adv-100"]
        assert merged[0]["gen"] == "new"

    def test_out_of_scope_sites_are_not_recovered(self, tmp_path: Path):
        shards_dir = tmp_path / "shards"
        shards_dir.mkdir()
        (shards_dir / "shopping-shard-0.json").write_text(
            json.dumps([self._plan("adv-shop-1", site="shopping")])
        )
        (shards_dir / "gitlab-shard-0.json").write_text(json.dumps([self._plan("adv-gl-1")]))
        merged, recovered = phase_2_injections._recover_orphaned_shards(
            shards_dir, [], allowed_sites={"gitlab", "reddit"}
        )
        # shopping is out of the WASP-aligned scope and stays on disk only.
        assert recovered == ["adv-gl-1"]
        assert {plan["id"] for plan in merged} == {"adv-gl-1"}

    def test_missing_shards_dir_returns_input_unchanged(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist"
        in_memory = [self._plan("adv-1")]
        merged, recovered = phase_2_injections._recover_orphaned_shards(
            missing, in_memory, allowed_sites={"gitlab"}
        )
        assert recovered == []
        assert merged == in_memory

    def test_malformed_shard_is_skipped(self, tmp_path: Path):
        shards_dir = tmp_path / "shards"
        shards_dir.mkdir()
        (shards_dir / "gitlab-shard-0.json").write_text("not-json-at-all")
        (shards_dir / "gitlab-shard-1.json").write_text(json.dumps([self._plan("adv-valid")]))
        _, recovered = phase_2_injections._recover_orphaned_shards(
            shards_dir, [], allowed_sites={"gitlab"}
        )
        assert recovered == ["adv-valid"]

    def test_reconstructs_bare_host_start_url_from_anchors(self, tmp_path: Path):
        """Orphans written before Fix A (commit 4b023aea) carry
        `start_url_resolved = "https://reddit.local"` etc. The helper must
        re-run `_reconstruct_start_url_from_anchors` so the probe lands
        at the concrete entity, not the host root."""
        shards_dir = tmp_path / "shards"
        shards_dir.mkdir()
        stale_orphan = {
            "id": "adv-stale",
            "site": "reddit",
            "sites": ["reddit"],
            "benign_target_resource": {
                "kind": "reddit_submission",
                "anchors": {"forum_name": "books", "submission_id": "12345"},
                "start_url_resolved": "https://reddit.local",
            },
            "seed_template": {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "reddit",
                        "method": "create_comment",
                        "args": {
                            "forum_name": "{benign_forum_name}",
                            "submission_id": "{benign_submission_id}",
                            "body": "{{PAYLOAD_TEXT}}",
                        },
                    }
                ],
            },
        }
        (shards_dir / "reddit-shard-0.json").write_text(json.dumps([stale_orphan]))

        merged, recovered = phase_2_injections._recover_orphaned_shards(
            shards_dir, [], allowed_sites={"reddit"}
        )
        assert recovered == ["adv-stale"]
        recovered_url = merged[0]["benign_target_resource"]["start_url_resolved"]
        # Must escape the host root and point at the concrete entity.
        assert recovered_url != "https://reddit.local"
        assert "/f/books/12345" in recovered_url

    def test_backfills_project_name_template_from_path(self, tmp_path: Path):
        """Orphan shards from pre-template-standardization runs carry
        ``project_path_template`` on editor_calls[].args but not the
        paired ``project_name_template`` that GitLab's editor
        arg-validator requires. Recovery must derive the name template
        from the path's leaf so Phase 2c doesn't fail these orphans with
        ``invalid_args: project_id or project_name_template is required``.
        """
        shards_dir = tmp_path / "shards"
        shards_dir.mkdir()
        orphan = {
            **self._plan("adv-name-backfill"),
            "benign_target_resource": {
                "kind": "gitlab_issue",
                "start_url_resolved": "https://gitlab.local",
                "anchors": {
                    "project_path": "a11yproject/a11yproject.com",
                    "issue_iid": 1064,
                },
            },
            # Placement-valid seed_template that references
            # {benign_project_path} (reachable from the override anchors).
            "seed_template": {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "gitlab",
                        "method": "create_issue_note",
                        "args": {
                            "project_path_template": "{benign_project_path}",
                            "issue_iid": "{benign_issue_iid}",
                            "body": "{{PAYLOAD_TEXT}}",
                        },
                    }
                ],
            },
            "adversarial_data_seed": {
                "editor_calls": [
                    {
                        "method": "create_issue_note",
                        "args": {
                            "project_path_template": "a11yproject/a11yproject.com",
                            # project_name_template intentionally missing.
                        },
                    }
                ],
            },
        }
        (shards_dir / "gitlab-shard-0.json").write_text(json.dumps([orphan]))
        merged, recovered = phase_2_injections._recover_orphaned_shards(
            shards_dir, [], allowed_sites={"gitlab"}
        )
        assert recovered == ["adv-name-backfill"]
        recovered_args = merged[0]["adversarial_data_seed"]["editor_calls"][0]["args"]
        assert recovered_args["project_path_template"] == "a11yproject/a11yproject.com"
        assert recovered_args["project_name_template"] == "a11yproject.com"

    def test_preserves_existing_project_name_template(self, tmp_path: Path):
        """Backfill must not stomp an already-populated template."""
        shards_dir = tmp_path / "shards"
        shards_dir.mkdir()
        orphan = {
            **self._plan("adv-already-named"),
            "benign_target_resource": {
                "kind": "gitlab_issue",
                "start_url_resolved": "https://gitlab.local",
                "anchors": {
                    "project_path": "byteblaze/dotfiles",
                    "issue_iid": 7,
                },
            },
            "seed_template": {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "gitlab",
                        "method": "create_issue_note",
                        "args": {
                            "project_path_template": "{benign_project_path}",
                            "issue_iid": "{benign_issue_iid}",
                            "body": "{{PAYLOAD_TEXT}}",
                        },
                    }
                ],
            },
            "adversarial_data_seed": {
                "editor_calls": [
                    {
                        "method": "create_issue_note",
                        "args": {
                            "project_path_template": "byteblaze/dotfiles",
                            "project_name_template": "webagent-task-{salt}",
                        },
                    }
                ],
            },
        }
        (shards_dir / "gitlab-shard-0.json").write_text(json.dumps([orphan]))
        merged, _ = phase_2_injections._recover_orphaned_shards(
            shards_dir, [], allowed_sites={"gitlab"}
        )
        recovered_args = merged[0]["adversarial_data_seed"]["editor_calls"][0]["args"]
        assert recovered_args["project_name_template"] == "webagent-task-{salt}"

    def test_name_backfill_skipped_for_non_gitlab(self, tmp_path: Path):
        """The backfill is gitlab-specific — reddit orphans have no
        project_name_template concept and must not acquire one."""
        shards_dir = tmp_path / "shards"
        shards_dir.mkdir()
        orphan = {
            **self._plan("adv-reddit-passthrough", site="reddit"),
            "benign_target_resource": {
                "kind": "reddit_submission",
                "start_url_resolved": "https://reddit.local/f/books/12345",
                "anchors": {"forum_name": "books", "submission_id": "12345"},
            },
            "adversarial_data_seed": {
                "editor_calls": [
                    {
                        "method": "create_comment",
                        "args": {
                            # Pathological payload but must pass through
                            # untouched — reddit does not use this field.
                            "project_path_template": "someone/something",
                        },
                    }
                ],
            },
        }
        (shards_dir / "reddit-shard-0.json").write_text(json.dumps([orphan]))
        merged, _ = phase_2_injections._recover_orphaned_shards(
            shards_dir, [], allowed_sites={"reddit"}
        )
        recovered_args = merged[0]["adversarial_data_seed"]["editor_calls"][0]["args"]
        assert "project_name_template" not in recovered_args

    def test_drops_pre_sunset_api_mechanism_orphans(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """Stale shards from pre-ff8381d5 runs carry
        ``seed_template.mechanism="api"`` with ``api_calls`` instead of
        ``editor_calls``. Without re-validation, those orphans flow
        through to ``adversarial_plans.json`` and crash Phase 2b text
        fill at ``validate_data_seed``. Recovery must drop them and let
        clean editor-mechanism orphans through."""
        shards_dir = tmp_path / "shards"
        shards_dir.mkdir()
        valid = self._plan("adv-valid-editor")
        invalid = {
            "id": "adv-stale-api",
            "site": "gitlab",
            "sites": ["gitlab"],
            "benign_target_resource": {
                "kind": "gitlab_project_milestone",
                "anchors": {
                    "project_path": "kkroening/ffmpeg-python",
                    "milestone_iid": "1",
                },
                "start_url_resolved": (
                    "https://gitlab.local/kkroening/ffmpeg-python/-/milestones/1"
                ),
                "layer": "L2",
            },
            "seed_template": {
                "mechanism": "api",
                "api_calls": [
                    {
                        "method": "PUT",
                        "path": "/api/v4/projects/{project_id}/milestones/{milestone_iid}",
                        "body": {"description": "{{PAYLOAD_TEXT}}"},
                    }
                ],
            },
        }
        (shards_dir / "gitlab-shard-0.json").write_text(json.dumps([valid]))
        (shards_dir / "gitlab-shard-1.json").write_text(json.dumps([invalid]))

        with caplog.at_level("WARNING", logger="worldsim.phases.phase_2_injections"):
            merged, recovered = phase_2_injections._recover_orphaned_shards(
                shards_dir, [], allowed_sites={"gitlab"}
            )

        assert {plan["id"] for plan in merged} == {"adv-valid-editor"}
        assert recovered == ["adv-valid-editor"]
        assert any(
            "skip-on-reject" in record.message and "adv-stale-api" in record.message
            for record in caplog.records
        )

    def test_drops_orphans_failing_contract_with_immutable_field_drift(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """Orphans whose immutable fields drift from the benign parent
        (e.g. ``instruction`` mutated post-validation) pass placement but
        fail the contract validator. Mirror the live two-validator chain
        and drop them with a ``(contract)`` qualifier."""
        shards_dir = tmp_path / "shards"
        shards_dir.mkdir()
        bad = {
            **self._plan("adv-contract-fail"),
            "benign_task_id": "benign-bad",
            "instruction": "drifted instruction not present on benign parent",
        }
        (shards_dir / "gitlab-shard-0.json").write_text(json.dumps([bad]))
        benign_by_id = {"benign-bad": {"id": "benign-bad", "site": "gitlab", "sites": ["gitlab"]}}

        with caplog.at_level("WARNING", logger="worldsim.phases.phase_2_injections"):
            merged, recovered = phase_2_injections._recover_orphaned_shards(
                shards_dir,
                [],
                allowed_sites={"gitlab"},
                benign_by_id=benign_by_id,
                site_profiles={"gitlab": {}},
            )

        assert merged == []
        assert recovered == []
        assert any(
            "skip-on-reject" in record.message
            and "adv-contract-fail" in record.message
            and "(contract)" in record.message
            and "instruction changed from benign task" in record.message
            for record in caplog.records
        )
