# ruff: noqa
# Auto-split from tests/test_phase_2_injections.py; shared helpers live in tests/phase_2/_fixtures.py.
from .._fixtures import *  # noqa: F403,F401
from warp_taskgen.phase_2 import plan_validation, reuse
from warp_taskgen.phase_2.phase_2c import stage as phase_2_stage


@pytest.mark.asyncio
async def test_phase_2_feasibility_stage_verifies_only_filtered_sites(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    output_dir = tmp_path / "phase_2"
    output_dir.mkdir(parents=True)
    output_path = output_dir / "adversarial_tasks.json"
    reddit_verified = {
        "id": "reddit-verified",
        "benchmark": "webarena_verified",
        "site": "reddit",
        "feasibility": {"status": "verified"},
    }
    shopping_task = _finalized_plan_task()
    shopping_task["id"] = "shopping-task"
    output_path.write_text(json.dumps([reddit_verified, shopping_task]))
    instances_path = tmp_path / "instances.smoke.json"
    instances_path.write_text(
        json.dumps(
            {
                "benchmark_name": "WebArena Verified",
                "instances": [
                    {"site_name": "shopping", "site_url": "http://shopping.test"},
                    {"site_name": "reddit", "site_url": "http://reddit.test"},
                ],
            }
        )
    )

    async def fake_verify_feasibility(path, *args, **kwargs):
        tasks = json.loads(Path(path).read_text())
        assert [task["id"] for task in tasks] == ["shopping-task"]
        assert [instance["site_name"] for instance in kwargs["instances"]] == ["shopping"]
        return phase_2_stage.FeasibilityReport(
            verified=[_with_feasibility_status(tasks[0], "verified")],
            infeasible=[],
            skipped_already_verified=[],
            cleanup_warnings=[],
            host_fingerprint={"host_config": "instances.smoke.json"},
            elapsed_seconds=0.0,
            per_site_counts={},
            phase_2_status="complete",
            dropped_source_data=[],
        )

    monkeypatch.setattr(phase_2_stage, "verify_feasibility", fake_verify_feasibility)

    rc = await phase_2_stage._run_feasibility_stage(
        args=Namespace(
            skip_feasibility=False,
            feasibility_only=True,
            feasibility_instances=str(instances_path),
            feasibility_concurrency=1,
            feasibility_retry_count=0,
            feasibility_ttl_hours=None,
            force_reverify=False,
        ),
        output_path=output_path,
        output_dir=output_dir,
        state_metadata={"feasibility_only": True, "sites": "shopping"},
        prior_phase_2_status="complete",
    )

    assert rc == 0
    output = json.loads(output_path.read_text())
    assert output[0] == reddit_verified
    assert output[1]["id"] == "shopping-task"
    assert output[1]["feasibility"]["status"] == "verified"
    report = json.loads((output_dir / "feasibility_report.json").read_text())
    assert report["verified_count"] == 2


@pytest.mark.asyncio
async def test_phase_2_feasibility_stage_preserves_partial_complete_terminal_status(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    output_dir = tmp_path / "phase_2"
    output_dir.mkdir(parents=True)
    output_path = output_dir / "adversarial_tasks.json"
    output_path.write_text(json.dumps([_finalized_plan_task()]))
    instances_path = tmp_path / "instances.smoke.json"
    instances_path.write_text(
        json.dumps(
            {
                "benchmark_name": "WebArena Verified",
                "instances": [
                    {
                        "site_name": "shopping",
                        "site_url": "http://shopping.test",
                    }
                ],
            }
        )
    )

    async def fake_verify_feasibility(*args, **kwargs):
        return phase_2_stage.FeasibilityReport(
            verified=[
                _with_feasibility_status(task, "verified")
                for task in json.loads(output_path.read_text())
            ],
            infeasible=[],
            skipped_already_verified=[],
            cleanup_warnings=[],
            host_fingerprint={"host_config": "instances.smoke.json"},
            elapsed_seconds=0.0,
            per_site_counts={},
            phase_2_status="partial_complete",
        )

    monkeypatch.setattr(phase_2_stage, "verify_feasibility", fake_verify_feasibility)

    rc = await phase_2_stage._run_feasibility_stage(
        args=Namespace(
            skip_feasibility=False,
            feasibility_only=True,
            feasibility_instances=str(instances_path),
            feasibility_concurrency=1,
            feasibility_retry_count=0,
            feasibility_ttl_hours=None,
            force_reverify=False,
        ),
        output_path=output_path,
        output_dir=output_dir,
        state_metadata={"feasibility_only": True},
        prior_phase_2_status="partial_complete",
    )

    assert rc == 0
    state = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert state["status"] == "partial_complete"


@pytest.mark.asyncio
async def test_phase_2_skip_feasibility_completes_after_resuming_running_checkpoint(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    output_dir = tmp_path / "phase_2"
    output_dir.mkdir(parents=True)
    output_path = output_dir / "adversarial_tasks.json"
    output_path.write_text(json.dumps([_finalized_plan_task()]))

    rc = await phase_2_stage._run_feasibility_stage(
        args=Namespace(
            skip_feasibility=True,
            feasibility_only=True,
            feasibility_instances="instances.smoke.json",
            feasibility_concurrency=1,
            feasibility_retry_count=0,
            feasibility_ttl_hours=None,
            force_reverify=False,
        ),
        output_path=output_path,
        output_dir=output_dir,
        state_metadata={"feasibility_only": True},
        prior_phase_2_status="running",
    )

    assert rc == 0
    state = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert state["status"] == "complete"
    report = json.loads((output_dir / "feasibility_report.json").read_text())
    assert report["source_data_dropped_count"] == 0
    assert report["unverified_count"] == 1
    assert report["verified_count"] == 0
    assert report["per_site"]["shopping"]["unverified"] == 1
    assert report["per_site"]["shopping"]["verified"] == 0


@pytest.mark.asyncio
async def test_phase_2_skip_feasibility_clears_stale_infeasible_sidecar(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    output_dir = tmp_path / "phase_2"
    output_dir.mkdir(parents=True)
    output_path = output_dir / "adversarial_tasks.json"
    output_path.write_text(json.dumps([_finalized_plan_task()]))
    infeasible_path = output_dir / "adversarial_tasks.infeasible.json"
    infeasible_path.write_text(
        json.dumps([{"id": "stale", "feasibility": {"status": "infeasible"}}])
    )

    rc = await phase_2_stage._run_feasibility_stage(
        args=Namespace(
            skip_feasibility=True,
            feasibility_only=True,
            feasibility_instances="instances.smoke.json",
            feasibility_concurrency=1,
            feasibility_retry_count=0,
            feasibility_ttl_hours=None,
            force_reverify=False,
        ),
        output_path=output_path,
        output_dir=output_dir,
        state_metadata={"feasibility_only": True},
        prior_phase_2_status="complete",
    )

    assert rc == 0
    assert json.loads(infeasible_path.read_text()) == []


@pytest.mark.asyncio
async def test_phase_2_skip_feasibility_preserves_unfiltered_sites(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    output_dir = tmp_path / "phase_2"
    output_dir.mkdir(parents=True)
    output_path = output_dir / "adversarial_tasks.json"
    reddit_verified = {
        "id": "reddit-verified",
        "benchmark": "webarena_verified",
        "site": "reddit",
        "feasibility": {
            "status": "verified",
            "last_reverify_skipped_at": "2026-04-24T00:00:00Z",
        },
    }
    shopping_task = {
        "id": "shopping-task",
        "benchmark": "webarena_verified",
        "site": "shopping",
    }
    output_path.write_text(json.dumps([reddit_verified, shopping_task]))

    rc = await phase_2_stage._run_feasibility_stage(
        args=Namespace(
            skip_feasibility=True,
            feasibility_only=True,
            feasibility_instances="missing-instances.json",
            feasibility_concurrency=1,
            feasibility_retry_count=0,
            feasibility_ttl_hours=None,
            force_reverify=False,
        ),
        output_path=output_path,
        output_dir=output_dir,
        state_metadata={"feasibility_only": True, "sites": "shopping"},
        prior_phase_2_status="complete",
    )

    assert rc == 0
    output = json.loads(output_path.read_text())
    assert output[0] == reddit_verified
    assert output[1]["id"] == "shopping-task"
    assert output[1]["feasibility"]["status"] == "unverified"
    report = json.loads((output_dir / "feasibility_report.json").read_text())
    assert report["verified_count"] == 1
    assert report["unverified_count"] == 1
    assert report["skipped_already_verified_count"] == 1
    assert report["per_site"]["reddit"]["skipped"] == 1
    assert report["per_site"]["shopping"]["unverified"] == 1
    assert report["per_site"]["shopping"]["verified"] == 0
    state = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert state["feasibility_skipped_count"] == 1


@pytest.mark.asyncio
async def test_phase_2_skip_feasibility_preserves_partial_complete_terminal_status(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    output_dir = tmp_path / "phase_2"
    output_dir.mkdir(parents=True)
    output_path = output_dir / "adversarial_tasks.json"
    output_path.write_text(json.dumps([_finalized_plan_task()]))

    rc = await phase_2_stage._run_feasibility_stage(
        args=Namespace(
            skip_feasibility=True,
            feasibility_only=True,
            feasibility_instances="instances.smoke.json",
            feasibility_concurrency=1,
            feasibility_retry_count=0,
            feasibility_ttl_hours=None,
            force_reverify=False,
        ),
        output_path=output_path,
        output_dir=output_dir,
        state_metadata={"feasibility_only": True},
        prior_phase_2_status="partial_complete",
    )

    assert rc == 0
    state = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert state["status"] == "partial_complete"


def test_validate_generated_adversarial_task_rejects_preseeded_feasibility():
    task = _plan_task()
    task["feasibility"] = {"status": "verified"}

    problem = plan_validation._validate_generated_adversarial_task(
        task,
        0,
        {"benign-1": _benign_task()},
        _single_surface_profile(),
    )

    assert "must not include Phase 2c output fields" in problem


def test_validate_reusable_phase_2_task_rejects_preseeded_phase_2c_fields():
    task = _finalized_plan_task()
    task["feasibility"] = {"status": "verified"}

    problem = reuse._validate_reusable_phase_2_task(
        task,
        task_index=0,
        texts_per_plan=1,
        benign_by_id={"benign-1": _benign_task()},
        site_profiles={"shopping": _single_surface_profile()},
    )

    assert "must not include Phase 2c output fields" in problem
