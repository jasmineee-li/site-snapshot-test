# ruff: noqa
# Auto-split from tests/test_phase_2_injections.py; shared helpers live in tests/phase_2/_fixtures.py.
from .._fixtures import *  # noqa: F403,F401

def test_phase_2c_artifact_writer_recomputes_per_site_after_partial_merge(tmp_path):
    output_path = tmp_path / "adversarial_tasks.json"
    infeasible_path = tmp_path / "adversarial_tasks.infeasible.json"
    dropped_source_path = tmp_path / "adversarial_tasks.dropped_source_data.json"
    report_path = tmp_path / "feasibility_report.json"
    output_path.write_text(
        json.dumps(
            [
                {
                    "id": "old-reddit",
                    "site": "reddit",
                    "feasibility": {"status": "verified"},
                }
            ]
        )
    )
    infeasible_path.write_text(json.dumps([]))
    dropped_source_path.write_text(json.dumps([]))

    result = phase_2_injections._write_phase_2c_artifacts(
        output_path=output_path,
        infeasible_path=infeasible_path,
        dropped_source_path=dropped_source_path,
        report_path=report_path,
        verified=[
            {
                "id": "new-gitlab",
                "site": "gitlab",
                "feasibility": {"status": "verified"},
            }
        ],
        infeasible=[],
        dropped_source_data=[],
        report_summary={
            "verified_count": 1,
            "infeasible_count": 0,
            "source_data_dropped_count": 0,
            "source_data_dropped_by_kind": {},
            "per_site": {"gitlab": {"verified": 1, "infeasible": 0, "skipped": 0}},
        },
        sites_filter={"gitlab"},
    )

    assert result.summary["verified_count"] == 2
    assert result.summary["per_site"] == {
        "reddit": {"verified": 1, "infeasible": 0, "skipped": 0, "unverified": 0},
        "gitlab": {"verified": 1, "infeasible": 0, "skipped": 0, "unverified": 0},
    }
    assert isinstance(result, phase_2_injections.Phase2cArtifactWriteResult)

def test_phase_2c_artifact_writer_validates_before_any_write(tmp_path):
    output_path = tmp_path / "adversarial_tasks.json"
    infeasible_path = tmp_path / "adversarial_tasks.infeasible.json"
    dropped_source_path = tmp_path / "adversarial_tasks.dropped_source_data.json"
    report_path = tmp_path / "feasibility_report.json"
    output_path.write_text(json.dumps([{"id": "old-output"}]))
    infeasible_path.write_text(json.dumps([{"id": "old-infeasible"}]))
    dropped_source_path.write_text(json.dumps([{"id": "old-drop"}]))
    report_path.write_text(json.dumps({"old": True}))

    with pytest.raises(ValueError, match="verified dataset contains"):
        phase_2_injections._write_phase_2c_artifacts(
            output_path=output_path,
            infeasible_path=infeasible_path,
            dropped_source_path=dropped_source_path,
            report_path=report_path,
            verified=[{"id": "bad", "feasibility": {"status": "infeasible"}}],
            infeasible=[],
            dropped_source_data=[],
            report_summary={
                "verified_count": 1,
                "infeasible_count": 0,
                "source_data_dropped_count": 0,
                "source_data_dropped_by_kind": {},
            },
            sites_filter=None,
        )

    assert json.loads(output_path.read_text()) == [{"id": "old-output"}]
    assert json.loads(infeasible_path.read_text()) == [{"id": "old-infeasible"}]
    assert json.loads(dropped_source_path.read_text()) == [{"id": "old-drop"}]
    assert json.loads(report_path.read_text()) == {"old": True}

def test_phase_2c_artifact_writer_observes_facade_helper_monkeypatch(
    monkeypatch,
    tmp_path,
):
    output_path = tmp_path / "adversarial_tasks.json"
    infeasible_path = tmp_path / "adversarial_tasks.infeasible.json"
    dropped_source_path = tmp_path / "adversarial_tasks.dropped_source_data.json"
    report_path = tmp_path / "feasibility_report.json"
    output_path.write_text(json.dumps([]))
    infeasible_path.write_text(json.dumps([]))
    dropped_source_path.write_text(json.dumps([]))
    report_path.write_text(json.dumps({"old": True}))

    def fail_validation(**_kwargs):
        raise RuntimeError("facade validation was used")

    monkeypatch.setattr(
        phase_2_injections,
        "_validate_phase_2c_artifact_payloads",
        fail_validation,
    )

    with pytest.raises(RuntimeError, match="facade validation was used"):
        phase_2_injections._write_phase_2c_artifacts(
            output_path=output_path,
            infeasible_path=infeasible_path,
            dropped_source_path=dropped_source_path,
            report_path=report_path,
            verified=[
                {
                    "id": "adv-1",
                    "site": "gitlab",
                    "feasibility": {"status": "verified"},
                }
            ],
            infeasible=[],
            dropped_source_data=[],
            report_summary={
                "verified_count": 1,
                "infeasible_count": 0,
                "source_data_dropped_count": 0,
                "source_data_dropped_by_kind": {},
            },
            sites_filter=None,
        )

    assert json.loads(output_path.read_text()) == []

@pytest.mark.asyncio
async def test_phase_2_run_marks_feasibility_stage_running_before_2c(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_1").mkdir(parents=True)
    (tmp_path / "phase_0c").mkdir(parents=True)
    (tmp_path / "phase_1" / "benign_tasks.json").write_text(json.dumps([_benign_task()]))
    (tmp_path / "phase_0c" / "BENCHMARK_PROFILE_shopping.json").write_text(
        json.dumps(_single_surface_profile())
    )
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

    async def fake_generate(
        site_name, site_tasks, all_site_tasks=None, profile_path=None, label=None, **kwargs
    ):
        return phase_2_injections.SiteInjectionResult(site_name, [_plan_task()], [])

    async def fake_fill(*args, **kwargs):
        finalized = _finalized_plan_task()
        return [finalized], [
            {"task_id": finalized["id"], "site": finalized["site"], "status": "ok"}
        ]

    captured_state = {}

    async def fake_verify_feasibility(*args, **kwargs):
        captured_state.update(json.loads((tmp_path / "pipeline_state.json").read_text()))
        tasks_path = args[0]
        return phase_2_injections.FeasibilityReport(
            verified=[
                _with_feasibility_status(task, "verified")
                for task in json.loads(tasks_path.read_text())
            ],
            infeasible=[],
            skipped_already_verified=[],
            cleanup_warnings=[],
            host_fingerprint={"host_config": "instances.smoke.json"},
            elapsed_seconds=0.0,
            per_site_counts={},
            phase_2_status="complete",
        )

    monkeypatch.setattr(phase_2_injections, "_generate_injections_for_site", fake_generate)
    monkeypatch.setattr(phase_2_injections, "fill_texts_for_tasks", fake_fill)
    monkeypatch.setattr(phase_2_injections, "verify_feasibility", fake_verify_feasibility)

    rc = await phase_2_injections.run(
        Namespace(
            skip_feasibility=False,
            feasibility_instances=str(instances_path),
            feasibility_concurrency=1,
            feasibility_retry_count=0,
            feasibility_ttl_hours=None,
            force_reverify=False,
            sandbox_model="claude-sonnet-4-6",
        )
    )

    assert rc == 0
    assert captured_state["status"] == "running"
    assert captured_state["phase_2_stage"] == "feasibility"

@pytest.mark.asyncio
async def test_phase_2_feasibility_only_marks_stage_running_before_2c(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_2").mkdir(parents=True)
    output_path = tmp_path / "phase_2" / "adversarial_tasks.json"
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

    captured_state = {}

    async def fake_verify_feasibility(*args, **kwargs):
        captured_state.update(json.loads((tmp_path / "pipeline_state.json").read_text()))
        return phase_2_injections.FeasibilityReport(
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
            phase_2_status="complete",
        )

    monkeypatch.setattr(phase_2_injections, "verify_feasibility", fake_verify_feasibility)

    rc = await phase_2_injections.run(
        Namespace(
            feasibility_only=True,
            skip_feasibility=False,
            feasibility_instances=str(instances_path),
            feasibility_concurrency=3,
            feasibility_retry_count=0,
            feasibility_ttl_hours=24.0,
            force_reverify=True,
            sandbox_model="claude-sonnet-4-6",
        )
    )

    assert rc == 0
    assert captured_state["status"] == "running"
    assert captured_state["phase_2_stage"] == "feasibility"
    assert captured_state["feasibility_only"] is True
    assert captured_state["feasibility_instances"] == str(instances_path)
    assert captured_state["feasibility_concurrency"] == 3
    assert captured_state["feasibility_retry_count"] == 0
    assert captured_state["feasibility_ttl_hours"] == 24.0
    assert captured_state["force_reverify"] is True

@pytest.mark.asyncio
async def test_phase_2_feasibility_only_completes_after_resuming_running_checkpoint(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_2").mkdir(parents=True)
    output_path = tmp_path / "phase_2" / "adversarial_tasks.json"
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
    save_state("phase_2", status="running", phase_2_stage="feasibility", sandbox_model="demo")

    async def fake_verify_feasibility(*args, **kwargs):
        return phase_2_injections.FeasibilityReport(
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
            phase_2_status="running",
        )

    monkeypatch.setattr(phase_2_injections, "verify_feasibility", fake_verify_feasibility)

    rc = await phase_2_injections.run(
        Namespace(
            feasibility_only=True,
            skip_feasibility=False,
            feasibility_instances=str(instances_path),
            feasibility_concurrency=1,
            feasibility_retry_count=0,
            feasibility_ttl_hours=None,
            force_reverify=False,
            sandbox_model="claude-sonnet-4-6",
        )
    )

    assert rc == 0
    state = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert state["status"] == "complete"
    assert state["phase_2_stage"] == "feasibility"

@pytest.mark.asyncio
async def test_phase_2_feasibility_stage_writes_report_after_dataset(monkeypatch, tmp_path):
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
        verified = _finalized_plan_task()
        verified["id"] = "adv-ok"
        verified = _with_feasibility_status(verified, "verified")
        infeasible = _finalized_plan_task()
        infeasible["id"] = "adv-bad"
        infeasible = _with_feasibility_status(infeasible, "infeasible")
        return phase_2_injections.FeasibilityReport(
            verified=[verified],
            infeasible=[infeasible],
            skipped_already_verified=[],
            cleanup_warnings=[],
            host_fingerprint={"host_config": "instances.smoke.json"},
            elapsed_seconds=0.0,
            per_site_counts={},
            phase_2_status="complete",
        )

    write_order: list[str] = []
    real_write_json_atomic = phase_2_injections.write_json_atomic

    def recording_write_json_atomic(path, payload, *, failpoint_base=None):
        write_order.append(Path(path).name)
        return real_write_json_atomic(path, payload, failpoint_base=failpoint_base)

    monkeypatch.setattr(phase_2_injections, "verify_feasibility", fake_verify_feasibility)
    monkeypatch.setattr(phase_2_injections, "write_json_atomic", recording_write_json_atomic)

    rc = await phase_2_injections._run_feasibility_stage(
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
        prior_phase_2_status="complete",
    )

    assert rc == 0
    assert write_order[-4:] == [
        "adversarial_tasks.infeasible.json",
        "adversarial_tasks.dropped_source_data.json",
        "adversarial_tasks.json",
        "feasibility_report.json",
    ]

@pytest.mark.asyncio
async def test_phase_2_feasibility_stage_preserves_unfiltered_source_sidecar_with_sites(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    output_dir = tmp_path / "phase_2"
    output_dir.mkdir(parents=True)
    output_path = output_dir / "adversarial_tasks.json"
    output_path.write_text(json.dumps([_finalized_plan_task()]))
    dropped_path = output_dir / "adversarial_tasks.dropped_source_data.json"
    dropped_path.write_text(
        json.dumps(
            [
                {
                    "id": "old-reddit-drop",
                    "site": "reddit",
                    "source_data_issue": {"kind": "gone"},
                }
            ]
        )
    )
    instances_path = tmp_path / "instances.smoke.json"
    instances_path.write_text(
        json.dumps(
            {
                "benchmark_name": "WebArena Verified",
                "instances": [{"site_name": "shopping", "site_url": "http://shopping.test"}],
            }
        )
    )

    async def fake_verify_feasibility(*args, **kwargs):
        return phase_2_injections.FeasibilityReport(
            verified=[],
            infeasible=[],
            skipped_already_verified=[],
            cleanup_warnings=[],
            host_fingerprint={"host_config": "instances.smoke.json"},
            elapsed_seconds=0.0,
            per_site_counts={},
            phase_2_status="complete",
            dropped_source_data=[],
        )

    monkeypatch.setattr(phase_2_injections, "verify_feasibility", fake_verify_feasibility)

    rc = await phase_2_injections._run_feasibility_stage(
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
    assert json.loads(dropped_path.read_text()) == [
        {
            "id": "old-reddit-drop",
            "site": "reddit",
            "source_data_issue": {"kind": "gone"},
        }
    ]
    report = json.loads((output_dir / "feasibility_report.json").read_text())
    assert report["source_data_dropped_count"] == 1
    state = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert state["feasibility_dropped_source_data_count"] == 1
    assert state["feasibility_dropped_source_data_path"] == str(dropped_path)
