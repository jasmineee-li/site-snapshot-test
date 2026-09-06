from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from warp_taskgen.cli.status import build_status_payload, format_status_payload
from warp_taskgen.cost_tracker import (
    CostReportMalformedError,
    CostTracker,
)
from warp_taskgen.cost_tracker import (
    tracker as cost_tracker,
)
from warp_taskgen.phase_1 import novel_task_cache
from warp_taskgen.phase_1.contract_bound_action_api import SelectedActionTaskContract
from warp_taskgen.phase_1.contract_bound_action_api import (
    slot_generation as contract_api,
)
from warp_taskgen.phases import (
    phase_1_generate_new_tasks,
    phase_1_tasks,
)


@pytest.mark.asyncio
async def test_contract_bound_paid_response_is_persisted_before_slot_extraction(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("WORLDSIM_STATE_DIR", raising=False)
    monkeypatch.setattr(cost_tracker, "entries", [])
    monkeypatch.setattr(contract_api, "get_client", lambda: object())

    response = SimpleNamespace(
        id="response-1",
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        content=[],
    )

    async def fake_call_with_retry(*args, **kwargs):
        return response

    monkeypatch.setattr(contract_api, "call_with_retry", fake_call_with_retry)

    def fail_after_paid_response(_response):
        raise ValueError("invalid tool payload")

    monkeypatch.setattr(contract_api, "_extract_slots", fail_after_paid_response)

    contract = SelectedActionTaskContract(
        site="gitlab",
        card_id="card-1",
        card={"id": "card-1"},
        route={"id": "route-1"},
        route_id="route-1",
        action_kind="create_issue",
        count=1,
        anchor_assignments=({"start_url": "__GITLAB__/issues"},),
    )

    with pytest.raises(ValueError, match="invalid tool payload"):
        await contract_api._call_slots_api(
            contract=contract,
            profile={},
            requested_slots=1,
            feedback=[],
            sandbox_model="claude-sonnet-4-6",
        )

    report = json.loads((tmp_path / "cost_report.json").read_text(encoding="utf-8"))
    assert report["entries"][-1]["total_cost_usd"] == pytest.approx(0.0006)


@pytest.mark.asyncio
async def test_contract_bound_missing_usage_is_unknown_not_zero(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(cost_tracker, "entries", [])
    monkeypatch.setattr(contract_api, "get_client", lambda: object())

    response = SimpleNamespace(id="response-without-usage", usage=None, content=[])

    async def fake_call_with_retry(*args, **kwargs):
        return response

    monkeypatch.setattr(contract_api, "call_with_retry", fake_call_with_retry)

    contract = SelectedActionTaskContract(
        site="gitlab",
        card_id="card-1",
        card={"id": "card-1"},
        route={"id": "route-1"},
        route_id="route-1",
        action_kind="create_issue",
        count=1,
        anchor_assignments=({"start_url": "__GITLAB__/issues"},),
    )

    with pytest.raises(ValueError, match="no emit_action_task_slots"):
        await contract_api._call_slots_api(
            contract=contract,
            profile={},
            requested_slots=1,
            feedback=[],
            sandbox_model="claude-sonnet-4-6",
        )

    report = json.loads((tmp_path / "cost_report.json").read_text(encoding="utf-8"))
    assert report["entries"][-1]["total_cost_usd"] is None


@pytest.mark.asyncio
async def test_contract_bound_partial_returned_usage_is_unknown_not_zero(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(cost_tracker, "entries", [])
    monkeypatch.setattr(contract_api, "get_client", lambda: object())

    response = SimpleNamespace(
        id="response-with-partial-usage",
        usage=SimpleNamespace(output_tokens=20),
        content=[],
    )

    async def fake_call_with_retry(*args, **kwargs):
        return response

    monkeypatch.setattr(contract_api, "call_with_retry", fake_call_with_retry)
    contract = SelectedActionTaskContract(
        site="gitlab",
        card_id="card-1",
        card={"id": "card-1"},
        route={"id": "route-1"},
        route_id="route-1",
        action_kind="create_issue",
        count=1,
        anchor_assignments=({"start_url": "__GITLAB__/issues"},),
    )

    with pytest.raises(ValueError, match="no emit_action_task_slots"):
        await contract_api._call_slots_api(
            contract=contract,
            profile={},
            requested_slots=1,
            feedback=[],
            sandbox_model="claude-sonnet-4-6",
        )

    report = json.loads((tmp_path / "cost_report.json").read_text(encoding="utf-8"))
    assert report["entries"][-1]["total_cost_usd"] is None


@pytest.mark.asyncio
async def test_contract_bound_paid_exception_records_one_unknown_observation(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(cost_tracker, "entries", [])
    monkeypatch.setattr(contract_api, "get_client", lambda: object())

    async def fail_call_with_retry(*args, **kwargs):
        raise TimeoutError("provider unavailable")

    monkeypatch.setattr(contract_api, "call_with_retry", fail_call_with_retry)
    contract = SelectedActionTaskContract(
        site="gitlab",
        card_id="card-1",
        card={"id": "card-1"},
        route={"id": "route-1"},
        route_id="route-1",
        action_kind="create_issue",
        count=1,
        anchor_assignments=({"start_url": "__GITLAB__/issues"},),
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await contract_api._call_slots_api(
            contract=contract,
            profile={},
            requested_slots=1,
            feedback=[],
            sandbox_model="claude-sonnet-4-6",
        )

    report = json.loads((tmp_path / "cost_report.json").read_text(encoding="utf-8"))
    assert len(report["entries"]) == 1
    assert report["entries"][0]["total_cost_usd"] is None


@pytest.mark.asyncio
async def test_contract_bound_partial_errored_usage_is_unknown_not_zero(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(cost_tracker, "entries", [])
    monkeypatch.setattr(contract_api, "get_client", lambda: object())

    provider_response = SimpleNamespace(
        id="errored-partial-response",
        usage={"input_tokens": 100},
        content=[],
    )
    failure = RuntimeError("provider returned an errored response")
    failure.raw_response = provider_response

    async def fail_call_with_retry(*args, **kwargs):
        raise failure

    monkeypatch.setattr(contract_api, "call_with_retry", fail_call_with_retry)
    contract = SelectedActionTaskContract(
        site="gitlab",
        card_id="card-1",
        card={"id": "card-1"},
        route={"id": "route-1"},
        route_id="route-1",
        action_kind="create_issue",
        count=1,
        anchor_assignments=({"start_url": "__GITLAB__/issues"},),
    )

    with pytest.raises(RuntimeError, match="provider returned an errored response"):
        await contract_api._call_slots_api(
            contract=contract,
            profile={},
            requested_slots=1,
            feedback=[],
            sandbox_model="claude-sonnet-4-6",
        )

    report = json.loads((tmp_path / "cost_report.json").read_text(encoding="utf-8"))
    assert report["entries"][0]["total_cost_usd"] is None


@pytest.mark.asyncio
async def test_contract_bound_setup_exception_does_not_record_observation(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(cost_tracker, "entries", [])
    monkeypatch.setattr(
        contract_api,
        "get_client",
        lambda: (_ for _ in ()).throw(RuntimeError("client setup failed")),
    )
    contract = SelectedActionTaskContract(
        site="gitlab",
        card_id="card-1",
        card={"id": "card-1"},
        route={"id": "route-1"},
        route_id="route-1",
        action_kind="create_issue",
        count=1,
        anchor_assignments=({"start_url": "__GITLAB__/issues"},),
    )

    with pytest.raises(RuntimeError, match="client setup failed"):
        await contract_api._call_slots_api(
            contract=contract,
            profile={},
            requested_slots=1,
            feedback=[],
            sandbox_model="claude-sonnet-4-6",
        )

    assert not (tmp_path / "cost_report.json").exists()


@pytest.mark.asyncio
async def test_contract_bound_error_response_usage_is_retained(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(cost_tracker, "entries", [])
    monkeypatch.setattr(contract_api, "get_client", lambda: object())

    provider_response = SimpleNamespace(
        id="errored-response-1",
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        content=[],
    )
    failure = RuntimeError("provider returned an errored response")
    failure.raw_response = provider_response

    async def fail_call_with_retry(*args, **kwargs):
        raise failure

    monkeypatch.setattr(contract_api, "call_with_retry", fail_call_with_retry)
    contract = SelectedActionTaskContract(
        site="gitlab",
        card_id="card-1",
        card={"id": "card-1"},
        route={"id": "route-1"},
        route_id="route-1",
        action_kind="create_issue",
        count=1,
        anchor_assignments=({"start_url": "__GITLAB__/issues"},),
    )

    with pytest.raises(RuntimeError, match="provider returned an errored response"):
        await contract_api._call_slots_api(
            contract=contract,
            profile={},
            requested_slots=1,
            feedback=[],
            sandbox_model="claude-sonnet-4-6",
        )

    report = json.loads((tmp_path / "cost_report.json").read_text(encoding="utf-8"))
    assert len(report["entries"]) == 1
    assert report["entries"][0]["total_cost_usd"] == pytest.approx(0.0006)


def test_explicit_cost_survives_missing_usage_metadata(tmp_path) -> None:
    tracker = CostTracker()
    tracker.record(
        "phase_1",
        json.dumps({"total_cost_usd": 1.25, "usage": "not available"}),
        site="gitlab",
    )
    tracker.save(tmp_path / "cost_report.json")

    inspection = tracker.inspect_report(tmp_path / "cost_report.json")
    assert inspection.status == "valid"
    assert inspection.known_total_cost_usd == pytest.approx(1.25)
    assert inspection.unknown_entry_count == 0


def test_switching_tracker_to_missing_root_discards_prior_entries(tmp_path) -> None:
    root_a = tmp_path / "root-a" / "cost_report.json"
    root_b = tmp_path / "root-b" / "cost_report.json"
    tracker = CostTracker()
    tracker.record("phase_1", json.dumps({"total_cost_usd": 1.0}), site="gitlab")
    tracker.save(root_a)
    tracker.record_and_save(
        "phase_1",
        json.dumps({"total_cost_usd": 1.5}),
        root_a,
        site="gitlab",
    )
    report_a = json.loads(root_a.read_text(encoding="utf-8"))
    assert len(report_a["entries"]) == 2
    tracker.record_and_save(
        "phase_1",
        json.dumps({"total_cost_usd": 2.0}),
        root_b,
        site="reddit",
    )

    report = json.loads(root_b.read_text(encoding="utf-8"))
    assert len(report["entries"]) == 1
    assert report["entries"][0]["site"] == "reddit"
    assert report["total_cost_usd"] == pytest.approx(2.0)


def test_record_and_save_starts_new_run_after_same_path_is_deleted(tmp_path) -> None:
    path = tmp_path / "cost_report.json"
    tracker = CostTracker()
    tracker.record_and_save(
        "phase_1",
        json.dumps({"total_cost_usd": 1.0}),
        path,
        site="gitlab",
    )
    path.unlink()

    tracker.record_and_save(
        "phase_1",
        json.dumps({"total_cost_usd": 2.0}),
        path,
        site="reddit",
    )

    report = json.loads(path.read_text(encoding="utf-8"))
    assert len(report["entries"]) == 1
    assert report["entries"][0]["site"] == "reddit"
    assert report["total_cost_usd"] == pytest.approx(2.0)


def test_phase1_finalization_preserves_malformed_cost_report(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cost_report.json"
    path.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(cost_tracker, "entries", [])
    monkeypatch.setattr(
        cost_tracker,
        "save",
        lambda *_args, **_kwargs: pytest.fail("malformed evidence must not be replaced"),
    )

    phase_1_tasks._save_phase1_cost_report(tmp_path)

    assert path.read_text(encoding="utf-8") == "{broken"
    with pytest.raises(CostReportMalformedError):
        cost_tracker.ensure_phase1_paid_dispatch_allowed(path)


def test_phase1_finalization_preserves_valid_cost_report(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cost_report.json"
    report = {
        "entries": [
            {
                "phase": "phase_1",
                "task_id": None,
                "site": "gitlab",
                "total_cost_usd": 3.5,
                "num_turns": None,
                "duration_ms": None,
                "session_id": None,
                "model_usage": None,
                "timestamp": "2026-09-03T00:00:00+00:00",
            }
        ]
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(cost_tracker, "entries", [])
    monkeypatch.setattr(cost_tracker, "_loaded_path", None)

    phase_1_tasks._save_phase1_cost_report(tmp_path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert len(saved["entries"]) == 1
    assert saved["entries"][0]["total_cost_usd"] == pytest.approx(3.5)


def test_missing_and_malformed_status_are_unobserved(tmp_path) -> None:
    missing = build_status_payload(tmp_path)
    assert missing["cost_observation"]["status"] == "missing"
    assert missing["cost_observation"]["known_total_cost_usd"] is None
    assert "known_total_cost_usd=unknown" in format_status_payload(missing)

    (tmp_path / "cost_report.json").write_text("{broken", encoding="utf-8")
    malformed = build_status_payload(tmp_path)
    assert malformed["cost_observation"]["status"] == "malformed"
    assert malformed["cost_observation"]["known_total_cost_usd"] is None
    assert "known_total_cost_usd=unknown" in format_status_payload(malformed)


def test_valid_empty_report_has_zero_observed_total(tmp_path) -> None:
    (tmp_path / "cost_report.json").write_text(
        json.dumps({"entries": []}),
        encoding="utf-8",
    )
    observation = build_status_payload(tmp_path)["cost_observation"]
    assert observation["status"] == "valid"
    assert observation["known_total_cost_usd"] == 0.0


def test_overflow_cost_report_is_malformed(tmp_path) -> None:
    path = tmp_path / "cost_report.json"
    path.write_text(
        json.dumps({"total_cost_usd": 10**400, "entries": []}),
        encoding="utf-8",
    )
    inspection = cost_tracker.inspect_report(path)
    assert inspection.status == "malformed"


@pytest.mark.asyncio
async def test_sandbox_setup_exception_does_not_record_observation(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "phase_1"
    output_dir.mkdir()
    monkeypatch.setattr(cost_tracker, "entries", [])
    setup_failure = RuntimeError("sandbox setup failed")
    sandbox = AsyncMock(side_effect=setup_failure)
    monkeypatch.setattr(phase_1_generate_new_tasks, "run_claude_in_sandbox", sandbox)
    monkeypatch.setattr(
        phase_1_generate_new_tasks, "_load_site_agent_context", lambda _site: ({}, [])
    )
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "build_task_route_contracts",
        lambda **_kwargs: {"route_families": [{"id": "route-1"}]},
    )

    with pytest.raises(RuntimeError, match="sandbox setup failed"):
        await phase_1_generate_new_tasks.generate_new_tasks_for_site(
            site=_site(tmp_path),
            benchmark_volume=object(),
            output_dir=output_dir,
            cache_fingerprint="cache-fingerprint",
        )

    assert not (tmp_path / "cost_report.json").exists()


@pytest.mark.asyncio
async def test_zero_requested_site_does_not_inspect_cost_report(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "phase_1"
    output_dir.mkdir()
    monkeypatch.setattr(cost_tracker, "entries", [])
    monkeypatch.setattr(
        cost_tracker,
        "ensure_phase1_paid_dispatch_allowed",
        lambda *_args, **_kwargs: pytest.fail("zero work must not inspect the report"),
    )

    result = await phase_1_generate_new_tasks.generate_new_tasks_for_site(
        site=_site(tmp_path),
        benchmark_volume=object(),
        output_dir=output_dir,
        cache_fingerprint="cache-fingerprint",
        novel_tasks_per_site=0,
    )

    assert result.benign_tasks == []
    assert result.errors == []


@pytest.mark.asyncio
async def test_zero_requested_run_does_not_gate_or_preflight(monkeypatch, tmp_path) -> None:
    state_dir = tmp_path / "state"
    (state_dir / "phase_0c").mkdir(parents=True)
    output_dir = state_dir / "phase_1"
    output_dir.mkdir()
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(state_dir))
    site = _site(tmp_path)
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "load_generate_new_tasks_eligible_sites",
        lambda **_kwargs: [site],
    )
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "_load_all_cached_site_results",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "_load_site_agent_context",
        lambda _site: ({}, []),
    )
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "build_task_route_contracts",
        lambda **_kwargs: {"route_families": [{"id": "route-1"}]},
    )
    monkeypatch.setattr(
        novel_task_cache,
        "build_task_route_contracts",
        lambda **_kwargs: {"route_families": [{"id": "route-1"}]},
    )
    monkeypatch.setattr(
        cost_tracker,
        "ensure_phase1_paid_dispatch_allowed",
        lambda *_args, **_kwargs: pytest.fail("zero work must not inspect the report"),
    )
    preflight = AsyncMock(side_effect=pytest.fail)
    monkeypatch.setattr(phase_1_generate_new_tasks, "preflight_sandbox_environment", preflight)

    generated = await phase_1_generate_new_tasks.run_generate_new_tasks(
        manifest={"evaluation": {"eval_types": []}},
        benchmark_root=tmp_path,
        output_dir=output_dir,
        novel_tasks_per_site=0,
    )

    assert generated == []
    assert preflight.await_count == 0
    assert not (state_dir / "cost_report.json").exists()


def _site(tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text("{}", encoding="utf-8")
    return phase_1_generate_new_tasks.EligibleSiteProfile(
        site_name="shopping",
        profile_path=profile_path,
        profile={},
    )


def _stub_sandbox_generation(monkeypatch, *, outputs):
    monkeypatch.setattr(
        phase_1_generate_new_tasks, "_load_site_agent_context", lambda _site: ({}, [])
    )
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "build_task_route_contracts",
        lambda **_kwargs: {"route_families": [{"id": "route-1"}]},
    )
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "validate_generated_novel_tasks_detailed",
        lambda tasks, **_kwargs: (tasks, []),
    )
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "_compile_phase1_model_owned_features",
        lambda tasks, **_kwargs: tasks,
    )
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "_compile_phase1_feature_tasks",
        lambda tasks, **_kwargs: tasks,
    )
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "run_claude_in_sandbox",
        AsyncMock(return_value=outputs),
    )


@pytest.mark.asyncio
async def test_sandbox_paid_response_is_saved_before_invalid_output_is_rejected(
    monkeypatch,
    tmp_path,
) -> None:
    output_dir = tmp_path / "phase_1"
    output_dir.mkdir()
    monkeypatch.setattr(cost_tracker, "entries", [])
    _stub_sandbox_generation(
        monkeypatch,
        outputs={
            phase_1_generate_new_tasks.NOVEL_TASK_OUTPUT_PATH: "{broken",
            "_summary": json.dumps({"total_cost_usd": 1.25}),
        },
    )

    result = await phase_1_generate_new_tasks.generate_new_tasks_for_site(
        site=_site(tmp_path),
        benchmark_volume=object(),
        output_dir=output_dir,
        cache_fingerprint="cache-fingerprint",
    )

    assert "invalid sandbox JSON" in result.errors[0]
    report = json.loads((tmp_path / "cost_report.json").read_text(encoding="utf-8"))
    assert report["entries"][-1]["total_cost_usd"] == pytest.approx(1.25)


@pytest.mark.asyncio
async def test_sandbox_paid_exception_records_one_unknown_observation(
    monkeypatch, tmp_path
) -> None:
    output_dir = tmp_path / "phase_1"
    output_dir.mkdir()
    monkeypatch.setattr(cost_tracker, "entries", [])
    paid_failure = RuntimeError("provider unavailable")
    paid_failure._warp_paid_call_started = True
    sandbox = AsyncMock(side_effect=paid_failure)
    monkeypatch.setattr(phase_1_generate_new_tasks, "run_claude_in_sandbox", sandbox)
    monkeypatch.setattr(
        phase_1_generate_new_tasks, "_load_site_agent_context", lambda _site: ({}, [])
    )
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "build_task_route_contracts",
        lambda **_kwargs: {"route_families": [{"id": "route-1"}]},
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await phase_1_generate_new_tasks.generate_new_tasks_for_site(
            site=_site(tmp_path),
            benchmark_volume=object(),
            output_dir=output_dir,
            cache_fingerprint="cache-fingerprint",
        )

    assert sandbox.await_count == 1
    report = json.loads((tmp_path / "cost_report.json").read_text(encoding="utf-8"))
    assert len(report["entries"]) == 1
    assert report["entries"][0]["total_cost_usd"] is None


@pytest.mark.asyncio
async def test_cost_report_write_failure_does_not_retry_paid_sandbox_call(
    monkeypatch, tmp_path
) -> None:
    output_dir = tmp_path / "phase_1"
    output_dir.mkdir()
    _stub_sandbox_generation(
        monkeypatch,
        outputs={
            phase_1_generate_new_tasks.NOVEL_TASK_OUTPUT_PATH: "{broken",
            "_summary": json.dumps({"total_cost_usd": 1.25}),
        },
    )
    sandbox = phase_1_generate_new_tasks.run_claude_in_sandbox
    monkeypatch.setattr(
        cost_tracker, "save", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full"))
    )

    with pytest.raises(OSError, match="disk full"):
        await phase_1_generate_new_tasks.generate_new_tasks_for_site(
            site=_site(tmp_path),
            benchmark_volume=object(),
            output_dir=output_dir,
            cache_fingerprint="cache-fingerprint",
        )

    assert sandbox.await_count == 1


@pytest.mark.asyncio
async def test_malformed_report_blocks_phase1_before_sandbox_preflight(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_0c").mkdir()
    (tmp_path / "cost_report.json").write_text("{broken", encoding="utf-8")
    site = _site(tmp_path)
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "load_generate_new_tasks_eligible_sites",
        lambda **_kwargs: [site],
    )
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "_load_all_cached_site_results",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "_fail_if_task_card_plan_missing_sites",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "_fail_if_action_counts_unavailable",
        lambda **_kwargs: None,
    )
    preflight = AsyncMock()
    monkeypatch.setattr(phase_1_generate_new_tasks, "preflight_sandbox_environment", preflight)

    with pytest.raises(CostReportMalformedError, match="paid dispatch refused"):
        await phase_1_generate_new_tasks.run_generate_new_tasks(
            manifest={"evaluation": {"eval_types": []}},
            benchmark_root=tmp_path,
            output_dir=tmp_path / "phase_1",
        )

    assert preflight.await_count == 0


def test_status_projects_lower_bound_and_malformed_cost_state(tmp_path) -> None:
    valid = {
        "entries": [
            {
                "phase": "phase_1",
                "task_id": None,
                "site": "gitlab",
                "total_cost_usd": 2.5,
                "num_turns": 1,
                "duration_ms": 10,
                "session_id": "response-1",
                "model_usage": None,
                "timestamp": "2026-09-03T00:00:00+00:00",
            },
            {
                "phase": "phase_1",
                "task_id": None,
                "site": "reddit",
                "total_cost_usd": None,
                "num_turns": None,
                "duration_ms": None,
                "session_id": None,
                "model_usage": None,
                "timestamp": "2026-09-03T00:00:01+00:00",
            },
        ]
    }
    (tmp_path / "cost_report.json").write_text(json.dumps(valid), encoding="utf-8")

    payload = build_status_payload(tmp_path)
    observation = payload["cost_observation"]
    assert observation == {
        "path": str(tmp_path / "cost_report.json"),
        "status": "valid",
        "known_total_cost_usd": 2.5,
        "known_entry_count": 1,
        "unknown_entry_count": 1,
        "recorded_entry_count": 2,
        "completeness": "lower_bound",
        "reason_code": None,
    }
    assert payload["cost_report"] == valid
    assert "known_total_cost_usd=2.5000" in format_status_payload(payload)

    (tmp_path / "cost_report.json").write_text("{broken", encoding="utf-8")
    malformed = build_status_payload(tmp_path)
    assert malformed["cost_observation"]["status"] == "malformed"
    assert "cost_report" not in malformed
    assert "Observed cost: status=malformed" in format_status_payload(malformed)


def test_missing_and_valid_null_reports_remain_dispatchable(tmp_path) -> None:
    missing_path = tmp_path / "missing.json"
    assert cost_tracker.ensure_phase1_paid_dispatch_allowed(missing_path).status == "missing"
    (tmp_path / "null.json").write_text(json.dumps({"entries": []}), encoding="utf-8")
    assert cost_tracker.ensure_phase1_paid_dispatch_allowed(tmp_path / "null.json").status == (
        "valid"
    )


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"entries": {"not": "a-list"}},
        {"entries": [{"phase": "phase_1"}]},
        {
            "entries": [
                {
                    "phase": "phase_1",
                    "task_id": None,
                    "site": "gitlab",
                    "total_cost_usd": -1,
                    "num_turns": None,
                    "duration_ms": None,
                    "session_id": None,
                    "model_usage": None,
                    "timestamp": "2026-09-03T00:00:00+00:00",
                }
            ]
        },
    ],
)
def test_invalid_cost_reports_are_not_dispatchable(tmp_path, payload) -> None:
    path = tmp_path / "cost_report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CostReportMalformedError, match="paid dispatch refused"):
        cost_tracker.ensure_phase1_paid_dispatch_allowed(path)
