from __future__ import annotations

import json
from pathlib import Path

from warp_taskgen.phase_4.existing_family_coverage import build_existing_family_coverage
from warp_taskgen.phase_4.scenario_funnel_export import build_scenario_funnel_export


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _task(
    task_id: str,
    *,
    site: str = "gitlab",
    origin: str = "new_task",
    surface: str = "issue.description",
    archetype: str = "field_status_check",
    feasibility: str | None = None,
) -> dict[str, object]:
    task: dict[str, object] = {
        "id": task_id,
        "site": site,
        "origin": origin,
        "target_surface_id": surface,
        "task_provenance": {"archetype_id": archetype},
    }
    if feasibility is not None:
        task["feasibility"] = {"status": feasibility}
    return task


def _complete_fixture(run: Path) -> None:
    _write_json(
        run / "phase_1" / "benign_tasks.json",
        [
            _task("candidate-1"),
            _task(
                "candidate-2",
                site="reddit",
                origin="existing_task",
                surface="submission.body",
                archetype="follow_up",
            ),
            _task("candidate-3", feasibility="unverified"),
        ],
    )
    _write_json(
        run / "phase_3" / "contracts.json",
        [
            {"id": "candidate-1", "validity_status": "valid", "task": _task("candidate-1")},
            {
                "id": "candidate-2",
                "validity_status": "valid",
                "task": _task(
                    "candidate-2",
                    site="reddit",
                    origin="existing_task",
                    surface="submission.body",
                    archetype="follow_up",
                ),
            },
            {
                "id": "candidate-3",
                "validity_status": "invalid",
                "validity_errors": ["missing reward"],
                "task": _task("candidate-3"),
            },
        ],
    )
    _write_json(
        run / "phase_2" / "adversarial_tasks.json",
        [
            {
                **_task("candidate-1", feasibility="verified"),
                "feasibility": {
                    "status": "verified",
                    "last_reverify_skipped_at": "2026-09-01T00:00:00Z",
                },
            }
        ],
    )
    _write_json(
        run / "phase_2" / "adversarial_tasks.infeasible.json",
        [
            _task(
                "candidate-2",
                site="reddit",
                surface="submission.body",
                archetype="follow_up",
                feasibility="infeasible",
            )
        ],
    )
    _write_json(
        run / "phase_2" / "adversarial_tasks.dropped_source_data.json",
        [
            {
                **_task("candidate-3"),
                "drop_reason": "source_data_not_found",
                "source_data_issue": {"kind": "not_found"},
            }
        ],
    )
    _write_json(
        run / "phase_2" / "feasibility_report.json",
        {
            "phase_2_status": "complete",
            "verified_count": 1,
            "infeasible_count": 1,
            "skipped_already_verified_count": 1,
            "unverified_count": 0,
            "source_data_dropped_count": 1,
            "source_data_dropped_by_kind": {"not_found": 1},
        },
    )
    _write_json(
        run / "phase_4" / "results.json",
        [
            {"task_id": "candidate-1", "final_status": "resistant"},
            {"task_id": "candidate-9", "final_status": "error"},
        ],
    )

    bank_event = {
        "schema_version": 1,
        "event_type": "admit_task",
        "event_id": "admit:candidate-1",
        "created_at": "2026-09-01T00:00:00Z",
        "task_id": "candidate-1",
        "site": "gitlab",
        "origin": "new_task",
        "task_signature": "sig-1",
        "carrier_contract": {"target_surface_id": "issue.description"},
        "archetype_id": "field_status_check",
    }
    run.joinpath("task_bank").mkdir(parents=True, exist_ok=True)
    run.joinpath("task_bank", "events.jsonl").write_text(
        json.dumps(bank_event) + "\n", encoding="utf-8"
    )


def test_existing_family_coverage_composes_complete_funnel(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)

    coverage = build_existing_family_coverage(tmp_path)

    assert coverage["schema_version"] == "existing_family_coverage_v1"
    assert coverage["funnel"]["candidate"] == {
        "count": 3,
        "status": "available",
        "source": "phase_1/benign_tasks.json",
    }
    assert coverage["funnel"]["validated"]["count"] == 2
    assert coverage["funnel"]["admitted"]["count"] == 1
    assert coverage["funnel"]["active"]["count"] == 1
    assert coverage["funnel"]["retired"]["count"] == 0
    assert coverage["funnel"]["evaluated"]["count"] == 2
    assert coverage["funnel"]["failed"]["count"] == 1
    assert coverage["phase2c_statuses"] == {
        "verified": 1,
        "infeasible": 1,
        "skipped": 1,
        "unverified": 0,
        "dropped_source": 1,
    }
    assert coverage["breakdowns"]["site"]["gitlab"]["admitted"] == 1
    assert coverage["breakdowns"]["origin"]["new_task"]["candidate"] == 2
    assert coverage["breakdowns"]["surface"]["issue.description"]["active"] == 1
    assert coverage["breakdowns"]["task_archetype"]["field_status_check"]["evaluated"] == 1
    assert coverage["warnings"]
    assert any("result task candidate-9" in warning for warning in coverage["warnings"])


def test_existing_family_coverage_marks_missing_phase1_candidates_unavailable(
    tmp_path: Path,
) -> None:
    _write_json(tmp_path / "phase_2" / "adversarial_tasks.json", [_task("admitted-1")])
    _write_json(
        tmp_path / "phase_2" / "feasibility_report.json",
        {"verified_count": 1, "infeasible_count": 0},
    )

    coverage = build_existing_family_coverage(tmp_path)

    assert coverage["funnel"]["candidate"] == {
        "count": None,
        "status": "unavailable",
        "source": "phase_1/benign_tasks.json",
        "reason": "artifact not found",
    }
    assert coverage["unavailable"]["candidate"] == ("artifact not found: phase_1/benign_tasks.json")
    assert coverage["phase2c_statuses"]["verified"] is None
    assert coverage["phase2c_statuses"]["infeasible"] is None
    assert coverage["phase2c_statuses"]["dropped_source"] is None
    assert "invalid feasibility.status" in coverage["unavailable"]["phase2c.verified"]
    assert "<missing>" in coverage["unavailable"]["phase2c.verified"]
    assert coverage["unavailable"]["phase2c.infeasible"] == (
        "artifact not found: phase_2/adversarial_tasks.infeasible.json"
    )


def test_existing_family_coverage_counts_skip_feasibility_as_unverified(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "phase_2" / "adversarial_tasks.json",
        [_task("skipped-1", feasibility="unverified")],
    )

    coverage = build_existing_family_coverage(tmp_path)

    assert coverage["phase2c_statuses"]["verified"] == 0
    assert coverage["phase2c_statuses"]["unverified"] == 1
    assert coverage["funnel"]["admitted"]["count"] == 0
    assert "phase2c.verified" not in coverage["unavailable"]


def test_existing_family_coverage_preserves_dropped_source_and_total_mismatches(
    tmp_path: Path,
) -> None:
    _write_json(tmp_path / "phase_1" / "benign_tasks.json", [_task("candidate-1")])
    _write_json(
        tmp_path / "phase_2" / "adversarial_tasks.json",
        [_task("candidate-1", feasibility="verified")],
    )
    _write_json(
        tmp_path / "phase_2" / "adversarial_tasks.dropped_source_data.json",
        [
            {
                **_task("dropped-1"),
                "source_data_issue": {"kind": "not_found"},
            },
            {
                **_task("dropped-2"),
                "source_data_issue": {"kind": "gone"},
            },
        ],
    )
    _write_json(
        tmp_path / "phase_2" / "feasibility_report.json",
        {
            "verified_count": 4,
            "infeasible_count": 3,
            "source_data_dropped_count": 1,
            "source_data_dropped_by_kind": {"not_found": 1},
        },
    )

    coverage = build_existing_family_coverage(tmp_path)

    assert coverage["phase2c_statuses"]["dropped_source"] == 2
    assert coverage["dropped_source_by_kind"] == {"gone": 1, "not_found": 1}
    assert any("verified_count" in warning for warning in coverage["warnings"])
    assert any("source_data_dropped_count" in warning for warning in coverage["warnings"])


def test_existing_family_coverage_keeps_malformed_owner_unavailable(
    tmp_path: Path,
) -> None:
    _write_json(tmp_path / "phase_1" / "benign_tasks.json", [_task("candidate-1"), "bad-row"])

    coverage = build_existing_family_coverage(tmp_path)

    assert coverage["funnel"]["candidate"] == {
        "count": None,
        "status": "unavailable",
        "source": "phase_1/benign_tasks.json",
        "reason": "Phase 1 candidates artifact contains non-object rows",
    }
    assert coverage["sources"]["phase1_candidates"]["status"] == "unavailable"
    assert any("non-object rows" in warning for warning in coverage["warnings"])


def test_scenario_funnel_includes_existing_family_coverage_summary(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)

    export = build_scenario_funnel_export(tmp_path)

    assert export["summary"]["coverage"]["funnel"]["admitted"]["count"] == 1


def test_existing_family_coverage_reports_phase2c_failed_state_and_reason(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "phase_2" / "feasibility_report.json",
        {"phase_2_status": "complete"},
    )
    _write_json(
        tmp_path / "pipeline_state.json",
        {
            "step": "phase_2",
            "phase_2_stage": "feasibility",
            "status": "failed",
            "reason": "feasibility_preflight",
            "feasibility_error": "benchmark instance unavailable",
        },
    )

    coverage = build_existing_family_coverage(tmp_path)

    assert coverage["phase2c_state"] == {
        "status": "failed",
        "reason": "feasibility_preflight",
        "source": "pipeline_state.json",
    }


def test_existing_family_coverage_rejects_unknown_phase2c_status_and_phase4_outcome(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "phase_2" / "adversarial_tasks.json",
        [
            _task("bad", feasibility="mystery"),
            {**_task("malformed"), "feasibility": {"status": 42}},
        ],
    )
    _write_json(
        tmp_path / "phase_2" / "adversarial_tasks.infeasible.json",
        [_task("missing")],
    )
    _write_json(
        tmp_path / "phase_4" / "results.json",
        [
            {"task_id": "bad", "final_status": "injection_not_encountered"},
            {"task_id": "unknown", "final_status": "not_a_phase4_status"},
        ],
    )

    coverage = build_existing_family_coverage(tmp_path)

    assert coverage["phase2c_statuses"]["verified"] is None
    assert "invalid feasibility.status" in coverage["unavailable"]["phase2c.verified"]
    assert "mystery" in coverage["unavailable"]["phase2c.verified"]
    assert "42" in coverage["unavailable"]["phase2c.verified"]
    assert coverage["phase2c_statuses"]["infeasible"] is None
    assert coverage["funnel"]["admitted"]["count"] is None
    assert coverage["funnel"]["failed"]["count"] == 2
    assert sum("retained as failed/attrition" in warning for warning in coverage["warnings"]) == 2


def test_existing_family_coverage_scopes_result_warning_to_supplied_task_bank(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "phase_4" / "results.json", [{"task_id": "bank-only", "final_status": "error"}]
    )
    bank_path = tmp_path / "external-events.jsonl"
    bank_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_type": "admit_task",
                "event_id": "event:bank-only",
                "created_at": "2026-09-01T00:00:00Z",
                "task_id": "bank-only",
                "site": "gitlab",
                "origin": "new_task",
                "task_signature": "sig-bank-only",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    coverage = build_existing_family_coverage(tmp_path, task_bank_path=bank_path)

    assert not any(
        "bank-only" in warning and "not present" in warning for warning in coverage["warnings"]
    )
