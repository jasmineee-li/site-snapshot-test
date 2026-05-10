from __future__ import annotations

import json

from scripts import drop_image_alt_text_tasks as drop_image_alt_text


def test_sync_feasibility_report_counts_matches_current_artifacts(tmp_path):
    base = tmp_path / "phase_2"
    base.mkdir()
    (base / "adversarial_tasks.json").write_text(
        json.dumps(
            [
                {"id": "v1", "site": "gitlab", "feasibility": {"status": "verified"}},
                {
                    "id": "v2",
                    "site": "reddit",
                    "feasibility": {
                        "status": "verified",
                        "last_reverify_skipped_at": "2026-04-23T00:00:00Z",
                    },
                },
            ]
        )
    )
    (base / "adversarial_tasks.infeasible.json").write_text(
        json.dumps([{"id": "i1", "site": "gitlab"}])
    )
    (base / "adversarial_tasks.dropped_source_data.json").write_text(
        json.dumps([{"id": "d1", "site": "gitlab", "source_data_issue": {"kind": "not_found"}}])
    )
    (base / "feasibility_report.json").write_text(
        json.dumps({"verified_count": 99, "infeasible_count": 99})
    )

    assert drop_image_alt_text._sync_feasibility_report_counts(base)

    report = json.loads((base / "feasibility_report.json").read_text())
    assert report["verified_count"] == 2
    assert report["infeasible_count"] == 1
    assert report["skipped_already_verified_count"] == 1
    assert report["source_data_dropped_count"] == 1
    assert report["source_data_dropped_by_kind"] == {"not_found": 1}
    assert report["per_site"] == {
        "gitlab": {"verified": 1, "infeasible": 1, "skipped": 0},
        "reddit": {"verified": 0, "infeasible": 0, "skipped": 1},
    }


def test_sync_pipeline_state_counts_matches_current_artifacts(tmp_path):
    base = tmp_path / "phase_2"
    base.mkdir()
    (base / "adversarial_tasks.json").write_text(
        json.dumps(
            [
                {"id": "v1", "site": "gitlab", "feasibility": {"status": "verified"}},
                {
                    "id": "v2",
                    "site": "reddit",
                    "feasibility": {"last_reverify_skipped_at": "2026-04-23T00:00:00Z"},
                },
            ]
        )
    )
    (base / "adversarial_tasks.infeasible.json").write_text(
        json.dumps([{"id": "i1", "site": "gitlab"}])
    )
    (base / "adversarial_tasks.dropped_source_data.json").write_text(
        json.dumps([{"id": "d1", "site": "gitlab", "source_data_issue": {"kind": "not_found"}}])
    )
    (tmp_path / "pipeline_state.json").write_text(
        json.dumps(
            {
                "step": "phase_2",
                "status": "complete",
                "feasibility_verified_count": 99,
                "feasibility_infeasible_count": 99,
                "feasibility_dropped_source_data_count": 99,
            }
        )
    )

    assert drop_image_alt_text._sync_pipeline_state_counts(tmp_path)

    state = json.loads((tmp_path / "pipeline_state.json").read_text())
    assert state["feasibility_verified_count"] == 2
    assert state["feasibility_infeasible_count"] == 1
    assert state["feasibility_skipped_count"] == 1
    assert state["feasibility_dropped_source_data_count"] == 1
    assert state["feasibility_dropped_source_data_path"] == str(
        base / "adversarial_tasks.dropped_source_data.json"
    )


def test_drop_image_alt_text_audit_preserves_per_file_provenance(tmp_path):
    base = tmp_path / "phase_2"
    base.mkdir()
    record = {"id": "adv-img", "site": "gitlab", "concealment": "image_alt_text"}
    (base / "adversarial_plans.json").write_text(json.dumps([record]))
    (base / "adversarial_tasks.json").write_text(json.dumps([record]))
    (base / "adversarial_tasks.infeasible.json").write_text(json.dumps([]))
    (base / "adversarial_tasks.dropped_source_data.json").write_text(json.dumps([]))
    (base / "feasibility_report.json").write_text(json.dumps({}))

    rc = drop_image_alt_text.main(["--state-dir", str(tmp_path), "--apply"])

    assert rc == 0
    sidecar = json.loads((base / "adversarial_tasks.dropped_image_alt_text.json").read_text())
    assert sorted(item["dropped_from"] for item in sidecar) == [
        "adversarial_plans.json",
        "adversarial_tasks.json",
    ]


def test_drop_image_alt_text_filters_dropped_source_data_sidecar(tmp_path):
    base = tmp_path / "phase_2"
    base.mkdir()
    record = {"id": "adv-img", "site": "gitlab", "concealment": "image_alt_text"}
    for name in (
        "adversarial_plans.json",
        "adversarial_tasks.json",
        "adversarial_tasks.infeasible.json",
    ):
        (base / name).write_text(json.dumps([]))
    (base / "adversarial_tasks.dropped_source_data.json").write_text(json.dumps([record]))
    (base / "feasibility_report.json").write_text(json.dumps({}))

    rc = drop_image_alt_text.main(["--state-dir", str(tmp_path), "--apply"])

    assert rc == 0
    assert json.loads((base / "adversarial_tasks.dropped_source_data.json").read_text()) == []
    sidecar = json.loads((base / "adversarial_tasks.dropped_image_alt_text.json").read_text())
    assert [(item["id"], item["dropped_from"]) for item in sidecar] == [
        ("adv-img", "adversarial_tasks.dropped_source_data.json")
    ]
