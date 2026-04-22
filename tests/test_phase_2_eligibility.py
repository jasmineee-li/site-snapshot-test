"""Tests for the Phase 2a pre-shard eligibility filter (commit 7).

Exercises :func:`worldsim.phases.phase_2_injections._phase_2a_eligible_tasks`
and the drop log at ``logs/phase_2/dropped_no_contract.json``.

A task is ineligible iff:

* ``kind`` is not None AND
* the contract has no valid methods for this kind on this site, OR
* the only reachable token is ``{benign_user_handle}`` AND no spec for
  the kind has a ``free_text`` body-accepting binding.

Dashboard-list kinds are *eligible* (routable via body-mention). Null
kinds are dropped with their resolver reason so unresolved L3/L4 tasks
never leak into planning.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldsim.phases.phase_2_injections import (
    _phase_2a_eligible_tasks,
    _write_eligibility_drops,
)


class TestEligibility:
    def test_null_kind_is_dropped(self) -> None:
        tasks = [{"id": "t1"}]
        resources = {"t1": {"kind": None, "anchors": {}, "reason": "L3 probe raised: timeout"}}
        eligible, dropped = _phase_2a_eligible_tasks(tasks, resources, "gitlab")
        assert eligible == []
        assert dropped == [
            {
                "task_id": "t1",
                "kind": None,
                "reason": "L3 probe raised: timeout",
                "anchors": {},
                "available_tokens": [],
            }
        ]

    def test_gitlab_issue_with_anchors_stays_eligible(self) -> None:
        tasks = [{"id": "t1"}]
        resources = {
            "t1": {
                "kind": "gitlab_issue",
                "anchors": {"project_path": "foo/bar", "issue_iid": "42"},
            }
        }
        eligible, dropped = _phase_2a_eligible_tasks(tasks, resources, "gitlab")
        assert eligible and dropped == []

    def test_dashboard_list_kind_stays_eligible(self) -> None:
        """Dashboard-list kinds route via @{benign_user_handle} body mention;
        the filter MUST keep them, not drop them."""
        tasks = [{"id": "t1"}, {"id": "t2"}]
        resources = {
            "t1": {
                "kind": "gitlab_dashboard_list",
                "anchors": {"dashboard": "todos"},
            },
            "t2": {
                "kind": "reddit_dashboard_list",
                "anchors": {"dashboard": "submitted"},
            },
        }
        eligible_gitlab, dropped_gitlab = _phase_2a_eligible_tasks([tasks[0]], resources, "gitlab")
        eligible_reddit, dropped_reddit = _phase_2a_eligible_tasks([tasks[1]], resources, "reddit")
        assert dropped_gitlab == []
        assert dropped_reddit == []
        assert len(eligible_gitlab) == 1
        assert len(eligible_reddit) == 1

    def test_synthetic_unknown_kind_dropped(self) -> None:
        tasks = [{"id": "t1"}]
        resources = {"t1": {"kind": "never_registered_kind", "anchors": {"x": "y"}}}
        eligible, dropped = _phase_2a_eligible_tasks(tasks, resources, "gitlab")
        assert eligible == []
        assert len(dropped) == 1
        assert dropped[0]["task_id"] == "t1"
        assert dropped[0]["reason"] == "no_addressable_method_on_site"
        assert dropped[0]["kind"] == "never_registered_kind"
        assert "available_tokens" in dropped[0]
        assert "anchors" in dropped[0]

    def test_reddit_kind_on_gitlab_site_dropped(self) -> None:
        """reddit_submission has no gitlab editor methods."""
        tasks = [{"id": "t1"}]
        resources = {
            "t1": {
                "kind": "reddit_submission",
                "anchors": {"forum_name": "books", "submission_id": "59421"},
            }
        }
        eligible, dropped = _phase_2a_eligible_tasks(tasks, resources, "gitlab")
        assert eligible == []
        assert len(dropped) == 1
        assert dropped[0]["reason"] == "no_addressable_method_on_site"

    def test_mixed_shard_partitions_correctly(self) -> None:
        tasks = [
            {"id": "t1"},  # valid
            {"id": "t2"},  # dropped (unknown kind)
            {"id": "t3"},  # valid (dashboard-list)
            {"id": "t4"},  # null kind dropped
        ]
        resources = {
            "t1": {"kind": "gitlab_issue", "anchors": {"project_path": "a/b", "issue_iid": "1"}},
            "t2": {"kind": "synthetic_kind", "anchors": {}},
            "t3": {"kind": "gitlab_dashboard_list", "anchors": {"dashboard": "todos"}},
            "t4": {"kind": None, "anchors": {}, "reason": "unresolved_target_resource"},
        }
        eligible, dropped = _phase_2a_eligible_tasks(tasks, resources, "gitlab")
        assert {t["id"] for t in eligible} == {"t1", "t3"}
        assert [d["task_id"] for d in dropped] == ["t2", "t4"]

    def test_drop_log_schema(self) -> None:
        tasks = [{"id": "t1"}]
        resources = {"t1": {"kind": "synthetic", "anchors": {"x": "1"}}}
        _, dropped = _phase_2a_eligible_tasks(tasks, resources, "gitlab")
        assert len(dropped) == 1
        row = dropped[0]
        expected_keys = {"task_id", "kind", "reason", "anchors", "available_tokens"}
        assert set(row.keys()) == expected_keys


class TestWriteEligibilityDrops:
    def test_writes_and_appends_per_site(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
        _write_eligibility_drops(
            "gitlab",
            [
                {
                    "task_id": "t1",
                    "kind": "synthetic",
                    "reason": "no_addressable_method_on_site",
                    "anchors": {},
                    "available_tokens": ["{benign_user_handle}"],
                }
            ],
        )
        path = tmp_path / "phase_2" / "dropped_no_contract.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert "gitlab" in data
        assert len(data["gitlab"]) == 1

        # Append another site
        _write_eligibility_drops(
            "reddit",
            [
                {
                    "task_id": "r1",
                    "kind": "other",
                    "reason": "no_addressable_method_on_site",
                    "anchors": {},
                    "available_tokens": [],
                }
            ],
        )
        data = json.loads(path.read_text())
        assert {"gitlab", "reddit"} <= set(data.keys())

    def test_noop_on_empty_dropped_list_still_writes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
        _write_eligibility_drops("gitlab", [])
        # Empty list still creates the site entry (even if empty)
        path = tmp_path / "phase_2" / "dropped_no_contract.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data.get("gitlab") == []
