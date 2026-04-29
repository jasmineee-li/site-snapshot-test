from __future__ import annotations

import json

import pytest

from scripts import pvpo_live_validation
from worldsim.config import BenchmarkConfig


def test_pvpo_live_validation_requires_explicit_tasks_path() -> None:
    parser = pvpo_live_validation.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_pvpo_live_validation_load_json_array_reports_missing_path(tmp_path) -> None:
    missing = tmp_path / "adversarial_tasks.json"

    with pytest.raises(FileNotFoundError, match="Pass an explicit artifact path"):
        pvpo_live_validation.load_json_array(missing, label="Phase 2 tasks artifact")


def test_pvpo_live_validation_derives_wasp_sites_from_tasks() -> None:
    config = BenchmarkConfig.model_validate(
        {
            "benchmark_name": "WebArena Verified",
            "benchmark_codebase": "vendors/webarena-verified",
            "instances": [
                {
                    "site_name": "gitlab",
                    "site_url": "http://host:8023",
                    "agent_auth": {"type": "none"},
                },
                {
                    "site_name": "reddit",
                    "site_url": "http://host:9999",
                    "agent_auth": {"type": "none"},
                },
                {
                    "site_name": "shopping",
                    "site_url": "http://host:7770",
                    "agent_auth": {"type": "none"},
                },
            ],
        }
    )
    tasks = [
        {"site": "shopping"},
        {"site": "reddit"},
        {"site": "gitlab", "delivery_channel": {"delivery_site": "reddit"}},
    ]

    assert pvpo_live_validation.sites_to_validate(tasks, config) == ["gitlab", "reddit"]


def test_pvpo_live_validation_load_json_array_filters_non_objects(tmp_path) -> None:
    path = tmp_path / "adversarial_tasks.json"
    path.write_text(json.dumps([{"id": "task"}, None, "bad"]))

    assert pvpo_live_validation.load_json_array(path, label="Phase 2 tasks artifact") == [
        {"id": "task"}
    ]
