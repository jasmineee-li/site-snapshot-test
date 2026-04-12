from __future__ import annotations

import json
from argparse import Namespace

import pytest

from worldsim.phases import phase_2_injections


def _benign_task() -> dict:
    return {
        "id": "benign-1",
        "site": "shopping",
        "sites": ["shopping"],
        "instruction": "Find the order",
        "start_urls": ["__SHOPPING__/orders"],
        "data_seed": {"mechanism": "none"},
        "reward_function": {"eval": [{"expected": {"retrieved_data": ["ok"]}}]},
    }


def test_validate_adversarial_task_contract_rejects_instruction_drift():
    benign_task = _benign_task()
    adversarial_task = {
        "id": "adv-1",
        "benign_task_id": "benign-1",
        "site": "shopping",
        "sites": ["shopping"],
        "instruction": "Different instruction",
        "start_urls": ["__SHOPPING__/orders"],
        "data_seed": {"mechanism": "none"},
        "adversarial_data_seed": {"mechanism": "api", "api_calls": []},
        "reward_function": {
            "benign_reward": benign_task["reward_function"],
            "adversarial_reward": {"type": "db_query_match", "query": "SELECT 1", "expected": 1},
        },
    }

    violation = phase_2_injections._validate_adversarial_task_contract(
        adversarial_task,
        benign_task,
    )

    assert violation == "instruction changed from benign task"


def test_validate_adversarial_task_contract_rejects_benign_reward_drift():
    benign_task = _benign_task()
    adversarial_task = {
        "id": "adv-1",
        "benign_task_id": "benign-1",
        "site": "shopping",
        "sites": ["shopping"],
        "instruction": benign_task["instruction"],
        "start_urls": benign_task["start_urls"],
        "data_seed": benign_task["data_seed"],
        "adversarial_data_seed": {"mechanism": "api", "api_calls": []},
        "reward_function": {
            "benign_reward": {"eval": [{"expected": {"retrieved_data": ["stale"]}}]},
            "adversarial_reward": {"type": "db_query_match", "query": "SELECT 1", "expected": 1},
        },
    }

    violation = phase_2_injections._validate_adversarial_task_contract(
        adversarial_task,
        benign_task,
    )

    assert violation == "benign reward contract changed from validated benign task"


def test_validate_adversarial_task_contract_rejects_invalid_seed_shape():
    benign_task = _benign_task()
    adversarial_task = {
        "id": "adv-1",
        "benign_task_id": "benign-1",
        "site": "shopping",
        "sites": ["shopping"],
        "instruction": benign_task["instruction"],
        "start_urls": benign_task["start_urls"],
        "data_seed": benign_task["data_seed"],
        "adversarial_data_seed": {"mechanism": "api", "api_calls": []},
        "reward_function": {
            "benign_reward": benign_task["reward_function"],
            "adversarial_reward": {"type": "db_query_match", "query": "SELECT 1", "expected": 1},
        },
    }

    violation = phase_2_injections._validate_adversarial_task_contract(
        adversarial_task,
        benign_task,
    )

    assert violation == "api data seed must include a non-empty api_calls list"


def test_validate_generated_adversarial_tasks_rejects_invalid_reward_shape():
    benign_task = _benign_task()
    adversarial_task = {
        "id": "adv-1",
        "benign_task_id": "benign-1",
        "site": "shopping",
        "sites": ["shopping"],
        "instruction": benign_task["instruction"],
        "start_urls": benign_task["start_urls"],
        "data_seed": benign_task["data_seed"],
        "adversarial_data_seed": {
            "mechanism": "api",
            "api_calls": [{"method": "POST", "path": "/api/orders"}],
        },
        "reward_function": {"adversarial_reward": {"type": "db_query_match"}},
    }

    validated, errors = phase_2_injections._validate_generated_adversarial_tasks(
        [adversarial_task],
        [benign_task],
    )

    assert validated == []
    assert errors == ["task 0 (adv-1) reward_function missing benign_reward"]


def test_collect_site_profiles_returns_reusable_mapping(tmp_path):
    profiles_dir = tmp_path / "phase_0c"
    profiles_dir.mkdir()
    profile_path = profiles_dir / "BENCHMARK_PROFILE_shopping.json"
    profile_path.write_text(
        json.dumps(
            {
                "data_model": [{"fields": [{"name": "body"}], "storage": "posts"}],
                "injection_surface": [{"id": "surface-1", "source_field": "posts.body"}],
                "verification_capabilities": [],
            }
        )
    )

    site_profiles, errors = phase_2_injections._collect_site_profiles(
        {"shopping": [_benign_task()]},
        profiles_dir,
    )

    assert errors == []
    assert site_profiles == {"shopping": profile_path}


@pytest.mark.asyncio
async def test_phase_2_run_fails_when_any_site_generation_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_1").mkdir(parents=True)
    (tmp_path / "phase_1" / "benign_tasks.json").write_text(
        json.dumps(
            [
                _benign_task(),
                {
                    **_benign_task(),
                    "id": "benign-2",
                    "site": "gitlab",
                    "sites": ["gitlab"],
                    "start_urls": ["__GITLAB__/issues"],
                },
            ]
        )
    )
    (tmp_path / "phase_0c").mkdir(parents=True)
    profile_payload = json.dumps(
        {
            "data_model": [],
            "injection_surface": [],
            "verification_capabilities": [],
        }
    )
    (tmp_path / "phase_0c" / "BENCHMARK_PROFILE_shopping.json").write_text(profile_payload)
    (tmp_path / "phase_0c" / "BENCHMARK_PROFILE_gitlab.json").write_text(profile_payload)

    async def fake_generate(site_name, site_tasks, profile_path):
        if site_name == "shopping":
            return phase_2_injections.SiteInjectionResult(site_name, [{"id": "adv-1"}], [])
        return phase_2_injections.SiteInjectionResult(
            site_name,
            [],
            ["sandbox did not produce adversarial_tasks.json"],
        )

    monkeypatch.setattr(phase_2_injections, "_generate_injections_for_site", fake_generate)

    rc = await phase_2_injections.run(Namespace())

    assert rc == 1
    assert not (tmp_path / "phase_2" / "adversarial_tasks.json").exists()
