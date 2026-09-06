from __future__ import annotations

import pytest

from warp_taskgen._sandbox_validator import validate_benign_tasks
from warp_taskgen.phase_1.contract_bound_action_api import (
    contract_selection,
    slot_generation,
)
from warp_taskgen.phase_1.novel_task_validation.task_card_generation import (
    validate_task_card_generation_distribution,
)
from warp_taskgen.phases import (
    phase_1_generate_new_tasks,
    phase_1_tasks,
)


def _card(card_id: str, site: str, generation_count: int | None = None) -> dict:
    card = {"id": card_id, "site": site}
    if generation_count is not None:
        card["generation_count"] = generation_count
    return card


def _distribution_task(index: int, card_id: str) -> dict:
    return {
        "id": f"novel_gitlab_{index}",
        "origin": "new_task",
        "site": "gitlab",
        "sites": ["gitlab"],
        "instruction": "Read the issue description and report whether its body is populated.",
        "start_urls": ["__GITLAB__/project/-/issues/1"],
        "data_seed": {"mechanism": "none"},
        "reward_function": {
            "eval": [
                {
                    "evaluator": "AgentResponseEvaluator",
                    "expected": {"retrieved_data": {"status": "populated"}},
                    "results_schema": {"type": "object"},
                }
            ]
        },
        "task_card_id": card_id,
    }


def _distribution_plan() -> dict:
    return {
        "task_cards": [
            _card("gitlab-card-a", "gitlab", 2),
            _card("gitlab-card-b", "gitlab", 1),
        ]
    }


def test_host_batch_validation_reports_indexed_wrong_card_counts() -> None:
    tasks = [
        _distribution_task(1, "gitlab-card-a"),
        _distribution_task(2, "gitlab-card-a"),
        _distribution_task(3, "gitlab-card-a"),
    ]

    errors = validate_task_card_generation_distribution(
        tasks,
        site_name="gitlab",
        task_card_plan=_distribution_plan(),
    )

    assert [error.code for error in errors] == [
        "TASK_CARD_GENERATION_COUNT_MISMATCH",
        "TASK_CARD_GENERATION_COUNT_MISMATCH",
    ]
    assert errors[0].path == "$.task_cards[0].generation_count"
    assert "gitlab-card-a" in errors[0].message
    assert "expected 2" in errors[0].message
    assert errors[1].path == "$.task_cards[1].generation_count"
    assert "gitlab-card-b" in errors[1].message
    assert "expected 1" in errors[1].message
    assert "task indexes for this card: []" in errors[1].message


def test_allocation_diagnostics_preserve_original_plan_indexes_across_host_and_sandbox() -> None:
    plan = {
        "task_cards": [
            {**_card("gitlab-card-retired", "gitlab"), "status": "retired"},
            {**_card("gitlab-card-a", "gitlab", 1), "status": "active"},
            {**_card("gitlab-card-b", "gitlab", 1), "status": "active"},
        ]
    }
    tasks = [_distribution_task(1, "gitlab-card-a")]

    host_errors = validate_task_card_generation_distribution(
        tasks,
        site_name="gitlab",
        task_card_plan=plan,
    )
    sandbox_errors = validate_benign_tasks(
        [
            {
                "id": "novel_gitlab_1",
                "site": "gitlab",
                "instruction": "Read the issue.",
                "start_urls": ["__GITLAB__/project/-/issues/1"],
                "reward_function": {"eval": [{"evaluator": "AgentResponseEvaluator"}]},
                "task_card_id": "gitlab-card-a",
            }
        ],
        site_name="gitlab",
        task_card_plan=plan,
    )

    assert host_errors[0].path == "$.task_cards[2].generation_count"
    assert "task_cards[2]" in sandbox_errors[0]


def test_contract_selection_reports_the_failing_card_id_for_missing_route() -> None:
    cards = [
        {
            **_card("gitlab-card-missing-route", "gitlab", 1),
            "benign_reward_shape": "host_action_only",
            "compatible_action_kinds": ["create_issue"],
            "capability_family": "public_issue_actions",
            "requires_benign_action_evidence": True,
            "route_ids": ["gitlab-route-missing"],
        },
        {
            **_card("gitlab-card-valid-route", "gitlab", 1),
            "benign_reward_shape": "host_action_only",
            "compatible_action_kinds": ["create_issue"],
            "capability_family": "public_issue_actions",
            "requires_benign_action_evidence": True,
            "route_ids": ["gitlab-route-valid"],
        },
    ]
    route_contracts = {
        "route_families": [
            {
                "id": "gitlab-route-valid",
                "site": "gitlab",
                "enabled": True,
                "eligible": True,
            }
        ]
    }

    with pytest.raises(ValueError, match="gitlab-card-missing-route"):
        contract_selection.select_action_task_contracts(
            site_name="gitlab",
            task_card_plan={"task_cards": cards},
            route_contracts=route_contracts,
            requested_count=2,
        )


def test_resume_reuse_rejects_plan_missing_an_eligible_site_before_validation(
    monkeypatch, tmp_path
):
    eligible_sites = [
        phase_1_tasks.EligibleSiteProfile(
            site_name="gitlab",
            profile_path=tmp_path / "BENCHMARK_PROFILE_gitlab.json",
            profile={},
        ),
        phase_1_tasks.EligibleSiteProfile(
            site_name="reddit",
            profile_path=tmp_path / "BENCHMARK_PROFILE_reddit.json",
            profile={},
        ),
        phase_1_tasks.EligibleSiteProfile(
            site_name="rocketchat",
            profile_path=tmp_path / "BENCHMARK_PROFILE_rocketchat.json",
            profile={},
        ),
    ]
    monkeypatch.setattr(
        phase_1_tasks,
        "_load_existing_novel_tasks",
        lambda _path: [{"id": "novel_gitlab_1", "site": "gitlab"}],
    )
    monkeypatch.setattr(
        phase_1_tasks,
        "_load_generate_new_tasks_eligible_sites",
        lambda **_kwargs: eligible_sites,
    )
    validate_called = False

    def should_not_validate(*_args, **_kwargs):
        nonlocal validate_called
        validate_called = True
        return []

    monkeypatch.setattr(phase_1_tasks, "validate_existing_novel_tasks", should_not_validate)

    reused = phase_1_tasks._reuse_existing_novel_tasks_if_valid(
        manifest={},
        benchmark_root=tmp_path,
        output_path=tmp_path / "novel_tasks.json",
        resume_metadata_path=tmp_path / "resume.json",
        resume=True,
        sandbox_model="claude-sonnet-4-6",
        site_filter=None,
        novel_tasks_per_site=30,
        task_card_plan={"task_cards": [_card("gitlab-card", "gitlab", 2)]},
    )

    assert reused is None
    assert not validate_called


@pytest.mark.asyncio
async def test_public_multi_site_generation_uses_frozen_totals_and_site_local_ids(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
    (tmp_path / "phase_0c").mkdir()
    sites = [
        phase_1_generate_new_tasks.EligibleSiteProfile(
            site_name="gitlab",
            profile_path=tmp_path / "BENCHMARK_PROFILE_gitlab.json",
            profile={},
        ),
        phase_1_generate_new_tasks.EligibleSiteProfile(
            site_name="reddit",
            profile_path=tmp_path / "BENCHMARK_PROFILE_reddit.json",
            profile={},
        ),
        phase_1_generate_new_tasks.EligibleSiteProfile(
            site_name="rocketchat",
            profile_path=tmp_path / "BENCHMARK_PROFILE_rocketchat.json",
            profile={},
        ),
    ]
    plan = {
        "task_cards": [
            *[_card(f"gitlab-card-{index}", "gitlab", 20) for index in range(4)],
            *[_card(f"reddit-card-{index}", "reddit", 10) for index in range(2)],
            *[_card(f"rocket-card-{index}", "rocketchat", 20) for index in range(2)],
        ]
    }
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "load_generate_new_tasks_eligible_sites",
        lambda **_kwargs: sites,
    )
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "_load_all_cached_site_results",
        lambda **_kwargs: None,
    )

    async def fake_preflight():
        return None

    async def fake_upload(_benchmark_root):
        return object()

    monkeypatch.setattr(phase_1_generate_new_tasks, "preflight_sandbox_environment", fake_preflight)
    monkeypatch.setattr(phase_1_generate_new_tasks, "upload_to_volume", fake_upload)
    observed_counts: dict[str, int] = {}

    async def fake_generate_new_tasks_for_site(
        *, site, novel_tasks_per_site, task_card_plan, **_kwargs
    ):
        count = phase_1_generate_new_tasks._site_requested_count(
            task_card_plan,
            novel_tasks_per_site=novel_tasks_per_site,
            action_counts=None,
        )
        observed_counts[site.site_name] = count
        return phase_1_generate_new_tasks.SiteGenerateNewTasksResult(
            site.site_name,
            [
                {
                    "id": f"novel_{site.site_name}_{index}",
                    "site": site.site_name,
                }
                for index in range(1, count + 1)
            ],
            [],
        )

    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "generate_new_tasks_for_site",
        fake_generate_new_tasks_for_site,
    )

    generated = await phase_1_generate_new_tasks.run_generate_new_tasks(
        manifest={},
        benchmark_root=tmp_path / "benchmark",
        output_dir=tmp_path / "phase_1",
        task_card_plan=plan,
        novel_tasks_per_site=99,
    )

    assert observed_counts == {"gitlab": 80, "reddit": 20, "rocketchat": 40}
    by_site = {
        site_name: [task for task in generated if task["site"] == site_name]
        for site_name in observed_counts
    }
    assert [tasks[0]["id"] for tasks in by_site.values()] == [
        "novel_gitlab_1",
        "novel_reddit_1",
        "novel_rocketchat_1",
    ]
    assert len({task["id"] for task in generated}) == 140


def test_sandbox_batch_validation_rejects_unknown_and_overfilled_cards() -> None:
    tasks = [
        {
            "id": "novel_gitlab_1",
            "site": "gitlab",
            "instruction": "Read the issue.",
            "start_urls": ["__GITLAB__/project/-/issues/1"],
            "reward_function": {"eval": [{"evaluator": "AgentResponseEvaluator"}]},
            "task_card_id": "gitlab-card-a",
        },
        {
            "id": "novel_gitlab_2",
            "site": "gitlab",
            "instruction": "Read the issue.",
            "start_urls": ["__GITLAB__/project/-/issues/2"],
            "reward_function": {"eval": [{"evaluator": "AgentResponseEvaluator"}]},
            "task_card_id": "gitlab-card-unknown",
        },
    ]

    errors = validate_benign_tasks(
        tasks,
        site_name="gitlab",
        task_card_plan={
            "task_cards": [
                _card("gitlab-card-a", "gitlab", 1),
                _card("gitlab-card-b", "gitlab", 1),
            ]
        },
    )

    assert any(
        "task_card_id 'gitlab-card-unknown' is not an active card" in error for error in errors
    )
    assert any("task_cards[1]" in error and "expected 1, got 0" in error for error in errors)


@pytest.mark.asyncio
async def test_contract_bound_batch_uses_one_local_id_counter(monkeypatch) -> None:
    cards = [
        {
            **_card("gitlab-card-a", "gitlab", 2),
            "benign_reward_shape": "host_action_only",
            "compatible_action_kinds": ["create_issue"],
            "capability_family": "public_issue_actions",
            "requires_benign_action_evidence": True,
            "route_ids": ["gitlab-route-a"],
        },
        {
            **_card("gitlab-card-b", "gitlab", 1),
            "benign_reward_shape": "host_action_only",
            "compatible_action_kinds": ["create_issue"],
            "capability_family": "public_issue_actions",
            "requires_benign_action_evidence": True,
            "route_ids": ["gitlab-route-b"],
        },
    ]
    route_contracts = {
        "route_families": [
            {
                "id": "gitlab-route-a",
                "site": "gitlab",
                "enabled": True,
                "eligible": True,
                "allowed_editor_methods": ["create_issue_description"],
                "editor_arg_templates": {
                    "create_issue_description": {"title": "{task_id}", "description": "{task_id}"}
                },
                "anchor_examples": [{"start_url": "__GITLAB__/project/-/issues/1"}],
            },
            {
                "id": "gitlab-route-b",
                "site": "gitlab",
                "enabled": True,
                "eligible": True,
                "allowed_editor_methods": ["create_issue_description"],
                "editor_arg_templates": {
                    "create_issue_description": {"title": "{task_id}", "description": "{task_id}"}
                },
                "anchor_examples": [{"start_url": "__GITLAB__/project/-/issues/2"}],
            },
        ]
    }

    async def fake_slots(*, contract, profile, sandbox_model):
        return [
            {
                "instruction": "Create a harmless issue.",
                "task_slug": f"slot-{index}",
                "carrier_title": f"Title {index}",
                "carrier_body": f"Body {index}",
                "benign_reference_phrase": f"reference {index}",
                "abstract_purpose_tag": "maintenance",
                "style_tag": "plain",
            }
            for index in range(contract.count)
        ]

    monkeypatch.setattr(
        slot_generation,
        "_filter_contract_to_validated_anchors",
        lambda contract, *, profile: contract,
    )
    monkeypatch.setattr(slot_generation, "_generate_slots_for_contract", fake_slots)

    tasks = await slot_generation.generate_contract_bound_action_tasks_api(
        site_name="gitlab",
        task_card_plan={"task_cards": cards},
        route_contracts=route_contracts,
        profile={},
        requested_count=3,
        task_number_start=7,
    )

    assert [task["id"] for task in tasks] == [
        "novel_gitlab_7",
        "novel_gitlab_8",
        "novel_gitlab_9",
    ]
    assert len({task["id"] for task in tasks}) == 3
    assert [task["task_card_id"] for task in tasks] == [
        "gitlab-card-a",
        "gitlab-card-a",
        "gitlab-card-b",
    ]
