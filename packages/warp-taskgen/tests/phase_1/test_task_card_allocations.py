from __future__ import annotations

import pytest

from warp_taskgen._sandbox_validator import validate_benign_tasks
from warp_taskgen.phase_1.generated_workflows import generation_prompt_addendum
from warp_taskgen.phase_1.novel_task_validation._impl import (
    _validate_task_card_generation_distribution,
)
from warp_taskgen.phases import phase_1_contract_bound_action_api, phase_1_generate_new_tasks
from warp_taskgen.phases.phase_1_task_cards import (
    TaskCardPlanError,
    task_card_generation_count,
    task_card_generation_counts,
    validate_task_card_plan,
)


def _card(card_id: str, site: str, generation_count: int | None = None) -> dict:
    card = {"id": card_id, "site": site}
    if generation_count is not None:
        card["generation_count"] = generation_count
    return card


def test_generation_count_sums_per_site_for_frozen_allocations() -> None:
    plan = {
        "schema_version": 1,
        "task_cards": [
            *[_card(f"gitlab-card-{index}", "gitlab", 20) for index in range(4)],
            *[_card(f"reddit-card-{index}", "reddit", 10) for index in range(2)],
            *[_card(f"rocket-card-{index}", "rocketchat", 20) for index in range(2)],
        ],
    }

    validate_task_card_plan(plan)

    assert task_card_generation_count(plan, site_name="gitlab") == 80
    assert task_card_generation_count(plan, site_name="rocketchat") == 40
    assert task_card_generation_counts(plan, site_name="gitlab") == {
        f"gitlab-card-{index}": 20 for index in range(4)
    }


def test_generation_count_is_optional_and_partial_sites_fail_closed() -> None:
    legacy_plan = {"task_cards": [_card("legacy", "gitlab")]}
    assert task_card_generation_counts(legacy_plan, site_name="gitlab") is None

    partial_plan = {
        "task_cards": [
            _card("gitlab-card-a", "gitlab", 20),
            _card("gitlab-card-b", "gitlab"),
        ]
    }
    with pytest.raises(TaskCardPlanError, match="must all declare generation_count"):
        validate_task_card_plan(partial_plan)


def test_generation_count_overrides_uniform_fallback_and_conflicting_action_counts_fail() -> None:
    plan = {
        "task_cards": [
            {**_card("gitlab-card-a", "gitlab", 2), "compatible_action_kinds": ["create_issue"]},
            {**_card("gitlab-card-b", "gitlab", 1), "compatible_action_kinds": ["create_issue"]},
        ]
    }

    assert (
        phase_1_generate_new_tasks._site_requested_count(
            plan,
            novel_tasks_per_site=99,
            action_counts=None,
        )
        == 3
    )
    with pytest.raises(ValueError, match="generation_count/action_counts conflict"):
        phase_1_generate_new_tasks._site_requested_count(
            plan,
            novel_tasks_per_site=99,
            action_counts={"create_issue": 2},
        )


def test_generation_prompt_describes_exact_card_ranges() -> None:
    plan = {
        "task_cards": [
            _card("gitlab-card-a", "gitlab", 2),
            _card("gitlab-card-b", "gitlab", 1),
        ]
    }

    prompt = generation_prompt_addendum(plan, site_name="gitlab")

    assert "Generate exactly 3 tasks" in prompt
    assert "task_card_id `gitlab-card-a`: exactly 2" in prompt
    assert "novel_gitlab_1` through `novel_gitlab_2" in prompt
    assert "task_card_id `gitlab-card-b`: exactly 1" in prompt
    assert "novel_gitlab_3` through `novel_gitlab_3" in prompt
    assert "one global 1-based counter" in prompt


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

    errors = _validate_task_card_generation_distribution(
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
        phase_1_contract_bound_action_api,
        "_filter_contract_to_validated_anchors",
        lambda contract, *, profile: contract,
    )
    monkeypatch.setattr(
        phase_1_contract_bound_action_api, "_generate_slots_for_contract", fake_slots
    )

    tasks = await phase_1_contract_bound_action_api.generate_contract_bound_action_tasks_api(
        site_name="gitlab",
        task_card_plan={"task_cards": cards},
        route_contracts=route_contracts,
        profile={},
        requested_count=3,
    )

    assert [task["id"] for task in tasks] == [
        "novel_gitlab_1",
        "novel_gitlab_2",
        "novel_gitlab_3",
    ]
    assert len({task["id"] for task in tasks}) == 3
    assert [task["task_card_id"] for task in tasks] == [
        "gitlab-card-a",
        "gitlab-card-a",
        "gitlab-card-b",
    ]
