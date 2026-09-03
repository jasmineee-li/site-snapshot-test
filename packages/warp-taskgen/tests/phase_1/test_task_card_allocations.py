from __future__ import annotations

import pytest

from warp_taskgen.phase_1 import generated_workflows
from warp_taskgen.phase_1.generated_workflows import generation_prompt_addendum
from warp_taskgen.phases import phase_1_generate_new_tasks
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
    assert task_card_generation_count(plan, site_name="reddit") == 20
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


@pytest.mark.parametrize("action_counts", [{}, {"create_issue": 0}, {"create_issue_note": 0}])
def test_generation_count_rejects_multi_action_card_with_any_action_counts(action_counts) -> None:
    plan = {
        "task_cards": [
            {
                **_card("gitlab-card-a", "gitlab", 2),
                "compatible_action_kinds": ["create_issue", "create_issue_note"],
            }
        ]
    }

    with pytest.raises(ValueError, match="multi-action"):
        phase_1_generate_new_tasks._site_requested_count(
            plan,
            novel_tasks_per_site=99,
            action_counts=action_counts,
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


def test_generation_prompt_allocation_is_part_of_shared_and_site_cache_fingerprints(
    monkeypatch, tmp_path
) -> None:
    plan = {
        "task_cards": [
            _card("gitlab-card", "gitlab", 2),
            _card("reddit-card", "reddit", 1),
            _card("rocket-card", "rocketchat", 3),
        ]
    }
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    manifest = {}
    shared = phase_1_generate_new_tasks.compute_generate_new_tasks_shared_inputs_fingerprint(
        benchmark_root=benchmark_root,
        manifest=manifest,
        task_card_plan=plan,
    )
    site = phase_1_generate_new_tasks.EligibleSiteProfile(
        site_name="gitlab",
        profile_path=tmp_path / "BENCHMARK_PROFILE_gitlab.json",
        profile={},
    )
    cache = phase_1_generate_new_tasks.compute_site_cache_fingerprint(
        shared_inputs_fingerprint=shared,
        site=site,
        task_card_plan=plan,
    )
    original = generated_workflows.task_card_generation_prompt_addendum

    def changed_addendum(plan, *, site_name=None):
        return original(plan, site_name=site_name) + "\nallocation prompt revision"

    monkeypatch.setattr(
        generated_workflows, "task_card_generation_prompt_addendum", changed_addendum
    )

    changed_shared = (
        phase_1_generate_new_tasks.compute_generate_new_tasks_shared_inputs_fingerprint(
            benchmark_root=benchmark_root,
            manifest=manifest,
            task_card_plan=plan,
        )
    )
    changed_cache = phase_1_generate_new_tasks.compute_site_cache_fingerprint(
        shared_inputs_fingerprint=changed_shared,
        site=site,
        task_card_plan=plan,
    )

    assert changed_shared != shared
    assert changed_cache != cache

    intermediate_path = tmp_path / "novel_tasks_gitlab.json"
    intermediate_path.write_text("[]")
    intermediate_path.with_suffix(".json.metadata.json").write_text(
        f'{{"fingerprint": "{cache}", "site_name": "gitlab"}}'
    )
    assert (
        phase_1_generate_new_tasks.load_cached_novel_tasks(
            intermediate_path=intermediate_path,
            site_name="gitlab",
            profile={},
            cache_fingerprint=changed_cache,
            expected_task_count=0,
        )
        is None
    )
