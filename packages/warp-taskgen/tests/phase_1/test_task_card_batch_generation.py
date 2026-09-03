from __future__ import annotations

import asyncio
import json

import pytest

from warp_taskgen.phase_1 import task_card_batch_generation
from warp_taskgen.phases import phase_1_generate_new_tasks


def _card(card_id: str, site: str, generation_count: int | None = None) -> dict:
    card = {"id": card_id, "site": site}
    if generation_count is not None:
        card["generation_count"] = generation_count
    return card


@pytest.mark.asyncio
async def test_multi_card_generation_slices_backends_and_rekeys_ids(monkeypatch, tmp_path) -> None:
    """Each explicit card quota gets its own backend while the site keeps one ID namespace."""

    profile = {
        "site_name": "gitlab",
        "verification_capabilities": [{"eval_type": "AgentResponseEvaluator"}],
    }
    profile_path = tmp_path / "BENCHMARK_PROFILE_gitlab.json"
    profile_path.write_text(json.dumps(profile))
    output_dir = tmp_path / "phase_1"
    output_dir.mkdir()
    site = phase_1_generate_new_tasks.EligibleSiteProfile(
        site_name="gitlab",
        profile_path=profile_path,
        profile=profile,
    )
    plan = {
        "schema_version": 1,
        "task_capability_profile": "tier2_pure_action_paper",
        "task_cards": [
            {
                **_card("gitlab-action", "gitlab", 1),
                "benign_reward_shape": "host_action_only",
                "compatible_action_kinds": ["create_issue"],
                "capability_family": "public_issue_actions",
                "requires_benign_action_evidence": True,
                "route_ids": ["gitlab-route"],
            },
            {
                **_card("gitlab-feature", "gitlab", 1),
                "benign_reward_shape": "agent_response",
                "compatible_action_kinds": [],
                "route_ids": ["gitlab-route"],
            },
        ],
    }
    route_contracts = {"route_families": [{"id": "gitlab-route"}]}
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "build_task_route_contracts",
        lambda **_: route_contracts,
    )
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "load_cached_novel_tasks",
        lambda **_: None,
    )
    validation_calls: list[dict] = []

    def fake_validate(tasks, **kwargs):
        validation_calls.append(kwargs)
        return tasks, []

    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "validate_generated_novel_tasks_detailed",
        fake_validate,
    )
    monkeypatch.setattr(
        task_card_batch_generation,
        "validate_generated_novel_tasks_detailed",
        fake_validate,
    )
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "_compile_phase1_model_owned_features",
        lambda tasks, **_: tasks,
    )
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "_compile_phase1_feature_tasks",
        lambda tasks, **_: tasks,
    )
    monkeypatch.setattr(
        task_card_batch_generation,
        "restore_compiled_tasks",
        lambda tasks, **_: tasks,
    )
    api_calls: list[dict] = []

    async def fake_api(**kwargs):
        api_calls.append(kwargs)
        return [
            {
                "id": "novel_gitlab_1",
                "origin": "new_task",
                "site": "gitlab",
                "sites": ["gitlab"],
                "task_card_id": "gitlab-action",
            }
        ]

    sandbox_calls: list[str] = []

    async def fake_sandbox(**kwargs):
        sandbox_calls.append(kwargs["prompt"])
        return {
            phase_1_generate_new_tasks.NOVEL_TASK_OUTPUT_PATH: json.dumps(
                [
                    {
                        "id": "novel_gitlab_1",
                        "origin": "new_task",
                        "site": "gitlab",
                        "sites": ["gitlab"],
                        "task_card_id": "gitlab-feature",
                    }
                ]
            ),
            "_summary": None,
        }

    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "generate_contract_bound_action_tasks_api",
        fake_api,
    )
    monkeypatch.setattr(phase_1_generate_new_tasks, "run_claude_in_sandbox", fake_sandbox)

    result = await phase_1_generate_new_tasks.generate_new_tasks_for_site(
        site=site,
        benchmark_volume=object(),
        output_dir=output_dir,
        cache_fingerprint="multi-card-test",
        novel_tasks_per_site=99,
        task_card_plan=plan,
    )

    assert result.errors == []
    assert len(api_calls) == 1
    assert len(sandbox_calls) == 1
    assert [task["id"] for task in result.benign_tasks] == [
        "novel_gitlab_1",
        "novel_gitlab_2",
    ]
    assert [task["task_card_id"] for task in result.benign_tasks] == [
        "gitlab-action",
        "gitlab-feature",
    ]
    assert any(
        call["task_card_plan"]["task_cards"] == plan["task_cards"]
        and call["expected_task_count"] == 2
        for call in validation_calls
    )


def test_compare_act_feature_owner_never_uses_contract_bound_backend() -> None:
    """A host-action-only compare-act card remains model-owned generation."""

    card = {
        "id": "gitlab_compare_act",
        "site": "gitlab",
        "benign_reward_shape": "host_action_only",
        "generation_contract": {
            "family": "gitlab_compare_act",
            "version": 1,
            "record_keys": ["release-blocker", "docs-gap", "closed-bug"],
            "decision_rule": {"state": "open", "dependency": "release-4"},
        },
    }

    assert (
        phase_1_generate_new_tasks._task_card_plan_is_host_action_only({"task_cards": [card]})
        is False
    )


def test_task_card_slices_preserve_root_metadata_and_authored_order() -> None:
    plan = {
        "schema_version": 4,
        "task_capability_profile": "tier2_pure_action_paper",
        "task_cards": [
            _card("gitlab-card-first", "gitlab", 2),
            {**_card("retired", "gitlab", 7), "status": "retired"},
            _card("gitlab-card-second", "gitlab", 3),
        ],
    }

    slices = task_card_batch_generation.task_card_generation_slices(
        plan,
        site_name="gitlab",
    )

    assert [item.task_number_start for item in slices] == [1, 3]
    assert [item.task_card_plan["task_cards"][0]["id"] for item in slices] == [
        "gitlab-card-first",
        "gitlab-card-second",
    ]
    assert all(item.task_card_plan["schema_version"] == 4 for item in slices)
    assert all(
        item.task_card_plan["task_capability_profile"] == "tier2_pure_action_paper"
        for item in slices
    )


@pytest.mark.parametrize("raw_id", ["not-a-novel-id", "novel_reddit_1"])
def test_sandbox_slice_rekey_rejects_malformed_or_foreign_ids(raw_id: str) -> None:
    with pytest.raises(ValueError, match="before canonical rekey"):
        task_card_batch_generation.rekey_sandbox_task_ids(
            [{"id": raw_id}],
            site_name="gitlab",
            task_number_start=2,
        )


@pytest.mark.asyncio
async def test_multi_card_slice_failure_does_not_promote_site_cache(monkeypatch, tmp_path) -> None:
    """A failed child leaves no combined cache for a partially generated site."""

    profile = {
        "site_name": "gitlab",
        "verification_capabilities": [{"eval_type": "AgentResponseEvaluator"}],
    }
    profile_path = tmp_path / "BENCHMARK_PROFILE_gitlab.json"
    profile_path.write_text(json.dumps(profile))
    output_dir = tmp_path / "phase_1"
    output_dir.mkdir()
    site = phase_1_generate_new_tasks.EligibleSiteProfile(
        site_name="gitlab",
        profile_path=profile_path,
        profile=profile,
    )
    plan = {
        "task_capability_profile": "tier2_pure_action_paper",
        "task_cards": [
            {
                **_card("gitlab-action", "gitlab", 1),
                "benign_reward_shape": "host_action_only",
                "compatible_action_kinds": ["create_issue"],
                "capability_family": "public_issue_actions",
                "requires_benign_action_evidence": True,
                "route_ids": ["gitlab-route"],
            },
            {
                **_card("gitlab-feature", "gitlab", 1),
                "benign_reward_shape": "agent_response",
                "route_ids": ["gitlab-route"],
            },
        ],
    }
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "build_task_route_contracts",
        lambda **_: {"route_families": [{"id": "gitlab-route"}]},
    )
    monkeypatch.setattr(phase_1_generate_new_tasks, "load_cached_novel_tasks", lambda **_: None)
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "validate_generated_novel_tasks_detailed",
        lambda tasks, **_: (tasks, []),
    )
    monkeypatch.setattr(
        task_card_batch_generation,
        "validate_generated_novel_tasks_detailed",
        lambda tasks, **_: (tasks, []),
    )
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "_compile_phase1_model_owned_features",
        lambda tasks, **_: tasks,
    )
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "_compile_phase1_feature_tasks",
        lambda tasks, **_: tasks,
    )
    monkeypatch.setattr(
        task_card_batch_generation,
        "restore_compiled_tasks",
        lambda tasks, **_: tasks,
    )

    async def fake_api(**_kwargs):
        return [{"id": "novel_gitlab_1", "task_card_id": "gitlab-action"}]

    async def failed_sandbox(**_kwargs):
        return {"_summary": None}

    monkeypatch.setattr(
        phase_1_generate_new_tasks, "generate_contract_bound_action_tasks_api", fake_api
    )
    monkeypatch.setattr(phase_1_generate_new_tasks, "run_claude_in_sandbox", failed_sandbox)

    result = await phase_1_generate_new_tasks.generate_new_tasks_for_site(
        site=site,
        benchmark_volume=object(),
        output_dir=output_dir,
        cache_fingerprint="multi-card-failure",
        task_card_plan=plan,
        novel_tasks_per_site=99,
    )

    assert result.benign_tasks == []
    assert any("sandbox did not produce benign_tasks.json" in error for error in result.errors)
    assert not (output_dir / "novel_tasks_gitlab.json").exists()
    assert not (output_dir / "novel_tasks_gitlab.json.metadata.json").exists()


@pytest.mark.asyncio
async def test_collect_card_slices_waits_for_siblings_before_aggregating_exception() -> None:
    """A failed slice does not leave a concurrent sibling running in the background."""

    sibling_finished = asyncio.Event()
    slices = (
        task_card_batch_generation.TaskCardGenerationSlice({"task_cards": []}, 1),
        task_card_batch_generation.TaskCardGenerationSlice({"task_cards": []}, 1),
    )

    async def generate_slice(_card_slice, index):
        if index == 0:
            raise RuntimeError("slice failed")
        await asyncio.sleep(0.01)
        sibling_finished.set()
        return task_card_batch_generation.CardSliceResult([], [])

    result = await task_card_batch_generation.collect_card_slices(
        card_slices=slices,
        generate_slice=generate_slice,
        expected_task_count=0,
        site_name="gitlab",
        profile={},
        route_contracts={},
        task_card_plan={"task_cards": []},
        host_compiled_evaluator_types=frozenset(),
    )

    assert sibling_finished.is_set()
    assert result.benign_tasks == []
    assert result.errors == ["card slice 1: RuntimeError: slice failed"]


@pytest.mark.asyncio
async def test_collect_card_slices_propagates_cancelled_slice() -> None:
    slices = (task_card_batch_generation.TaskCardGenerationSlice({"task_cards": []}, 1),)

    async def generate_slice(_card_slice, _index):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await task_card_batch_generation.collect_card_slices(
            card_slices=slices,
            generate_slice=generate_slice,
            expected_task_count=0,
            site_name="gitlab",
            profile={},
            route_contracts={},
            task_card_plan={"task_cards": []},
            host_compiled_evaluator_types=frozenset(),
        )
