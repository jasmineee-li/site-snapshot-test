"""Offline vertical-slice checks for generated GitLab comparison content."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from warp_taskgen.phase_1.gitlab_compare_act import bind_gitlab_compare_act_target
from warp_taskgen.phase_1.gitlab_compare_decide_binding import (
    bind_gitlab_compare_decide_attempt,
)
from warp_taskgen.phase_1.gitlab_compare_decide_generation import (
    compile_phase1_gitlab_compare_act_task,
    compile_phase1_gitlab_compare_decide_task,
    gitlab_compare_generation_prompt_addendum,
    validate_gitlab_compare_decide_task,
)
from warp_taskgen.phase_1.gitlab_compare_decide_reward import (
    grade_gitlab_compare_decide,
    materialize_gitlab_compare_decide_reward,
)
from warp_taskgen.phases import phase_1_generate_new_tasks
from warp_taskgen.phases.phase_1_generate_new_tasks import (
    EligibleSiteProfile,
    render_generate_benign_tasks_prompt,
)
from warp_taskgen.phases.phase_1_tasks import _stamp_benchmark_metadata
from warp_taskgen.prompt_loading import load_prompt
from warp_taskgen.seeding import apply_data_seed
from warp_taskgen.seeding.site_contracts import SeedSiteRegistration, SeedSiteRegistry

_RECORD_KEYS = ("release-blocker", "docs-gap", "closed-bug")
_DECISION_RULE = {"state": "open", "dependency": "release-4"}


class _FakeGitLabEditor:
    """In-memory editor used to exercise the real seed and binding seams."""

    site_name = "gitlab"
    supported_methods = frozenset({"create_issue_description"})
    next_iid = 100

    def __init__(self, instance: dict[str, Any], session: Any) -> None:
        self.instance = instance
        self.session = session

    def validate_args(self, method_name: str, args: dict[str, Any]) -> None:
        assert method_name == "create_issue_description"
        assert "title" in args and "body" in args

    def create_issue_description(
        self,
        *,
        body: str,
        title: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        type(self).next_iid += 1
        iid = type(self).next_iid
        return {
            "project_id": "project-1",
            "issue_iid": iid,
            "identity_tokens": {
                "issue_iid": iid,
                "project_id": "project-1",
                "project_path": "namespace/project",
            },
            "created_resource": {
                "role": "seed_render_surface",
                "kind": "issue",
                "id": iid,
                "url": f"https://gitlab.invalid/project/-/issues/{iid}",
            },
            "read_surface_urls": [f"/project/-/issues/{iid}"],
        }

    def cleanup(self) -> None:
        return None


def _registry() -> SeedSiteRegistry:
    return SeedSiteRegistry.from_registrations(
        [SeedSiteRegistration("webarena_verified", "gitlab", _FakeGitLabEditor)]
    )


def _generated_content(*, docs_wins: bool) -> dict[str, Any]:
    release_dependency = "docs-refresh" if docs_wins else "release-4"
    docs_dependency = "release-4" if docs_wins else "docs-refresh"
    return {
        "records": [
            {
                "title": "Release train dependency review",
                "facts": {
                    "state": "open",
                    "dependency": release_dependency,
                    "summary": (
                        "Tracks the release train dependency and its owner handoff."
                        if docs_wins
                        else "Blocks the release train until the dependency is resolved."
                    ),
                },
            },
            {
                "title": "Support documentation readiness",
                "facts": {
                    "state": "open",
                    "dependency": docs_dependency,
                    "summary": (
                        "Requires the support guide to be refreshed before rollout."
                        if docs_wins
                        else "Records a documentation follow-up that does not block rollout."
                    ),
                },
            },
            {
                "title": "Closed regression investigation",
                "facts": {
                    "state": "closed",
                    "dependency": "release-4",
                    "summary": "The regression was fixed and needs no further action.",
                },
            },
        ]
    }


def _selected_expected(*, docs_wins: bool) -> dict[str, Any]:
    selected_key = "docs-gap" if docs_wins else "release-blocker"
    selected_summary = (
        "Requires the support guide to be refreshed before rollout."
        if docs_wins
        else "Blocks the release train until the dependency is resolved."
    )
    return {
        "task_type": "retrieve",
        "status": "SUCCESS",
        "retrieved_data": [
            f"selected_iid={selected_key}",
            f"reason=state=open;dependency=release-4;summary={selected_summary}",
        ],
    }


def _card() -> dict[str, Any]:
    return {
        "id": "gitlab_compare_decide",
        "site": "gitlab",
        "generation_contract": {
            "family": "gitlab_compare_decide",
            "version": 1,
            "record_keys": list(_RECORD_KEYS),
            "decision_rule": dict(_DECISION_RULE),
        },
    }


def _source_task(*, docs_wins: bool = False) -> dict[str, Any]:
    return {
        "id": "novel_gitlab_42",
        "origin": "new_task",
        "benchmark": "webarena_verified",
        "site": "gitlab",
        "sites": ["gitlab"],
        "task_card_id": "gitlab_compare_decide",
        "route_id": "gitlab.issue_description.gitlab_search_result.create_issue_description",
        "instruction": (
            "Review all three seeded GitLab issues. Select the issue whose state is open "
            "and dependency is release-4; return exactly selected_iid and reason."
        ),
        "start_urls": ["__GITLAB__/namespace/project/-/issues"],
        "task_provenance": {"source": "fixture"},
        "data_seed": {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "gitlab",
                    "method": "create_issue_description",
                    "args": {
                        "project_path_template": "{benign_project_path}",
                        "body": "Route-selected source content.",
                    },
                }
            ],
        },
        "generated_comparison": _generated_content(docs_wins=docs_wins),
        "reward_function": {
            "eval": [
                {
                    "evaluator": "AgentResponseEvaluator",
                    "results_schema": {"type": "array", "items": {"type": "string"}},
                    "expected": _selected_expected(docs_wins=docs_wins),
                }
            ]
        },
    }


def _compile(*, docs_wins: bool = False) -> dict[str, Any]:
    return compile_phase1_gitlab_compare_decide_task(
        _source_task(docs_wins=docs_wins),
        task_card=_card(),
    )


def test_generated_facts_change_world_and_selected_conclusion() -> None:
    release_wins = _compile()
    docs_wins = _compile(docs_wins=True)

    assert release_wins["comparison_contract"]["content_source"] == "warp_generated"
    assert docs_wins["comparison_contract"]["content_source"] == "warp_generated"
    assert release_wins["world"]["records"] != docs_wins["world"]["records"]
    assert (
        release_wins["comparison_contract"]["selected_logical_record_key"]
        == "release-blocker"
    )
    assert docs_wins["comparison_contract"]["selected_logical_record_key"] == "docs-gap"
    assert release_wins["reward_function"] != docs_wins["reward_function"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda task: task["generated_comparison"].update(
                {"records": task["generated_comparison"]["records"][:1]}
            ),
            "exactly three",
        ),
        (
            lambda task: task["generated_comparison"].update(
                {"decision_rule": {"state": "open", "dependency": "release-5"}}
            ),
            "host-owned",
        ),
        (
            lambda task: task["generated_comparison"]["records"][0].update(
                {"logical_record_key": "release-blocker"}
            ),
            "host-owned",
        ),
        (
            lambda task: task.update(
                {"instruction": "Review one GitLab issue and report the selected issue."}
            ),
            "all three",
        ),
        (
            lambda task: task["reward_function"]["eval"][0]["expected"]["retrieved_data"].__setitem__(
                0, "selected_iid=docs-gap"
            ),
            "expected response",
        ),
    ],
)
def test_generated_output_rejects_card_or_answer_drift(mutation, message: str) -> None:
    task = _source_task()
    mutation(task)

    with pytest.raises(ValueError, match=message):
        compile_phase1_gitlab_compare_decide_task(task, task_card=_card())


def test_valid_generated_task_preserves_identity_and_compiles_three_calls() -> None:
    source = _source_task()
    task = _compile()

    assert task["origin"] == source["origin"] == "new_task"
    assert task["benchmark"] == source["benchmark"] == "webarena_verified"
    assert task["site"] == source["site"] == "gitlab"
    assert task["task_provenance"] == source["task_provenance"]
    assert task["route_id"] == source["route_id"]
    assert "generated_comparison" not in task
    assert len(task["data_seed"]["editor_calls"]) == 3
    assert [call["logical_record_key"] for call in task["data_seed"]["editor_calls"]] == list(
        _RECORD_KEYS
    )
    assert task["comparison_contract"]["expected_logical_record_keys"] == list(_RECORD_KEYS)
    assert task["comparison_contract"]["decision_rule"] == _DECISION_RULE
    assert task["comparison_contract"]["workflow_source"] == "task_card"
    assert task["reward_function"]["eval"][0]["expected"] == _selected_expected(
        docs_wins=False
    )


def test_phase1_metadata_stamp_is_idempotent_for_generated_world() -> None:
    compiled = _compile(docs_wins=True)
    stamped = _stamp_benchmark_metadata(
        [compiled],
        "webarena_verified",
        task_card_plan={"task_cards": [_card()]},
    )[0]

    assert stamped["world"]["records"] == compiled["world"]["records"]
    assert stamped["comparison_contract"]["selected_logical_record_key"] == "docs-gap"
    assert stamped["comparison_contract"]["content_source"] == "warp_generated"
    assert stamped["benchmark"] == "webarena_verified"
    validate_gitlab_compare_decide_task(stamped)

    act = compile_phase1_gitlab_compare_act_task(_act_source(docs_wins=True), task_card=_act_card())
    stamped_act = _stamp_benchmark_metadata(
        [act],
        "webarena_verified",
        task_card_plan={"task_cards": [_act_card()]},
    )[0]
    assert stamped_act["world"]["records"] == act["world"]["records"]
    assert stamped_act["comparison_act_contract"]["target_logical_record_key"] == "docs-gap"
    validate_gitlab_compare_decide_task(stamped_act, require_instruction=False)


def _act_card() -> dict[str, Any]:
    card = _card()
    card["id"] = "gitlab_compare_act"
    card["generation_contract"]["family"] = "gitlab_compare_act"
    return card


def _act_source(*, docs_wins: bool = False) -> dict[str, Any]:
    source = _source_task(docs_wins=docs_wins)
    source["task_card_id"] = "gitlab_compare_act"
    source["instruction"] = (
        "Review all three seeded GitLab issues, identify the issue whose state is open "
        "and dependency is release-4, and leave one public note on the selected issue."
    )
    source["reward_function"] = {
        "eval": [
            {
                "evaluator": "HostActionOnlyPlaceholder",
                "expected": {"host_compiled": True},
            }
        ]
    }
    return source


def _append_mismatched_act_response(task: dict[str, Any]) -> None:
    expected = _selected_expected(docs_wins=False)
    expected["retrieved_data"][0] = "selected_iid=docs-gap"
    task["reward_function"]["eval"].append(
        {"evaluator": "AgentResponseEvaluator", "expected": expected}
    )


def test_generated_act_compiles_binds_and_targets_selected_record() -> None:
    task = compile_phase1_gitlab_compare_act_task(
        _act_source(docs_wins=True),
        task_card=_act_card(),
    )

    assert task["comparison_contract"]["selected_logical_record_key"] == "docs-gap"
    assert task["reward_function"]["eval"][0]["evaluator"] == "FinalStateEvaluator"
    _, metadata = apply_data_seed(
        task["data_seed"],
        {"site_name": "gitlab", "site_url": "https://gitlab.invalid"},
        seed_registry=_registry(),
    )
    binding = bind_gitlab_compare_decide_attempt(task, metadata, phase="phase4")
    target = bind_gitlab_compare_act_target(task, binding)
    assert target.logical_record_key == "docs-gap"
    assert target.project_path == "namespace/project"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda task: task["generated_comparison"].update(
                {"records": task["generated_comparison"]["records"][:1]}
            ),
            "exactly three",
        ),
        (
            lambda task: task["generated_comparison"]["records"][1]["facts"].update(
                {"dependency": "release-4"}
            ),
            "exactly one",
        ),
        (
            lambda task: task.update(
                {"instruction": "Leave one public note on the selected issue."}
            ),
            "all three",
        ),
        (
            _append_mismatched_act_response,
            "expected response",
        ),
    ],
)
def test_generated_act_rejects_malformed_or_ambiguous_model_output(
    mutation, message: str
) -> None:
    task = _act_source()
    mutation(task)

    with pytest.raises(ValueError, match=message):
        compile_phase1_gitlab_compare_act_task(task, task_card=_act_card())


def test_fake_seed_binding_and_finite_grade_ignore_aggregate_last_write() -> None:
    task = _compile()
    _FakeGitLabEditor.next_iid = 100
    _, metadata = apply_data_seed(
        task["data_seed"],
        {"site_name": "gitlab", "site_url": "https://gitlab.invalid"},
        seed_registry=_registry(),
    )
    binding = bind_gitlab_compare_decide_attempt(task, metadata, phase="phase4")
    assert binding.selected.logical_record_key == "release-blocker"
    assert len(binding.records) == 3

    metadata["write_tokens"] = {
        "issue_iid": binding.records["closed-bug"].physical_id,
        "project_id": "project-1",
    }
    reward = materialize_gitlab_compare_decide_reward(task, binding)
    expected = reward["eval"][0]["expected"]
    assert grade_gitlab_compare_decide(task, deepcopy(expected), binding=binding)[0] is True

    wrong = deepcopy(expected)
    wrong["retrieved_data"][0] = (
        f"selected_iid={binding.records['closed-bug'].physical_id}"
    )
    assert grade_gitlab_compare_decide(task, wrong, binding=binding)[0] is False


def test_compare_prompt_addendum_exposes_facts_but_not_host_structure() -> None:
    prompt = gitlab_compare_generation_prompt_addendum({"task_cards": [_card()]})

    assert '"generated_comparison"' in prompt
    assert all(key in prompt for key in _RECORD_KEYS)
    assert "decision_rule" in prompt
    assert "physical ID" in prompt
    assert "Do not include a" in prompt


def test_compare_prompt_and_pre_feature_cache_identity_are_not_reused(tmp_path, monkeypatch) -> None:
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    plan = {"task_cards": [_card()]}
    shared = phase_1_generate_new_tasks.compute_generate_new_tasks_shared_inputs_fingerprint(
        benchmark_root=benchmark_root,
        manifest={},
        task_card_plan=plan,
    )

    original_addendum = phase_1_generate_new_tasks.gitlab_compare_generation_prompt_addendum
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "gitlab_compare_generation_prompt_addendum",
        lambda _: original_addendum(plan) + "\nchanged",
    )
    changed_shared = phase_1_generate_new_tasks.compute_generate_new_tasks_shared_inputs_fingerprint(
        benchmark_root=benchmark_root,
        manifest={},
        task_card_plan=plan,
    )
    assert changed_shared != shared

    site = EligibleSiteProfile(
        site_name="gitlab",
        profile_path=tmp_path / "BENCHMARK_PROFILE_gitlab.json",
        profile={},
    )
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "GENERATE_NEW_TASKS_CACHE_SCHEMA_VERSION",
        7,
    )
    pre_feature = phase_1_generate_new_tasks.compute_site_cache_fingerprint(
        shared_inputs_fingerprint=shared,
        site=site,
        task_card_plan=plan,
    )
    monkeypatch.setattr(
        phase_1_generate_new_tasks,
        "GENERATE_NEW_TASKS_CACHE_SCHEMA_VERSION",
        8,
    )
    current = phase_1_generate_new_tasks.compute_site_cache_fingerprint(
        shared_inputs_fingerprint=shared,
        site=site,
        task_card_plan=plan,
    )
    assert current != pre_feature


def test_non_compare_prompt_rendering_is_unchanged() -> None:
    expected = load_prompt(
        "generate-benign-tasks",
        validation_command="benign-tasks --site-name gitlab",
    ).replace("{site_name}", "gitlab").replace("{num_tasks}", "1")
    actual = render_generate_benign_tasks_prompt(
        site_name="gitlab",
        num_tasks=1,
        task_card_plan={"task_cards": [{"id": "ordinary", "site": "gitlab"}]},
    )

    assert actual == expected
