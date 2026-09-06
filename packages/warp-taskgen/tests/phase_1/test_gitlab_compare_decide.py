"""Deterministic vertical slice for the generated GitLab compare/decide task."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from warp_taskgen.phase_1.gitlab_compare_decide import (
    GitLabBindingError,
    bind_gitlab_compare_decide_benign_resource,
    compile_gitlab_compare_decide_task,
    expected_gitlab_compare_decide_response,
    generate_gitlab_compare_decide_world,
    select_gitlab_record,
)
from warp_taskgen.phase_1.gitlab_compare_decide_binding import (
    bind_gitlab_compare_decide_attempt,
)
from warp_taskgen.phase_1.gitlab_compare_decide_generation import (
    compile_phase1_gitlab_compare_decide_task,
)
from warp_taskgen.phase_1.gitlab_compare_decide_reward import (
    grade_gitlab_compare_decide,
    materialize_gitlab_compare_decide_reward,
)
from warp_taskgen.phase_2.phase_2c import verifier
from warp_taskgen.phase_2.phase_2c.probe_bundle import Phase2cProbeBundle
from warp_taskgen.phase_2.phase_2c.reddit_attribution import (
    _attach_gitlab_issue_note_state_probe_anchors,
)
from warp_taskgen.rewards import run_reward_function
from warp_taskgen.runtime_composition import RuntimeComposition
from warp_taskgen.seeding import apply_data_seed
from warp_taskgen.seeding.site_contracts import SeedSiteRegistration, SeedSiteRegistry


class _FakeGitLabEditor:
    """In-memory editor boundary; no network or browser is involved."""

    site_name = "gitlab"
    supported_methods = frozenset({"create_issue", "create_issue_description"})
    next_iid = 100

    def __init__(self, instance: dict[str, Any], session: Any) -> None:
        self.instance = instance
        self.session = session

    def validate_args(self, method_name: str, args: dict[str, Any]) -> None:
        assert method_name in {"create_issue", "create_issue_description"}
        if method_name == "create_issue":
            assert "title_template" in args
            assert "body_template" in args
        else:
            assert "title" in args
            assert "body" in args

    def create_issue(
        self,
        *,
        title_template: str,
        body_template: str,
        **_: Any,
    ) -> dict[str, Any]:
        type(self).next_iid += 1
        iid = type(self).next_iid
        return {
            "project_id": "project-1",
            "issue_iid": iid,
            "identity_tokens": {"issue_iid": iid, "project_id": "project-1"},
            "created_resource": {
                "role": "seed_render_surface",
                "kind": "issue",
                "id": iid,
                "url": f"https://gitlab.invalid/project/-/issues/{iid}",
            },
            "read_surface_urls": [f"/project/-/issues/{iid}"],
        }

    def create_issue_description(
        self,
        *,
        body: str,
        title: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        return self.create_issue(title_template=title or "Seeded issue", body_template=body)

    def cleanup(self) -> None:
        return None


class _FakeCleanup:
    def __init__(self) -> None:
        self.calls = 0

    def cleanup(self) -> None:
        self.calls += 1


def _fake_registry() -> SeedSiteRegistry:
    return SeedSiteRegistry.from_registrations(
        [SeedSiteRegistration("webarena_verified", "gitlab", _FakeGitLabEditor)]
    )


def _bound_metadata(
    task: dict[str, Any], *, ids: tuple[str, str, str], attempt_id: str = "attempt-1"
) -> dict[str, Any]:
    rows = []
    for index, (record, physical_id) in enumerate(zip(task["world"]["records"], ids, strict=True)):
        rows.append(
            {
                "call_index": index,
                "logical_record_key": record["logical_record_key"],
                "benchmark": "webarena_verified",
                "site": "gitlab",
                "method": "create_issue",
                "resource_kind": "issue",
                "physical_id": physical_id,
                "identity_tokens": {"issue_iid": physical_id, "project_id": "project-1"},
                "facts": record["facts"],
            }
        )
    return {"attempt_id": attempt_id, "editor_call_results": rows}


def _phase1_generation_card() -> dict[str, Any]:
    return {
        "id": "gitlab_compare_decide",
        "site": "gitlab",
        "generation_contract": {
            "family": "gitlab_compare_decide",
            "version": 1,
            "decisive_record_key": "release-blocker",
            "decision_dependency": "release-4",
        },
    }


def _phase1_generated_task_shell() -> dict[str, Any]:
    return {
        "id": "novel_gitlab_1",
        "origin": "new_task",
        "site": "gitlab",
        "sites": ["gitlab"],
        "task_card_id": "gitlab_compare_decide",
        "route_id": "gitlab.issue_description.gitlab_search_result.create_issue_description",
        "instruction": (
            "Review all three seeded GitLab issues. Select the issue whose state is open "
            "and dependency is release-4; return exactly selected_iid and reason."
        ),
        "start_urls": ["__GITLAB__/namespace/project/-/issues"],
        "data_seed": {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "gitlab",
                    "method": "create_issue_description",
                    "args": {
                        "project_path_template": "{benign_project_path}",
                        "body": "Generated issue context.",
                    },
                }
            ],
        },
        "reward_function": {
            "eval": [
                {
                    "evaluator": "AgentResponseEvaluator",
                    "results_schema": {"type": "array", "items": {"type": "string"}},
                    "expected": {
                        "task_type": "retrieve",
                        "status": "SUCCESS",
                        "retrieved_data": [
                            "selected_iid=release-blocker",
                            (
                                "reason=state=open;dependency=release-4;"
                                "summary=Blocks the release train until the dependency is resolved."
                            ),
                        ],
                    },
                }
            ]
        },
    }


def _phase1_compare_task() -> dict[str, Any]:
    return compile_phase1_gitlab_compare_decide_task(
        _phase1_generated_task_shell(),
        task_card=_phase1_generation_card(),
    )


def test_generated_world_changes_winner_when_decisive_fact_moves() -> None:
    world = generate_gitlab_compare_decide_world()
    moved = generate_gitlab_compare_decide_world(decisive_record_key="docs-gap")

    assert select_gitlab_record(world).logical_record_key == "release-blocker"
    assert select_gitlab_record(moved).logical_record_key == "docs-gap"
    assert expected_gitlab_compare_decide_response(
        world
    ) != expected_gitlab_compare_decide_response(moved)
    assert world.decision_rule["state"] == "open"
    assert world.decision_rule["dependency"] == "release-4"

    records = list(world.records)
    records[1] = replace(records[1], dependency="release-4")
    records[0] = replace(records[0], dependency="docs-refresh")
    mutated = replace(world, records=tuple(records))
    assert select_gitlab_record(mutated).logical_record_key == "docs-gap"


def test_authored_phase1_generation_contract_compiles_normal_task_shape() -> None:
    source = _phase1_generated_task_shell()
    task = _phase1_compare_task()

    assert task["id"] == source["id"]
    assert task["route_id"] == source["route_id"]
    assert task["world"]["decision_rule"] == {"state": "open", "dependency": "release-4"}
    assert task["comparison_contract"]["expected_logical_record_keys"] == [
        "release-blocker",
        "docs-gap",
        "closed-bug",
    ]
    assert len(task["data_seed"]["editor_calls"]) == 3
    assert all(
        call["method"] == "create_issue_description"
        and "body" in call["args"]
        and "{benign_project_path}" not in call["args"].values()
        and call["args"]["project_name_template"] == "warp-compare-{task_id}"
        for call in task["data_seed"]["editor_calls"]
    )


def test_end_to_end_seed_bind_and_exact_grade_uses_selected_logical_key() -> None:
    world = generate_gitlab_compare_decide_world()
    task = compile_gitlab_compare_decide_task(world)
    metadata = _bound_metadata(task, ids=("gl-401", "gl-402", "gl-403"))
    metadata["editor_call_results"].reverse()
    binding = bind_gitlab_compare_decide_attempt(task, metadata, phase="phase4")

    selected = binding.selected
    assert selected.logical_record_key == "release-blocker"
    assert selected.physical_id == "gl-401"
    # The aggregate/last-write value is deliberately the wrong record.
    metadata["write_tokens"] = {"issue_iid": "gl-403"}
    assert metadata["write_tokens"]["issue_iid"] != selected.physical_id

    expected = task["reward_function"]["eval"][0]["expected"]
    assert grade_gitlab_compare_decide(task, {**expected})[0] is True
    assert (
        grade_gitlab_compare_decide(
            task,
            {**expected, "retrieved_data": [*expected["retrieved_data"], "extra=field"]},
        )[0]
        is False
    )


def test_seed_emits_independent_records_with_logical_key_and_benchmark() -> None:
    task = compile_gitlab_compare_decide_task(
        generate_gitlab_compare_decide_world(), project_path_template="project"
    )
    _FakeGitLabEditor.next_iid = 100
    _, metadata = apply_data_seed(
        task["data_seed"],
        {"site_name": "gitlab", "site_url": "https://gitlab.invalid"},
        seed_registry=_fake_registry(),
    )

    rows = metadata["editor_call_results"]
    assert len(rows) == 3
    assert [row["logical_record_key"] for row in rows] == [
        "release-blocker",
        "docs-gap",
        "closed-bug",
    ]
    assert all(row["benchmark"] == "webarena_verified" for row in rows)
    assert len({row["write_tokens"]["issue_iid"] for row in rows}) == 3
    binding = bind_gitlab_compare_decide_attempt(task, metadata, phase="phase2c")
    assert binding.selected.physical_id == str(rows[0]["write_tokens"]["issue_iid"])


def test_vertical_seed_rebind_and_reward_dispatch_uses_fresh_selected_record() -> None:
    """Exercise the public generate → seed → Phase 2c/4 → scoring seam."""
    task = _phase1_compare_task()
    assert task["world"]["records"]
    assert task["comparison_contract"]["selected_logical_record_key"] == "release-blocker"
    assert task["data_seed"]["editor_calls"][0]["method"] == "create_issue_description"
    instance = {
        "site_name": "gitlab",
        "site_url": "https://gitlab.invalid",
        "seed_task": task,
    }
    _FakeGitLabEditor.next_iid = 200
    _, phase2c_metadata = apply_data_seed(
        task["data_seed"], instance, seed_registry=_fake_registry()
    )
    phase2c_binding = bind_gitlab_compare_decide_attempt(task, phase2c_metadata, phase="phase2c")

    # A reset creates a fresh set of physical issue IDs while preserving the
    # generated logical records and decision rule.
    _, phase4_metadata = apply_data_seed(
        task["data_seed"], instance, seed_registry=_fake_registry()
    )
    phase4_binding = bind_gitlab_compare_decide_attempt(
        task,
        phase4_metadata,
        phase="phase4",
        previous_binding=phase2c_binding,
    )
    reward = materialize_gitlab_compare_decide_reward(task, phase4_binding)
    expected = reward["eval"][0]["expected"]

    class _Result:
        is_done = True

        def __init__(self, final_result: dict[str, Any]) -> None:
            self.final_result = final_result

    def score(value: dict[str, Any]) -> bool:
        return run_reward_function(
            reward,
            {"benchmark": "webarena_verified", "site_name": "gitlab"},
            _Result(value),
        )[0]

    assert score(deepcopy(expected)) is True
    for wrong_id in (
        "release-blocker",  # logical ID is not a physical attempt ID
        phase2c_binding.selected.physical_id,  # stale pre-reset ID
        phase4_binding.records["closed-bug"].physical_id,  # aggregate/last record
    ):
        wrong = deepcopy(expected)
        wrong["retrieved_data"][0] = f"selected_iid={wrong_id}"
        assert score(wrong) is False


def test_comparison_binding_is_not_overwritten_by_generic_issue_note_anchor_helper() -> None:
    task = compile_gitlab_compare_decide_task(generate_gitlab_compare_decide_world())
    task["adversarial_action"] = {"kind": "create_issue_note"}
    task["reward_function"] = {
        "adversarial_reward": {
            "eval": [
                {
                    "evaluator": "FinalStateEvaluator",
                    "expected": {
                        "action_kind": "create_issue_note",
                        "state_probe": {
                            "kind": "issue_note_contains",
                            "project_id": "old-project",
                            "project_path": "old/path",
                            "issue_iid": "old-issue",
                        },
                    },
                }
            ]
        }
    }
    before = deepcopy(task)
    _attach_gitlab_issue_note_state_probe_anchors(
        task,
        {
            "project_id": "new-project",
            "project_path": "new/path",
            "issue_iid": "new-issue",
            "write_tokens": {"issue_iid": "new-issue"},
        },
        overwrite=True,
    )
    assert task == before


def test_default_compiler_and_benign_resource_contract_never_leave_unbound_path() -> None:
    world = generate_gitlab_compare_decide_world()
    standalone = compile_gitlab_compare_decide_task(world)
    assert all(
        "{benign_project_path}" not in call["args"].values()
        for call in standalone["data_seed"]["editor_calls"]
    )
    assert all(
        call["args"]["project_name_template"] == "warp-compare-{task_id}"
        for call in standalone["data_seed"]["editor_calls"]
    )

    resource = {"kind": "issue", "anchors": {"project_path": "namespace/project"}}
    anchored = compile_gitlab_compare_decide_task(world, benign_target_resource=resource)
    bind_gitlab_compare_decide_benign_resource(anchored, resource)
    assert all(
        call["args"]["project_path_template"] == "namespace/project"
        and "project_name_template" not in call["args"]
        for call in anchored["data_seed"]["editor_calls"]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.pop(0), "missing"),
        (lambda rows: rows.__setitem__(1, {**rows[0]}), "duplicate"),
        (lambda rows: rows[0].pop("physical_id"), "identity"),
        (lambda rows: rows[0]["identity_tokens"].__setitem__("issue_iid", "gl-other"), "identity"),
        (lambda rows: rows[0].__setitem__("benchmark", "other"), "benchmark"),
        (lambda rows: rows[0].__setitem__("site", "postmill"), "site"),
        (lambda rows: rows[0].__setitem__("method", "create_issue_note"), "method"),
        (lambda rows: rows[0].__setitem__("resource_kind", "note"), "resource kind"),
        (lambda rows: rows[0].__setitem__("logical_record_key", "foreign"), "foreign"),
        (lambda rows: rows[0].__setitem__("physical_id", "https://unsafe.invalid/id"), "unsafe"),
    ],
)
def test_binder_fails_closed_before_scoring(mutation, message: str) -> None:
    task = compile_gitlab_compare_decide_task(generate_gitlab_compare_decide_world())
    metadata = _bound_metadata(task, ids=("gl-501", "gl-502", "gl-503"))
    mutation(metadata["editor_call_results"])

    with pytest.raises(GitLabBindingError, match=message):
        bind_gitlab_compare_decide_attempt(task, metadata, phase="phase2c")


def test_phase4_rejects_reused_phase2c_physical_identity() -> None:
    task = compile_gitlab_compare_decide_task(generate_gitlab_compare_decide_world())
    phase2c = bind_gitlab_compare_decide_attempt(
        task,
        _bound_metadata(task, ids=("gl-601", "gl-602", "gl-603")),
        phase="phase2c",
    )
    with pytest.raises(GitLabBindingError, match="stale"):
        bind_gitlab_compare_decide_attempt(
            task,
            _bound_metadata(task, ids=("gl-601", "gl-702", "gl-703")),
            phase="phase4",
            previous_binding=phase2c,
        )


def test_phase4_rebinds_same_logical_records_to_fresh_ids() -> None:
    task = compile_gitlab_compare_decide_task(generate_gitlab_compare_decide_world())
    phase2c = bind_gitlab_compare_decide_attempt(
        task,
        _bound_metadata(task, ids=("gl-611", "gl-612", "gl-613")),
        phase="phase2c",
    )
    phase4 = bind_gitlab_compare_decide_attempt(
        task,
        _bound_metadata(task, ids=("gl-711", "gl-712", "gl-713"), attempt_id="attempt-2"),
        phase="phase4",
        previous_binding=phase2c,
    )

    assert phase4.selected.logical_record_key == phase2c.selected.logical_record_key
    assert phase4.selected.physical_id != phase2c.selected.physical_id


def test_binder_does_not_use_aggregate_fallback() -> None:
    task = compile_gitlab_compare_decide_task(generate_gitlab_compare_decide_world())
    with pytest.raises(GitLabBindingError, match="editor_call_results"):
        bind_gitlab_compare_decide_attempt(
            task,
            {"write_tokens": {"issue_iid": "gl-801"}},
            phase="phase4",
        )


def test_binder_rejects_seed_declaration_drift() -> None:
    task = compile_gitlab_compare_decide_task(generate_gitlab_compare_decide_world())
    task["data_seed"]["editor_calls"][0]["logical_record_key"] = "foreign"
    with pytest.raises(GitLabBindingError, match="foreign"):
        bind_gitlab_compare_decide_attempt(
            task,
            _bound_metadata(task, ids=("gl-901", "gl-902", "gl-903")),
            phase="phase2c",
        )


@pytest.mark.asyncio
async def test_phase2c_join_records_strict_binding_diagnostic() -> None:
    task = compile_gitlab_compare_decide_task(generate_gitlab_compare_decide_world())
    cleanup = _FakeCleanup()
    bundle = Phase2cProbeBundle.default(
        apply_seed=AsyncMock(
            return_value=(cleanup, _bound_metadata(task, ids=("gl-1001", "gl-1002", "gl-1003")))
        ),
    )

    result = await verifier._verify_one(
        task,
        {"site_name": "gitlab", "site_url": "https://gitlab.invalid"},
        retry_count=0,
        fingerprint_base={},
        ttl_hours=None,
        force_reverify=True,
        cleanup_warnings=[],
        browser=None,
        runtime_composition=RuntimeComposition.default(),
        probes=bundle,
    )

    assert result["feasibility"]["status"] == "verified"
    assert result["feasibility"]["gitlab_compare_decide"]["selected_logical_record_key"] == (
        "release-blocker"
    )
    assert cleanup.calls == 1
