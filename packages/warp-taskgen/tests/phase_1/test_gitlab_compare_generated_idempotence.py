"""Regression checks for repeated GitLab comparison compilation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from tests.phase_1.test_gitlab_compare_generated_content import (
    _act_card,
    _act_source,
    _card,
    _compile,
)
from warp_taskgen.phase_1.gitlab_compare_decide_generation import (
    compile_phase1_gitlab_compare_act_task,
    compile_phase1_gitlab_compare_decide_task,
    validate_gitlab_compare_decide_task,
)
from warp_taskgen.phase_1.novel_task_validation._impl import (
    _validate_task_card_alignment,
)
from warp_taskgen.phases.phase_1_tasks import _stamp_benchmark_metadata


def _generic_task_card_validation(task: dict[str, Any], card: dict[str, Any]) -> None:
    problem = _validate_task_card_alignment(
        task,
        index=0,
        site_name="gitlab",
        card_index={card["id"]: card},
        route_index=None,
    )
    assert problem is None


def test_phase1_metadata_stamp_is_idempotent_for_generated_world() -> None:
    compiled = _compile(docs_wins=True)
    _generic_task_card_validation(compiled, _card())
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
    expected_action_provenance = deepcopy(act["task_provenance"]["benign_action_contract"])
    _generic_task_card_validation(act, _act_card())
    assert "benign_action_contract" not in act["task_provenance"]
    stamped_act = _stamp_benchmark_metadata(
        [act],
        "webarena_verified",
        task_card_plan={"task_cards": [_act_card()]},
    )[0]
    assert stamped_act["world"]["records"] == act["world"]["records"]
    assert stamped_act["comparison_act_contract"]["target_logical_record_key"] == "docs-gap"
    assert stamped_act["task_provenance"]["benign_action_contract"] == expected_action_provenance
    validate_gitlab_compare_decide_task(stamped_act, require_instruction=False)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda task: task["world"].update({"decisive_record_key": "closed-bug"}),
            "decisive record",
        ),
        (
            lambda task: task["comparison_contract"].update({"method": "create_issue"}),
            "contract is not canonical",
        ),
        (
            lambda task: task["reward_function"].update({"extra": True}),
            "reward is not canonical",
        ),
        (
            lambda task: task["task_provenance"].update({"task_card_id": "other"}),
            "task-card provenance",
        ),
    ],
)
def test_host_compiled_decide_rejects_derived_field_tampering(mutation, message: str) -> None:
    task = _compile()
    mutation(task)
    with pytest.raises(ValueError, match=message):
        compile_phase1_gitlab_compare_decide_task(task, task_card=_card())


def test_host_compiled_act_rejects_reward_metadata_tampering() -> None:
    task = compile_phase1_gitlab_compare_act_task(_act_source(), task_card=_act_card())
    task["reward_function"]["eval"][0]["extra"] = True
    with pytest.raises(ValueError, match="reward disagrees"):
        compile_phase1_gitlab_compare_act_task(task, task_card=_act_card())
