from __future__ import annotations

import copy

import pytest

from warp_taskgen.phase_1.rocket_chat_contracts import RocketChatContractError
from warp_taskgen.phase_1.rocket_chat_decisions import (
    compile_rocket_chat_task,
    generate_rocket_chat_conversation,
)
from warp_taskgen.phase_1.rocket_chat_task_envelope import (
    ROCKET_CHAT_CONTRACT_FIELD,
    compile_rocket_chat_benign_task,
    project_rocket_chat_static_contract,
    validate_rocket_chat_benign_task,
)


def test_benign_envelope_compiles_static_contract_into_normal_warp_task() -> None:
    conversation = generate_rocket_chat_conversation(run_marker="WARP-E1-ENVELOPE")

    task = compile_rocket_chat_benign_task(
        conversation,
        task_id="novel_rocketchat_1",
        instruction="Read the complete project thread and return its current owner and due date.",
    )

    static = compile_rocket_chat_task(conversation)
    assert task["id"] == "novel_rocketchat_1"
    assert task["origin"] == "new_task"
    assert task["benchmark"] == "theagentcompany"
    assert task["site"] == "rocketchat"
    assert task["sites"] == ["rocketchat"]
    assert task["start_urls"] == static["start_urls"]
    assert task["reward_function"] == static["reward_function"]
    assert task[ROCKET_CHAT_CONTRACT_FIELD] == static
    assert task["data_seed"] == {
        "mechanism": "editor",
        "render_signature": conversation.message("correction").body,
        "editor_calls": [
            {
                "benchmark": "theagentcompany",
                "site": "rocketchat",
                "method": "seed_rocket_chat_conversation",
                "args": {"conversation": conversation.as_dict()},
            }
        ],
    }
    assert project_rocket_chat_static_contract(task) == static
    validate_rocket_chat_benign_task(task)


def test_envelope_projection_keeps_exact_static_compiler_as_supported_input() -> None:
    static = compile_rocket_chat_task(generate_rocket_chat_conversation())

    assert project_rocket_chat_static_contract(static) is static


@pytest.mark.parametrize(
    "mutate",
    (
        lambda task: task.update({"task_id": "native-tac-task"}),
        lambda task: task.update({"site": "gitlab"}),
        lambda task: task["reward_function"]["eval"][0]["expected"].update(
            {"owner": "wrong-owner"}
        ),
        lambda task: task["data_seed"]["editor_calls"][0].update(
            {"method": "send_notification"}
        ),
        lambda task: task[ROCKET_CHAT_CONTRACT_FIELD]["conversation"]["messages"][0].update(
            {"body": "forged body"}
        ),
    ),
)
def test_benign_envelope_rejects_outer_or_inner_contract_drift(mutate) -> None:
    task = compile_rocket_chat_benign_task(
        generate_rocket_chat_conversation(),
        task_id="novel_rocketchat_2",
        instruction="Return the current decision.",
    )
    drifted = copy.deepcopy(task)
    mutate(drifted)

    with pytest.raises(RocketChatContractError):
        validate_rocket_chat_benign_task(drifted)
