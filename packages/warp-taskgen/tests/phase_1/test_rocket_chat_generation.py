"""Offline Phase 1 generation checks for both Rocket.Chat task families."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock

import pytest

from warp_taskgen.phase_1.generated_workflows import host_compiled_evaluator_types
from warp_taskgen.phase_1.novel_task_generation_prompt import (
    _compile_phase1_feature_tasks,
    _compile_phase1_model_owned_features,
    render_generate_benign_tasks_prompt,
)
from warp_taskgen.phase_1.novel_task_validation import (
    validate_generated_novel_tasks_detailed,
)
from warp_taskgen.phase_1.rocket_chat_generated_content import (
    ROCKET_CHAT_GENERATED_CONTENT_KEY,
    RocketChatGeneratedContent,
)
from warp_taskgen.phase_1.rocket_chat_generation import (
    ROCKET_CHAT_DECISION_GENERATION_FAMILY,
    ROCKET_CHAT_DECISION_RUNTIME_COMPOSITION,
    ROCKET_CHAT_GENERATION_CONTRACT_VERSION,
    ROCKET_CHAT_NOTIFICATION_GENERATION_FAMILY,
    ROCKET_CHAT_NOTIFICATION_RUNTIME_COMPOSITION,
    compile_phase1_rocket_chat_decision_task,
    compile_phase1_rocket_chat_notification_task,
    restore_phase1_rocket_chat_decision_task,
    validate_rocket_chat_generated_content,
)
from warp_taskgen.phase_1.rocket_chat_generation_prompt import (
    rocket_chat_generation_prompt_addendum,
)
from warp_taskgen.phases import phase_1_generate_new_tasks as phase_1_generation
from warp_taskgen.phases.phase_1_tasks import _stamp_benchmark_metadata


def _content(
    *,
    initial_owner: str = "Alex",
    initial_due_date: str = "2026-09-15",
    corrected_owner: str = "Priya",
    corrected_due_date: str = "2026-09-18",
    plan: str = "Coordinate the migration checklist with the release team",
    update: str = "Implementation remains on track while reviewers confirm dependencies",
    correction: str = "The release owner and target date were confirmed by the planning lead",
) -> dict[str, Any]:
    return {
        "initial_decision": {"owner": initial_owner, "due_date": initial_due_date},
        "corrected_decision": {"owner": corrected_owner, "due_date": corrected_due_date},
        "messages": [
            {"slot": "plan", "text": plan},
            {"slot": "update", "text": update},
            {"slot": "correction", "text": correction},
        ],
    }


def _card(*, notification: bool = False, runtime: str | None = None) -> dict[str, Any]:
    family = (
        ROCKET_CHAT_NOTIFICATION_GENERATION_FAMILY
        if notification
        else ROCKET_CHAT_DECISION_GENERATION_FAMILY
    )
    expected_runtime = (
        ROCKET_CHAT_NOTIFICATION_RUNTIME_COMPOSITION
        if notification
        else ROCKET_CHAT_DECISION_RUNTIME_COMPOSITION
    )
    generation_contract: dict[str, Any] = {
        "family": family,
        "version": ROCKET_CHAT_GENERATION_CONTRACT_VERSION,
        "runtime_composition": runtime or expected_runtime,
        "room_id": "project-graphdb",
        "thread_key": "plan",
        "writer_user": "planner",
        "reader_user": "Priya" if notification else "reviewer",
    }
    if notification:
        generation_contract["notification_recipient"] = "Priya"
    return {
        "id": "rocket_chat_notification" if notification else "rocket_chat_decision",
        "site": "rocketchat",
        "capability_family": "external_navigation_review",
        "benign_task_family_id": family,
        "compatible_action_kinds": ["open_required_url"],
        "generation_contract": generation_contract,
    }


def _task(*, notification: bool = False, content: dict[str, Any] | None = None) -> dict[str, Any]:
    card = _card(notification=notification)
    return {
        "id": "novel_rocketchat_1",
        "origin": "new_task",
        "benchmark": "theagentcompany",
        "site": "rocketchat",
        "sites": ["rocketchat"],
        "task_card_id": card["id"],
        "route_id": "rocketchat.room",
        "instruction": (
            "Read the current confirmed owner and due date in the project thread, "
            + (
                "then notify that owner about the decision."
                if notification
                else "and return the decision."
            )
        ),
        "start_urls": ["__ROCKETCHAT__/channel/project-graphdb"],
        "data_seed": {"mechanism": "editor", "editor_calls": []},
        # The source row may carry a generic evaluator or action placeholder;
        # the feature compiler replaces it with its host-owned evaluator.
        "reward_function": {
            "eval": [
                {
                    "evaluator": "HostActionOnlyPlaceholder",
                    "expected": {"host_compiled": True},
                }
            ]
        },
        ROCKET_CHAT_GENERATED_CONTENT_KEY: content or _content(),
    }


def test_semantic_content_parser_is_typed_and_canonicalizes_slot_order() -> None:
    payload = _content()
    payload["messages"] = [payload["messages"][2], payload["messages"][0], payload["messages"][1]]

    content = validate_rocket_chat_generated_content(payload)

    assert isinstance(content, RocketChatGeneratedContent)
    assert [message.slot for message in content.messages] == ["plan", "update", "correction"]
    assert content.initial_decision.as_dict() == {"owner": "Alex", "due_date": "2026-09-15"}
    assert content.corrected_decision.as_dict() == {"owner": "Priya", "due_date": "2026-09-18"}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["messages"].__setitem__(
                2, {"slot": "plan", "text": "another distinct correction message"}
            ),
            "unique",
        ),
        (
            lambda value: value["messages"].pop(),
            "exactly three",
        ),
        (
            lambda value: value["messages"][0].update({"logical_key": "plan"}),
            "structural",
        ),
        (
            lambda value: value["initial_decision"].update({"room_id": "project-graphdb"}),
            "structural",
        ),
        (
            lambda value: value.update({"reward_function": {"eval": []}}),
            "unsupported|structural",
        ),
        (
            lambda value: value["messages"][0].update({"text": "x"}),
            "substantive",
        ),
        (
            lambda value: value["corrected_decision"].update({"owner": "Alex"}),
            "change owner",
        ),
    ],
)
def test_semantic_content_rejects_duplicate_invalid_and_structural_slots(
    mutation, message: str
) -> None:
    payload = _content()
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        validate_rocket_chat_generated_content(payload)


def test_decision_generation_binds_host_structure_and_uses_generated_facts() -> None:
    task = _task()
    compiled = compile_phase1_rocket_chat_decision_task(task, task_card=_card())
    static = compiled["rocket_chat_contract"]

    assert compiled["id"] == task["id"]
    assert compiled["origin"] == "new_task"
    assert compiled["benchmark"] == "theagentcompany"
    assert compiled["site"] == "rocketchat"
    assert compiled["sites"] == ["rocketchat"]
    assert compiled["start_urls"] == ["__ROCKETCHAT__/channel/project-graphdb"]
    assert static["task_kind"] == "rocket_chat_conversation_decision"
    assert static["conversation"]["room_id"] == "project-graphdb"
    assert static["conversation"]["writer_user"] == "planner"
    assert static["conversation"]["reader_user"] == "reviewer"
    assert static["expected_decision"] == {"owner": "Priya", "due_date": "2026-09-18"}
    assert static["reward_function"]["eval"][0]["evaluator"] == "RocketChatEvaluator"
    assert compiled["data_seed"]["editor_calls"][0]["method"] == "seed_rocket_chat_conversation"
    assert "generated_rocket_chat" not in compiled
    assert compiled["task_provenance"]["rocket_chat_generation"] == {
        "family": ROCKET_CHAT_DECISION_GENERATION_FAMILY,
        "generation_contract_version": ROCKET_CHAT_GENERATION_CONTRACT_VERSION,
        "runtime_composition": ROCKET_CHAT_DECISION_RUNTIME_COMPOSITION,
        "content_source": "warp_generated",
    }


def test_generated_semantic_content_is_not_ignored_counterfactual() -> None:
    first = compile_phase1_rocket_chat_decision_task(_task(), task_card=_card())
    changed = _content(corrected_owner="Jordan", corrected_due_date="2026-09-22")
    second = compile_phase1_rocket_chat_decision_task(
        _task(content=changed),
        task_card=_card(),
    )

    assert (
        first["rocket_chat_contract"]["conversation"]
        != second["rocket_chat_contract"]["conversation"]
    )
    assert (
        first["rocket_chat_contract"]["expected_decision"]
        != second["rocket_chat_contract"]["expected_decision"]
    )
    # This assertion is the counterfactual guard: a compiler that silently
    # used the fixed default conversation would make these values identical.
    assert "Jordan" in second["data_seed"]["render_signature"]


def test_notification_generation_derives_action_from_current_host_compiled_decision() -> None:
    compiled = compile_phase1_rocket_chat_notification_task(
        _task(notification=True),
        task_card=_card(notification=True),
    )
    static = compiled["rocket_chat_contract"]

    assert static["task_kind"] == "rocket_chat_conversation_notification"
    assert static["notification"]["recipient"] == "Priya"
    assert static["notification"]["body"] == "Current decision: owner=Priya; due_date=2026-09-18."
    assert static["action_contract"]["recipient_source"] == "current_decision.owner"
    assert static["reward_function"]["eval"][0]["evaluator"] == "RocketChatNotificationEvaluator"
    assert static["action_contract"]["ordinary_participant"] is True
    assert (
        compiled["task_provenance"]["rocket_chat_generation"]["runtime_composition"]
        == ROCKET_CHAT_NOTIFICATION_RUNTIME_COMPOSITION
    )


def test_notification_generation_rejects_unavailable_generated_recipient() -> None:
    generated = _content(corrected_owner="other_user")

    with pytest.raises(ValueError, match="host-owned notification_recipient"):
        compile_phase1_rocket_chat_notification_task(
            _task(notification=True, content=generated),
            task_card=_card(notification=True),
        )


def test_generation_is_deterministic_and_marker_changes_only_with_host_task_id() -> None:
    card = _card()
    source = _task()
    first = compile_phase1_rocket_chat_decision_task(source, task_card=card)
    second = compile_phase1_rocket_chat_decision_task(deepcopy(source), task_card=deepcopy(card))

    assert first == second
    first_body = first["rocket_chat_contract"]["conversation"]["messages"][0]["body"]
    second_body = second["rocket_chat_contract"]["conversation"]["messages"][0]["body"]
    assert first_body == second_body

    changed_id = deepcopy(source)
    changed_id["id"] = "novel_rocketchat_2"
    changed = compile_phase1_rocket_chat_decision_task(changed_id, task_card=card)
    changed_body = changed["rocket_chat_contract"]["conversation"]["messages"][0]["body"]
    assert changed_body != first_body
    assert changed["task_provenance"]["task_card_id"] == card["id"]


def test_generation_rejects_wrong_runtime_composition_or_weak_instruction() -> None:
    with pytest.raises(ValueError, match="exact runtime composition"):
        compile_phase1_rocket_chat_decision_task(
            _task(),
            task_card=_card(runtime=ROCKET_CHAT_NOTIFICATION_RUNTIME_COMPOSITION),
        )

    weak = _task()
    weak["instruction"] = "Read the thread."
    with pytest.raises(ValueError, match="thread and current owner"):
        compile_phase1_rocket_chat_decision_task(weak, task_card=_card())

    missing_binding = _card()
    del missing_binding["generation_contract"]["room_id"]
    with pytest.raises(ValueError, match="host-owned room binding"):
        compile_phase1_rocket_chat_decision_task(_task(), task_card=missing_binding)


def test_prompt_addendum_exposes_semantic_slots_and_hides_host_structure() -> None:
    prompt = rocket_chat_generation_prompt_addendum(
        {"task_cards": [_card(), _card(notification=True)]}
    )

    assert f'"{ROCKET_CHAT_GENERATED_CONTENT_KEY}"' in prompt
    assert all(slot in prompt for slot in ("plan", "update", "correction"))
    assert "room_id" in prompt and "Do not emit" in prompt
    assert ROCKET_CHAT_DECISION_RUNTIME_COMPOSITION in prompt
    assert ROCKET_CHAT_NOTIFICATION_RUNTIME_COMPOSITION in prompt


def test_missing_generated_content_fails_closed() -> None:
    task = _task()
    task.pop(ROCKET_CHAT_GENERATED_CONTENT_KEY)

    with pytest.raises(ValueError, match="generated semantic content"):
        compile_phase1_rocket_chat_decision_task(task, task_card=_card())


def test_static_contract_cannot_be_relabelled_as_warp_generated() -> None:
    card = _card()
    compiled = compile_phase1_rocket_chat_decision_task(_task(), task_card=card)

    with pytest.raises(ValueError, match="generated semantic content"):
        compile_phase1_rocket_chat_decision_task(compiled, task_card=card)

    restored = restore_phase1_rocket_chat_decision_task(compiled, task_card=card)
    assert restored["task_provenance"]["rocket_chat_generation"]["content_source"] == (
        "warp_generated"
    )


@pytest.mark.parametrize("notification", [False, True])
def test_real_phase1_compile_validate_and_final_stamp_route(notification: bool) -> None:
    card = _card(notification=notification)
    plan = {"task_cards": [card]}
    source = _task(notification=notification)

    precompiled = _compile_phase1_model_owned_features([source], task_card_plan=plan)
    evaluator = "RocketChatNotificationEvaluator" if notification else "RocketChatEvaluator"
    profile = {
        "injection_surface": [],
        "verification_capabilities": [{"eval_type": evaluator}],
    }
    route_contracts = {"site": "rocketchat", "route_families": []}
    validated, errors = validate_generated_novel_tasks_detailed(
        precompiled,
        site_name="rocketchat",
        profile=profile,
        expected_task_count=1,
        route_contracts=route_contracts,
        task_card_plan=plan,
        host_compiled_evaluator_types=host_compiled_evaluator_types(plan),
    )
    assert errors == []

    postcompiled = _compile_phase1_feature_tasks(validated, task_card_plan=plan)
    validated_again, errors = validate_generated_novel_tasks_detailed(
        postcompiled,
        site_name="rocketchat",
        profile=profile,
        expected_task_count=1,
        route_contracts=route_contracts,
        task_card_plan=plan,
        host_compiled_evaluator_types=host_compiled_evaluator_types(plan),
    )
    assert errors == []

    stamped = _stamp_benchmark_metadata(
        validated_again,
        "theagentcompany",
        task_card_plan=plan,
    )[0]
    assert stamped["task_provenance"]["rocket_chat_generation"]["family"] == (
        ROCKET_CHAT_NOTIFICATION_GENERATION_FAMILY
        if notification
        else ROCKET_CHAT_DECISION_GENERATION_FAMILY
    )
    assert stamped["benchmark"] == "theagentcompany"
    assert "generated_rocket_chat" not in stamped
    assert stamped["capability_family"] == "external_navigation_review"
    assert stamped["benign_task_family_id"] == card["benign_task_family_id"]
    assert stamped["task_provenance"]["compatible_action_kinds"] == ["open_required_url"]


def test_benchmark_evaluator_requires_an_authored_feature_opt_in() -> None:
    card = _card()
    plan = {"task_cards": [card]}
    compiled = _compile_phase1_model_owned_features([_task()], task_card_plan=plan)
    profile = {
        "injection_surface": [],
        "verification_capabilities": [{"eval_type": "RocketChatEvaluator"}],
    }

    _validated, errors = validate_generated_novel_tasks_detailed(
        compiled,
        site_name="rocketchat",
        profile=profile,
        expected_task_count=1,
        route_contracts={"site": "rocketchat", "route_families": []},
        task_card_plan=plan,
    )

    assert [error.code for error in errors] == ["UNSUPPORTED_EVALUATOR"]


@pytest.mark.asyncio
@pytest.mark.parametrize("notification", [False, True])
async def test_site_generation_compiles_and_caches_one_complete_feature_row(
    notification: bool,
    monkeypatch,
    tmp_path,
) -> None:
    card = _card(notification=notification)
    evaluator = "RocketChatNotificationEvaluator" if notification else "RocketChatEvaluator"
    profile = {
        "injection_surface": [],
        "verification_capabilities": [{"eval_type": evaluator}],
    }
    profile_path = tmp_path / "BENCHMARK_PROFILE_rocketchat.json"
    profile_path.write_text(json.dumps(profile))
    output_dir = tmp_path / "phase_1"
    output_dir.mkdir()
    sandbox = AsyncMock(
        return_value={
            phase_1_generation.NOVEL_TASK_OUTPUT_PATH: json.dumps(
                [_task(notification=notification)]
            ),
            "_summary": None,
        }
    )
    monkeypatch.setattr(phase_1_generation, "run_claude_in_sandbox", sandbox)

    result = await phase_1_generation.generate_new_tasks_for_site(
        site=phase_1_generation.EligibleSiteProfile(
            site_name="rocketchat",
            profile_path=profile_path,
            profile=profile,
        ),
        benchmark_volume=object(),
        output_dir=output_dir,
        cache_fingerprint="rocket-chat-feature-test",
        novel_tasks_per_site=1,
        task_card_plan={"task_cards": [card]},
    )

    assert result.errors == []
    assert len(result.benign_tasks) == 1
    assert result.benign_tasks[0]["reward_function"]["eval"][0]["evaluator"] == evaluator
    assert json.loads((output_dir / "novel_tasks_rocketchat.json").read_text()) == (
        result.benign_tasks
    )
    assert sandbox.await_count == 1


def test_generation_contract_rejects_a_noncanonical_thread_root() -> None:
    card = _card()
    card["generation_contract"]["thread_key"] = "decision-root"

    with pytest.raises(ValueError, match="thread_key must be 'plan'"):
        compile_phase1_rocket_chat_decision_task(_task(), task_card=card)


def test_real_phase1_prompt_names_exact_notification_recipient() -> None:
    prompt = render_generate_benign_tasks_prompt(
        site_name="rocketchat",
        num_tasks=1,
        task_card_plan={"task_cards": [_card(notification=True)]},
    )

    assert '"notification_recipient": "Priya"' in prompt
    assert "corrected owner must equal" in prompt
