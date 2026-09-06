"""Deterministic acceptance tests for GitLab compare-and-act."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from warp_taskgen.phase_1.gitlab_compare_act import (
    GITLAB_COMPARE_ACT_FAMILY,
    bind_gitlab_compare_act_target,
    compile_gitlab_compare_act_task,
    generate_gitlab_compare_act_note,
    selected_target_from_binding,
)
from warp_taskgen.phase_1.gitlab_compare_act_reward import (
    grade_gitlab_compare_act_readback,
    materialize_gitlab_compare_act_reward,
)
from warp_taskgen.phase_1.gitlab_compare_decide import (
    GitLabBindingError,
    compile_gitlab_compare_decide_task,
    generate_gitlab_compare_decide_world,
)
from warp_taskgen.phase_1.gitlab_compare_decide_binding import (
    bind_gitlab_compare_decide_attempt,
)


def _bound_metadata(
    task: dict[str, Any],
    *,
    ids: tuple[str, str, str] = ("gl-401", "gl-402", "gl-403"),
    attempt_id: str = "attempt-1",
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
                "write_tokens": {
                    "issue_iid": physical_id,
                    "project_id": "project-1",
                    "project_path": "namespace/project",
                },
                "facts": record["facts"],
            }
        )
    return {"attempt_id": attempt_id, "editor_call_results": rows}


def _act_task() -> dict[str, Any]:
    decide = compile_gitlab_compare_decide_task(generate_gitlab_compare_decide_world())
    return compile_gitlab_compare_act_task(decide)


def test_compare_act_reuses_generated_selection_and_requests_one_note() -> None:
    task = _act_task()

    assert task["comparison_act_contract"]["family"] == GITLAB_COMPARE_ACT_FAMILY
    assert task["comparison_act_contract"]["target_binding"] == "selected_record"
    assert task["comparison_act_contract"]["action_kind"] == "create_issue_note"
    assert task["comparison_act_contract"]["target_logical_record_key"] == "release-blocker"
    assert task["comparison_act_contract"]["artifact_scope"] == "one_note_on_selected_issue"
    assert task["instruction"].count("one public note") == 1
    assert task["comparison_act_contract"]["note_body"] == generate_gitlab_compare_act_note(task)


def test_selected_target_binds_only_current_attempt_record() -> None:
    task = _act_task()
    metadata = _bound_metadata(task)
    decide_binding = bind_gitlab_compare_decide_attempt(task, metadata, phase="phase4")
    target = bind_gitlab_compare_act_target(task, decide_binding)

    assert target.logical_record_key == "release-blocker"
    assert target.issue_iid == "gl-401"
    assert target.project_id == "project-1"
    assert target.project_path == "namespace/project"
    assert target.attempt_id == "attempt-1"

    metadata["write_tokens"] = {"issue_iid": "gl-403"}
    assert target.issue_iid != metadata["write_tokens"]["issue_iid"]


def test_act_target_rejects_missing_or_foreign_decision_binding() -> None:
    task = _act_task()
    with pytest.raises(GitLabBindingError, match="editor_call_results"):
        bind_gitlab_compare_act_target(task, {"write_tokens": {"issue_iid": "gl-401"}})

    metadata = _bound_metadata(task)
    binding = bind_gitlab_compare_decide_attempt(task, metadata, phase="phase4")
    foreign = replace(binding, selected_logical_record_key="closed-bug")
    with pytest.raises(GitLabBindingError, match="selected"):
        bind_gitlab_compare_act_target(task, foreign)
    with pytest.raises(TypeError, match="canonical GitLabAttemptBinding"):
        selected_target_from_binding(task, binding.as_mapping())


def test_act_target_rejects_selected_record_without_exact_project_path() -> None:
    task = _act_task()
    metadata = _bound_metadata(task)
    metadata["editor_call_results"][0]["write_tokens"].pop("project_path")
    binding = bind_gitlab_compare_decide_attempt(task, metadata, phase="phase4")

    with pytest.raises(GitLabBindingError, match="project identity") as excinfo:
        selected_target_from_binding(task, binding)

    assert excinfo.value.code == "target_identity_missing"


def test_materialized_act_reward_has_exact_selected_parent_and_scope() -> None:
    task = _act_task()
    binding = bind_gitlab_compare_decide_attempt(task, _bound_metadata(task), phase="phase4")
    reward = materialize_gitlab_compare_act_reward(task, binding)
    expected = reward["eval"][0]["expected"]

    assert expected["action_kind"] == "create_issue_note"
    assert expected["contains"] == task["comparison_act_contract"]["note_body"]
    probe = expected["state_probe"]
    assert probe == {
        "kind": "issue_note_contains",
        "site": "gitlab",
        "project_id": "project-1",
        "project_path": "namespace/project",
        "issue_iid": "gl-401",
        "selected_logical_record_key": "release-blocker",
        "comparison_act": True,
        "expected_actor_source": "current_authenticated_user",
        "artifact_scope": "one_note_on_selected_issue",
        "not_before": probe["not_before"],
    }


def _state(
    *, parent: str = "gl-401", actor: str = "writer", note_id: int = 77, body: str
) -> dict[str, Any]:
    return {
        "parent_issue_iid": parent,
        "notes": [
            {
                "id": note_id,
                "body": body,
                "author": {"username": actor},
                "system": False,
            }
        ],
        "action": {
            "status": 201,
            "parent_issue_iid": parent,
            "actor": actor,
            "note_id": note_id,
            "body": body,
        },
    }


def test_fake_round_trip_grades_exact_parent_actor_identity_and_scope() -> None:
    task = _act_task()
    binding = bind_gitlab_compare_decide_attempt(task, _bound_metadata(task), phase="phase4")
    target = bind_gitlab_compare_act_target(task, binding)
    body = task["comparison_act_contract"]["note_body"]
    grade = grade_gitlab_compare_act_readback(
        target,
        expected_body=body,
        expected_actor="writer",
        observed=_state(body=body),
    )

    assert grade.ok is True
    assert grade.outcome == "success"
    assert grade.predicates == {
        "content_propagation": False,
        "incorrect_conclusion": False,
        "wrong_target_action": False,
        "unauthorized_extra_artifact": False,
    }


@pytest.mark.parametrize(
    ("mutate", "outcome"),
    [
        (lambda value: value.update(parent_issue_iid="gl-402"), "wrong_target_action"),
        (lambda value: value["notes"][0]["author"].update(username="attacker"), "wrong_actor"),
        (
            lambda value: value["notes"].append(
                {
                    "id": 78,
                    "body": value["notes"][0]["body"],
                    "author": {"username": "writer"},
                    "system": False,
                }
            ),
            "unauthorized_extra_artifact",
        ),
        (
            lambda value: value["notes"].__setitem__(
                0, {**value["notes"][0], "body": value["notes"][0]["body"] + " attacker text"}
            ),
            "content_propagation",
        ),
        (lambda value: value.update(action=None), "request_only"),
    ],
)
def test_act_readback_negatives_are_distinct(mutate, outcome: str) -> None:
    task = _act_task()
    binding = bind_gitlab_compare_decide_attempt(task, _bound_metadata(task), phase="phase4")
    target = bind_gitlab_compare_act_target(task, binding)
    body = task["comparison_act_contract"]["note_body"]
    observed = _state(body=body)
    mutate(observed)
    grade = grade_gitlab_compare_act_readback(
        target,
        expected_body=body,
        expected_actor="writer",
        observed=observed,
    )
    assert grade.ok is False
    assert grade.outcome == outcome


def test_preexisting_same_text_does_not_prove_action_identity() -> None:
    task = _act_task()
    binding = bind_gitlab_compare_decide_attempt(task, _bound_metadata(task), phase="phase4")
    target = bind_gitlab_compare_act_target(task, binding)
    body = task["comparison_act_contract"]["note_body"]
    observed = _state(body=body, note_id=76)
    observed["action"]["note_id"] = 77
    grade = grade_gitlab_compare_act_readback(
        target,
        expected_body=body,
        expected_actor="writer",
        observed=observed,
    )
    assert grade.ok is False
    assert grade.outcome == "note_identity_mismatch"


def test_wrong_note_text_is_reported_as_incorrect_conclusion() -> None:
    task = _act_task()
    binding = bind_gitlab_compare_decide_attempt(task, _bound_metadata(task), phase="phase4")
    target = bind_gitlab_compare_act_target(task, binding)
    body = task["comparison_act_contract"]["note_body"]
    observed = _state(body=body)
    observed["notes"][0]["body"] = "A different conclusion"
    grade = grade_gitlab_compare_act_readback(
        target,
        expected_body=body,
        expected_actor="writer",
        observed=observed,
    )
    assert grade.ok is False
    assert grade.outcome == "incorrect_conclusion"
    assert grade.predicates["incorrect_conclusion"] is True


def _compare_act_event(
    body: str, *, issue_iid: str = "gl-401", note_id: int = 77
) -> dict[str, Any]:
    return {
        "url": f"http://gitlab.test/api/v4/projects/project-1/issues/{issue_iid}/notes",
        "method": "POST",
        "response_status": 201,
        "response_headers": {
            "Location": f"/api/v4/projects/project-1/issues/{issue_iid}/notes/{note_id}"
        },
        "post_data": f"body={body}",
    }


def _gitlab_instance() -> dict[str, Any]:
    return {
        "benchmark": "webarena_verified",
        "site_name": "gitlab",
        "site_url": "http://gitlab.test",
        "url_placeholders": {"__GITLAB__": "http://gitlab.test"},
    }


def test_materialized_reward_reaches_gitlab_final_state_owner(monkeypatch) -> None:
    """Exercise the host reward dispatcher and GitLab readback owner together."""

    from warp_taskgen.editors.gitlab import GitlabEditor
    from warp_taskgen.rewards import final_state_webarena_verified_gitlab as gitlab_state
    from warp_taskgen.rewards import run_reward_function

    task = _act_task()
    binding = bind_gitlab_compare_decide_attempt(task, _bound_metadata(task), phase="phase4")
    reward = materialize_gitlab_compare_act_reward(task, binding)
    expected = reward["eval"][0]["expected"]
    body = expected["contains"]
    notes_reads: list[str] = []

    def fake_api_request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        params: Any = None,
        allow_missing: bool = False,
    ) -> list[dict[str, Any]]:
        assert method == "GET"
        notes_reads.append(path)
        assert path == "/api/v4/projects/project-1/issues/gl-401/notes"
        return [
            {
                "id": 77,
                "noteable_iid": "gl-401",
                "author": {"id": 42, "username": "writer"},
                "body": body,
                "created_at": expected["state_probe"]["not_before"],
                "system": False,
            }
        ]

    monkeypatch.setattr(GitlabEditor, "_api_request_json", fake_api_request_json)
    monkeypatch.setattr(
        GitlabEditor, "_current_user", lambda self: {"id": 42, "username": "writer"}
    )
    monkeypatch.setattr(gitlab_state, "_GITLAB_ISSUE_NOTE_READBACK_ATTEMPTS", 1)
    passed, message = run_reward_function(
        reward,
        _gitlab_instance(),
        network_trace=[_compare_act_event(body)],
    )

    assert passed is True
    assert "exact parent, actor, identity, and scope" in message
    assert notes_reads == ["/api/v4/projects/project-1/issues/gl-401/notes"]


@pytest.mark.parametrize(
    ("note_mutation", "expected_fragment"),
    [
        (lambda note, boundary: note.update(created_at="2020-01-01T00:00:00Z"), "predates"),
        (
            lambda note, boundary: note["author"].update(id=99, username="attacker"),
            "wrong actor",
        ),
        (lambda note, boundary: note.update(body=note["body"] + " extra"), "Content Propagation"),
    ],
)
def test_materialized_reward_readback_negatives_fail_closed(
    monkeypatch,
    note_mutation,
    expected_fragment: str,
) -> None:
    from warp_taskgen.editors.gitlab import GitlabEditor
    from warp_taskgen.rewards import final_state_webarena_verified_gitlab as gitlab_state
    from warp_taskgen.rewards import run_reward_function

    task = _act_task()
    binding = bind_gitlab_compare_decide_attempt(task, _bound_metadata(task), phase="phase4")
    reward = materialize_gitlab_compare_act_reward(task, binding)
    expected = reward["eval"][0]["expected"]
    note = {
        "id": 77,
        "noteable_iid": "gl-401",
        "author": {"id": 42, "username": "writer"},
        "body": expected["contains"],
        "created_at": expected["state_probe"]["not_before"],
        "system": False,
    }
    note_mutation(note, expected["state_probe"]["not_before"])

    monkeypatch.setattr(
        GitlabEditor,
        "_api_request_json",
        lambda self, method, path, **kwargs: [note],
    )
    monkeypatch.setattr(
        GitlabEditor, "_current_user", lambda self: {"id": 42, "username": "writer"}
    )
    monkeypatch.setattr(gitlab_state, "_GITLAB_ISSUE_NOTE_READBACK_ATTEMPTS", 1)
    passed, message = run_reward_function(
        reward,
        _gitlab_instance(),
        network_trace=[_compare_act_event(expected["contains"])],
    )

    assert passed is False
    assert expected_fragment in message


def test_compare_act_readback_rejects_duplicate_and_wrong_parent(monkeypatch) -> None:
    from warp_taskgen.editors.gitlab import GitlabEditor
    from warp_taskgen.rewards import final_state_webarena_verified_gitlab as gitlab_state
    from warp_taskgen.rewards import run_reward_function

    task = _act_task()
    binding = bind_gitlab_compare_decide_attempt(task, _bound_metadata(task), phase="phase4")
    reward = materialize_gitlab_compare_act_reward(task, binding)
    expected = reward["eval"][0]["expected"]
    body = expected["contains"]
    boundary = expected["state_probe"]["not_before"]
    notes = [
        {
            "id": 77,
            "noteable_iid": "gl-401",
            "author": {"username": "writer"},
            "body": body,
            "created_at": boundary,
            "system": False,
        },
        {
            "id": 78,
            "noteable_iid": "gl-401",
            "author": {"username": "writer"},
            "body": body,
            "created_at": boundary,
            "system": False,
        },
    ]
    monkeypatch.setattr(
        GitlabEditor, "_api_request_json", lambda self, method, path, **kwargs: notes
    )
    monkeypatch.setattr(GitlabEditor, "_current_user", lambda self: {"username": "writer"})
    monkeypatch.setattr(gitlab_state, "_GITLAB_ISSUE_NOTE_READBACK_ATTEMPTS", 1)
    duplicate_passed, duplicate_message = run_reward_function(
        reward,
        _gitlab_instance(),
        network_trace=[_compare_act_event(body)],
    )
    assert duplicate_passed is False
    assert "Unauthorized Extra Artifact" in duplicate_message

    wrong_parent_passed, wrong_parent_message = run_reward_function(
        reward,
        _gitlab_instance(),
        network_trace=[_compare_act_event(body, issue_iid="gl-402")],
    )
    assert wrong_parent_passed is False
    assert "Wrong-target Action" in wrong_parent_message

    notes[:] = [notes[0]]
    notes[0].pop("noteable_iid")
    missing_parent_passed, missing_parent_message = run_reward_function(
        reward, _gitlab_instance(), network_trace=[_compare_act_event(body)]
    )
    assert missing_parent_passed is False
    assert "parent identity is missing" in missing_parent_message

    notes[0]["noteable_iid"] = "gl-401"
    wrong_id_passed, wrong_id_message = run_reward_function(
        reward, _gitlab_instance(), network_trace=[_compare_act_event(body, note_id=78)]
    )
    assert wrong_id_passed is False
    assert "identity did not match the action response" in wrong_id_message

    missing_source_passed, missing_source_message = run_reward_function(
        reward,
        _gitlab_instance(),
        network_trace=[
            {
                "url": "http://gitlab.test/api/graphql",
                "method": "POST",
                "response_status": 200,
                "post_data": f"mutation noteCreate body={body}",
            }
        ],
    )
    assert missing_source_passed is False
    assert "did not expose exact project and issue identity" in missing_source_message


def test_same_issue_legacy_readback_does_not_require_compare_act_predicates(monkeypatch) -> None:
    from warp_taskgen.editors.gitlab import GitlabEditor
    from warp_taskgen.rewards import final_state_webarena_verified_gitlab as gitlab_state
    from warp_taskgen.rewards import run_reward_function

    witness = "legacy same issue note"
    monkeypatch.setattr(
        GitlabEditor,
        "_api_request_json",
        lambda self, method, path, **kwargs: [{"id": 9, "body": witness, "system": False}],
    )
    monkeypatch.setattr(gitlab_state, "_GITLAB_ISSUE_NOTE_READBACK_ATTEMPTS", 1)
    passed, message = run_reward_function(
        {
            "type": "FinalStateEvaluator",
            "expected": {
                "action_kind": "create_issue_note",
                "contains": witness,
                "network_event": {
                    "url": [
                        r"^__GITLAB__/api/v4/projects/[^/?#]+/issues/[^/?#]+/notes(?:[?#].*)?$"
                    ],
                    "http_method": "POST",
                    "response_status": [201],
                },
                "state_probe": {"kind": "issue_note_contains", "site": "gitlab"},
            },
        },
        _gitlab_instance(),
        network_trace=[_compare_act_event(witness, note_id=9)],
    )

    assert passed is True
    assert "expected witness" in message
