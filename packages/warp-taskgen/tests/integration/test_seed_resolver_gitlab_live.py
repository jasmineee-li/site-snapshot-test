from __future__ import annotations

import copy

import pytest
import requests

from warp_taskgen.auth_tokens import acquire_tokens_for_instances
from warp_taskgen.editors.base import EditorError
from warp_taskgen.editors.gitlab import GitlabEditor

pytestmark = pytest.mark.integration


def test_gitlab_live_probe_create_reuse_and_cleanup(live_instance, unique_suffix):
    instance = live_instance("gitlab")
    assert acquire_tokens_for_instances([instance]) == []

    with requests.Session() as session:
        editor = GitlabEditor(instance, session)
        GitlabEditor.probe_base_state(instance)

        slug = f"webagent-task-int-{unique_suffix}"
        first = editor.create_project(
            name_template=slug,
            description_template=f"Live integration project {unique_suffix}",
        )
        second = editor.create_project(name_template=slug)
        fetched = editor._gitlab_get_json(
            f"/api/v4/projects/{editor._quote(first['project_path'])}",
            allow_missing=False,
        )

        assert first["project_id"] == second["project_id"] == fetched["id"]
        assert fetched["path_with_namespace"].endswith(slug)

        editor.cleanup()
        deleted = editor._gitlab_get_json(
            f"/api/v4/projects/{editor._quote(first['project_path'])}",
            allow_missing=True,
        )
        assert deleted is None


def test_gitlab_live_probe_rejects_invalid_auth(live_instance):
    instance = live_instance("gitlab")
    assert acquire_tokens_for_instances([instance]) == []
    invalid = copy.deepcopy(instance)
    invalid["auth"] = {
        "type": "bearer_token",
        "header_name": "PRIVATE-TOKEN",
        "token": "invalid-token",
        "validation_endpoint": "/api/v4/user",
    }

    with pytest.raises(EditorError) as excinfo:
        GitlabEditor.probe_base_state(invalid)

    assert excinfo.value.kind == "auth_missing"


def test_gitlab_live_issue_mr_repo_and_profile_methods(live_instance, unique_suffix):
    instance = live_instance("gitlab")
    assert acquire_tokens_for_instances([instance]) == []

    with requests.Session() as session:
        editor = GitlabEditor(instance, session)
        GitlabEditor.probe_base_state(instance)

        current_user = editor._current_user()
        original_name = str(current_user.get("name") or "")
        original_bio = str(current_user.get("bio") or "")
        original_status = editor._gitlab_request_json("GET", "/api/v4/user/status")
        original_status_message = ""
        original_status_emoji = ""
        if isinstance(original_status, dict):
            original_status_message = str(original_status.get("message") or "")
            original_status_emoji = str(original_status.get("emoji") or "")

        slug = f"webagent-live-{unique_suffix}"
        issue_title = f"Issue {unique_suffix}"
        issue_note_body = f"Issue note {unique_suffix}"
        mr_title = f"MR {unique_suffix}"
        mr_note_body = f"MR note {unique_suffix}"
        repo_path = f"live-{unique_suffix}.md"
        repo_content = f"content {unique_suffix}\n"
        source_branch = f"seed-{unique_suffix}"
        updated_name = f"WorldSim {unique_suffix[:8]}"
        updated_bio = f"WorldSim bio {unique_suffix}"
        updated_status_message = f"WorldSim status {unique_suffix}"
        status_updated = False
        profile_updated = False

        try:
            issue = editor.create_issue(
                project_name_template=slug,
                title_template=issue_title,
                body_template=f"Issue body {unique_suffix}",
            )
            project_payload = editor._gitlab_get_json(
                f"/api/v4/projects/{issue['project_id']}",
                allow_missing=False,
            )
            issue_payload = editor._gitlab_get_json(
                f"/api/v4/projects/{issue['project_id']}/issues/{issue['issue_iid']}",
                allow_missing=False,
            )
            issue_note = editor.create_issue_note(
                project_id=issue["project_id"],
                issue_iid=issue["issue_iid"],
                note_body=issue_note_body,
            )
            issue_notes = editor._gitlab_request_json(
                "GET",
                f"/api/v4/projects/{issue['project_id']}/issues/{issue['issue_iid']}/notes",
                params={"per_page": 100},
            )
            merge_request = editor.create_mr(
                project_id=issue["project_id"],
                title_template=mr_title,
                body_template=f"MR body {unique_suffix}",
                source_branch=source_branch,
            )
            merge_request_payload = editor._gitlab_get_json(
                f"/api/v4/projects/{merge_request['project_id']}/merge_requests/{merge_request['mr_iid']}",
                allow_missing=False,
            )
            mr_note = editor.create_mr_note(
                project_id=merge_request["project_id"],
                mr_iid=merge_request["mr_iid"],
                note_body=mr_note_body,
            )
            mr_notes = editor._gitlab_request_json(
                "GET",
                f"/api/v4/projects/{merge_request['project_id']}/merge_requests/{merge_request['mr_iid']}/notes",
                params={"per_page": 100},
            )
            repo_file = editor.create_repo_file(
                project_id=issue["project_id"],
                branch=str(project_payload.get("default_branch") or "main"),
                path=repo_path,
                content=repo_content,
            )
            repo_file_content = editor._gitlab_get_file_content(
                issue["project_id"],
                file_path=repo_path,
                ref=str(project_payload.get("default_branch") or "main"),
            )
            editor.update_user_status(message=updated_status_message, emoji="speech_balloon")
            status_updated = True
            updated_status = editor._gitlab_request_json("GET", "/api/v4/user/status")
            editor.update_user_profile(name=updated_name, bio=updated_bio)
            profile_updated = True
            updated_user = editor._gitlab_request_json("GET", "/api/v4/user")

            assert issue_payload["title"] == issue_title
            assert isinstance(issue_notes, list) and any(
                isinstance(note, dict)
                and note.get("id") == issue_note["note_id"]
                and note.get("body") == issue_note_body
                for note in issue_notes
            )
            assert merge_request_payload["title"] == mr_title
            assert isinstance(mr_notes, list) and any(
                isinstance(note, dict)
                and note.get("id") == mr_note["note_id"]
                and note.get("body") == mr_note_body
                for note in mr_notes
            )
            assert repo_file["commit_id"] not in (None, "")
            assert repo_file_content == repo_content
            assert isinstance(updated_status, dict)
            assert str(updated_status.get("message") or "") == updated_status_message
            assert isinstance(updated_user, dict)
            assert str(updated_user.get("name") or "") == updated_name
            assert str(updated_user.get("bio") or "") == updated_bio
        finally:
            if status_updated:
                editor.update_user_status(
                    message=original_status_message,
                    emoji=original_status_emoji or None,
                )
            if profile_updated:
                try:
                    editor.update_user_profile(
                        name=original_name,
                        bio=original_bio,
                    )
                except EditorError as exc:
                    if not (exc.kind == "request_failed" and "HTTP 404" in exc.detail):
                        raise
            editor.cleanup()
