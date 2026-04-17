from __future__ import annotations

import pytest

from worldsim.auth_tokens import acquire_tokens_for_instances
from worldsim.seed_resolvers import gitlab

pytestmark = pytest.mark.integration


def test_gitlab_live_reuses_existing_project(live_instance, unique_suffix):
    instance = live_instance("gitlab")
    errors = acquire_tokens_for_instances([instance])
    assert errors == []

    slug = f"webagent-task-int-{unique_suffix}"
    target = {
        "create": {
            "project": {
                "name_template": slug,
                "path_template": slug,
                "description_template": f"Live integration project {unique_suffix}",
            }
        }
    }
    seed_context = {"task_id": unique_suffix}

    first = gitlab._ensure_project(target, instance, seed_context)
    second = gitlab._ensure_project(target, instance, seed_context)

    assert first["id"] == second["id"]
    assert first["path_with_namespace"] == second["path_with_namespace"]
    assert first["path_with_namespace"].endswith(slug)


def test_gitlab_live_creates_issue_and_merge_request_parents(live_instance, unique_suffix):
    instance = live_instance("gitlab")
    errors = acquire_tokens_for_instances([instance])
    assert errors == []

    issue_task_id = f"{unique_suffix}-issue"
    mr_task_id = f"{unique_suffix}-mr"

    issue_call = gitlab.create(
        {
            "mechanism": "api",
            "method": "POST",
            "resource_type": "issue",
            "create": {
                "project": {"name_template": f"webagent-task-int-{issue_task_id}"},
                "issue": {
                    "title_template": f"Live issue {issue_task_id}",
                    "body_template": f"Live issue body {issue_task_id}",
                },
            },
        },
        instance,
        {"task_id": issue_task_id},
    )
    assert "/issues" in issue_call.url
    assert issue_call.context_additions["project_id"]

    mr_call = gitlab.create(
        {
            "mechanism": "api",
            "method": "POST",
            "resource_type": "mr",
            "create": {
                "project": {"name_template": f"webagent-task-int-{mr_task_id}"},
                "mr": {
                    "title_template": f"Live MR {mr_task_id}",
                    "body_template": f"Live MR body {mr_task_id}",
                    "source_branch_template": f"live-int-{mr_task_id}",
                    "target_branch_template": "main",
                },
            },
        },
        instance,
        {"task_id": mr_task_id},
    )
    assert "/merge_requests" in mr_call.url
    assert mr_call.context_additions["project_id"]
