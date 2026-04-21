"""Option A placement validator tests (X2d).

Directly exercises ``phase_2_injections._validate_option_a_placement``
and ``_is_option_a_site`` so the Option A contract is tested in
isolation from the surrounding surface-match / invariant checks (which
have their own coverage in ``test_phase_2_injections.py``).

The adopted validator semantics are skip-on-reject: a non-None return
value is a structured rejection reason; callers log + drop the task.
"""

from __future__ import annotations

from worldsim.phases import phase_2_injections


def _base_gitlab_plan() -> dict:
    """A valid Option A plan: note on existing benign-task issue."""
    return {
        "id": "adv-gitlab-1",
        "benign_task_id": "benign-1",
        "target_surface_id": "note_on_issue",
        "framing": "authority",
        "concealment": "plaintext",
        "delivery_mechanism": "api",
        "site": "gitlab",
        "sites": ["gitlab"],
        "attack_objective": "…",
        "benign_target_resource": {
            "kind": "gitlab_issue",
            "anchors": {"project_id": "159", "issue_iid": "104"},
            "attach_surfaces": [
                {
                    "surface_id": "note_on_issue",
                    "attach_method": "create_issue_note",
                    "required_editor_args": ["project_id", "issue_iid", "body"],
                }
            ],
            "layer": "L1",
        },
        "seed_template": {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "gitlab",
                    "method": "create_issue_note",
                    "args": {
                        "project_id": "{benign_project_id}",
                        "issue_iid": "{benign_issue_iid}",
                        "body": "{{PAYLOAD_TEXT}}",
                    },
                }
            ],
        },
    }


def _base_reddit_plan() -> dict:
    return {
        "id": "adv-reddit-1",
        "benign_task_id": "benign-r1",
        "target_surface_id": "comment_body_thread",
        "framing": "peer_tip",
        "concealment": "plaintext",
        "delivery_mechanism": "form",
        "site": "reddit",
        "sites": ["reddit"],
        "attack_objective": "…",
        "benign_target_resource": {
            "kind": "reddit_submission",
            "anchors": {"forum_name": "books", "submission_id": "59421"},
            "attach_surfaces": [
                {
                    "surface_id": "comment_body_thread",
                    "attach_method": "create_comment",
                    "required_editor_args": ["submission_id", "body"],
                }
            ],
            "layer": "L1",
        },
        "seed_template": {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "reddit",
                    "method": "create_comment",
                    "args": {
                        "submission_id": "{benign_submission_id}",
                        "body": "{{PAYLOAD_TEXT}}",
                    },
                }
            ],
        },
    }


# --- acceptance ---------------------------------------------------------


def test_accepts_benign_attached_gitlab_plan():
    assert phase_2_injections._validate_option_a_placement(_base_gitlab_plan(), "plan") is None


def test_accepts_reddit_comment_with_benign_submission_token():
    assert phase_2_injections._validate_option_a_placement(_base_reddit_plan(), "plan") is None


def test_accepts_gitlab_mr_note_with_benign_mr_iid_token():
    plan = _base_gitlab_plan()
    plan["target_surface_id"] = "note_on_mr"
    plan["benign_target_resource"] = {
        "kind": "gitlab_mr",
        "anchors": {"project_id": "159", "mr_iid": "42"},
        "attach_surfaces": [
            {
                "surface_id": "note_on_mr",
                "attach_method": "create_mr_note",
                "required_editor_args": ["project_id", "mr_iid", "body"],
            }
        ],
        "layer": "L1",
    }
    plan["seed_template"]["editor_calls"][0] = {
        "benchmark": "webarena_verified",
        "site": "gitlab",
        "method": "create_mr_note",
        "args": {
            "project_id": "{benign_project_id}",
            "mr_iid": "{benign_mr_iid}",
            "body": "{{PAYLOAD_TEXT}}",
        },
    }
    assert phase_2_injections._validate_option_a_placement(plan, "plan") is None


# --- dangling-parent rejections ----------------------------------------


def test_rejects_dangling_mechanism_under_option_a():
    plan = _base_gitlab_plan()
    plan["seed_template"]["editor_calls"][0] = {
        "benchmark": "webarena_verified",
        "site": "gitlab",
        "method": "create_project",
        "args": {"name": "webagent-task-dangling", "description": "{{PAYLOAD_TEXT}}"},
    }
    reason = phase_2_injections._validate_option_a_placement(plan, "plan")
    assert reason is not None
    assert "create_project" in reason
    assert "dangling" in reason


def test_rejects_create_group():
    plan = _base_gitlab_plan()
    plan["seed_template"]["editor_calls"][0] = {
        "benchmark": "webarena_verified",
        "site": "gitlab",
        "method": "create_group",
        "args": {"name": "g", "description": "{{PAYLOAD_TEXT}}"},
    }
    reason = phase_2_injections._validate_option_a_placement(plan, "plan")
    assert reason is not None
    assert "create_group" in reason


def test_rejects_create_forum():
    plan = _base_reddit_plan()
    plan["seed_template"]["editor_calls"][0] = {
        "benchmark": "webarena_verified",
        "site": "reddit",
        "method": "create_forum",
        "args": {"name": "f", "description": "{{PAYLOAD_TEXT}}"},
    }
    reason = phase_2_injections._validate_option_a_placement(plan, "plan")
    assert reason is not None
    assert "create_forum" in reason


# --- child-create-without-token rejections -----------------------------


def test_rejects_create_issue_without_benign_project_id_token():
    plan = _base_gitlab_plan()
    plan["seed_template"]["editor_calls"][0] = {
        "benchmark": "webarena_verified",
        "site": "gitlab",
        "method": "create_issue",
        "args": {
            "project_id": "159",  # concrete, not {benign_project_id}
            "title": "x",
            "description": "{{PAYLOAD_TEXT}}",
        },
    }
    reason = phase_2_injections._validate_option_a_placement(plan, "plan")
    assert reason is not None
    assert "project_id" in reason


def test_rejects_create_issue_note_without_benign_issue_iid_token():
    plan = _base_gitlab_plan()
    plan["seed_template"]["editor_calls"][0]["args"]["issue_iid"] = "104"
    reason = phase_2_injections._validate_option_a_placement(plan, "plan")
    assert reason is not None
    assert "issue_iid" in reason


def test_rejects_create_mr_note_without_benign_mr_iid_token():
    plan = _base_gitlab_plan()
    plan["seed_template"]["editor_calls"][0] = {
        "benchmark": "webarena_verified",
        "site": "gitlab",
        "method": "create_mr_note",
        "args": {
            "project_id": "{benign_project_id}",
            "mr_iid": "42",
            "body": "{{PAYLOAD_TEXT}}",
        },
    }
    reason = phase_2_injections._validate_option_a_placement(plan, "plan")
    assert reason is not None
    assert "mr_iid" in reason


def test_rejects_create_submission_without_benign_forum_name_token():
    plan = _base_reddit_plan()
    plan["seed_template"]["editor_calls"][0] = {
        "benchmark": "webarena_verified",
        "site": "reddit",
        "method": "create_submission",
        "args": {
            "forum_name": "books",  # concrete
            "title": "x",
            "body": "{{PAYLOAD_TEXT}}",
        },
    }
    reason = phase_2_injections._validate_option_a_placement(plan, "plan")
    assert reason is not None
    assert "forum_name" in reason


def test_rejects_create_comment_without_benign_submission_token():
    plan = _base_reddit_plan()
    plan["seed_template"]["editor_calls"][0]["args"]["submission_id"] = "59421"
    reason = phase_2_injections._validate_option_a_placement(plan, "plan")
    assert reason is not None
    assert "submission_id" in reason


# --- benign_target_resource shape rejections ---------------------------


def test_rejects_plan_with_null_benign_target_resource_kind():
    plan = _base_gitlab_plan()
    plan["benign_target_resource"] = {"kind": None, "anchors": {}}
    reason = phase_2_injections._validate_option_a_placement(plan, "plan")
    assert reason is not None
    assert "null kind" in reason


def test_rejects_plan_missing_benign_target_resource_entirely():
    plan = _base_gitlab_plan()
    plan.pop("benign_target_resource")
    reason = phase_2_injections._validate_option_a_placement(plan, "plan")
    assert reason is not None
    assert "missing" in reason


def test_rejects_plan_with_empty_editor_calls():
    plan = _base_gitlab_plan()
    plan["seed_template"]["editor_calls"] = []
    reason = phase_2_injections._validate_option_a_placement(plan, "plan")
    assert reason is not None
    assert "editor_calls" in reason


# --- site gating --------------------------------------------------------


def test_is_option_a_site_recognises_gitlab_and_reddit():
    assert phase_2_injections._is_option_a_site({"site": "gitlab"}) is True
    assert phase_2_injections._is_option_a_site({"site": "reddit"}) is True
    assert phase_2_injections._is_option_a_site({"sites": ["gitlab"]}) is True
    assert phase_2_injections._is_option_a_site({"sites": ["reddit", "gitlab"]}) is True


def test_is_option_a_site_rejects_legacy_sites():
    assert phase_2_injections._is_option_a_site({"site": "shopping"}) is False
    assert phase_2_injections._is_option_a_site({"sites": ["shopping_admin"]}) is False
    assert phase_2_injections._is_option_a_site({}) is False
