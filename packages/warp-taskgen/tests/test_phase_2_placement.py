"""Option A placement validator tests (X2d).

Directly exercises ``option_a._validate_option_a_placement``
and ``_is_option_a_site`` so the Option A contract is tested in
isolation from the surrounding surface-match / invariant checks (which
have their own coverage in ``test_phase_2_injections.py``).

The adopted validator semantics are skip-on-reject: a non-None return
value is a structured rejection reason; callers log + drop the task.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from warp_taskgen.phase_2 import option_a as phase_2_injections


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
    # Post-flag-flip (commit 8): the registry validator phrases this as
    # "not a valid Option A attach for kind=...". Accept either wording.
    assert ("dangling" in reason) or ("not a valid Option A attach" in reason)


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
    # Legacy phrased this as "project_id must reference ..."; the registry
    # validator rejects earlier as "create_issue is not a valid attach for
    # kind='gitlab_issue'" (create_issue has kinds=frozenset()). Accept
    # either — the key guarantee is that the plan is rejected.
    assert ("project_id" in reason) or ("create_issue" in reason)


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
    # Legacy rejected on the literal mr_iid; registry rejects because
    # create_mr_note isn't a valid attach for kind='gitlab_issue' (only
    # create_issue_note is). Accept either wording.
    assert ("mr_iid" in reason) or ("create_mr_note" in reason)


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
    # Legacy rejected on the literal forum_name; registry rejects because
    # create_submission isn't a valid attach for kind='reddit_submission'
    # (only create_comment is; create_submission addresses reddit_forum).
    assert ("forum_name" in reason) or ("create_submission" in reason)


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


# ------------------------------------------------------------------
# Commit 4: registry-driven validator + dual-run flag
# ------------------------------------------------------------------


class TestRegistryValidator:
    """Focused on the new registry validator's semantics. Most legacy-
    equivalent cases are already covered above via the dual-run wrapper."""

    def test_accepts_valid_plan(self) -> None:
        plan = _base_gitlab_plan()
        assert phase_2_injections._validate_option_a_placement_registry(plan, "plan") is None

    def test_rejects_unknown_method(self) -> None:
        plan = _base_gitlab_plan()
        plan["seed_template"]["editor_calls"][0]["method"] = "create_nonexistent_thing"
        reason = phase_2_injections._validate_option_a_placement_registry(plan, "plan")
        assert reason is not None
        assert "create_nonexistent_thing" in reason
        assert "not a valid Option A attach" in reason

    def test_rejects_kind_not_registered(self) -> None:
        plan = _base_gitlab_plan()
        plan["benign_target_resource"]["kind"] = "synthetic_unknown_kind"
        reason = phase_2_injections._validate_option_a_placement_registry(plan, "plan")
        assert reason is not None
        assert "kind_not_registered" in reason

    def test_rejects_mixed_benchmark_metadata(self) -> None:
        plan = _base_gitlab_plan()
        plan["benchmark"] = "wasp"
        plan["seed_template"]["editor_calls"][0]["benchmark"] = "webarena_verified"
        reason = phase_2_injections._validate_option_a_placement_registry(plan, "plan")
        assert reason is not None
        assert "benchmark metadata is invalid" in reason

    def test_rejects_empty_selector_group(self) -> None:
        """If neither project_id nor project_path_template nor
        project_name_template is populated, the project selector group is
        unsatisfied. Legacy validator missed this (only checked
        issue_iid)."""
        plan = _base_gitlab_plan()
        # Drop project identifier, keep issue_iid + body.
        plan["seed_template"]["editor_calls"][0]["args"] = {
            "issue_iid": "{benign_issue_iid}",
            "body": "x",
        }
        reason = phase_2_injections._validate_option_a_placement_registry(plan, "plan")
        assert reason is not None
        assert "selector group 'project'" in reason

    def test_rejects_token_not_reachable_via_anchors(self) -> None:
        """Plan uses project_id with {benign_project_id} token, but
        anchors only carry project_path — the substitution would render
        empty at apply time. Legacy validator silently accepted this."""
        plan = _base_gitlab_plan()
        plan["benign_target_resource"]["anchors"] = {
            "project_path": "foo/bar",
            "issue_iid": "42",
        }
        # Keep project_id in args — now unreachable via anchors.
        plan["seed_template"]["editor_calls"][0]["args"] = {
            "project_id": "{benign_project_id}",
            "issue_iid": "{benign_issue_iid}",
            "body": "x",
        }
        reason = phase_2_injections._validate_option_a_placement_registry(plan, "plan")
        assert reason is not None
        assert "selector group 'project'" in reason or "not reachable" in reason

    def test_accepts_project_path_template_instead_of_project_id(self) -> None:
        """SelectorGroup OR-logic: project_path_template satisfies the
        project selector when anchors carry project_path."""
        plan = _base_gitlab_plan()
        plan["benign_target_resource"]["anchors"] = {
            "project_path": "foo/bar",
            "issue_iid": "42",
        }
        plan["seed_template"]["editor_calls"][0]["args"] = {
            "project_path_template": "{benign_project_path}",
            "issue_iid": "{benign_issue_iid}",
            "body": "x",
        }
        assert phase_2_injections._validate_option_a_placement_registry(plan, "plan") is None


class TestDualRunFlag:
    def test_source_default_post_flag_flip(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Commit 8 flipped RIGOROUS_OPTION_A_DEFAULT to True after the
        dev-profile dual-run validated the registry validator's verdicts.
        The env-var opt-out is still honored (tested separately)."""
        monkeypatch.delenv("WORLDSIM_RIGOROUS_OPTION_A", raising=False)
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
        assert phase_2_injections.RIGOROUS_OPTION_A_DEFAULT is True
        assert phase_2_injections._rigorous_option_a_enabled() is True

    def test_env_override_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLDSIM_RIGOROUS_OPTION_A", "true")
        assert phase_2_injections._rigorous_option_a_enabled() is True

    def test_env_override_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLDSIM_RIGOROUS_OPTION_A", "FALSE")
        assert phase_2_injections._rigorous_option_a_enabled() is False

    def test_env_override_accepts_varied_truth_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for value in ("1", "yes", "on", "TRUE"):
            monkeypatch.setenv("WORLDSIM_RIGOROUS_OPTION_A", value)
            assert phase_2_injections._rigorous_option_a_enabled() is True

    def test_discrepancy_log_written_when_verdicts_diverge(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("WORLDSIM_RIGOROUS_OPTION_A", raising=False)
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
        plan = _base_gitlab_plan()
        # Drop project_id → legacy accepts (only checks issue_iid), new rejects.
        plan["seed_template"]["editor_calls"][0]["args"] = {
            "issue_iid": "{benign_issue_iid}",
            "body": "x",
        }
        phase_2_injections._validate_option_a_placement(plan, "test-plan")
        log = tmp_path / "phase_2" / "option_a_validator_discrepancy.ndjson"
        assert log.exists(), "discrepancy NDJSON should have been written"
        line = log.read_text().strip()
        record = json.loads(line)
        assert record["task_name"] == "test-plan"
        assert record["legacy_verdict"] is None
        assert record["new_verdict"] is not None

    def test_no_discrepancy_log_when_verdicts_agree(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("WORLDSIM_RIGOROUS_OPTION_A", raising=False)
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
        # Both validators accept a well-formed plan.
        phase_2_injections._validate_option_a_placement(_base_gitlab_plan(), "ok")
        log = tmp_path / "phase_2" / "option_a_validator_discrepancy.ndjson"
        assert not log.exists()

    def test_flag_on_enforces_new_verdict(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("WORLDSIM_RIGOROUS_OPTION_A", "true")
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
        plan = _base_gitlab_plan()
        # Legacy accepts, new rejects (no project selector populated).
        plan["seed_template"]["editor_calls"][0]["args"] = {
            "issue_iid": "{benign_issue_iid}",
            "body": "x",
        }
        reason = phase_2_injections._validate_option_a_placement(plan, "plan")
        assert reason is not None, "new validator should reject"
        assert "selector group" in reason

    def test_flag_off_still_enforces_registry_verdict(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The registry validator is production authority; the env flag only
        controls legacy comparison logging and no longer changes admission."""
        monkeypatch.setenv("WORLDSIM_RIGOROUS_OPTION_A", "false")
        monkeypatch.setenv("WORLDSIM_STATE_DIR", str(tmp_path))
        plan = _base_gitlab_plan()
        # Legacy accepts, new rejects.
        plan["seed_template"]["editor_calls"][0]["args"] = {
            "issue_iid": "{benign_issue_iid}",
            "body": "x",
        }
        reason = phase_2_injections._validate_option_a_placement(plan, "plan")
        assert reason is not None, "registry validator should remain authoritative"
        assert "selector group" in reason


# --- malformed-token rejection (Bug B) ---------------------------------


class TestMalformedTokenRejection:
    """Values like ``"{benign_submission_id"`` (missing close brace) must
    be rejected by both validators — seeding.py's substitution pattern
    requires ``{name}``, not ``{name``, so the literal token would leak
    into the rendered payload if the validator accepted it."""

    def test_legacy_rejects_unclosed_benign_submission_id(self) -> None:
        plan = _base_reddit_plan()
        plan["seed_template"]["editor_calls"][0]["args"]["submission_id"] = "{benign_submission_id"
        reason = phase_2_injections._validate_option_a_placement(plan, "plan")
        assert reason is not None
        assert "submission_id" in reason

    def test_legacy_rejects_unclosed_benign_issue_iid(self) -> None:
        plan = _base_gitlab_plan()
        plan["seed_template"]["editor_calls"][0]["args"]["issue_iid"] = "{benign_issue_iid"
        reason = phase_2_injections._validate_option_a_placement(plan, "plan")
        assert reason is not None

    def test_legacy_accepts_well_formed_token_with_trailing_suffix(self) -> None:
        plan = _base_reddit_plan()
        plan["seed_template"]["editor_calls"][0]["args"]["submission_id"] = (
            "{benign_submission_id}/extra"
        )
        # Trailing non-token content beyond the closing brace is still
        # accepted (legacy prefix semantics preserved for well-formed
        # tokens) — seeding substitution handles the trailing content.
        reason = phase_2_injections._validate_option_a_placement(plan, "plan")
        assert reason is None

    def test_value_starts_with_token_helper_requires_close_brace(self) -> None:
        assert phase_2_injections._value_starts_with_token(
            "{benign_submission_id}", "{benign_submission_id}"
        )
        assert not phase_2_injections._value_starts_with_token(
            "{benign_submission_id", "{benign_submission_id}"
        )
        # Legacy call sites passed the brace-less form of the expected
        # token — the helper normalises it to the closed form and
        # enforces a well-formed match against the value.
        assert phase_2_injections._value_starts_with_token(
            "{benign_submission_id}", "{benign_submission_id"
        )
        assert not phase_2_injections._value_starts_with_token(
            "{benign_submission_id", "{benign_submission_id"
        )

    def test_value_starts_with_token_rejects_empty(self) -> None:
        assert not phase_2_injections._value_starts_with_token("", "{benign_submission_id}")

    def test_value_starts_with_token_rejects_non_benign_prefix(self) -> None:
        assert not phase_2_injections._value_starts_with_token(
            "{some_other_token}", "{benign_submission_id}"
        )
