"""Serialized live proof for the generated GitLab comparison workflow.

Run this test alone through the existing integration wrapper::

    PYTEST_GITLAB_COMPARE_VERTICAL_SLICE=1 \
      bash scripts/run_integration_tests.sh \
      --host-config configs/benchmark_hosts/r8a.local.yaml --quiet -- \
      tests/integration/test_gitlab_compare_vertical_slice_live.py

The explicit opt-in matters: this test performs Golden-State Reset before and
after its attempt, so it must never share a reset-sensitive benchmark instance
with another task.

This is a source/evaluator smoke, not a browser-action or HAR-capture claim.
It uses a fresh browser context for exposure, then projects the actual editor
note receipt into the network-event shape consumed by the production reward
dispatcher and verifies the persisted state through a separate API readback.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import pytest
import requests

from warp_taskgen.agent_auth import resolve_agent_auth
from warp_taskgen.agent_config import bind_task_to_instance, task_reset_endpoints
from warp_taskgen.auth_tokens import acquire_tokens_for_instances, clear_run_token_cache
from warp_taskgen.editors.gitlab import GitlabEditor
from warp_taskgen.phase_1.gitlab_compare_act import (
    compile_gitlab_compare_act_task,
    selected_target_from_binding,
)
from warp_taskgen.phase_1.gitlab_compare_act_reward import (
    materialize_gitlab_compare_act_reward,
)
from warp_taskgen.phase_1.gitlab_compare_decide import (
    generate_gitlab_compare_decide_world,
)
from warp_taskgen.phase_1.gitlab_compare_decide_binding import (
    bind_gitlab_compare_decide_attempt,
)
from warp_taskgen.phase_1.gitlab_compare_decide_reward import (
    grade_gitlab_compare_decide,
)
from warp_taskgen.phase_4.execution_helpers import _reset_task_environment
from warp_taskgen.phases.phase_2_render_check import verify_seed_renders
from warp_taskgen.rewards import run_reward_function
from warp_taskgen.seeding import SeedCleanupHandle, apply_data_seed

pytestmark = pytest.mark.integration

_OPT_IN_ENV = "PYTEST_GITLAB_COMPARE_VERTICAL_SLICE"


def _require_explicit_opt_in() -> None:
    if os.getenv(_OPT_IN_ENV) != "1":
        pytest.skip(
            f"set {_OPT_IN_ENV}=1 and run this file alone; "
            "the GitLab comparison proof resets its selected instance"
        )


def _gitlab_instance_model(live_config):
    matches = [instance for instance in live_config.instances if instance.site_name == "gitlab"]
    if not matches:
        pytest.skip("live instances file does not define a GitLab instance")
    if len(matches) != 1:
        pytest.fail(
            "the serialized GitLab comparison proof requires exactly one selected "
            f"GitLab instance, found {len(matches)}"
        )
    return matches[0]


def _instance_payload(live_config, instance_model) -> dict[str, Any]:
    instance = instance_model.model_dump()
    instance["benchmark"] = live_config.benchmark_name or "webarena_verified"
    return instance


def _selected_call_row(metadata: dict[str, Any], logical_key: str) -> dict[str, Any]:
    rows = metadata.get("editor_call_results")
    assert isinstance(rows, list), "seed metadata did not include per-call results"
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("logical_record_key") == logical_key
    ]
    assert len(matches) == 1, (
        f"expected one per-call result for {logical_key!r}, found {len(matches)}"
    )
    return matches[0]


async def _assert_selected_issue_visible(
    instance: dict[str, Any],
    *,
    selected_row: dict[str, Any],
    expected_project_path: str,
    expected_issue_iid: str,
    signature: str,
) -> None:
    """Observe the selected issue through a fresh agent-authenticated context."""

    from playwright.async_api import async_playwright

    raw_auth = instance.get("agent_auth")
    benchmark_root = instance.get("benchmark_root")
    resolved_auth = resolve_agent_auth(
        raw_auth if isinstance(raw_auth, dict) else None,
        site_name="gitlab",
        site_url=str(instance["site_url"]),
        benchmark_root=Path(str(benchmark_root)) if benchmark_root else None,
        storage_state_override=instance.get("storage_state_path"),
    )
    assert resolved_auth.usable, (
        f"configured GitLab reader authentication is unusable: {resolved_auth.unusable_reason}"
    )
    urls = selected_row.get("read_surface_urls")
    assert isinstance(urls, list) and urls, (
        "selected GitLab seed call emitted no independent read surface"
    )
    expected_path = f"/{expected_project_path.strip('/')}/-/issues/{expected_issue_iid}"
    assert any(urlsplit(str(url)).path.rstrip("/") == expected_path for url in urls), (
        "selected call read surfaces did not address its exact bound issue: "
        f"expected_path={expected_path!r}, urls={urls!r}"
    )

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            outcome = await verify_seed_renders(
                browser=browser,
                urls=[str(url) for url in urls],
                site_name="gitlab",
                site_url=str(instance["site_url"]),
                signature=signature,
                browser_context_kwargs=resolved_auth.browser_context_kwargs,
                write_tokens=dict(selected_row.get("write_tokens") or {}),
                diagnostics={
                    "reader_session": "fresh_agent_authenticated_context",
                    "logical_record_key": selected_row.get("logical_record_key"),
                },
            )
        finally:
            await browser.close()

    assert outcome.ok, f"fresh reader did not observe selected issue: {outcome.evidence()}"
    assert outcome.matched_signature == signature
    assert outcome.matched_url is not None
    assert urlsplit(outcome.matched_url).path.rstrip("/") == expected_path


def _note_network_event(
    instance: dict[str, Any],
    *,
    project_id: str,
    issue_iid: str,
    note_id: Any,
    body: str,
) -> dict[str, Any]:
    return {
        "url": (
            f"{str(instance['site_url']).rstrip('/')}/api/v4/projects/"
            f"{quote(project_id, safe='')}/issues/{quote(issue_iid, safe='')}/notes"
        ),
        "method": "POST",
        "response_status": 201,
        "response_note_id": str(note_id),
        "post_data": json.dumps({"body": body}),
    }


def _assert_resource_absent_after_reset(
    instance: dict[str, Any],
    *,
    project_path: str,
    issue_iid: str | None,
) -> None:
    clear_run_token_cache()
    token_errors = acquire_tokens_for_instances([instance])
    assert token_errors == [], f"post-reset GitLab auth refresh failed: {token_errors}"
    with requests.Session() as session:
        editor = GitlabEditor(instance, session)
        project = editor._gitlab_get_json(
            f"/api/v4/projects/{editor._quote(project_path)}",
            allow_missing=True,
        )
        issue = None
        if (
            issue_iid is not None
            and isinstance(project, dict)
            and project.get("id") not in (None, "")
        ):
            issue = editor._gitlab_get_json(
                f"/api/v4/projects/{editor._quote(project['id'])}/issues/"
                f"{editor._quote(issue_iid)}",
                allow_missing=True,
            )
    assert project is None and issue is None, (
        "Golden-State Reset left the exact comparison resource behind: "
        f"project_path={project_path!r}, issue_iid={issue_iid!r}, "
        f"project_present={project is not None}, issue_present={issue is not None}"
    )


def test_gitlab_compare_decide_and_act_vertical_slice(live_config) -> None:
    """Prove generated world -> bind -> read -> decide -> act -> readback -> reset."""

    _require_explicit_opt_in()
    instance_model = _gitlab_instance_model(live_config)
    instance = _instance_payload(live_config, instance_model)
    task = compile_gitlab_compare_act_task(generate_gitlab_compare_decide_world())
    task["id"] = f"gitlab_compare_vertical_{uuid.uuid4().hex[:12]}"
    bound_task = bind_task_to_instance(task, instance_model, list(live_config.instances))
    reset_endpoints = task_reset_endpoints(bound_task)
    assert len(reset_endpoints) == 1, (
        "the serialized GitLab comparison proof requires exactly one configured "
        f"reset endpoint, found {reset_endpoints!r}"
    )

    handle: SeedCleanupHandle | None = None
    action_editor: GitlabEditor | None = None
    action_session: requests.Session | None = None
    project_path: str | None = None
    issue_iid: str | None = None

    # Reset before auth acquisition: restoring the snapshot can invalidate a
    # previously minted API token. The same ordering is used again below.
    asyncio.run(_reset_task_environment(bound_task))
    clear_run_token_cache()
    token_errors = acquire_tokens_for_instances([instance])
    assert token_errors == [], f"pre-seed GitLab auth acquisition failed: {token_errors}"
    instance["seed_task"] = task

    first_call = task["adversarial_data_seed"]["editor_calls"][0]
    preview_args = dict(first_call["args"])
    preview_args["project_name_template"] = str(preview_args["project_name_template"]).format(
        task_id=task["id"]
    )
    with requests.Session() as preview_session:
        preview_editor = GitlabEditor(instance, preview_session)
        preview = preview_editor.preview_context("create_issue", preview_args)
    project_path = str(preview.get("project_path") or "").strip() or None
    assert project_path is not None, "GitLab seed preview did not resolve its exact project path"

    try:
        handle, metadata = apply_data_seed(task["adversarial_data_seed"], instance)
        assert handle is not None, "three-record GitLab seed returned no cleanup handle"

        original_rows = metadata.get("editor_call_results")
        assert isinstance(original_rows, list) and len(original_rows) == 3
        reordered_metadata = copy.deepcopy(metadata)
        reordered_metadata["editor_call_results"] = list(reversed(original_rows))
        # Deliberately poison the aggregate compatibility field. The strict
        # binder must still select from current-attempt per-call evidence.
        reordered_metadata["write_tokens"] = dict(original_rows[-1].get("write_tokens") or {})

        binding = bind_gitlab_compare_decide_attempt(
            task,
            reordered_metadata,
            phase="phase4",
        )
        target = selected_target_from_binding(task, binding)
        assert binding.diagnostics["aggregate_fallback_used"] is False
        assert target.logical_record_key == "release-blocker"
        aggregate_issue_iid = str(reordered_metadata["write_tokens"].get("issue_iid") or "")
        assert aggregate_issue_iid and target.issue_iid != aggregate_issue_iid

        assert target.project_path == project_path
        issue_iid = target.issue_iid
        selected_row = _selected_call_row(metadata, target.logical_record_key)
        signature = f"Logical record: {target.logical_record_key}"
        asyncio.run(
            _assert_selected_issue_visible(
                instance,
                selected_row=selected_row,
                expected_project_path=project_path,
                expected_issue_iid=issue_iid,
                signature=signature,
            )
        )

        selected_record = next(
            record
            for record in task["world"]["records"]
            if record["logical_record_key"] == target.logical_record_key
        )
        selected_facts = selected_record["facts"]
        decision = {
            "task_type": "retrieve",
            "status": "SUCCESS",
            "retrieved_data": [
                f"selected_iid={target.issue_iid}",
                (
                    f"reason=state={selected_facts['state']};"
                    f"dependency={selected_facts['dependency']};"
                    f"summary={selected_facts['summary']}"
                ),
            ],
        }
        decision_passed, decision_message = grade_gitlab_compare_decide(
            task,
            decision,
            binding=binding,
        )
        assert decision_passed, f"finite compare decision did not grade: {decision_message}"

        reward = materialize_gitlab_compare_act_reward(task, binding)
        expected = reward["eval"][0]["expected"]
        note_body = str(expected["contains"])
        action_session = requests.Session()
        action_editor = GitlabEditor(instance, action_session)
        note = action_editor.create_issue_note(
            project_id=target.project_id,
            issue_iid=target.issue_iid,
            note_body=note_body,
        )
        note_id = note.get("note_id")
        assert note_id not in (None, ""), "GitLab note create returned no stable identity"
        action_event = _note_network_event(
            instance,
            project_id=target.project_id,
            issue_iid=target.issue_iid,
            note_id=note_id,
            body=note_body,
        )
        action_passed, action_message = run_reward_function(
            reward,
            instance,
            network_trace=[action_event],
        )
        assert action_passed, f"production GitLab action readback failed: {action_message}"
        assert "exact parent, actor, identity, and scope" in action_message
    finally:
        reset_verified = False
        reset_errors: list[str] = []
        cleanup_errors: list[str] = []

        # Reset while the exact project still exists. This makes the absence
        # check a proof of Golden-State Reset rather than a consequence of the
        # editors' ordinary cleanup callbacks.
        try:
            asyncio.run(_reset_task_environment(bound_task))
            if project_path is not None:
                _assert_resource_absent_after_reset(
                    instance,
                    project_path=project_path,
                    issue_iid=issue_iid,
                )
                reset_verified = True
        except Exception as exc:  # pragma: no cover - live failure diagnostic
            reset_errors.append(f"final reset or absence verification failed: {exc}")

        # Always run the existing cleanup contracts as a second, idempotent
        # safety net. Post-reset auth refresh above mutates the shared instance
        # mapping, so these callbacks use a current token when reset rotates it.
        if action_editor is not None:
            try:
                action_editor.cleanup()
            except Exception as exc:  # pragma: no cover - live failure diagnostic
                cleanup_errors.append(f"action cleanup failed: {exc}")
        if action_session is not None:
            action_session.close()
        if handle is not None:
            try:
                handle.cleanup()
            except Exception as exc:  # pragma: no cover - live failure diagnostic
                # The ordinary GitLab cleanup stack closes issues before it
                # deletes their project, and that close operation is not
                # idempotent after /init has already removed the project.
                # Accept only that expected 404 after exact reset absence was
                # independently verified; every other cleanup failure remains
                # fatal.
                if not (reset_verified and "HTTP 404" in str(exc)):
                    cleanup_errors.append(f"seed cleanup failed: {exc}")
        finalization_errors = reset_errors + cleanup_errors
        assert not finalization_errors, "; ".join(finalization_errors)
