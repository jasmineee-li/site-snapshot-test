"""Unit tests for Phase 2c feasibility verification.

Uses the ``_FakeSession``/``_FakeResponse`` monkeypatch pattern from
``tests/test_seeding.py`` — the repo does not depend on ``respx``,
``aresponses``, or ``tenacity`` and this file does not introduce them.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest

from worldsim.editors import EditorError
from worldsim.phases import phase_2_feasibility as feas

# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


class _FakeHandle:
    def __init__(self, *, raises: bool = False) -> None:
        self.cleaned = False
        self._raises = raises

    def cleanup(self) -> None:
        self.cleaned = True
        if self._raises:
            raise EditorError("cleanup_failed", "synthetic cleanup failure")


def _gitlab_instance(**extras: Any) -> dict[str, Any]:
    payload = {
        "site_name": "gitlab",
        "site_url": "https://gitlab.example/",
        "benchmark": "webarena_verified",
    }
    payload.update(extras)
    return payload


def _shopping_instance(**extras: Any) -> dict[str, Any]:
    payload = {
        "site_name": "shopping",
        "site_url": "https://shop.example/",
        "benchmark": "webarena_verified",
    }
    payload.update(extras)
    return payload


def _task(
    task_id: str = "AT-001",
    site: str = "gitlab",
    *,
    method: str = "create_group",
    detail: str = "hello",
    feasibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task: dict[str, Any] = {
        "id": task_id,
        "site": site,
        "benign_task_id": f"benign-{task_id}",
        "adversarial_data_seed": {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": site,
                    "method": method,
                    "args": {"name": f"{task_id}-resource", "description": detail},
                }
            ],
        },
    }
    if feasibility is not None:
        task["feasibility"] = feasibility
    return task


def _at009_oversize_task() -> dict[str, Any]:
    # Mirrors the AT-009 shape: a GitLab create_group with a 624-char
    # description. The exact content doesn't matter for classification;
    # the mock editor is what asserts the length_exceeded outcome.
    description = "X" * 624
    return _task(task_id="AT-009", method="create_group", detail=description)


@pytest.fixture(autouse=True)
def _stable_git_fingerprint(monkeypatch):
    """Deterministic editor/dataset commit for fingerprint matching tests."""
    monkeypatch.setenv("WORLDSIM_EDITOR_COMMIT_OVERRIDE", "cafebabe1234")
    yield


@pytest.fixture(autouse=True)
def _bypass_preflight(monkeypatch):
    """Stub out probe_base_state + token acquisition by default.

    Individual tests that care about the pre-flight behavior re-patch these
    locally via ``monkeypatch.setattr``.
    """
    monkeypatch.setattr(feas, "acquire_tokens_for_instances", lambda instances: [])

    class _StubEditorCls:
        @classmethod
        def probe_base_state(cls, instance: dict[str, Any]) -> None:
            return None

    monkeypatch.setattr(
        feas,
        "EDITOR_REGISTRY",
        {
            ("webarena_verified", "gitlab"): _StubEditorCls,
            ("webarena_verified", "shopping"): _StubEditorCls,
            ("webarena_verified", "reddit"): _StubEditorCls,
        },
    )
    yield


def _write_tasks(tmp_path: Path, tasks: list[dict[str, Any]]) -> Path:
    target = tmp_path / "adversarial_tasks.json"
    target.write_text(json.dumps(tasks))
    return target


def _patch_apply(
    monkeypatch,
    responder,
) -> list[int]:
    """Patch ``apply_data_seed_async`` to call ``responder(attempt_index)``.

    ``responder`` may return a fake handle, raise ``EditorError``, raise
    ``ValueError``, or return ``None`` (the "empty_seed" path).
    """
    counter = {"n": 0}
    attempt_log: list[int] = []

    async def fake(seed, instance):
        idx = counter["n"]
        counter["n"] += 1
        attempt_log.append(idx)
        return responder(idx, seed, instance)

    monkeypatch.setattr(feas, "apply_data_seed_async", fake)
    return attempt_log


# ---------------------------------------------------------------------------
# Case 1 — happy path
# ---------------------------------------------------------------------------


def test_case_01_2xx_create_verifies_and_cleans(tmp_path, monkeypatch):
    handle = _FakeHandle()

    def responder(idx, seed, instance):
        return handle

    _patch_apply(monkeypatch, responder)
    tasks_path = _write_tasks(tmp_path, [_task()])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            instances=[_gitlab_instance()],
            instances_label="instances.smoke.json",
            concurrency=1,
            retry_count=0,
        )
    )
    assert len(report.verified) == 1
    assert report.verified[0]["feasibility"]["status"] == "verified"
    assert handle.cleaned is True
    assert not report.infeasible
    assert not report.cleanup_warnings


# ---------------------------------------------------------------------------
# Case 2 — 400 too long → length_exceeded
# ---------------------------------------------------------------------------


def test_case_02_length_exceeded_classification(tmp_path, monkeypatch):
    def responder(idx, seed, instance):
        raise EditorError(
            "length_exceeded",
            "gitlab group description is too long (maximum is 255 characters)",
            http_status=400,
            response_snippet='{"message":"is too long"}',
        )

    _patch_apply(monkeypatch, responder)
    tasks_path = _write_tasks(tmp_path, [_task()])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=0,
        )
    )
    assert len(report.infeasible) == 1
    entry = report.infeasible[0]["feasibility"]
    assert entry["status"] == "infeasible"
    assert entry["errors"][0]["kind"] == "length_exceeded"
    assert entry["errors"][0]["http_status"] == 400


# ---------------------------------------------------------------------------
# Case 3 — 401 / auth_missing
# ---------------------------------------------------------------------------


def test_case_03_auth_missing_does_not_retry(tmp_path, monkeypatch):
    calls = {"n": 0}

    def responder(idx, seed, instance):
        calls["n"] += 1
        raise EditorError("auth_missing", "401 on POST")

    _patch_apply(monkeypatch, responder)
    tasks_path = _write_tasks(tmp_path, [_task()])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=3,
        )
    )
    assert len(report.infeasible) == 1
    assert report.infeasible[0]["feasibility"]["errors"][0]["kind"] == "auth_missing"
    assert calls["n"] == 1  # auth_missing must not retry


# ---------------------------------------------------------------------------
# Case 4 — 500 then 2xx → retry success
# ---------------------------------------------------------------------------


def test_case_04_retry_after_request_failed_succeeds(tmp_path, monkeypatch):
    handle = _FakeHandle()

    def responder(idx, seed, instance):
        if idx == 0:
            raise EditorError("request_failed", "upstream 500", http_status=500)
        return handle

    _patch_apply(monkeypatch, responder)
    tasks_path = _write_tasks(tmp_path, [_task()])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=1,
        )
    )
    assert len(report.verified) == 1
    attempts = report.verified[0]["feasibility"]["attempts"]
    assert [a["attempt"] for a in attempts] == [0, 1]
    assert attempts[-1]["status"] == "success"


# ---------------------------------------------------------------------------
# Case 5 — 500 twice → exhausts retries, reports request_failed
# ---------------------------------------------------------------------------


def test_case_05_retry_exhausted_yields_request_failed(tmp_path, monkeypatch):
    def responder(idx, seed, instance):
        raise EditorError("request_failed", "upstream 503", http_status=503)

    _patch_apply(monkeypatch, responder)
    tasks_path = _write_tasks(tmp_path, [_task()])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=1,
        )
    )
    assert len(report.infeasible) == 1
    errors = report.infeasible[0]["feasibility"]["errors"]
    assert errors[0]["kind"] == "request_failed"


# ---------------------------------------------------------------------------
# Case 6 — cleanup raises but verification still recorded
# ---------------------------------------------------------------------------


def test_case_06_cleanup_error_yields_warning(tmp_path, monkeypatch):
    handle = _FakeHandle(raises=True)

    def responder(idx, seed, instance):
        return handle

    _patch_apply(monkeypatch, responder)
    tasks_path = _write_tasks(tmp_path, [_task()])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=0,
        )
    )
    assert len(report.verified) == 1
    assert report.cleanup_warnings
    assert "cleanup_failed" in report.cleanup_warnings[0]


# ---------------------------------------------------------------------------
# Case 7 — fingerprint match ⇒ skip
# ---------------------------------------------------------------------------


def test_case_07_fingerprint_match_skips_http(tmp_path, monkeypatch):
    seed = _task()["adversarial_data_seed"]["editor_calls"]
    content_hash = feas._task_content_hash(seed)
    prior_feas = {
        "status": "verified",
        "verified_at": "2026-04-18T00:00:00Z",
        "host_fingerprint": {
            "host_config": "instances.smoke.json",
            "editor_commit": "cafebabe1234",
            "dataset_commit": "cafebabe1234",
            "task_content_hash": content_hash,
        },
        "attempts": [{"attempt": 0, "status": "success", "elapsed_ms": 100}],
    }
    task = _task(feasibility=prior_feas)

    calls = {"n": 0}

    def responder(idx, seed_payload, instance):
        calls["n"] += 1
        raise AssertionError("should have skipped due to fingerprint match")

    _patch_apply(monkeypatch, responder)
    tasks_path = _write_tasks(tmp_path, [task])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=0,
        )
    )
    assert calls["n"] == 0
    # Idempotency-skip preserves the prior ``status="verified"`` record so
    # Phase 4's strict admission gate still admits the task. The skip fact
    # is recorded on a sibling field (``last_reverify_skipped_at``) and the
    # task is additionally surfaced via ``report.skipped_already_verified``
    # for reporting.
    assert len(report.verified) == 1
    stanza = report.verified[0]["feasibility"]
    assert stanza["status"] == "verified"
    assert stanza["verified_at"] == "2026-04-18T00:00:00Z"
    assert stanza["last_reverify_skipped_at"]
    assert stanza["last_reverify_skip_reason"] == "fingerprint_match"
    assert len(report.skipped_already_verified) == 1
    assert report.skipped_already_verified[0] is report.verified[0]


# ---------------------------------------------------------------------------
# Case 8 — fingerprint drift (different editor_commit) ⇒ re-verify
# ---------------------------------------------------------------------------


def test_case_08_fingerprint_drift_reverifies(tmp_path, monkeypatch):
    prior = {
        "status": "verified",
        "verified_at": "2026-04-18T00:00:00Z",
        "host_fingerprint": {
            "host_config": "instances.smoke.json",
            "editor_commit": "olddeadbeef",
            "dataset_commit": "olddeadbeef",
            "task_content_hash": "deadbeef0000",
        },
    }
    task = _task(feasibility=prior)

    handle = _FakeHandle()

    def responder(idx, seed, instance):
        return handle

    _patch_apply(monkeypatch, responder)
    tasks_path = _write_tasks(tmp_path, [task])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=0,
        )
    )
    assert len(report.verified) == 1
    assert report.verified[0]["feasibility"]["status"] == "verified"
    assert handle.cleaned is True


# ---------------------------------------------------------------------------
# Case 9 — task_content drift forces re-verify even with matching git commit
# ---------------------------------------------------------------------------


def test_case_09_task_content_hash_drift_reverifies(tmp_path, monkeypatch):
    prior = {
        "status": "verified",
        "verified_at": "2026-04-18T00:00:00Z",
        "host_fingerprint": {
            "host_config": "instances.smoke.json",
            "editor_commit": "cafebabe1234",
            "dataset_commit": "cafebabe1234",
            "task_content_hash": "aaaaaaaaaaaa",  # mismatches the live seed
        },
    }
    task = _task(feasibility=prior)

    handle = _FakeHandle()

    def responder(idx, seed, instance):
        return handle

    _patch_apply(monkeypatch, responder)
    tasks_path = _write_tasks(tmp_path, [task])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=0,
        )
    )
    assert len(report.verified) == 1
    assert report.verified[0]["feasibility"]["status"] == "verified"


# ---------------------------------------------------------------------------
# Case 10 — multi-call chain, second call fails → cleanup tears down
# ---------------------------------------------------------------------------


def test_case_10_multi_call_chain_cleanup_on_second_failure(tmp_path, monkeypatch):
    # apply_data_seed internally iterates calls; its exception handler
    # cleans up partial state. We simulate the end-state: the outer wrapper
    # raises EditorError after having torn down.
    def responder(idx, seed, instance):
        raise EditorError(
            "length_exceeded",
            "second call exceeded column length",
            http_status=400,
        )

    _patch_apply(monkeypatch, responder)
    task = _task()
    task["adversarial_data_seed"]["editor_calls"].append(
        {
            "benchmark": "webarena_verified",
            "site": "gitlab",
            "method": "create_issue",
            "args": {"project": "AT-001-resource", "description": "Y" * 10000},
        }
    )
    tasks_path = _write_tasks(tmp_path, [task])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=0,
        )
    )
    assert len(report.infeasible) == 1
    assert report.infeasible[0]["feasibility"]["errors"][0]["kind"] == "length_exceeded"


# ---------------------------------------------------------------------------
# Case 11 — ValueError from validator ⇒ schema_mismatch
# ---------------------------------------------------------------------------


def test_case_11_value_error_remaps_to_schema_mismatch(tmp_path, monkeypatch):
    def responder(idx, seed, instance):
        raise ValueError("editor_calls[0].args is missing required 'name'")

    _patch_apply(monkeypatch, responder)
    tasks_path = _write_tasks(tmp_path, [_task()])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=0,
        )
    )
    assert len(report.infeasible) == 1
    assert report.infeasible[0]["feasibility"]["errors"][0]["kind"] == "schema_mismatch"


# ---------------------------------------------------------------------------
# Case 12 — AT-009 regression: 624-char GitLab description is load-bearing
# ---------------------------------------------------------------------------


def test_case_12_at009_regression_classifies_length_exceeded(tmp_path, monkeypatch):
    def responder(idx, seed, instance):
        raise EditorError(
            "length_exceeded",
            "gitlab group description is too long (maximum is 255 characters)",
            http_status=400,
            response_snippet='{"message":"Failed to save group {:description=>[\\"is too long (maximum is 255 characters)\\"]}"}',
        )

    _patch_apply(monkeypatch, responder)
    tasks_path = _write_tasks(tmp_path, [_at009_oversize_task()])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=0,
        )
    )
    assert len(report.infeasible) == 1
    stanza = report.infeasible[0]["feasibility"]
    assert stanza["errors"][0]["kind"] == "length_exceeded"
    assert stanza["errors"][0]["http_status"] == 400
    # Response snippet must be preserved verbatim so reviewers can audit the
    # raw platform rejection when triaging an infeasible task.
    assert "is too long (maximum is 255 characters)" in stanza["errors"][0]["response_snippet"]


# ---------------------------------------------------------------------------
# Case 13 — --force-reverify bypasses fingerprint short-circuit
# ---------------------------------------------------------------------------


def test_case_13_force_reverify_bypasses_skip(tmp_path, monkeypatch):
    seed = _task()["adversarial_data_seed"]["editor_calls"]
    content_hash = feas._task_content_hash(seed)
    prior = {
        "status": "verified",
        "verified_at": "2026-04-18T00:00:00Z",
        "host_fingerprint": {
            "host_config": "instances.smoke.json",
            "editor_commit": "cafebabe1234",
            "dataset_commit": "cafebabe1234",
            "task_content_hash": content_hash,
        },
    }
    task = _task(feasibility=prior)
    handle = _FakeHandle()

    def responder(idx, seed_payload, instance):
        return handle

    _patch_apply(monkeypatch, responder)
    tasks_path = _write_tasks(tmp_path, [task])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=0,
            force_reverify=True,
        )
    )
    assert handle.cleaned is True
    assert len(report.verified) == 1
    assert report.verified[0]["feasibility"]["status"] == "verified"


# ---------------------------------------------------------------------------
# Case 14 — TTL short-circuit when fingerprint drifts but verified_at recent
# ---------------------------------------------------------------------------


def test_case_14_ttl_hours_preserves_recent_verification(tmp_path, monkeypatch):
    from datetime import datetime

    verified_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    prior = {
        "status": "verified",
        "verified_at": verified_at,
        "host_fingerprint": {
            "host_config": "instances.smoke.json",
            "editor_commit": "olddeadbeef",  # drifts
            "dataset_commit": "olddeadbeef",
            "task_content_hash": "deadbeef0000",
        },
    }
    task = _task(feasibility=prior)

    def responder(idx, seed, instance):
        raise AssertionError("TTL short-circuit should have skipped this task")

    _patch_apply(monkeypatch, responder)
    tasks_path = _write_tasks(tmp_path, [task])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=0,
            ttl_hours=24.0,
        )
    )
    assert len(report.verified) == 1
    stanza = report.verified[0]["feasibility"]
    # TTL-skip preserves the prior ``status="verified"`` and records the skip
    # reason so the per-run summary can distinguish it from a fresh verify.
    assert stanza["status"] == "verified"
    assert stanza["last_reverify_skip_reason"] == "ttl_hours"


# ---------------------------------------------------------------------------
# Case 15 — token acquisition failure raises before launching workers
# ---------------------------------------------------------------------------


def test_case_15_token_cache_miss_raises_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(
        feas,
        "acquire_tokens_for_instances",
        lambda instances: ["gitlab: could not acquire bearer"],
    )
    tasks_path = _write_tasks(tmp_path, [_task()])
    with pytest.raises(RuntimeError, match="token acquisition failed"):
        asyncio.run(
            feas.verify_feasibility(
                tasks_path,
                instances=[_gitlab_instance()],
                concurrency=1,
                retry_count=0,
            )
        )


# ---------------------------------------------------------------------------
# Case 16 — task references a site with no instance → unsupported_site
# ---------------------------------------------------------------------------


def test_case_16_missing_instance_raises_preflight(tmp_path, monkeypatch):
    tasks_path = _write_tasks(tmp_path, [_task(site="reddit")])
    with pytest.raises(RuntimeError, match="no matching instance"):
        asyncio.run(
            feas.verify_feasibility(
                tasks_path,
                instances=[_gitlab_instance()],
                concurrency=1,
                retry_count=0,
            )
        )


# ---------------------------------------------------------------------------
# Extra — empty editor_calls classified as empty_seed
# ---------------------------------------------------------------------------


def test_empty_editor_calls_marks_task_empty_seed(tmp_path, monkeypatch):
    task = _task()
    task["adversarial_data_seed"]["editor_calls"] = []

    def responder(idx, seed, instance):
        return None  # apply_data_seed_async returns None on empty calls

    _patch_apply(monkeypatch, responder)
    tasks_path = _write_tasks(tmp_path, [task])

    # Empty seed should be flagged before the dispatcher is even called —
    # but if it slips past, apply_data_seed_async returning None also trips
    # the explicit handle-is-None check. Both paths mark the task infeasible.
    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=0,
        )
    )
    assert len(report.infeasible) == 1
    kind = report.infeasible[0]["feasibility"]["errors"][0]["kind"]
    assert kind in {"empty_seed", "schema_mismatch"}


# ---------------------------------------------------------------------------
# Idempotency decision unit tests
# ---------------------------------------------------------------------------


def test_idempotency_decision_truth_table():
    fp = {
        "host_config": "a",
        "editor_commit": "b",
        "dataset_commit": "c",
        "task_content_hash": "d",
    }
    drift = {**fp, "task_content_hash": "other"}

    def _decide(existing, *, ttl=None, force=False):
        return feas._idempotency_decision(
            existing, current_fingerprint=fp, ttl_hours=ttl, force_reverify=force
        )

    # missing → verify
    assert _decide(None) == ("verify", None)
    # verified + match → skip (reason=fingerprint_match)
    assert _decide({"status": "verified", "host_fingerprint": fp}) == (
        "skip",
        "fingerprint_match",
    )
    # verified + drift → re-verify
    assert _decide({"status": "verified", "host_fingerprint": drift}) == ("verify", None)
    # verified + drift + TTL covers it → skip (reason=ttl_hours)
    from datetime import datetime

    recent = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert _decide(
        {"status": "verified", "host_fingerprint": drift, "verified_at": recent},
        ttl=24.0,
    ) == ("skip", "ttl_hours")
    # infeasible → always re-verify
    assert _decide({"status": "infeasible", "host_fingerprint": fp}) == ("verify", None)
    # unverified (skip flag) → verify
    assert _decide({"status": "unverified"}) == ("verify", None)
    # force overrides skip
    assert _decide({"status": "verified", "host_fingerprint": fp}, force=True) == (
        "verify",
        None,
    )


# ---------------------------------------------------------------------------
# Case 7b — double-run invariant: running the verifier twice in a row must
# converge to a byte-identical dataset (modulo timestamps). This is the
# regression test for the idempotency-skip status-mutation bug.
# ---------------------------------------------------------------------------


def test_case_07b_double_run_converges_without_status_drift(tmp_path, monkeypatch):
    handle = _FakeHandle()

    def responder(idx, seed, instance):
        return handle

    _patch_apply(monkeypatch, responder)
    tasks_path = _write_tasks(tmp_path, [_task()])

    # First run: fresh verify.
    first = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=0,
        )
    )
    assert len(first.verified) == 1
    first_feas = first.verified[0]["feasibility"]
    assert first_feas["status"] == "verified"
    assert "last_reverify_skipped_at" not in first_feas

    # Persist exactly what the Phase 2c caller would persist — the
    # ``report.verified`` list — and re-run. The second run must hit the
    # idempotency shortcut, leaving ``status="verified"`` intact so Phase 4
    # strict admission still admits the task.
    tasks_path.write_text(json.dumps(first.verified))

    def blow_up_if_called(idx, seed, instance):
        raise AssertionError("second run should short-circuit via idempotency")

    _patch_apply(monkeypatch, blow_up_if_called)
    second = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=0,
        )
    )
    assert len(second.verified) == 1
    second_feas = second.verified[0]["feasibility"]
    # The critical invariant: Phase 4's admission gate reads this field.
    assert second_feas["status"] == "verified"
    # Reused-via-idempotency evidence must surface on the report for audit
    # trail without trampling the original verification.
    assert second_feas["verified_at"] == first_feas["verified_at"]
    assert second_feas["last_reverify_skip_reason"] == "fingerprint_match"
    assert len(second.skipped_already_verified) == 1
    # Third run over the *twice-processed* dataset must also converge —
    # i.e. ``last_reverify_skipped_at`` updates without status drifting.
    tasks_path.write_text(json.dumps(second.verified))
    third = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=0,
        )
    )
    assert third.verified[0]["feasibility"]["status"] == "verified"
    assert third.verified[0]["feasibility"]["verified_at"] == first_feas["verified_at"]
