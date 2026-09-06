"""``verify_feasibility`` end-to-end cases: classification, retry, idempotency skip."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC
from typing import Any

import pytest

from warp_taskgen.editors import EditorError
from warp_taskgen.phase_2.phase_2c import fingerprints
from warp_taskgen.phase_2.phase_2c import runner as feas

from ._fixtures import (
    _STUB_SEED_REGISTRY,
    _bundle,
    _bypass_preflight,  # noqa: F401
    _FakeHandle,
    _gitlab_instance,
    _seed_bundle,
    _stable_git_fingerprint,  # noqa: F401
    _task,
    _write_tasks,
)


def _at009_oversize_task() -> dict[str, Any]:
    # Mirrors the AT-009 shape: a GitLab create_group with a 624-char
    # description. The exact content doesn't matter for classification;
    # the mock editor is what asserts the length_exceeded outcome.
    description = "X" * 624
    return _task(task_id="AT-009", method="create_group", detail=description)


def _host_fingerprint_for_test(
    *,
    instances: list[dict[str, Any]] | None = None,
    instances_label: str = "instances.smoke.json",
    editor_commit: str = "cafebabe1234",
    dataset_commit: str = "cafebabe1234",
    task_content_hash: str = "deadbeef0000",
) -> dict[str, str]:
    active_instances = instances or [_gitlab_instance()]
    return {
        "host_config": instances_label,
        "instances_digest": fingerprints._instances_digest(active_instances),
        "editor_commit": editor_commit,
        "dataset_commit": dataset_commit,
        "task_content_hash": task_content_hash,
    }


# ---------------------------------------------------------------------------
# Case 1 — happy path
# ---------------------------------------------------------------------------


def test_case_01_2xx_create_verifies_and_cleans(tmp_path, monkeypatch):
    handle = _FakeHandle()

    def responder(idx, seed, instance):
        return handle

    bundle = _seed_bundle(responder)
    tasks_path = _write_tasks(tmp_path, [_task()])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
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

    bundle = _seed_bundle(responder)
    tasks_path = _write_tasks(tmp_path, [_task()])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
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

    bundle = _seed_bundle(responder)
    tasks_path = _write_tasks(tmp_path, [_task()])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
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
    sleep_calls: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    def responder(idx, seed, instance):
        if idx == 0:
            raise EditorError("request_failed", "upstream 500", http_status=500)
        return handle

    bundle = _seed_bundle(responder, retry_sleep=_fake_sleep)
    tasks_path = _write_tasks(tmp_path, [_task()])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=1,
        )
    )
    assert len(report.verified) == 1
    attempts = report.verified[0]["feasibility"]["attempts"]
    assert [a["attempt"] for a in attempts] == [0, 1]
    assert attempts[-1]["status"] == "success"
    assert sleep_calls == [1.0]


# ---------------------------------------------------------------------------
# Case 5 — 500 twice → exhausts retries, reports request_failed
# ---------------------------------------------------------------------------


def test_case_05_retry_exhausted_yields_request_failed(tmp_path, monkeypatch):
    sleep_calls: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    def responder(idx, seed, instance):
        raise EditorError("request_failed", "upstream 503", http_status=503)

    bundle = _seed_bundle(responder, retry_sleep=_fake_sleep)
    tasks_path = _write_tasks(tmp_path, [_task()])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=1,
        )
    )
    assert len(report.infeasible) == 1
    errors = report.infeasible[0]["feasibility"]["errors"]
    assert errors[0]["kind"] == "request_failed"
    attempts = report.infeasible[0]["feasibility"]["attempts"]
    assert [a["attempt"] for a in attempts] == [0, 1]
    assert [a["status"] for a in attempts] == ["request_failed", "request_failed"]
    assert sleep_calls == [1.0]


# ---------------------------------------------------------------------------
# Case 6 — cleanup raises but verification still recorded
# ---------------------------------------------------------------------------


def test_case_06_cleanup_error_yields_warning(tmp_path, monkeypatch):
    handle = _FakeHandle(raises=True)

    def responder(idx, seed, instance):
        return handle

    bundle = _seed_bundle(responder)
    tasks_path = _write_tasks(tmp_path, [_task()])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
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
    content_hash = fingerprints._task_content_hash(seed)
    prior_feas = {
        "status": "verified",
        "verified_at": "2026-04-18T00:00:00Z",
        "host_fingerprint": _host_fingerprint_for_test(task_content_hash=content_hash),
        "attempts": [{"attempt": 0, "status": "success", "elapsed_ms": 100}],
    }
    task = _task(feasibility=prior_feas)

    calls = {"n": 0}

    def responder(idx, seed_payload, instance):
        calls["n"] += 1
        raise AssertionError("should have skipped due to fingerprint match")

    bundle = _seed_bundle(responder)
    tasks_path = _write_tasks(tmp_path, [task])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
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
        "host_fingerprint": _host_fingerprint_for_test(
            editor_commit="olddeadbeef",
            dataset_commit="olddeadbeef",
        ),
    }
    task = _task(feasibility=prior)

    handle = _FakeHandle()

    def responder(idx, seed, instance):
        return handle

    bundle = _seed_bundle(responder)
    tasks_path = _write_tasks(tmp_path, [task])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
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
        "host_fingerprint": _host_fingerprint_for_test(
            task_content_hash="aaaaaaaaaaaa"  # mismatches the live seed
        ),
    }
    task = _task(feasibility=prior)

    handle = _FakeHandle()

    def responder(idx, seed, instance):
        return handle

    bundle = _seed_bundle(responder)
    tasks_path = _write_tasks(tmp_path, [task])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=0,
        )
    )
    assert len(report.verified) == 1
    assert report.verified[0]["feasibility"]["status"] == "verified"


# ---------------------------------------------------------------------------
# Case 9b — instances file content drift behind the same basename ⇒ re-verify
# ---------------------------------------------------------------------------


def test_case_09b_instance_identity_drift_reverifies(tmp_path, monkeypatch):
    prior = {
        "status": "verified",
        "verified_at": "2026-04-18T00:00:00Z",
        "host_fingerprint": _host_fingerprint_for_test(
            instances=[_gitlab_instance(site_url="https://gitlab-old.example/")]
        ),
    }
    task = _task(feasibility=prior)
    handle = _FakeHandle()

    def responder(idx, seed, instance):
        return handle

    bundle = _seed_bundle(responder)
    tasks_path = _write_tasks(tmp_path, [task])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
            instances=[_gitlab_instance(site_url="https://gitlab-new.example/")],
            instances_label="instances.smoke.json",
            concurrency=1,
            retry_count=0,
        )
    )
    assert len(report.verified) == 1
    assert report.verified[0]["feasibility"]["status"] == "verified"
    assert handle.cleaned is True


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

    bundle = _seed_bundle(responder)
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
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
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

    bundle = _seed_bundle(responder)
    tasks_path = _write_tasks(tmp_path, [_task()])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
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

    bundle = _seed_bundle(responder)
    tasks_path = _write_tasks(tmp_path, [_at009_oversize_task()])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
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
    content_hash = fingerprints._task_content_hash(seed)
    prior = {
        "status": "verified",
        "verified_at": "2026-04-18T00:00:00Z",
        "host_fingerprint": _host_fingerprint_for_test(task_content_hash=content_hash),
    }
    task = _task(feasibility=prior)
    handle = _FakeHandle()

    def responder(idx, seed_payload, instance):
        return handle

    bundle = _seed_bundle(responder)
    tasks_path = _write_tasks(tmp_path, [task])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
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
        "host_fingerprint": _host_fingerprint_for_test(
            editor_commit="olddeadbeef",  # drifts
            dataset_commit="olddeadbeef",
        ),
    }
    task = _task(feasibility=prior)

    def responder(idx, seed, instance):
        raise AssertionError("TTL short-circuit should have skipped this task")

    bundle = _seed_bundle(responder)
    tasks_path = _write_tasks(tmp_path, [task])

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
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
    bundle = _bundle(acquire_tokens=lambda instances: ["gitlab: could not acquire bearer"])
    tasks_path = _write_tasks(tmp_path, [_task()])
    with pytest.raises(RuntimeError, match="token acquisition failed"):
        asyncio.run(
            feas.verify_feasibility(
                tasks_path,
                probes=bundle,
                seed_registry=_STUB_SEED_REGISTRY,
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
    bundle = _bundle()
    with pytest.raises(RuntimeError, match="no matching instance"):
        asyncio.run(
            feas.verify_feasibility(
                tasks_path,
                probes=bundle,
                seed_registry=_STUB_SEED_REGISTRY,
                instances=[_gitlab_instance()],
                concurrency=1,
                retry_count=0,
            )
        )


# ---------------------------------------------------------------------------
# Case 17 — unexpected verifier exception marks task infeasible, phase continues
# ---------------------------------------------------------------------------


def test_case_17_unexpected_verifier_exception_marks_task_infeasible(tmp_path, monkeypatch):
    """One task's unhandled exception must NOT cancel siblings.

    Before the per-task-browser cutover, the worker re-raised unexpected
    exceptions which then propagated through asyncio.gather(
    return_exceptions=False), cancelling every other worker mid-flight.
    At concurrency 8 that turned one real error into 7 TargetClosedError
    casualties. The new contract: mark the offending task as
    verification_crashed and let the other tasks run to completion.
    """

    def responder(idx, seed, instance):
        raise TypeError("boom")

    bundle = _seed_bundle(responder)
    task_a = _task()
    task_a["id"] = "AT-001"
    task_b = _task()
    task_b["id"] = "AT-002"
    tasks_path = _write_tasks(tmp_path, [task_a, task_b])

    monkeypatch.setenv("WORLDSIM_PHASE_2C_SKIP_RENDER_CHECK", "1")
    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
            instances=[_gitlab_instance()],
            concurrency=2,
            retry_count=0,
        )
    )

    # Both tasks hit the raising responder, so both end infeasible with
    # kind="verification_crashed" — but the key property is that
    # asyncio.gather completed rather than propagating the TypeError.
    assert len(report.infeasible) == 2
    for entry in report.infeasible:
        errors = entry["feasibility"]["errors"]
        assert any(e["kind"] == "verification_crashed" for e in errors)
        assert any("TypeError" in e.get("detail", "") for e in errors)


# ---------------------------------------------------------------------------
# Extra — empty editor_calls classified as empty_seed
# ---------------------------------------------------------------------------


def test_empty_editor_calls_marks_task_empty_seed(tmp_path, monkeypatch):
    task = _task()
    task["adversarial_data_seed"]["editor_calls"] = []

    def responder(idx, seed, instance):
        return None  # apply_data_seed_async returns None on empty calls

    bundle = _seed_bundle(responder)
    tasks_path = _write_tasks(tmp_path, [task])

    # Empty seed should be flagged before the dispatcher is even called —
    # but if it slips past, apply_data_seed_async returning None also trips
    # the explicit handle-is-None check. Both paths mark the task infeasible.
    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=0,
        )
    )
    assert len(report.infeasible) == 1
    kind = report.infeasible[0]["feasibility"]["errors"][0]["kind"]
    assert kind in {"empty_seed", "schema_mismatch"}


# ---------------------------------------------------------------------------
# Case 7b — double-run invariant: running the verifier twice in a row must
# converge to a byte-identical dataset (modulo timestamps). This is the
# regression test for the idempotency-skip status-mutation bug.
# ---------------------------------------------------------------------------


def test_case_07b_double_run_converges_without_status_drift(tmp_path, monkeypatch):
    handle = _FakeHandle()

    def responder(idx, seed, instance):
        return handle

    bundle = _seed_bundle(responder)
    tasks_path = _write_tasks(tmp_path, [_task()])

    # First run: fresh verify.
    first = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
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

    bundle = _seed_bundle(blow_up_if_called)
    second = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
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
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
            instances=[_gitlab_instance()],
            concurrency=1,
            retry_count=0,
        )
    )
    assert third.verified[0]["feasibility"]["status"] == "verified"
    assert third.verified[0]["feasibility"]["verified_at"] == first_feas["verified_at"]
