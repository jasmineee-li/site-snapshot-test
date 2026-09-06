"""Replica fanout: same-site distribution, per-replica cap, stats summary."""

from __future__ import annotations

import asyncio

from warp_taskgen.phase_2.phase_2c import runner as feas

from ._fixtures import (
    _STUB_SEED_REGISTRY,
    _bundle,
    _bypass_preflight,  # noqa: F401
    _FakeHandle,
    _seed_bundle,
    _stable_git_fingerprint,  # noqa: F401
    _task,
    _write_tasks,
)

# ---------------------------------------------------------------------------
# Replica fanout — regression guard for the 2026-04-22 gitlab_18 crush bug
# ---------------------------------------------------------------------------


def test_replica_fanout_distributes_tasks_across_same_site_replicas(tmp_path, monkeypatch):
    """107 gitlab tasks over 21 gitlab replicas must fan out.

    Pre-fix Phase 2c built a ``dict[site, inst]`` that silently dropped every
    replica after the first, routing every task to a single upstream (the
    last-loaded one, gitlab_18 on r5.yaml). The fanout selector places tasks
    by SHA-256 hash of the task id; this test asserts every replica receives
    traffic and that the worst-case skew stays within statistical bounds.
    """
    replicas = [
        {
            "site_name": "gitlab",
            "site_url": f"http://172.17.0.1:{8023 + i * 10}",
            "replica_index": i,
            "replica_name": f"gitlab_{i}",
            "benchmark": "webarena_verified",
        }
        for i in range(21)
    ]
    tasks = [_task(task_id=f"AT-{i:03d}") for i in range(107)]

    observed: list[str] = []

    def responder(idx, seed, instance):
        observed.append(str(instance.get("replica_name")))
        return _FakeHandle()

    bundle = _seed_bundle(responder)
    tasks_path = _write_tasks(tmp_path, tasks)

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
            instances=replicas,
            concurrency=8,
            retry_count=0,
        )
    )

    assert len(report.verified) == 107
    assert len(observed) == 107
    distinct = set(observed)
    # Pre-fix: 1 distinct replica (the last-loaded), 107/107 tasks.
    # Post-fix: SHA-256 fanout over 21 buckets with 107 tasks — coupon-collector
    # variance means not every bucket is guaranteed, but at least 18/21 is a
    # tight regression bound that still catches a regression to the old
    # single-replica routing.
    assert len(distinct) >= 18, f"only {len(distinct)}/21 replicas received tasks: {distinct}"
    worst = max(observed.count(name) for name in distinct)
    # Mean 5.1 tasks per replica; SD ~2.2. 18 is generous 3-sigma headroom and
    # still catches any regression to the old single-replica behavior.
    assert worst <= 18, (
        f"skew too high; counts: {sorted([(n, observed.count(n)) for n in distinct])}"
    )


def test_per_replica_cap_bounds_in_flight_verifications(tmp_path, monkeypatch):
    """With a single replica and cap 2, no more than 2 verifications run at once.

    Forces 10 tasks onto one replica (so P2C's single-replica short-circuit
    picks it every time) with a per-replica cap override of 2. The fake
    ``apply_data_seed_async`` sleeps briefly while tracking the in-flight
    count via a shared dict. The cap must hold regardless of how high
    ``concurrency`` is set on the verify_feasibility call.
    """
    replicas = [
        {
            "site_name": "gitlab",
            "site_url": "http://172.17.0.1:8023",
            "replica_index": 0,
            "replica_name": "gitlab_solo",
            "benchmark": "webarena_verified",
        }
    ]
    tasks = [_task(task_id=f"AT-{i:03d}") for i in range(10)]

    monkeypatch.setitem(feas._PER_REPLICA_CAP_DEFAULT, "gitlab", 2)

    state: dict[str, int] = {"in_flight": 0, "max_in_flight": 0}

    async def fake_apply(seed, instance, **kwargs):
        state["in_flight"] += 1
        state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        try:
            await asyncio.sleep(0.02)
        finally:
            state["in_flight"] -= 1
        return _FakeHandle(), {}

    bundle = _bundle(apply_seed=fake_apply)
    tasks_path = _write_tasks(tmp_path, tasks)

    report = asyncio.run(
        feas.verify_feasibility(
            tasks_path,
            probes=bundle,
            seed_registry=_STUB_SEED_REGISTRY,
            instances=replicas,
            concurrency=10,  # outer memory sem relaxed to max(10, 64)=64
            retry_count=0,
        )
    )

    assert len(report.verified) == 10
    assert state["max_in_flight"] <= 2, (
        f"per-replica cap was 2 but observed {state['max_in_flight']} concurrent verifications"
    )
    # Sanity: we did actually exercise some parallelism, otherwise the
    # cap assertion is trivially satisfied by serial execution.
    assert state["max_in_flight"] >= 2, (
        f"expected ≥2 concurrent; observed max={state['max_in_flight']}"
    )


def test_replica_stats_summary_logged(tmp_path, monkeypatch, caplog):
    """End-of-run log emits one ``replica_stats`` line per replica touched.

    Confirms Layer 5 observability is actually reaching the logger so
    operators can tune per-replica caps from the data instead of guesses.
    """
    replicas = [
        {
            "site_name": "gitlab",
            "site_url": f"http://172.17.0.1:{8023 + i * 10}",
            "replica_index": i,
            "replica_name": f"gitlab_{i}",
            "benchmark": "webarena_verified",
        }
        for i in range(3)
    ]
    tasks = [_task(task_id=f"AT-{i:03d}") for i in range(6)]

    def responder(idx, seed, instance):
        return _FakeHandle()

    bundle = _seed_bundle(responder)
    tasks_path = _write_tasks(tmp_path, tasks)

    import logging

    with caplog.at_level(logging.INFO, logger="warp_taskgen.phase_2.phase_2c.runner"):
        asyncio.run(
            feas.verify_feasibility(
                tasks_path,
                probes=bundle,
                seed_registry=_STUB_SEED_REGISTRY,
                instances=replicas,
                concurrency=3,
                retry_count=0,
            )
        )

    summary_lines = [r.getMessage() for r in caplog.records if "replica_stats" in r.getMessage()]
    assert summary_lines, "expected at least one replica_stats summary line"
    # Every logged line mentions a real replica_name and the requests/errors
    # fields that tuning work needs.
    for line in summary_lines:
        assert "replica=" in line
        assert "requests=" in line
        assert "errors=" in line
