"""Unit tests for ``worldsim.instance_selection``.

Covers both the ``BenchmarkInstance``-typed :func:`select_task_site_instance`
and the dict-valued :func:`select_task_site_instance_dict` used by Phase 2c
(which threads raw dicts rather than Pydantic models).
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

import pytest

from worldsim.config import BenchmarkInstance
from worldsim.instance_selection import (
    replica_key,
    select_task_site_instance,
    select_task_site_instance_dict,
    select_task_site_instance_dict_p2c,
)


def _gitlab_dict_replica(index: int) -> dict[str, Any]:
    return {
        "site_name": "gitlab",
        "site_url": f"http://172.17.0.1:{8023 + index * 10}",
        "replica_index": index,
        "replica_name": f"gitlab_{index}",
        "benchmark": "webarena_verified",
    }


def _reddit_dict_replica(index: int) -> dict[str, Any]:
    return {
        "site_name": "reddit",
        "site_url": f"http://172.17.0.1:{9900 + index * 10}",
        "replica_index": index,
        "replica_name": f"reddit_{index}",
        "benchmark": "webarena_verified",
    }


def test_dict_selector_single_replica_short_circuits():
    instances = [_gitlab_dict_replica(0)]
    task = {"id": "AT-001"}
    assert select_task_site_instance_dict(task, "gitlab", instances) is instances[0]


def test_dict_selector_missing_site_raises():
    instances = [_gitlab_dict_replica(0)]
    with pytest.raises(ValueError):
        select_task_site_instance_dict({"id": "AT-001"}, "shopping", instances)


def test_dict_selector_is_deterministic_across_calls():
    instances = [_gitlab_dict_replica(i) for i in range(21)]
    task = {"id": "AT-042"}
    first = select_task_site_instance_dict(task, "gitlab", instances)
    for _ in range(10):
        assert select_task_site_instance_dict(task, "gitlab", instances) == first


def test_dict_selector_ignores_cross_site_instances():
    instances = [_gitlab_dict_replica(i) for i in range(3)] + [
        _reddit_dict_replica(i) for i in range(3)
    ]
    task = {"id": "AT-055"}
    picked = select_task_site_instance_dict(task, "gitlab", instances)
    assert picked["site_name"] == "gitlab"


def test_dict_selector_order_independent():
    """Shuffle the replica list; the same task picks the same physical replica."""
    forward = [_gitlab_dict_replica(i) for i in range(21)]
    reverse = list(reversed(forward))
    task = {"id": "AT-077"}
    assert select_task_site_instance_dict(
        task, "gitlab", forward
    ) == select_task_site_instance_dict(task, "gitlab", reverse)


def test_dict_selector_107_tasks_fan_out_across_21_replicas():
    """Statistical fanout bound: 107 tasks over 21 buckets should hit every
    replica at least once and nowhere near the old "single replica" worst
    case. SHA-256 hashing with mean 5.1 tasks/bucket and SD ~2.2 stays well
    under 18 (3-sigma+safety).
    """
    instances = [_gitlab_dict_replica(i) for i in range(21)]
    counts: Counter[str] = Counter()
    for index in range(107):
        task = {"id": f"AT-{index:03d}"}
        picked = select_task_site_instance_dict(task, "gitlab", instances)
        counts[picked["replica_name"]] += 1

    assert sum(counts.values()) == 107
    # The pre-fix bug sent 100% of 107 tasks to a single replica (gitlab_18,
    # 107/107). Real SHA-256 fanout over 107 tasks into 21 buckets has
    # coupon-collector variance — hitting every bucket is not guaranteed, but
    # at least 18/21 is a tight regression bound that still catches any
    # regression to single-replica behavior.
    assert len(counts) >= 18, f"only {len(counts)}/21 replicas received tasks: {counts}"
    worst = max(counts.values())
    # Mean 5.1 tasks/bucket, SD ~2.2; 18 is generous 3-sigma + safety.
    assert worst <= 18, f"replica skew too high: {counts}"


def test_dict_selector_matches_model_selector_on_same_topology():
    """The dict and Pydantic selectors share a hash space so a task lands on
    the same replica regardless of which caller routes it."""
    dict_instances = [_gitlab_dict_replica(i) for i in range(21)]
    model_instances = [BenchmarkInstance(**d) for d in dict_instances]
    for index in range(50):
        task = {"id": f"AT-{index:03d}"}
        dict_pick = select_task_site_instance_dict(task, "gitlab", dict_instances)
        model_pick = select_task_site_instance(task, "gitlab", model_instances)
        assert dict_pick["replica_name"] == model_pick.replica_name


# ---------------------------------------------------------------------------
# Power-of-two-choices (P2C) selector
# ---------------------------------------------------------------------------


def test_p2c_selector_single_replica_short_circuits():
    instances = [_gitlab_dict_replica(0)]
    in_flight: dict[str, int] = {}
    picked = select_task_site_instance_dict_p2c({"id": "x"}, "gitlab", instances, in_flight)
    assert picked is instances[0]


def test_p2c_selector_missing_site_raises():
    instances = [_gitlab_dict_replica(0)]
    with pytest.raises(ValueError):
        select_task_site_instance_dict_p2c({"id": "x"}, "shopping", instances, in_flight_counts={})


def test_p2c_selector_prefers_lower_in_flight():
    """When P2C samples a hot replica and an idle replica, it routes to idle.

    Forces the RNG to always sample replicas 0 (hot) and 1 (idle) so the
    decision is purely about load comparison, not sampling luck.
    """
    instances = [_gitlab_dict_replica(i) for i in range(5)]
    in_flight: dict[str, int] = {replica_key(instances[0]): 10}
    # Deterministic "sample" that always picks index 0 and 1.
    rng = random.Random(0)

    class FixedRng(random.Random):
        def sample(self, population, k):  # type: ignore[override]
            assert k == 2
            ordered = list(population)
            return [ordered[0], ordered[1]]

        def random(self) -> float:  # type: ignore[override]
            return rng.random()

    picked = select_task_site_instance_dict_p2c(
        {"id": "x"}, "gitlab", instances, in_flight, rng=FixedRng()
    )
    assert picked["replica_name"] == instances[1]["replica_name"]


def test_p2c_balances_load_better_than_deterministic_hash():
    """End-to-end: with a simulated workload where deterministic hash
    produces a skewed distribution, P2C's max-load should be strictly
    lower. Asserts the Mitzenmacher-1996 guarantee operationally.

    Simulation: 400 tasks x 21 replicas, P2C tracks in-flight as each
    task reserves its chosen replica. Since tasks "never finish" in this
    model, max-load keeps growing for both — but P2C's worst bucket ends
    meaningfully smaller than the deterministic-hash worst bucket.
    """
    instances = [_gitlab_dict_replica(i) for i in range(21)]
    task_count = 400

    det_counts: Counter[str] = Counter()
    for index in range(task_count):
        task = {"id": f"AT-{index:04d}"}
        det_counts[replica_key(select_task_site_instance_dict(task, "gitlab", instances))] += 1

    rng = random.Random(42)
    in_flight: dict[str, int] = {}
    p2c_counts: Counter[str] = Counter()
    for index in range(task_count):
        task = {"id": f"AT-{index:04d}"}
        picked = select_task_site_instance_dict_p2c(task, "gitlab", instances, in_flight, rng=rng)
        key = replica_key(picked)
        in_flight[key] = in_flight.get(key, 0) + 1
        p2c_counts[key] += 1

    assert max(p2c_counts.values()) < max(det_counts.values()), (
        f"P2C max={max(p2c_counts.values())} must beat hash max="
        f"{max(det_counts.values())}; counts: p2c={p2c_counts} det={det_counts}"
    )
    # P2C should also use every replica. Pure hashing on 400 tasks / 21
    # buckets does too, but keeping the bound tight catches regressions.
    assert len(p2c_counts) == 21
