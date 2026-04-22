"""Unit tests for ``worldsim.instance_selection``.

Covers both the ``BenchmarkInstance``-typed :func:`select_task_site_instance`
and the dict-valued :func:`select_task_site_instance_dict` used by Phase 2c
(which threads raw dicts rather than Pydantic models).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from worldsim.config import BenchmarkInstance
from worldsim.instance_selection import (
    select_task_site_instance,
    select_task_site_instance_dict,
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
