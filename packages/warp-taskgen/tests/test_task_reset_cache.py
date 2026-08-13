from __future__ import annotations

from types import SimpleNamespace

from warp_taskgen.agent_config import RUNTIME_METADATA_KEY
from warp_taskgen.task_reset_cache import (
    TaskResetCache,
    callable_accepts_keyword,
    result_likely_mutated_state,
)


def _bound_task() -> dict:
    return {
        "id": "task-1",
        "site": "shopping",
        "data_seed": {"mechanism": "none"},
        RUNTIME_METADATA_KEY: {
            "bound_instances": {
                "shopping": {
                    "site_url": "http://shopping.test:7770",
                    "reset_endpoint": "http://shopping.test:7771/init",
                }
            }
        },
    }


def test_task_reset_cache_requires_first_reset_and_skips_after_clean_run() -> None:
    cache = TaskResetCache()
    task = _bound_task()

    assert cache.should_reset(task) is True
    cache.mark_clean(task)
    assert cache.should_reset(task) is False


def test_task_reset_cache_marks_dirty_after_mutation() -> None:
    cache = TaskResetCache()
    task = _bound_task()

    cache.mark_clean(task)
    cache.mark_dirty(task)
    assert cache.should_reset(task) is True


def test_result_likely_mutated_state_detects_seed_or_post_requests() -> None:
    seeded_task = _bound_task()
    seeded_task["data_seed"] = {"mechanism": "form"}
    assert result_likely_mutated_state(seeded_task, SimpleNamespace(network_trace=[])) is True

    read_only_task = _bound_task()
    read_only_result = SimpleNamespace(
        network_trace=[{"method": "GET", "url": "http://shopping.test:7770/orders"}]
    )
    assert result_likely_mutated_state(read_only_task, read_only_result) is False

    mutating_result = SimpleNamespace(
        network_trace=[{"method": "POST", "url": "http://shopping.test:7770/cart/add"}]
    )
    assert result_likely_mutated_state(read_only_task, mutating_result) is True


def test_callable_accepts_keyword_detects_explicit_and_var_keyword_support() -> None:
    def explicit(*, reset_cache=None):
        return reset_cache

    def flexible(**kwargs):
        return kwargs

    def incompatible(task):
        return task

    assert callable_accepts_keyword(explicit, "reset_cache") is True
    assert callable_accepts_keyword(flexible, "reset_cache") is True
    assert callable_accepts_keyword(incompatible, "reset_cache") is False
