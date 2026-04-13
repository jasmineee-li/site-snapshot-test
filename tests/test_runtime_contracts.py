from __future__ import annotations

import pytest

from worldsim.agent_config import (
    RUNTIME_METADATA_KEY,
    bind_task_to_instance,
    execution_instance_dict,
    make_llm,
    prepare_task_for_execution,
    resolve_task_inputs,
)
from worldsim.config import BenchmarkConfig, BenchmarkInstance


def test_prepare_task_for_execution_supports_multi_site_placeholders_and_resets():
    instances = [
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping.test",
            reset_endpoint="http://shopping.test/init",
        ),
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://gitlab.test",
            reset_endpoint="http://gitlab.test/init",
        ),
    ]
    task = {
        "id": "task-1",
        "site": "shopping",
        "sites": ["shopping", "gitlab"],
        "instruction": "Visit __SHOPPING__ and then open __GITLAB__/issues.",
        "start_urls": ["__SHOPPING__/products", "__GITLAB__/issues"],
    }

    prepared, missing = prepare_task_for_execution(task, instances)

    assert missing == []
    runtime = prepared[RUNTIME_METADATA_KEY]
    assert runtime["sites"] == ["shopping", "gitlab"]
    assert runtime["reset_endpoints"] == []

    bound = bind_task_to_instance(prepared, instances[0], instances)
    assert bound[RUNTIME_METADATA_KEY]["reset_endpoints"] == [
        "http://shopping.test/init",
        "http://gitlab.test/init",
    ]

    instance_dict = execution_instance_dict(instances[0], bound)
    instruction, start_urls = resolve_task_inputs(bound, instance_dict)

    assert instruction == ("Visit http://shopping.test and then open http://gitlab.test/issues.")
    assert start_urls == [
        "http://shopping.test/products",
        "http://gitlab.test/issues",
    ]


def test_bind_task_to_instance_uses_chosen_primary_instance_metadata():
    instances = [
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping-primary.test",
            reset_endpoint="http://shopping-primary.test/init",
        ),
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping-secondary.test",
            reset_endpoint="http://shopping-secondary.test/init",
        ),
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://gitlab.test",
            reset_endpoint="http://gitlab.test/init",
        ),
    ]
    task = {
        "id": "task-bind",
        "site": "shopping",
        "sites": ["shopping", "gitlab"],
        "instruction": "Visit __SHOPPING__ then __GITLAB__.",
        "start_urls": ["__SHOPPING__/products", "__GITLAB__/issues"],
    }

    prepared, missing = prepare_task_for_execution(task, instances)

    assert missing == []

    bound = bind_task_to_instance(prepared, instances[1], instances)
    runtime = bound[RUNTIME_METADATA_KEY]

    assert runtime["reset_endpoints"] == [
        "http://shopping-secondary.test/init",
        "http://gitlab.test/init",
    ]

    instance_dict = execution_instance_dict(instances[1], bound)
    instruction, start_urls = resolve_task_inputs(bound, instance_dict)

    assert instruction == ("Visit http://shopping-secondary.test then http://gitlab.test.")
    assert start_urls == [
        "http://shopping-secondary.test/products",
        "http://gitlab.test/issues",
    ]


def test_prepare_task_for_execution_reports_missing_sites():
    instances = [BenchmarkInstance(site_name="shopping", site_url="http://shopping.test")]
    task = {
        "id": "task-2",
        "site": "shopping",
        "sites": ["shopping", "reddit"],
        "instruction": "Open __SHOPPING__ then __REDDIT__.",
        "start_urls": [],
    }

    _, missing = prepare_task_for_execution(task, instances)

    assert missing == ["reddit"]


def test_benchmark_config_accepts_top_level_url_placeholders():
    config = BenchmarkConfig.model_validate(
        {
            "benchmark_name": "WebArena Verified",
            "url_placeholders": {"__SHOPPING__": "http://shopping.test"},
            "instances": [
                {
                    "site_name": "shopping",
                    "site_url": "http://shopping-instance.test",
                    "url_placeholders": {"__GITLAB__": "http://gitlab.test"},
                }
            ],
            "benchmark_codebase": "/tmp/benchmark",
        }
    )

    assert config.url_placeholders == {
        "__SHOPPING__": "http://shopping.test",
        "__GITLAB__": "http://gitlab.test",
    }


def test_make_llm_rejects_unknown_model_family_without_provider():
    with pytest.raises(ValueError, match="Could not infer a provider"):
        make_llm(model="mystery-model-1")
