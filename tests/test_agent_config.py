from __future__ import annotations

from worldsim.agent_config import (
    execution_instance_dict,
    instances_for_site,
    prepare_task_for_execution,
    bind_task_to_instance,
    resolve_task_inputs,
)
from worldsim.config import BenchmarkInstance


def test_same_site_replicas_route_and_resolve_placeholders() -> None:
    instances = [
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping-1.test",
            reset_endpoint="http://shopping-1.test/init",
            replica_index=0,
            replica_name="shopping_0",
        ),
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping-2.test",
            reset_endpoint="http://shopping-2.test/init",
            replica_index=1,
            replica_name="shopping_1",
        ),
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://gitlab.test",
            reset_endpoint="http://gitlab.test/init",
            replica_index=0,
            replica_name="gitlab_0",
        ),
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://gitlab-2.test",
            reset_endpoint="http://gitlab-2.test/init",
            replica_index=1,
            replica_name="gitlab_1",
        ),
    ]
    prepared, missing = prepare_task_for_execution(
        {
            "id": "task-1",
            "site": "shopping",
            "sites": ["shopping", "gitlab"],
            "instruction": "Open __SHOPPING__/orders and __GITLAB__/merge_requests",
            "start_urls": ["__SHOPPING__/orders", "__GITLAB__/merge_requests"],
            "data_seed": {"mechanism": "none"},
            "reward_function": {"eval": []},
        },
        instances,
    )

    assert missing == []
    assert len(instances_for_site(instances, "shopping")) == 2

    bound = bind_task_to_instance(prepared, instances[1], instances)
    instance_dict = execution_instance_dict(instances[1], bound)
    instruction, start_urls = resolve_task_inputs(bound, instance_dict)

    assert instruction == (
        "Open http://shopping-2.test/orders and http://gitlab-2.test/merge_requests"
    )
    assert start_urls == [
        "http://shopping-2.test/orders",
        "http://gitlab-2.test/merge_requests",
    ]
    assert bound["_worldsim_runtime"]["reset_endpoints"] == [
        "http://shopping-2.test/init",
        "http://gitlab-2.test/init",
    ]


def test_multi_site_secondary_binding_is_deterministic_per_task() -> None:
    instances = [
        BenchmarkInstance(
            site_name="shopping",
            site_url="http://shopping-1.test",
            reset_endpoint="http://shopping-1.test/init",
            replica_index=0,
            replica_name="shopping_0",
        ),
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://gitlab-1.test",
            reset_endpoint="http://gitlab-1.test/init",
            replica_index=0,
            replica_name="gitlab_0",
        ),
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://gitlab-2.test",
            reset_endpoint="http://gitlab-2.test/init",
            replica_index=1,
            replica_name="gitlab_1",
        ),
    ]

    base_task = {
        "site": "shopping",
        "sites": ["shopping", "gitlab"],
        "instruction": "Open __SHOPPING__/orders and __GITLAB__/merge_requests",
        "start_urls": ["__SHOPPING__/orders", "__GITLAB__/merge_requests"],
        "data_seed": {"mechanism": "none"},
        "reward_function": {"eval": []},
    }

    bound_a = bind_task_to_instance({**base_task, "id": "task-1"}, instances[0], instances)
    bound_b = bind_task_to_instance({**base_task, "id": "task-1"}, instances[0], instances)
    bound_c = bind_task_to_instance({**base_task, "id": "task-9"}, instances[0], instances)

    assert bound_a["_worldsim_runtime"]["bound_instances"]["gitlab"]["site_url"] == (
        "http://gitlab-2.test"
    )
    assert bound_b["_worldsim_runtime"]["bound_instances"]["gitlab"]["site_url"] == (
        "http://gitlab-2.test"
    )
    assert bound_c["_worldsim_runtime"]["bound_instances"]["gitlab"]["site_url"] == (
        "http://gitlab-1.test"
    )


def test_bind_task_to_instance_rewrites_host_bound_auth_urls() -> None:
    instances = [
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://gitlab-new.test:8033",
            reset_endpoint="http://gitlab-new.test:8034/init",
            replica_index=1,
            replica_name="gitlab_1",
        ),
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://gitlab-old.test:8023",
            reset_endpoint="http://gitlab-old.test:8024/init",
            replica_index=0,
            replica_name="gitlab_0",
        ),
    ]

    bound = bind_task_to_instance(
        {
            "id": "task-host-rewrite",
            "site": "gitlab",
            "sites": ["gitlab"],
            "instruction": "Do the thing",
            "start_urls": ["__GITLAB__"],
            "data_seed": {"mechanism": "none"},
            "reward_function": {"eval": []},
            "agent_context": {
                "auth_mechanism": {
                    "type": "storage_state",
                    "storage_state": {
                        "path": "logs/phase_0d/gitlab/storage_state.json",
                        "form_login": {
                            "login_url": "http://gitlab-old.test:8023/users/sign_in",
                        },
                    },
                }
            },
        },
        instances[0],
        instances,
    )

    login_url = bound["agent_context"]["auth_mechanism"]["storage_state"]["form_login"]["login_url"]
    assert login_url == "http://gitlab-new.test:8033/users/sign_in"


def test_bind_task_to_instance_rewrites_host_bound_auth_scope_patterns() -> None:
    instances = [
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://gitlab-new.test:8033",
            reset_endpoint="http://gitlab-new.test:8034/init",
            replica_index=1,
            replica_name="gitlab_1",
        ),
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://gitlab-old.test:8023",
            reset_endpoint="http://gitlab-old.test:8024/init",
            replica_index=0,
            replica_name="gitlab_0",
        ),
    ]

    bound = bind_task_to_instance(
        {
            "id": "task-host-rewrite-pattern",
            "site": "gitlab",
            "sites": ["gitlab"],
            "instruction": "Do the thing",
            "start_urls": ["__GITLAB__"],
            "data_seed": {"mechanism": "none"},
            "reward_function": {"eval": []},
            "agent_context": {
                "auth_mechanism": {
                    "type": "http_headers",
                    "scope_url_pattern": "^http://gitlab-old.test:8023/.*$",
                    "headers": {"Authorization": "Bearer demo"},
                }
            },
        },
        instances[0],
        instances,
    )

    scope = bound["agent_context"]["auth_mechanism"]["scope_url_pattern"]
    assert scope == "^http://gitlab-new.test:8033/.*$"
