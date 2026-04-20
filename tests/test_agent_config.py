from __future__ import annotations

from worldsim.agent_config import (
    bind_task_to_instance,
    execution_instance_dict,
    instances_for_site,
    prepare_task_for_execution,
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


# ─── service_tier threading ────────────────────────────────────────────────


def _install_fake_llm_modules(monkeypatch):
    """Install fakes for the browser-use LLM classes that ``make_llm`` lazily imports.

    Each fake records the kwargs it was constructed with so tests can assert on them.
    """
    import sys
    import types

    recorded: dict[str, dict] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            recorded["openai"] = dict(kwargs)

    class FakeChatOpenRouter:
        def __init__(self, **kwargs):
            recorded["openrouter"] = dict(kwargs)

    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            recorded["anthropic"] = dict(kwargs)

    class FakeChatGoogle:
        def __init__(self, **kwargs):
            recorded["google"] = dict(kwargs)

    openai_mod = types.ModuleType("browser_use.llm.openai.chat")
    openai_mod.ChatOpenAI = FakeChatOpenAI
    router_mod = types.ModuleType("browser_use.llm.openrouter.chat")
    router_mod.ChatOpenRouter = FakeChatOpenRouter
    anthropic_mod = types.ModuleType("browser_use.llm.anthropic.chat")
    anthropic_mod.ChatAnthropic = FakeChatAnthropic
    google_mod = types.ModuleType("browser_use.llm.google.chat")
    google_mod.ChatGoogle = FakeChatGoogle

    monkeypatch.setitem(sys.modules, "browser_use.llm.openai.chat", openai_mod)
    monkeypatch.setitem(sys.modules, "browser_use.llm.openrouter.chat", router_mod)
    monkeypatch.setitem(sys.modules, "browser_use.llm.anthropic.chat", anthropic_mod)
    monkeypatch.setitem(sys.modules, "browser_use.llm.google.chat", google_mod)

    # Patch the Responses-API wrapper at its import site so make_llm's lazy
    # ``from worldsim.llm_wrapper import ChatOpenAIResponses`` picks up the fake.
    import worldsim.llm_wrapper as llm_wrapper

    def fake_create(**kwargs):
        recorded["openai_responses"] = dict(kwargs)
        return object()

    monkeypatch.setattr(llm_wrapper.ChatOpenAIResponses, "create", staticmethod(fake_create))

    return recorded


def test_service_tier_threads_to_openai_responses_arm(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    recorded = _install_fake_llm_modules(monkeypatch)
    from worldsim.agent_config import make_llm

    make_llm("gpt-5.4-mini", provider="openai", service_tier="priority")
    assert recorded["openai_responses"]["service_tier"] == "priority"
    assert recorded["openai_responses"]["model"] == "gpt-5.4-mini"


def test_service_tier_threads_to_openai_chat_arm_via_extra_body(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    recorded = _install_fake_llm_modules(monkeypatch)
    from worldsim.agent_config import make_llm

    make_llm("gpt-4o", provider="openai", service_tier="flex")
    assert recorded["openai"]["extra_body"] == {"service_tier": "flex"}


def test_service_tier_threads_to_openrouter_arm_nested(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    recorded = _install_fake_llm_modules(monkeypatch)
    from worldsim.agent_config import make_llm

    make_llm("gpt-5.4-mini", provider="openrouter", service_tier="priority")
    inner = recorded["openrouter"]["extra_body"]["extra_body"]
    assert inner["service_tier"] == "priority"
    # The existing reasoning + provider pins must still be present.
    assert inner["reasoning"] == {"effort": "none", "exclude": True}
    assert inner["provider"]["only"] == ["openai"]


def test_service_tier_ignored_and_warned_for_anthropic(monkeypatch, caplog) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    recorded = _install_fake_llm_modules(monkeypatch)
    import logging

    from worldsim.agent_config import make_llm

    with caplog.at_level(logging.WARNING, logger="worldsim.agent_config"):
        make_llm("claude-sonnet-4-6", provider="anthropic", service_tier="priority")

    # ChatAnthropic must not have received a service_tier kwarg.
    assert "service_tier" not in recorded["anthropic"]
    assert "extra_body" not in recorded["anthropic"]
    # And a warning must have been emitted.
    assert any(
        "service_tier" in rec.message and "OpenAI-only" in rec.message for rec in caplog.records
    )


def test_service_tier_omitted_leaves_baseline_unchanged(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    recorded = _install_fake_llm_modules(monkeypatch)
    from worldsim.agent_config import make_llm

    make_llm("gpt-5.4-mini", provider="openai")
    # ChatOpenAIResponses.create always receives the kwarg; the wrapper's
    # ainvoke closure gates it out of ``responses.parse`` when falsy.
    assert recorded["openai_responses"].get("service_tier") is None

    make_llm("gpt-5.4-mini", provider="openrouter")
    inner = recorded["openrouter"]["extra_body"]["extra_body"]
    assert "service_tier" not in inner
