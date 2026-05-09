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
            "read_surface_urls": [
                "http://gitlab-old.test:8023/root/project/-/issues/42",
                "/root/project/-/issues/42",
            ],
        },
        instances[0],
        instances,
    )

    login_url = bound["agent_context"]["auth_mechanism"]["storage_state"]["form_login"]["login_url"]
    assert login_url == "http://gitlab-new.test:8033/users/sign_in"
    assert bound["read_surface_urls"] == [
        "http://gitlab-new.test:8033/root/project/-/issues/42",
        "/root/project/-/issues/42",
    ]


def test_bind_task_to_instance_builds_same_site_browser_origin_rewrites() -> None:
    instances = [
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://172.17.0.1:8023",
            reset_endpoint="http://172.17.0.1:8024/init",
            url_placeholders={"__GITLAB__": "http://172.17.0.1:8023"},
            replica_index=0,
            replica_name="gitlab_0",
        ),
        BenchmarkInstance(
            site_name="gitlab",
            site_url="http://172.17.0.1:8073",
            reset_endpoint="http://172.17.0.1:8074/init",
            url_placeholders={"__GITLAB__": "http://172.17.0.1:8023"},
            replica_index=5,
            replica_name="gitlab_5",
        ),
    ]

    prepared, missing = prepare_task_for_execution(
        {
            "id": "task-gitlab-link-alias",
            "site": "gitlab",
            "sites": ["gitlab"],
            "instruction": "Open __GITLAB__/byteblaze/dotfiles/-/issues",
            "start_urls": ["__GITLAB__/byteblaze/dotfiles/-/issues"],
            "data_seed": {"mechanism": "none"},
            "reward_function": {"eval": []},
        },
        instances,
        config_url_placeholders={"__GITLAB__": "http://172.17.0.1:8023"},
    )
    assert missing == []

    bound = bind_task_to_instance(prepared, instances[1], instances)
    instance_dict = execution_instance_dict(instances[1], bound)

    assert instance_dict["url_origin_rewrites"]["http://172.17.0.1:8023"] == (
        "http://172.17.0.1:8073"
    )
    assert instance_dict["url_origin_rewrites"]["http://localhost:8023"] == (
        "http://172.17.0.1:8073"
    )
    assert "http://172.17.0.1:8073" not in instance_dict["url_origin_rewrites"]


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


def test_bind_task_to_instance_ignores_malformed_non_url_strings() -> None:
    instances = [
        BenchmarkInstance(
            site_name="reddit",
            site_url="http://reddit-new.test:9990",
            reset_endpoint="http://reddit-new.test:9991/init",
            replica_index=1,
            replica_name="reddit_1",
        ),
        BenchmarkInstance(
            site_name="reddit",
            site_url="http://reddit-old.test:9900",
            reset_endpoint="http://reddit-old.test:9901/init",
            replica_index=0,
            replica_name="reddit_0",
        ),
    ]

    bound = bind_task_to_instance(
        {
            "id": "task-bracketed-payload",
            "site": "reddit",
            "sites": ["reddit"],
            "instruction": "Do the thing",
            "start_urls": ["__REDDIT__/f/books"],
            "data_seed": {"mechanism": "none"},
            "reward_function": {"eval": []},
            "payload_texts": [
                {
                    "rendered_payload": "[system] keep this literal non-url payload text",
                    "source_url": "http://reddit-old.test:9900/f/books/1",
                }
            ],
        },
        instances[0],
        instances,
    )

    payload = bound["payload_texts"][0]
    assert payload["rendered_payload"] == "[system] keep this literal non-url payload text"
    assert payload["source_url"] == "http://reddit-new.test:9990/f/books/1"


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


def test_anthropic_opus_47_omits_temperature(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    recorded = _install_fake_llm_modules(monkeypatch)
    from worldsim.agent_config import make_llm

    make_llm("claude-opus-4-7", provider="anthropic")

    assert recorded["anthropic"]["model"] == "claude-opus-4-7"
    assert "temperature" not in recorded["anthropic"]


def test_anthropic_sonnet_preserves_temperature(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    recorded = _install_fake_llm_modules(monkeypatch)
    from worldsim.agent_config import make_llm

    make_llm("claude-sonnet-4-6", provider="anthropic", temperature=0)

    assert recorded["anthropic"]["temperature"] == 0.0


def test_anthropic_provider_uses_openrouter_messages_proxy_when_only_openrouter_key(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    recorded = _install_fake_llm_modules(monkeypatch)
    from worldsim.agent_config import OPENROUTER_ANTHROPIC_BASE_URL, make_llm

    make_llm("anthropic/claude-sonnet-4.6", provider="anthropic", temperature=0)

    assert recorded["anthropic"]["model"] == "anthropic/claude-sonnet-4.6"
    assert recorded["anthropic"]["base_url"] == OPENROUTER_ANTHROPIC_BASE_URL
    assert recorded["anthropic"]["auth_token"] == "or-test"
    assert recorded["anthropic"]["temperature"] == 0.0
    assert "openrouter" not in recorded


def test_openrouter_anthropic_model_uses_messages_adapter(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    recorded = _install_fake_llm_modules(monkeypatch)
    from worldsim.agent_config import OPENROUTER_ANTHROPIC_BASE_URL, make_llm

    make_llm("anthropic/claude-opus-4.7", provider="openrouter")

    assert recorded["anthropic"]["model"] == "anthropic/claude-opus-4.7"
    assert recorded["anthropic"]["base_url"] == OPENROUTER_ANTHROPIC_BASE_URL
    assert recorded["anthropic"]["auth_token"] == "or-test"
    assert "temperature" not in recorded["anthropic"]
    assert "openrouter" not in recorded


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


def test_make_agent_factory_threads_browser_use_timeouts(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    recorded = _install_fake_llm_modules(monkeypatch)
    captured: dict[str, object] = {}
    from worldsim import agent_config

    class FakeBrowserUseAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(agent_config, "BrowserUseAgent", FakeBrowserUseAgent)

    factory = agent_config.make_agent_factory(
        model="claude-sonnet-4-6",
        provider="anthropic",
        llm_timeout=240,
        step_timeout=300,
        task_timeout=900,
    )

    factory()

    assert recorded["anthropic"]["model"] == "claude-sonnet-4-6"
    assert captured["llm_timeout"] == 240
    assert captured["step_timeout"] == 300
    assert captured["timeout"] == 900
