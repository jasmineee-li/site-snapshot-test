from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from worldsim.agent_config import (
    RUNTIME_METADATA_KEY,
    bind_task_to_instance,
    cap_tasks_per_site,
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


def test_benchmark_config_accepts_instance_auth_blocks():
    config = BenchmarkConfig.model_validate(
        {
            "benchmark_name": "WebArena Verified",
            "instances": [
                {
                    "site_name": "gitlab",
                    "site_url": "http://gitlab.test",
                    "auth": {
                        "type": "bearer_token",
                        "header_name": "PRIVATE-TOKEN",
                        "token_source": "/tmp/token.txt",
                    },
                }
            ],
            "benchmark_codebase": "/tmp/benchmark",
        }
    )

    assert config.instances[0].auth == {
        "type": "bearer_token",
        "header_name": "PRIVATE-TOKEN",
        "token_source": "/tmp/token.txt",
    }


def test_make_llm_rejects_unknown_model_family_without_provider():
    with pytest.raises(ValueError, match="Could not infer a provider"):
        make_llm(model="mystery-model-1")


def test_cap_tasks_per_site_rejects_non_positive_values():
    with pytest.raises(ValueError, match="positive integer"):
        cap_tasks_per_site([], 0)


def test_cap_tasks_per_site_normalizes_primary_site_keys():
    tasks = [
        {"id": "a", "site": "Shopping"},
        {"id": "b", "site": "shopping"},
        {"id": "c", "site": "SHOPPING"},
    ]

    capped = cap_tasks_per_site(tasks, 2)

    assert len(capped) == 2


def test_cap_tasks_per_site_is_order_and_site_independent():
    tasks = [
        {"id": "shopping-3", "site": "shopping"},
        {"id": "gitlab-1", "site": "gitlab"},
        {"id": "shopping-1", "site": "shopping"},
        {"id": "shopping-2", "site": "shopping"},
        {"id": "gitlab-2", "site": "gitlab"},
    ]
    reordered = [
        {"id": "reddit-1", "site": "reddit"},
        {"id": "reddit-2", "site": "reddit"},
        {"id": "shopping-2", "site": "shopping"},
        {"id": "shopping-3", "site": "shopping"},
        {"id": "gitlab-2", "site": "gitlab"},
        {"id": "shopping-1", "site": "shopping"},
        {"id": "gitlab-1", "site": "gitlab"},
    ]

    capped = cap_tasks_per_site(tasks, 2)
    capped_reordered = cap_tasks_per_site(reordered, 2)

    assert {task["id"] for task in capped if task["site"] == "shopping"} == {
        task["id"] for task in capped_reordered if task["site"] == "shopping"
    }


# ── OpenRouter fallback tests ────────────────────────────────────────────


def _env_without(*keys: str) -> dict[str, str]:
    """Return a copy of os.environ with *keys* removed."""
    return {k: v for k, v in os.environ.items() if k not in keys}


def _mock_browser_use_module(parent_path: str, class_name: str):
    """Create mock modules for browser_use.llm.*.chat imports.

    Returns ``(mock_chat_cls, modules_dict)`` where *modules_dict* is suitable
    for ``patch.dict("sys.modules", ...)``.  The mock hierarchy must include
    every intermediate package so the ``from browser_use.llm.X.chat import Y``
    import resolves correctly.
    """
    from unittest.mock import MagicMock

    mock_chat_cls = MagicMock()
    # Build the chain: browser_use → browser_use.llm → browser_use.llm.<provider>
    #                   → browser_use.llm.<provider>.chat
    parts = parent_path.split(".")
    modules: dict[str, Any] = {}
    current = MagicMock()
    for i in range(len(parts)):
        modules[".".join(parts[: i + 1])] = current

    chat_module = MagicMock()
    setattr(chat_module, class_name, mock_chat_cls)
    modules[parent_path + ".chat"] = chat_module
    return mock_chat_cls, modules


def test_make_llm_falls_back_to_openrouter_when_google_key_missing():
    """gemini model + no GOOGLE_API_KEY + OPENROUTER_API_KEY → OpenRouter."""
    env = _env_without("GOOGLE_API_KEY")
    env["OPENROUTER_API_KEY"] = "or-test-key"
    mock_chat_cls, modules = _mock_browser_use_module(
        "browser_use.llm.openrouter", "ChatOpenRouter"
    )
    with (
        patch.dict(os.environ, env, clear=True),
        patch.dict("sys.modules", modules),
    ):
        llm = make_llm(model="gemini-3-flash-preview")
        mock_chat_cls.assert_called_once_with(
            model="google/gemini-3-flash-preview", temperature=0, api_key="or-test-key"
        )


def test_make_llm_no_fallback_when_google_key_present():
    """gemini model + GOOGLE_API_KEY set → uses Google directly (no fallback)."""
    env = _env_without("OPENROUTER_API_KEY")
    env["GOOGLE_API_KEY"] = "google-test-key"
    mock_chat_cls, modules = _mock_browser_use_module("browser_use.llm.google", "ChatGoogle")
    with (
        patch.dict(os.environ, env, clear=True),
        patch.dict("sys.modules", modules),
    ):
        llm = make_llm(model="gemini-3-flash-preview")
        mock_chat_cls.assert_called_once_with(model="gemini-3-flash-preview", temperature=0)


def test_make_llm_openrouter_model_no_fallback_needed():
    """Model with '/' already detects as OpenRouter — no fallback logic needed."""
    env = _env_without("GOOGLE_API_KEY")
    env["OPENROUTER_API_KEY"] = "or-test-key"
    mock_chat_cls, modules = _mock_browser_use_module(
        "browser_use.llm.openrouter", "ChatOpenRouter"
    )
    with (
        patch.dict(os.environ, env, clear=True),
        patch.dict("sys.modules", modules),
    ):
        llm = make_llm(model="google/gemini-3-flash-preview")
        # Model name should be passed as-is (already has prefix)
        mock_chat_cls.assert_called_once_with(
            model="google/gemini-3-flash-preview", temperature=0, api_key="or-test-key"
        )


def test_make_llm_falls_back_to_openrouter_for_openai_model():
    """OpenAI model + no OPENAI_API_KEY + OPENROUTER_API_KEY → OpenRouter."""
    env = _env_without("OPENAI_API_KEY")
    env["OPENROUTER_API_KEY"] = "or-test-key"
    mock_chat_cls, modules = _mock_browser_use_module(
        "browser_use.llm.openrouter", "ChatOpenRouter"
    )
    with (
        patch.dict(os.environ, env, clear=True),
        patch.dict("sys.modules", modules),
    ):
        llm = make_llm(model="gpt-4o")
        mock_chat_cls.assert_called_once_with(
            model="openai/gpt-4o", temperature=0, api_key="or-test-key"
        )


def test_make_llm_falls_back_to_openrouter_for_anthropic_model():
    """Anthropic model, no Anthropic envs at all, OPENROUTER_API_KEY set -> OpenRouter."""
    env = _env_without("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")
    env["OPENROUTER_API_KEY"] = "or-test-key"
    mock_chat_cls, modules = _mock_browser_use_module(
        "browser_use.llm.openrouter", "ChatOpenRouter"
    )
    with (
        patch.dict(os.environ, env, clear=True),
        patch.dict("sys.modules", modules),
    ):
        make_llm(model="claude-sonnet-4-6")
        mock_chat_cls.assert_called_once_with(
            model="anthropic/claude-sonnet-4-6", temperature=0, api_key="or-test-key"
        )


def test_make_llm_uses_anthropic_proxy_when_base_url_and_auth_token_set():
    """Anthropic base_url + auth_token set: skip OpenRouter fallback, use ChatAnthropic."""
    env = _env_without("ANTHROPIC_API_KEY")
    env["ANTHROPIC_BASE_URL"] = "https://openrouter.ai/api"
    env["ANTHROPIC_AUTH_TOKEN"] = "sk-or-proxy"
    env["OPENROUTER_API_KEY"] = "or-fallback-key"
    mock_chat_cls, modules = _mock_browser_use_module("browser_use.llm.anthropic", "ChatAnthropic")
    with (
        patch.dict(os.environ, env, clear=True),
        patch.dict("sys.modules", modules),
    ):
        make_llm(model="claude-sonnet-4-6")
        mock_chat_cls.assert_called_once_with(
            model="claude-sonnet-4-6",
            temperature=0,
            base_url="https://openrouter.ai/api",
            auth_token="sk-or-proxy",
        )


def test_make_llm_no_fallback_when_no_openrouter_key():
    """No provider key AND no OPENROUTER_API_KEY → normal error (no silent fallback)."""
    env = _env_without("GOOGLE_API_KEY", "OPENROUTER_API_KEY")
    with patch.dict(os.environ, env, clear=True):
        mock_chat_cls, modules = _mock_browser_use_module("browser_use.llm.google", "ChatGoogle")
        with patch.dict("sys.modules", modules):
            # No fallback — calls Google directly even without the key
            llm = make_llm(model="gemini-3-flash-preview")
            mock_chat_cls.assert_called_once_with(model="gemini-3-flash-preview", temperature=0)
