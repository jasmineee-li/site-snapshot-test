from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from worldsim import main as worldsim_main
from worldsim.agent_models import resolve_agent_model_profile, supported_agentlab_model_profiles
from worldsim.agentlab_cli import _prepare_single_task, _select_instance, _task_from_args
from worldsim.config import BenchmarkConfig
from worldsim.resume_metadata import RESULT_FINGERPRINT_KEY
from worldsim.runners import agentlab as agentlab_runner
from worldsim.runners import available_runners, get_runner_module
from worldsim.runners.agentlab import (
    AgentLabAgentWrapper,
    _browsergym_env_overrides,
    _build_phase4_sidecar_request,
    _build_sidecar_request,
    _parse_sidecar_result,
    _persist_result_sentinel,
    _sidecar_command,
)


class _FakeInstance(SimpleNamespace):
    def model_dump(self):
        return {
            "site_name": self.site_name,
            "site_url": self.site_url,
            "agent_auth": getattr(self, "agent_auth", None),
        }


def _minimal_config(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(
        {
            "benchmark_name": "WebArena Verified",
            "benchmark_codebase": str(tmp_path),
            "instances": [
                {
                    "site_name": "gitlab",
                    "site_url": "http://gitlab.test",
                    "reset_endpoint": "http://reset.test/reset",
                    "url_placeholders": {"__GITLAB__": "http://gitlab.test"},
                }
            ],
        }
    )


def _load_sidecar_cli_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "packages"
        / "worldsim-agentlab-runner"
        / "src"
        / "worldsim_agentlab_runner"
        / "cli.py"
    )
    spec = importlib.util.spec_from_file_location("worldsim_agentlab_runner_cli_test", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_sidecar_model_args_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "packages"
        / "worldsim-agentlab-runner"
        / "src"
        / "worldsim_agentlab_runner"
        / "model_args.py"
    )
    spec = importlib.util.spec_from_file_location(
        "worldsim_agentlab_runner_model_args_test",
        module_path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_sidecar_module(module_name: str):
    package_root = (
        Path(__file__).resolve().parents[1]
        / "packages"
        / "worldsim-agentlab-runner"
        / "src"
    )
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    return __import__(f"worldsim_agentlab_runner.{module_name}", fromlist=["*"])


def test_runner_registry_exposes_agentlab_without_importing_optional_deps():
    assert "agentlab" in available_runners()
    assert get_runner_module("agentlab") is agentlab_runner


@pytest.mark.asyncio
async def test_agentlab_wrapper_lifecycle_and_phase4_sidecar_bridge(monkeypatch, tmp_path):
    agent = AgentLabAgentWrapper(model="demo-model")
    captured: dict[str, object] = {}

    def fake_run_sidecar_request(request, task_dir, subcommand="run"):
        captured["request"] = request
        captured["task_dir"] = task_dir
        captured["subcommand"] = subcommand
        return {
            "status": "success",
            "elapsed": 1.25,
            "steps": 2,
            "is_done": True,
            "final_result": "done",
            "errors": [],
            "network_trace": [{"url": "http://example.test", "method": "GET"}],
        }

    monkeypatch.setattr(agentlab_runner, "_run_sidecar_request", fake_run_sidecar_request)

    await agent.setup("http://example.test")
    result = await agent.run(
        "task",
        "http://example.test",
        tmp_path,
        start_urls=["http://example.test/start"],
        payload_text="payload",
        pvpo_cdp_url="http://127.0.0.1:9222",
    )
    await agent.teardown()

    assert result.status == "success"
    assert result.steps == 2
    assert result.final_result == "done"
    assert result.network_trace == [{"url": "http://example.test", "method": "GET"}]
    assert captured["subcommand"] == "phase4-run"
    assert captured["task_dir"] == tmp_path
    assert captured["request"]["mode"] == "phase4"
    assert captured["request"]["start_urls"] == ["http://example.test/start"]
    assert captured["request"]["payload_text"] == "payload"


def test_parse_sidecar_result_marks_error_payload_as_error(tmp_path):
    parsed = _parse_sidecar_result(
        "task-1",
        tmp_path,
        {"status": "error", "error": "AgentLab did not write summary_info.json"},
    )

    assert parsed["task_id"] == "task-1"
    assert parsed["status"] == "error"
    assert parsed["outcome"] == "error"


def test_parse_sidecar_result_reads_agentlab_summary(tmp_path):
    parsed = _parse_sidecar_result(
        "task-2",
        tmp_path,
        {
            "summary_info": {
                "n_steps": 4,
                "cum_reward": 1,
                "err_msg": None,
                "terminated": True,
                "truncated": False,
            }
        },
    )

    assert parsed["passed"] is True
    assert parsed["status"] == "success"
    assert parsed["steps"] == 4
    assert parsed["is_done"] is True
    assert parsed["reward"] == 1.0
    assert parsed["agentlab_reward"] == 1.0


def test_parse_sidecar_result_preserves_explicit_failed_status(tmp_path):
    parsed = _parse_sidecar_result(
        "task-false",
        tmp_path,
        {
            "passed": False,
            "reward": 1,
            "status": "failure",
            "summary_info": {
                "n_steps": 2,
                "cum_reward": 1,
                "err_msg": None,
            },
        },
    )

    assert parsed["passed"] is False
    assert parsed["status"] == "failure"
    assert parsed["reward"] == 1.0


def test_persist_result_sentinel_writes_canonical_result_json(tmp_path):
    _persist_result_sentinel(
        {
            "id": "task-3",
            "benchmark_name": "webarena_verified",
            "agentlab_task_name": "webarena_verified.1",
            RESULT_FINGERPRINT_KEY: "fp-task-3",
        },
        tmp_path,
        {
            "task_id": "task-3",
            "passed": True,
            "message": "passed",
            "status": "success",
            "elapsed": 2.0,
            "steps": 4,
            "is_done": True,
            "errors": [],
            "agentlab_summary": {"cum_reward": 1},
        },
    )

    data = json.loads((tmp_path / "result.json").read_text())
    assert data["task_id"] == "task-3"
    assert data["passed"] is True
    assert data["status"] == "success"
    assert data["benchmark_name"] == "webarena_verified"
    assert data["agentlab_task_name"] == "webarena_verified.1"
    assert data["agentlab_summary"] == {"cum_reward": 1}
    assert data["reward"] == 0.0
    assert data["agentlab_reward"] == 0.0
    assert data[RESULT_FINGERPRINT_KEY] == "fp-task-3"


def test_browsergym_env_overrides_maps_instances_and_placeholders():
    overrides = _browsergym_env_overrides(
        {
            "site_name": "gitlab",
            "site_url": "http://gitlab.local",
            "url_placeholders": {
                "__REDDIT__": "http://reddit.local",
            },
        }
    )

    assert overrides["WA_GITLAB"] == "http://gitlab.local"
    assert overrides["WA_REDDIT"] == "http://reddit.local"


def test_sidecar_command_can_be_overridden(monkeypatch):
    monkeypatch.setenv("WORLDSIM_AGENTLAB_RUNNER_CMD", "custom-runner run")

    assert _sidecar_command() == ["custom-runner", "run"]


def test_build_sidecar_request_maps_worldsim_inputs(tmp_path):
    request = _build_sidecar_request(
        {
            "id": "42",
            "benchmark_name": "webarena_verified",
            "agentlab_task_name": "webarena_verified.42",
            "agentlab_task_seed": 7,
        },
        {
            "site_name": "gitlab",
            "site_url": "http://gitlab.test",
            "agent_auth": {"type": "storage_state", "path": "auth.json"},
        },
        AgentLabAgentWrapper(model="gpt52", provider="openrouter", service_tier="priority"),
        tmp_path,
        benchmark_name="webarena_verified",
        benchmark_prefix="webarena_verified",
        max_steps=17,
    )

    assert request["task_id"] == "42"
    assert request["browsergym_task_name"] == "webarena_verified.42"
    assert request["requested_model"] == "gpt52"
    assert request["requested_provider"] == "openrouter"
    assert request["model"] == "openai/gpt-5.2"
    assert request["provider"] == "openrouter"
    assert request["service_tier"] == "priority"
    assert request["model_profile"]["key"] == "gpt52"
    assert request["model_profile"]["transport"] == "openrouter"
    assert request["model_profile"]["transport_model"] == "openai/gpt-5.2"
    assert request["model_profile"]["temperature"] is None
    assert request["model_profile"]["extra_body"]["provider"]["only"] == ["openai"]
    assert request["model_profile"]["extra_body"]["service_tier"] == "priority"
    assert request["model_profile"]["extra_body"]["reasoning"] == {"effort": "none", "exclude": True}
    assert request["model_metadata_path"] == str(tmp_path / "worldsim_model_calls.jsonl")
    assert request["max_steps"] == 17
    assert request["vision_support"] is True
    assert request["storage_state"] == "auth.json"
    assert request["task_seed"] == 7
    assert request["env_overrides"] == {"WA_GITLAB": "http://gitlab.test"}


def test_build_sidecar_request_reads_nested_storage_state_schema(tmp_path):
    request = _build_sidecar_request(
        {
            "id": "43",
            "benchmark_name": "webarena_verified",
            "agentlab_task_name": "webarena_verified.43",
        },
        {
            "site_name": "gitlab",
            "site_url": "http://gitlab.test",
            "agent_auth": {
                "type": "storage_state",
                "storage_state": {"path": "nested-auth.json"},
            },
        },
        AgentLabAgentWrapper(model="gpt52", provider="openrouter"),
        tmp_path,
        benchmark_name="webarena_verified",
        benchmark_prefix="webarena_verified",
        max_steps=17,
    )

    assert request["storage_state"] == "nested-auth.json"


def test_build_phase4_sidecar_request_maps_runner_contract(tmp_path):
    request = _build_phase4_sidecar_request(
        "Do the task",
        "http://gitlab.test",
        tmp_path,
        AgentLabAgentWrapper(model="gpt52", provider="openrouter", max_steps=9),
        {
            "start_urls": ["http://gitlab.test/root"],
            "site_prompt": "site prompt",
            "task_site": "gitlab",
            "auth_mechanism": {
                "type": "storage_state",
                "storage_state": {"path": "auth.json"},
            },
            "payload_text": "payload",
            "payload_witnesses": [{"id": "w", "text": "payload"}],
            "pvpo_cdp_url": "http://127.0.0.1:9222",
            "instance_id": "gitlab-0",
            "url_origin_rewrites": {"http://canonical.test": "http://gitlab.test"},
        },
    )

    assert request["mode"] == "phase4"
    assert request["task"] == "Do the task"
    assert request["start_urls"] == ["http://gitlab.test/root"]
    assert request["site_prompt"] == "site prompt"
    assert request["storage_state"] == "auth.json"
    assert request["payload_text"] == "payload"
    assert request["payload_witnesses"] == [{"id": "w", "text": "payload"}]
    assert request["pvpo_cdp_url"] == "http://127.0.0.1:9222"
    assert request["max_steps"] == 9
    assert request["env_overrides"] == {"WA_GITLAB": "http://gitlab.test"}


def test_sidecar_normalizes_direct_provider_requests_for_gpt_5_2():
    module = _load_sidecar_cli_module()

    assert module._normalized_model_name("gpt-5.2", provider="openai") == "openai/gpt-5.2"
    assert module._normalized_model_name("openai/gpt-5.2", provider="openai") == "openai/gpt-5.2"
    assert module._normalized_model_name("claude-sonnet-4-6", provider="anthropic") == (
        "anthropic/claude-sonnet-4-6"
    )
    assert module._default_temperature("openai/gpt-5.2") is None
    assert module._default_temperature("openai/gpt-4.1") == 0


def test_sidecar_applies_agentlab_benchmark_config(monkeypatch):
    module = _load_sidecar_cli_module()

    class FakeBenchmark:
        name = "webarena_verified"
        is_multi_tab = True
        high_level_action_set_args = SimpleNamespace(subsets=("webarena",))

    class FakeBgym:
        pass

    FakeBgym.DEFAULT_BENCHMARKS = {"webarena_verified": lambda: FakeBenchmark()}

    class FakeAgentArgs:
        def __init__(self):
            self.flags = SimpleNamespace(
                obs=SimpleNamespace(use_tabs=False),
                action=SimpleNamespace(action_set=SimpleNamespace(subsets=("bid",))),
            )
            self.calls = []

        def set_benchmark(self, benchmark, demo_mode):
            self.calls.append((benchmark, demo_mode))
            self.flags.obs.use_tabs = benchmark.is_multi_tab
            self.flags.action.action_set = benchmark.high_level_action_set_args

    monkeypatch.setitem(sys.modules, "bgym", FakeBgym())
    agent_args = FakeAgentArgs()

    config = module._apply_benchmark_config(
        agent_args,
        {"benchmark_name": "webarena_verified", "demo_mode": False},
    )

    assert config["status"] == "applied"
    assert config["is_multi_tab"] is True
    assert config["obs_use_tabs"] is True
    assert agent_args.flags.action.action_set.subsets == ("webarena",)
    assert len(agent_args.calls) == 1


def test_sidecar_artifact_manifest_includes_phase4_outputs(tmp_path):
    module = _load_sidecar_cli_module()
    for relative in (
        "summary_info.json",
        "history.json",
        "final_response.json",
        "network_trace.json",
        "network.har",
        "browser_runtime.json",
        "needham_trace.json",
        "needham_trace.xml",
        "package_versions.txt",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (tmp_path / "pvpo").mkdir()
    (tmp_path / "pvpo" / "step_0.json").write_text("{}", encoding="utf-8")
    (tmp_path / "step_0.pkl.gz").write_bytes(b"x")

    manifest = module._artifact_manifest(tmp_path)

    assert manifest["history"] == str(tmp_path / "history.json")
    assert manifest["network_har"] == str(tmp_path / "network.har")
    assert manifest["needham_xml"] == str(tmp_path / "needham_trace.xml")
    assert manifest["pvpo_steps"] == [str(tmp_path / "pvpo" / "step_0.json")]
    assert manifest["steps"] == [str(tmp_path / "step_0.pkl.gz")]


def test_phase4_sidecar_helpers_normalize_final_message_and_pvpo_payload():
    worldsim_task = _load_sidecar_module("worldsim_task")
    sync_pvpo = _load_sidecar_module("sync_pvpo")

    final = worldsim_task.latest_assistant_message(
        [
            {"role": "assistant", "message": "Hi! I am your UI assistant, hello"},
            {"role": "user", "message": "task"},
            {"role": "assistant", "message": "Finished"},
        ]
    )
    payload = sync_pvpo._unwrap_runtime_payload(
        {
            "result": {
                "value": {
                    "entries": [{"char": "p", "layoutVisible": True}],
                    "backgroundColor": {"r": 1, "g": 2, "b": 3},
                    "matchFound": True,
                    "matchOffset": 0,
                    "matchedWitnessId": "w",
                    "matchedWitnessText": "payload",
                    "pageUrl": "http://example.test",
                }
            }
        }
    )

    assert final == "Finished"
    assert payload["background_color"] == [1, 2, 3]
    assert payload["match_found"] is True
    assert payload["match_offset"] == 0


def test_named_agent_model_profiles_cover_openrouter_matrix():
    profiles = {profile.key: profile for profile in supported_agentlab_model_profiles()}

    assert set(profiles) == {"opus47", "sonnet46", "gemini25pro", "kimik25", "gpt52", "glm5"}
    assert profiles["opus47"].transport_model == "anthropic/claude-opus-4.7"
    assert profiles["opus47"].temperature is None
    assert profiles["sonnet46"].transport_model == "anthropic/claude-sonnet-4.6"
    assert profiles["gemini25pro"].transport_model == "google/gemini-2.5-pro"
    assert profiles["kimik25"].transport_model == "moonshotai/kimi-k2.5"
    assert profiles["kimik25"].max_new_tokens == 4096
    assert profiles["kimik25"].temperature is None
    assert profiles["kimik25"].extra_body["provider"]["only"] == ["moonshotai"]
    assert profiles["kimik25"].vision_support is False
    assert profiles["gpt52"].transport_model == "openai/gpt-5.2"
    assert profiles["gpt52"].extra_body["provider"]["allow_fallbacks"] is False
    assert profiles["gpt52"].extra_body["reasoning"] == {"effort": "none", "exclude": True}
    assert profiles["glm5"].transport_model == "z-ai/glm-5"
    assert profiles["glm5"].extra_body["provider"]["only"] == ["z-ai"]


def test_agent_model_profile_supports_native_routes():
    profile = resolve_agent_model_profile("gpt52", provider="openai", service_tier="priority")

    assert profile.provider == "openai"
    assert profile.transport == "litellm"
    assert profile.transport_model == "openai/gpt-5.2"
    assert profile.required_env_var == "OPENAI_API_KEY"
    assert profile.temperature is None
    assert profile.extra_body == {"service_tier": "priority"}


def test_agent_model_profile_normalizes_native_google_slugs():
    profile = resolve_agent_model_profile("gemini25pro", provider="google")
    explicit = resolve_agent_model_profile("google/gemini-2.5-pro", provider="google")

    assert profile.transport_model == "gemini/gemini-2.5-pro"
    assert explicit.transport_model == "gemini/gemini-2.5-pro"


def test_agent_model_profile_supports_raw_openrouter_slugs_for_qwen_family():
    profile = resolve_agent_model_profile("qwen/some-explicit-slug", provider="openrouter")

    assert profile.provider == "openrouter"
    assert profile.transport == "openrouter"
    assert profile.transport_model == "qwen/some-explicit-slug"
    assert profile.required_env_var == "OPENROUTER_API_KEY"
    assert profile.vision_support is False


def test_sidecar_model_args_from_profile_request_builds_openrouter_params():
    module = _load_sidecar_cli_module()
    model_args_module = _load_sidecar_model_args_module()
    profile = resolve_agent_model_profile("gpt52", provider="openrouter").to_sidecar_dict()
    args = model_args_module.model_args_from_request(
        {"model": "openrouter/gpt52", "provider": "openrouter", "model_profile": profile}
    )
    chat = args.make_model()

    params = chat._build_api_params(
        [{"role": "user", "content": "hello"}],
        n_samples=1,
    )

    assert module._normalized_model_name("gpt52", provider="openrouter") == "openrouter/gpt52"
    assert args.model_name == "openai/gpt-5.2"
    assert args.transport == "openrouter"
    assert args.temperature is None
    assert params["model"] == "openai/gpt-5.2"
    assert "temperature" not in params
    assert params["max_tokens"] == 4096
    assert params["extra_body"]["provider"]["require_parameters"] is True


def test_build_sidecar_request_uses_profile_vision_support(tmp_path):
    request = _build_sidecar_request(
        {
            "id": "42",
            "benchmark_name": "webarena_verified",
            "agentlab_task_name": "webarena_verified.42",
        },
        {"site_name": "gitlab", "site_url": "http://gitlab.test"},
        AgentLabAgentWrapper(model="glm5", provider="openrouter"),
        tmp_path,
        benchmark_name="webarena_verified",
        benchmark_prefix="webarena_verified",
        max_steps=17,
    )

    assert request["model_profile"]["key"] == "glm5"
    assert request["vision_support"] is False


def test_sidecar_model_args_rejects_malformed_profiles():
    model_args_module = _load_sidecar_model_args_module()
    profile = resolve_agent_model_profile("gpt52", provider="openrouter").to_sidecar_dict()

    with pytest.raises(ValueError, match="extra_body"):
        model_args_module.model_args_from_request(
            {
                "model": "openai/gpt-5.2",
                "provider": "openrouter",
                "model_profile": profile | {"extra_body": []},
            }
        )
    with pytest.raises(ValueError, match="transport"):
        model_args_module.model_args_from_request(
            {
                "model": "openai/gpt-5.2",
                "provider": "openrouter",
                "model_profile": profile | {"transport": "bogus"},
            }
        )
    with pytest.raises(ValueError, match="vision_support"):
        model_args_module.model_args_from_request(
            {
                "model": "openai/gpt-5.2",
                "provider": "openrouter",
                "model_profile": profile | {"vision_support": "yes"},
            }
        )


def test_sidecar_chat_model_stats_are_agentlab_numeric_only():
    model_args_module = _load_sidecar_model_args_module()
    chat = model_args_module.WorldSimChatModelArgs(
        model_name="openai/gpt-5.2",
        transport="openrouter",
        provider="openrouter",
        profile_key="gpt52",
    ).make_model()

    stats = chat.get_stats()

    assert stats == {"n_retry_llm": 0}
    assert all(isinstance(value, int | float) for value in stats.values())


def test_sidecar_chat_model_writes_model_call_metadata(tmp_path):
    model_args_module = _load_sidecar_model_args_module()
    metadata_path = tmp_path / "worldsim_model_calls.jsonl"
    chat = model_args_module.WorldSimChatModelArgs(
        model_name="openai/gpt-5.2",
        transport="openrouter",
        provider="openrouter",
        profile_key="gpt52",
        display_name="GPT-5.2",
        metadata_path=str(metadata_path),
    ).make_model()
    response = SimpleNamespace(
        model="openai/gpt-5.2",
        provider="openai",
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7),
    )

    chat._record_usage(response)
    chat._record_call_metadata(response)

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["profile_key"] == "gpt52"
    assert payload["response_provider"] == "openai"
    assert payload["usage"] == {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}


def test_build_parser_accepts_agentlab_run_gpt_5_2_default(tmp_path):
    parser = worldsim_main.build_parser()

    args = parser.parse_args(
        [
            "agentlab",
            "run",
            "--instances",
            str(tmp_path / "instances.json"),
            "--browsergym-task-name",
            "webarena_verified.42",
        ]
    )

    assert args.command == "agentlab"
    assert args.agentlab_command == "run"
    assert args.agent_model == "gpt52"
    assert args.agent_provider == "openrouter"


def test_agentlab_runner_default_model_matches_cli_default():
    agent = agentlab_runner.make_agent_factory()()

    assert agent.model == "gpt52"


def test_build_parser_accepts_agentlab_models_command():
    parser = worldsim_main.build_parser()

    args = parser.parse_args(["agentlab", "models", "--json"])

    assert args.command == "agentlab"
    assert args.agentlab_command == "models"
    assert args.json is True


def test_agentlab_cli_builds_synthetic_task_and_prepares_instance(tmp_path):
    config = _minimal_config(tmp_path)
    args = SimpleNamespace(
        task_json=None,
        browsergym_task_name="webarena_verified.42",
        task_id="agentlab-42",
        benchmark_name=None,
        site=None,
        replica_name=None,
        replica_index=None,
    )

    task = _task_from_args(args, config)
    instance = _select_instance(config, task, args)
    prepared = _prepare_single_task(task, config)

    assert task["id"] == "agentlab-42"
    assert task["agentlab_task_name"] == "webarena_verified.42"
    assert task["benchmark_name"] == "webarena_verified"
    assert task["site"] == "gitlab"
    assert instance.site_url == "http://gitlab.test"
    assert prepared["site"] == "gitlab"


@pytest.mark.asyncio
async def test_make_task_runner_runs_comparison_adapter_contract(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    async def fake_reset(task):
        captured["reset_task_id"] = task["id"]

    def fake_run_sidecar_request(request, task_dir):
        captured["request"] = request
        captured["task_dir"] = task_dir
        return {
            "status": "success",
            "passed": True,
            "summary_info": {
                "n_steps": 1,
                "cum_reward": 1,
                "err_msg": None,
                "terminated": True,
                "truncated": False,
            },
        }

    monkeypatch.setattr(agentlab_runner, "_reset_task_environment", fake_reset)
    monkeypatch.setattr(agentlab_runner, "_run_sidecar_request", fake_run_sidecar_request)
    monkeypatch.setattr(agentlab_runner, "_persist_result_sentinel", lambda *args: None)

    runner = agentlab_runner.make_task_runner(max_steps=17)
    result = await runner(
        {
            "id": "42",
            "site": "gitlab",
            "sites": ["gitlab"],
            "benchmark_name": "webarena_verified",
            "agentlab_task_name": "webarena_verified.42",
            "data_seed": {"mechanism": "none"},
        },
        AgentLabAgentWrapper(model="demo-model"),
        _FakeInstance(site_name="gitlab", site_url="http://gitlab.test"),
        tmp_path,
    )

    assert result["status"] == "success"
    assert captured["reset_task_id"] == "42"
    assert captured["task_dir"] == tmp_path
    assert captured["request"]["browsergym_task_name"] == "webarena_verified.42"
    assert captured["request"]["max_steps"] == 17
    assert captured["request"]["env_overrides"] == {"WA_GITLAB": "http://gitlab.test"}


@pytest.mark.asyncio
async def test_make_task_runner_rejects_non_webarena_task_without_agentlab_name(
    monkeypatch, tmp_path
):
    async def fail_reset(task):
        raise AssertionError("reset should not run when task routing metadata is incomplete")

    monkeypatch.setattr(agentlab_runner, "_reset_task_environment", fail_reset)

    runner = agentlab_runner.make_task_runner()

    with pytest.raises(ValueError, match="missing agentlab_task_name"):
        await runner(
            {
                "id": "st-1",
                "site": "gitlab",
                "sites": ["gitlab"],
                "benchmark_name": "stwebagentbench",
                "data_seed": {"mechanism": "none"},
            },
            AgentLabAgentWrapper(model="demo-model"),
            _FakeInstance(site_name="gitlab", site_url="http://gitlab.test"),
            tmp_path,
        )


def test_make_task_runner_rejects_unknown_attack_mode():
    with pytest.raises(ValueError, match="unsupported AgentLab attack_mode"):
        agentlab_runner.make_task_runner(attack_mode="worldsim")
