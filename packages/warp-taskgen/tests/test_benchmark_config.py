import json
from pathlib import Path

import pytest

from warp_taskgen.config import (
    BenchmarkConfig,
    dump_verification_proxy_config,
    load_benchmark_config,
)


def _config_payload(tmp_path: Path, **extra):
    payload = {
        "instances": [{"site_name": "gitlab", "site_url": "http://gitlab.test"}],
        "benchmark_codebase": str(tmp_path),
    }
    payload.update(extra)
    return payload


def test_benchmark_config_accepts_benchmark_alias(tmp_path):
    config = BenchmarkConfig.model_validate(
        _config_payload(tmp_path, benchmark="WebArena Verified")
    )

    assert config.benchmark_name == "webarena_verified"
    assert config.instances[0].benchmark_name == "webarena_verified"


def test_benchmark_config_rejects_mixed_benchmark_aliases(tmp_path):
    with pytest.raises(ValueError, match="mixed benchmark metadata"):
        BenchmarkConfig.model_validate(
            _config_payload(
                tmp_path,
                benchmark_name="WebArena Verified",
                benchmark_adapter="st-webagentbench",
            )
        )


def test_benchmark_config_rejects_mixed_instance_benchmark_name(tmp_path):
    with pytest.raises(ValueError, match="mixed benchmark metadata"):
        BenchmarkConfig.model_validate(
            _config_payload(
                tmp_path,
                benchmark_name="WebArena Verified",
                instances=[
                    {
                        "site_name": "gitlab",
                        "site_url": "http://gitlab.test",
                        "benchmark_name": "wasp",
                    }
                ],
            )
        )


def test_benchmark_config_canonicalizes_known_benchmark_name(tmp_path):
    config = BenchmarkConfig.model_validate(
        _config_payload(tmp_path, benchmark_name="WebArena Verified")
    )

    assert config.benchmark_name == "webarena_verified"
    assert config.instances[0].benchmark_name == "webarena_verified"


def test_benchmark_config_rejects_unknown_top_level_benchmark_name(tmp_path):
    with pytest.raises(ValueError, match="unknown benchmark"):
        BenchmarkConfig.model_validate(_config_payload(tmp_path, benchmark_name="demo"))


def test_benchmark_config_accepts_top_level_http_headers_shape(tmp_path):
    config = BenchmarkConfig.model_validate(
        _config_payload(
            tmp_path,
            benchmark_name="WebArena Verified",
            instances=[
                {
                    "site_name": "reddit",
                    "site_url": "http://reddit.test",
                    "agent_auth": {
                        "type": "http_headers",
                        "headers": {"X-Postmill-Auto-Login": "alice:pw"},
                    },
                }
            ],
        )
    )

    assert config.instances[0].agent_auth["headers"] == {"X-Postmill-Auto-Login": "alice:pw"}


def test_load_benchmark_config_resolves_proxy_token_file_relative_to_config(tmp_path):
    config_dir = tmp_path / "run"
    config_dir.mkdir()
    (config_dir / ".proxy_token").write_text("file-token\n")
    config_path = config_dir / "instances.json"
    config_path.write_text(
        json.dumps(
            _config_payload(
                tmp_path,
                benchmark_name="WebArena Verified",
                verification_proxy={
                    "token_file": ".proxy_token",
                    "port_offset": 10000,
                },
            )
        )
    )

    config = load_benchmark_config(config_path)

    assert config.verification_proxy is not None
    assert config.verification_proxy.token == "file-token"


def test_verification_proxy_model_dump_json_serializes_token_file(tmp_path):
    config = BenchmarkConfig.model_validate(
        _config_payload(
            tmp_path,
            benchmark_name="WebArena Verified",
            verification_proxy={
                "token_file": str(tmp_path / ".proxy_token"),
                "port_offset": 10000,
            },
        )
    )

    assert config.verification_proxy is not None
    json.dumps(config.verification_proxy.model_dump(mode="json", exclude_none=True))


def test_dump_verification_proxy_config_omits_resolved_external_token(monkeypatch, tmp_path):
    token_file = tmp_path / ".proxy_token"
    token_file.write_text("file-token\n")
    monkeypatch.setenv("WORLDSIM_TEST_PROXY_TOKEN", "env-token")
    config = BenchmarkConfig.model_validate(
        _config_payload(
            tmp_path,
            benchmark_name="WebArena Verified",
            verification_proxy={
                "token_env": "WORLDSIM_TEST_PROXY_TOKEN",
                "token_file": str(token_file),
                "port_offset": 10000,
            },
        )
    )

    assert config.verification_proxy is not None
    dumped = dump_verification_proxy_config(config.verification_proxy)

    assert dumped == {
        "token_env": "WORLDSIM_TEST_PROXY_TOKEN",
        "token_file": str(token_file),
        "scheme": "http",
        "port_offset": 10000,
    }


def test_checked_in_smoke_example_config_is_token_safe_and_loadable(monkeypatch):
    monkeypatch.delenv("WARP_TASKGEN_VERIFICATION_PROXY_TOKEN", raising=False)
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "instances.smoke.example.json"
    payload = json.loads(config_path.read_text())

    assert payload["verification_proxy"] == {
        "token_env": "WARP_TASKGEN_VERIFICATION_PROXY_TOKEN",
        "token_file": ".proxy_token",
        "port_offset": 10000,
    }
    assert {instance["site_name"] for instance in payload["instances"]} == {"gitlab", "reddit"}

    config = load_benchmark_config(config_path)

    assert config.benchmark_name == "webarena_verified"
    assert {instance.site_name for instance in config.instances} == {"gitlab", "reddit"}
