from pathlib import Path

import pytest

from worldsim.config import BenchmarkConfig


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
