from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from worldsim.host_config import BenchmarkHostConfig, load_host_config


def test_load_host_config_defaults_db_bind_and_compose_file(tmp_path: Path) -> None:
    path = tmp_path / "host.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "r5",
                "access_mode": "remote_direct_restricted",
                "advertise_host": "3.12.221.9",
                "bind_host": "10.0.0.15",
            },
            sort_keys=False,
        )
    )

    cfg = load_host_config(path)
    assert cfg.db_bind_host == "10.0.0.15"
    assert cfg.compose_file_remote == "/home/ubuntu/docker-compose.yml"


def test_host_config_rejects_unapproved_public_binds() -> None:
    with pytest.raises(ValueError, match="allow_public_web_bind=true"):
        BenchmarkHostConfig.model_validate(
            {
                "name": "r5",
                "access_mode": "remote_direct_restricted",
                "advertise_host": "3.12.221.9",
                "bind_host": "0.0.0.0",
            }
        )

    with pytest.raises(ValueError, match="allow_public_db_bind=true"):
        BenchmarkHostConfig.model_validate(
            {
                "name": "r5",
                "access_mode": "remote_direct_restricted",
                "advertise_host": "3.12.221.9",
                "bind_host": "10.0.0.15",
                "db_bind_host": "0.0.0.0",
            }
        )


def test_host_config_rejects_loopback_remote_direct() -> None:
    with pytest.raises(ValueError, match="non-loopback advertise_host"):
        BenchmarkHostConfig.model_validate(
            {
                "name": "loopback",
                "access_mode": "remote_direct_restricted",
                "advertise_host": "127.0.0.1",
                "bind_host": "127.0.0.1",
            }
        )
