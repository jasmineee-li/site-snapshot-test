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
                "name": "r8a",
                "access_mode": "remote_direct_restricted",
                "advertise_host": "203.0.113.10",
                "bind_host": "192.0.2.10",
            },
            sort_keys=False,
        )
    )

    cfg = load_host_config(path)
    assert cfg.db_bind_host == "192.0.2.10"
    assert cfg.compose_file_remote == "/home/ubuntu/docker-compose.yml"


def test_host_config_rejects_unapproved_public_binds() -> None:
    with pytest.raises(ValueError, match="allow_public_web_bind=true"):
        BenchmarkHostConfig.model_validate(
            {
                "name": "r8a",
                "access_mode": "remote_direct_restricted",
                "advertise_host": "203.0.113.10",
                "bind_host": "0.0.0.0",
            }
        )

    with pytest.raises(ValueError, match="allow_public_db_bind=true"):
        BenchmarkHostConfig.model_validate(
            {
                "name": "r8a",
                "access_mode": "remote_direct_restricted",
                "advertise_host": "203.0.113.10",
                "bind_host": "192.0.2.10",
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


def test_host_config_rejects_world_open_trusted_operator_cidrs() -> None:
    with pytest.raises(ValueError, match="world-open CIDRs"):
        BenchmarkHostConfig.model_validate(
            {
                "name": "r8a",
                "access_mode": "remote_direct_restricted",
                "advertise_host": "203.0.113.10",
                "bind_host": "0.0.0.0",
                "allow_public_web_bind": True,
                "allow_public_db_bind": True,
                "trusted_operator_cidrs": ["0.0.0.0/0"],
            }
        )


def test_host_config_allows_narrow_trusted_operator_cidrs() -> None:
    cfg = BenchmarkHostConfig.model_validate(
        {
            "name": "r8a",
            "access_mode": "remote_direct_restricted",
            "advertise_host": "203.0.113.10",
            "bind_host": "0.0.0.0",
            "allow_public_web_bind": True,
            "allow_public_db_bind": True,
            "trusted_operator_cidrs": ["198.51.100.10/32"],
        }
    )
    assert cfg.trusted_operator_cidrs == ["198.51.100.10/32"]


def test_host_config_instance_id_optional() -> None:
    cfg = BenchmarkHostConfig.model_validate(
        {
            "name": "r8a",
            "access_mode": "remote_direct_restricted",
            "advertise_host": "203.0.113.10",
            "bind_host": "0.0.0.0",
            "allow_public_web_bind": True,
            "allow_public_db_bind": True,
        }
    )
    assert cfg.instance_id is None


def test_host_config_instance_id_accepts_well_formed() -> None:
    cfg = BenchmarkHostConfig.model_validate(
        {
            "name": "r8a",
            "access_mode": "remote_direct_restricted",
            "advertise_host": "203.0.113.10",
            "bind_host": "0.0.0.0",
            "allow_public_web_bind": True,
            "allow_public_db_bind": True,
            "instance_id": "i-0123456789abcdef0",
        }
    )
    assert cfg.instance_id == "i-0123456789abcdef0"


@pytest.mark.parametrize(
    "bad_id",
    [
        "i-0123456789abcdef",
        "i-0123456789abcdef00",
        "i-0123456789ABCDEF0",
        "0123456789abcdef0",
        "i-zzzzzzzzzzzzzzzzz",
        "i- 0123456789abcdef0",
    ],
)
def test_host_config_instance_id_rejects_malformed(bad_id: str) -> None:
    payload = {
        "name": "r8a",
        "access_mode": "remote_direct_restricted",
        "advertise_host": "203.0.113.10",
        "bind_host": "0.0.0.0",
        "allow_public_web_bind": True,
        "allow_public_db_bind": True,
        "instance_id": bad_id,
    }
    with pytest.raises(ValueError, match="instance_id"):
        BenchmarkHostConfig.model_validate(payload)


def test_host_config_instance_id_strips_outer_whitespace() -> None:
    cfg = BenchmarkHostConfig.model_validate(
        {
            "name": "r8a",
            "access_mode": "remote_direct_restricted",
            "advertise_host": "203.0.113.10",
            "bind_host": "0.0.0.0",
            "allow_public_web_bind": True,
            "allow_public_db_bind": True,
            "instance_id": "  i-0123456789abcdef0  ",
        }
    )
    assert cfg.instance_id == "i-0123456789abcdef0"
