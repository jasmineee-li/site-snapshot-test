"""Host deployment configuration for benchmark bring-up tooling."""

from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


_WIDE_BIND_HOSTS = frozenset({"0.0.0.0", "::"})
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _is_loopback_host(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _is_wide_bind_host(value: str) -> bool:
    return value.strip().lower() in _WIDE_BIND_HOSTS


class BenchmarkHostConfig(BaseModel):
    """Checked-in deployment contract for benchmark host access."""

    name: str = Field(..., description="Short host label such as r5.")
    access_mode: Literal["remote_direct_restricted", "host_local"]
    advertise_host: str = Field(
        ...,
        description="Host or IP advertised in runtime URLs and env-ctrl site URLs.",
    )
    bind_host: str = Field(
        ...,
        description="Host interface Docker should publish web/env-ctrl ports on.",
    )
    db_bind_host: str | None = Field(
        None,
        description="Host interface Docker should publish DB ports on. Defaults to bind_host.",
    )
    ssh_user: str = Field("ubuntu", description="Remote SSH user for host-side scripts.")
    compose_dir_remote: str = Field(
        "/home/ubuntu",
        description="Remote directory that holds compose files and helper scripts.",
    )
    compose_file_remote: str | None = Field(
        None,
        description="Remote compose file path. Defaults to compose_dir_remote/docker-compose.yml.",
    )
    allow_public_web_bind: bool = Field(
        False,
        description="Explicit opt-in for 0.0.0.0/:: web or env-ctrl publishing.",
    )
    allow_public_db_bind: bool = Field(
        False,
        description="Explicit opt-in for 0.0.0.0/:: DB publishing.",
    )
    trusted_operator_cidrs: list[str] = Field(
        default_factory=list,
        description="Advisory list of CIDRs that should retain access to exposed ports.",
    )
    security_group_id: str | None = None
    region: str | None = None

    @model_validator(mode="after")
    def _validate_contract(self) -> BenchmarkHostConfig:
        self.advertise_host = self.advertise_host.strip()
        self.bind_host = self.bind_host.strip()
        self.db_bind_host = (self.db_bind_host or self.bind_host).strip()
        self.ssh_user = self.ssh_user.strip()
        self.compose_dir_remote = self.compose_dir_remote.rstrip("/") or "/"
        self.compose_file_remote = (
            self.compose_file_remote.strip()
            if isinstance(self.compose_file_remote, str) and self.compose_file_remote.strip()
            else f"{self.compose_dir_remote}/docker-compose.yml"
        )

        if not self.advertise_host:
            raise ValueError("advertise_host must be a non-empty string")
        if not self.bind_host:
            raise ValueError("bind_host must be a non-empty string")
        if not self.db_bind_host:
            raise ValueError("db_bind_host must be a non-empty string")
        if not self.ssh_user:
            raise ValueError("ssh_user must be a non-empty string")

        if self.access_mode == "remote_direct_restricted" and _is_loopback_host(self.advertise_host):
            raise ValueError("remote_direct_restricted requires a non-loopback advertise_host")
        if self.access_mode == "remote_direct_restricted" and _is_loopback_host(self.bind_host):
            raise ValueError("remote_direct_restricted requires a non-loopback bind_host")
        if self.access_mode == "host_local" and not _is_loopback_host(self.advertise_host):
            raise ValueError("host_local requires a loopback advertise_host")

        if _is_wide_bind_host(self.bind_host) and not self.allow_public_web_bind:
            raise ValueError(
                "bind_host publishes on all interfaces; set allow_public_web_bind=true to opt in"
            )
        if _is_wide_bind_host(self.db_bind_host) and not self.allow_public_db_bind:
            raise ValueError(
                "db_bind_host publishes on all interfaces; set allow_public_db_bind=true to opt in"
            )
        for cidr in self.trusted_operator_cidrs:
            try:
                ipaddress.ip_network(cidr, strict=False)
            except ValueError as exc:
                raise ValueError(f"trusted_operator_cidrs contains invalid CIDR {cidr!r}") from exc
        return self


def load_host_config(path: str | Path) -> BenchmarkHostConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"host config must be a YAML object: {config_path}")
    return BenchmarkHostConfig.model_validate(data)
