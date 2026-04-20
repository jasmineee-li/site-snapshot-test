#!/usr/bin/env python3
"""Validate that benchmark ingress is not wide open on required runtime ports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

from worldsim.host_config import load_host_config


def _port_from_url(raw: str | None) -> int | None:
    if not raw:
        return None
    parts = urlsplit(raw)
    return parts.port


def _port_from_db(raw: str | None) -> int | None:
    return _port_from_url(raw)


def _parse_instances_payload(instances_path: Path) -> list[dict[str, str | None]]:
    payload = json.loads(instances_path.read_text())
    instances = payload.get("instances") if isinstance(payload, dict) else payload
    if not isinstance(instances, list):
        raise ValueError(f"instances payload is not a list: {instances_path}")

    parsed: list[dict[str, str | None]] = []
    for index, instance in enumerate(instances):
        if not isinstance(instance, dict):
            raise ValueError(f"instances[{index}] is not an object in {instances_path}")
        site_url = instance.get("site_url")
        if not isinstance(site_url, str) or not site_url.strip():
            raise ValueError(f"instances[{index}].site_url must be a non-empty string")
        if _port_from_url(site_url) is None:
            raise ValueError(f"instances[{index}].site_url must include an explicit port: {site_url!r}")
        reset_endpoint = instance.get("reset_endpoint")
        if reset_endpoint is not None:
            if not isinstance(reset_endpoint, str) or not reset_endpoint.strip():
                raise ValueError(
                    f"instances[{index}].reset_endpoint must be a non-empty string when present"
                )
            if _port_from_url(reset_endpoint) is None:
                raise ValueError(
                    f"instances[{index}].reset_endpoint must include an explicit port: {reset_endpoint!r}"
                )
        db_connection = instance.get("db_connection")
        if db_connection is not None:
            if not isinstance(db_connection, str) or not db_connection.strip():
                raise ValueError(
                    f"instances[{index}].db_connection must be a non-empty string when present"
                )
            if _port_from_db(db_connection) is None:
                raise ValueError(
                    f"instances[{index}].db_connection must include an explicit port: {db_connection!r}"
                )
        parsed.append(
            {
                "site_url": site_url,
                "reset_endpoint": reset_endpoint,
                "db_connection": db_connection,
            }
        )
    return parsed


def _required_ports(instances_path: Path) -> set[int]:
    instances = _parse_instances_payload(instances_path)
    ports: set[int] = set()
    for instance in instances:
        for candidate in (
            _port_from_url(instance["site_url"]),
            _port_from_url(instance["reset_endpoint"]),
            _port_from_db(instance["db_connection"]),
        ):
            if candidate is not None:
                ports.add(candidate)
    return ports


def _permission_covers_port(permission: dict, port: int) -> bool:
    if permission.get("IpProtocol") not in {"tcp", "-1"}:
        return False
    if permission.get("IpProtocol") == "-1":
        return True
    from_port = permission.get("FromPort")
    to_port = permission.get("ToPort")
    if from_port is None or to_port is None:
        return False
    return int(from_port) <= port <= int(to_port)


def _allowed_cidrs(permission: dict) -> set[str]:
    cidrs = {item.get("CidrIp", "") for item in permission.get("IpRanges", [])}
    cidrs.update(item.get("CidrIpv6", "") for item in permission.get("Ipv6Ranges", []))
    return {cidr for cidr in cidrs if cidr}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host-config", required=True)
    ap.add_argument("--instances", required=True)
    args = ap.parse_args()

    cfg = load_host_config(args.host_config)
    if not cfg.security_group_id or not cfg.region:
        print("ERROR: host config must include security_group_id and region", file=sys.stderr)
        return 2

    ports = sorted(_required_ports(Path(args.instances)))
    if not ports:
        print("ERROR: no runtime ports discovered from instances config", file=sys.stderr)
        return 2

    describe = subprocess.run(
        [
            "aws",
            "ec2",
            "describe-security-groups",
            "--group-ids",
            cfg.security_group_id,
            "--region",
            cfg.region,
            "--output",
            "json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if describe.returncode != 0:
        print(describe.stderr.strip() or "ERROR: failed to query security group", file=sys.stderr)
        return 2

    payload = json.loads(describe.stdout)
    groups = payload.get("SecurityGroups") or []
    if not groups:
        print(f"ERROR: security group not found: {cfg.security_group_id}", file=sys.stderr)
        return 2
    permissions = groups[0].get("IpPermissions") or []

    missing: list[int] = []
    world_open: list[int] = []
    for port in ports:
        matches = [perm for perm in permissions if _permission_covers_port(perm, port)]
        if not matches:
            missing.append(port)
            continue
        if any(cidr in {"0.0.0.0/0", "::/0"} for perm in matches for cidr in _allowed_cidrs(perm)):
            world_open.append(port)

    if missing:
        print(
            "ERROR: security group is missing inbound rules for required runtime ports: "
            + ", ".join(str(port) for port in missing),
            file=sys.stderr,
        )
        return 2
    if world_open:
        print(
            "ERROR: security group exposes required runtime ports to the public internet: "
            + ", ".join(str(port) for port in world_open),
            file=sys.stderr,
        )
        return 2

    print(f"security_group={cfg.security_group_id}")
    print("validated_ports=" + ",".join(str(port) for port in ports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
