from __future__ import annotations

from pathlib import Path

import yaml

from scripts.generate_compose_scale import build_proxy_ports, expand_site, make_ip_allocator


def test_generated_proxy_port_map_matches_scale_config() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scale_config = yaml.safe_load((repo_root / "scripts" / "scale_config.yml").read_text())
    proxy_port_offset = int(scale_config["proxy_port_offset"])

    ip_allocator = make_ip_allocator(scale_config["network"]["subnet"])
    records = []
    for site_name, site_cfg in scale_config["sites"].items():
        records.extend(
            expand_site(
                site_name=site_name,
                site_cfg=site_cfg,
                replica_count=int(site_cfg["replicas"]),
                orchestrator_host="127.0.0.1",
                bind_host="127.0.0.1",
                db_bind_host="127.0.0.1",
                proxy_port_offset=proxy_port_offset,
                ip_allocator=ip_allocator,
            )
        )
    expected_lines: list[str] = []
    for rec in records:
        real_port = rec["real_web_port"]
        proxy_port = real_port + proxy_port_offset
        expected_lines.append(f"{rec['service_name']}:{real_port}:{proxy_port}")
        # envctrl entry follows the replica's web entry so Phase 2c /
        # Phase 4 reset_endpoint hits reach nginx instead of a
        # docker-loopback port. Matches build_proxy_ports().
        expected_lines.append(
            f"{rec['service_name']}_envctrl:{real_port + 1}:{proxy_port + 1}"
        )
    actual_lines = [
        line.strip()
        for line in build_proxy_ports(records).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert actual_lines == expected_lines
