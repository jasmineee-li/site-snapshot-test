from __future__ import annotations

from pathlib import Path

import yaml


def test_checked_in_proxy_port_map_matches_scale_config() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scale_config = yaml.safe_load((repo_root / "scripts" / "scale_config.yml").read_text())
    proxy_port_offset = int(scale_config["proxy_port_offset"])

    expected_lines: list[str] = []
    for site_name, site_cfg in scale_config["sites"].items():
        port_step = int(site_cfg.get("port_step", 10))
        real_port_base = int(site_cfg["real_port_base"])
        for replica_index in range(int(site_cfg["replicas"])):
            real_port = real_port_base + replica_index * port_step
            proxy_port = real_port + proxy_port_offset
            expected_lines.append(f"{site_name}_{replica_index}:{real_port}:{proxy_port}")
            # envctrl entry follows the replica's web entry so Phase 2c /
            # Phase 4 reset_endpoint hits reach nginx instead of a
            # docker-loopback port. Matches build_proxy_ports().
            expected_lines.append(
                f"{site_name}_{replica_index}_envctrl:{real_port + 1}:{proxy_port + 1}"
            )

    actual_lines = [
        line.strip()
        for line in (repo_root / "scripts" / "proxy_ports.conf").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert actual_lines == expected_lines
