from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


def _write_host_config(tmp_path: Path, **overrides: object) -> Path:
    host_config = {
        "name": "test-r5",
        "access_mode": "remote_direct_restricted",
        "advertise_host": "203.0.113.10",
        "bind_host": "0.0.0.0",
        "db_bind_host": "0.0.0.0",
        "allow_public_web_bind": True,
        "allow_public_db_bind": True,
    }
    host_config.update(overrides)
    host_config_path = tmp_path / "host.yml"
    host_config_path.write_text(yaml.safe_dump(host_config, sort_keys=False))
    return host_config_path


def test_generate_compose_scale_preserves_runtime_contract(tmp_path: Path) -> None:
    base_config = {
        "benchmark_name": "WebArena Verified",
        "benchmark_codebase": "vendors/webarena-verified",
        "verification_proxy": {
            "token": "old-token",
            "port_offset": 10000,
            "scheme": "http",
        },
        "url_placeholders": {
            "__SHOPPING__": "http://old-host:7770",
            "__WIKIPEDIA__": "http://old-host:8888",
        },
        "instances": [
            {
                "site_name": "shopping",
                "site_url": "http://old-host:7770",
                "reset_endpoint": "http://old-host:7771/init",
                "db_connection": "mysql://user:pass@old-host:3306/shop",
                "auth": {
                    "type": "http_headers",
                    "headers": {"X-Test": "seed"},
                },
                "api_auth": {
                    "type": "bearer_token",
                    "token_endpoint": "/token",
                    "credentials": {"username": "admin", "password": "pw"},
                },
                "agent_auth": {
                    "type": "http_headers",
                    "http_headers": {"headers": {"X-Agent": "on"}},
                },
            },
            {
                "site_name": "wikipedia",
                "site_url": "http://old-host:8888",
                "reset_endpoint": "http://old-host:8889/init",
                "agent_auth": {"type": "none"},
            },
        ],
    }
    scale_config = {
        "network": {"name": "worldsim-bench", "subnet": "172.20.0.0/20"},
        "proxy_port_offset": 10000,
        "smoke_test_replicas": {"shopping": 1, "wikipedia": 1},
        "sites": {
            "shopping": {
                "image": "example/shopping:latest",
                "replicas": 2,
                "real_port_base": 7770,
                "container_web_port": 80,
                "port_step": 10,
                "db_port_base": 3306,
                "db_port_step": 1,
            },
            "wikipedia": {
                "image": "example/wikipedia:latest",
                "replicas": 1,
                "real_port_base": 8888,
                "container_web_port": 8080,
                "port_step": 10,
            },
        },
    }

    base_path = tmp_path / "instances.base.json"
    config_path = tmp_path / "scale.yml"
    base_path.write_text(json.dumps(base_config, indent=2) + "\n")
    config_path.write_text(json.dumps(scale_config))

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "generate_compose_scale.py"
    host_config_path = _write_host_config(tmp_path)

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--config",
            str(config_path),
            "--base-config",
            str(base_path),
            "--host-config",
            str(host_config_path),
            "--proxy-token",
            "new-token",
            "--proxy-scheme",
            "https",
            "--out-dir",
            str(tmp_path),
        ],
        check=True,
        cwd=repo_root,
    )

    instances_config = json.loads((tmp_path / "instances.json").read_text())
    instances = instances_config["instances"]

    assert [instance["site_name"] for instance in instances] == [
        "shopping",
        "shopping",
        "wikipedia",
    ]
    assert [instance["replica_name"] for instance in instances] == [
        "shopping_0",
        "shopping_1",
        "wikipedia_0",
    ]

    shopping_0, shopping_1 = instances[:2]
    assert shopping_0["site_url"] == "http://203.0.113.10:7770"
    assert shopping_0["reset_endpoint"] == "http://203.0.113.10:7771/init"
    assert shopping_1["site_url"] == "http://203.0.113.10:7780"
    assert shopping_1["reset_endpoint"] == "http://203.0.113.10:7781/init"
    assert shopping_0["db_connection"] == "mysql://user:pass@203.0.113.10:3306/shop"
    assert shopping_1["db_connection"] == "mysql://user:pass@203.0.113.10:3307/shop"

    assert shopping_0["auth"]["headers"]["X-Test"] == "seed"
    assert shopping_0["api_auth"]["token_endpoint"] == "/token"
    assert shopping_0["agent_auth"]["http_headers"]["headers"]["X-Agent"] == "on"

    assert instances_config["verification_proxy"] == {
        "token": "new-token",
        "port_offset": 10000,
        "scheme": "https",
    }
    assert instances_config["url_placeholders"]["__SHOPPING__"] == "http://203.0.113.10:7770"
    assert instances_config["url_placeholders"]["__WIKIPEDIA__"] == "http://203.0.113.10:8888"

    proxy_ports = (tmp_path / "proxy_ports.conf").read_text().splitlines()
    assert "shopping_0:7770:17770" in proxy_ports
    assert "shopping_1:7780:17780" in proxy_ports


def test_generate_compose_scale_require_pinned_images_rejects_tagged_refs(tmp_path: Path) -> None:
    base_config = {
        "benchmark_name": "WebArena Verified",
        "benchmark_codebase": str(tmp_path),
        "instances": [
            {
                "site_name": "shopping",
                "site_url": "http://old-host:7770",
                "reset_endpoint": "http://old-host:7771/init",
            }
        ],
    }
    scale_config = {
        "sites": {
            "shopping": {
                "image": "example/shopping:latest",
                "replicas": 1,
                "real_port_base": 7770,
                "container_web_port": 80,
            },
            "wikipedia": {
                "image": "worldsim/webarena-verified-wikipedia:amd64",
                "replicas": 1,
                "real_port_base": 8888,
                "container_web_port": 8080,
            },
        }
    }

    base_path = tmp_path / "instances.base.json"
    config_path = tmp_path / "scale.yml"
    base_path.write_text(json.dumps(base_config, indent=2) + "\n")
    config_path.write_text(json.dumps(scale_config))

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "generate_compose_scale.py"
    host_config_path = _write_host_config(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--config",
            str(config_path),
            "--base-config",
            str(base_path),
            "--host-config",
            str(host_config_path),
            "--require-pinned-images",
            "--out-dir",
            str(tmp_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "example/shopping:latest" in completed.stderr
    assert "worldsim/webarena-verified-wikipedia:amd64" in completed.stderr


def test_generate_compose_scale_uses_canonical_shared_volume_names_and_db_ports(
    tmp_path: Path,
) -> None:
    base_config = {
        "benchmark_name": "WebArena Verified",
        "benchmark_codebase": str(tmp_path),
        "instances": [
            {
                "site_name": "shopping",
                "site_url": "http://old-host:7770",
                "reset_endpoint": "http://old-host:7771/init",
                "db_connection": "mysql://user:pass@old-host:3306/shop",
            },
            {
                "site_name": "map",
                "site_url": "http://old-host:3030",
                "reset_endpoint": "http://old-host:3031/init",
            },
            {
                "site_name": "wikipedia",
                "site_url": "http://old-host:8888",
                "reset_endpoint": "http://old-host:8889/init",
            },
        ],
    }
    scale_config = {
        "network": {"name": "worldsim-bench", "subnet": "172.20.0.0/20"},
        "sites": {
            "shopping": {
                "image": "worldsim/webarena-verified-shopping:amd64",
                "replicas": 1,
                "real_port_base": 7770,
                "container_web_port": 80,
                "db_port_base": 3306,
            },
            "map": {
                "image": "am1n3e/webarena-verified-map:latest",
                "replicas": 1,
                "real_port_base": 3030,
                "container_web_port": 8080,
                "shared_ro_volumes": [
                    {
                        "volume_name": "webarena-verified-map-tile-db",
                        "container_path": "/data/database",
                    },
                    {
                        "volume_name": "webarena-verified-map-tiles",
                        "container_path": "/data/tiles",
                    },
                ],
            },
            "wikipedia": {
                "image": "worldsim/webarena-verified-wikipedia:amd64",
                "replicas": 1,
                "real_port_base": 8888,
                "container_web_port": 8080,
                "shared_ro_volumes": [
                    {
                        "volume_name": "webarena-verified-envs_webarena-verified-wikipedia-data",
                        "container_path": "/data",
                    }
                ],
            },
        },
    }

    base_path = tmp_path / "instances.base.json"
    config_path = tmp_path / "scale.yml"
    base_path.write_text(json.dumps(base_config, indent=2) + "\n")
    config_path.write_text(json.dumps(scale_config))

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "generate_compose_scale.py"
    host_config_path = _write_host_config(
        tmp_path,
        advertise_host="198.51.100.24",
        bind_host="10.0.0.5",
        db_bind_host="10.0.0.6",
        allow_public_web_bind=False,
        allow_public_db_bind=False,
    )

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--config",
            str(config_path),
            "--base-config",
            str(base_path),
            "--host-config",
            str(host_config_path),
            "--out-dir",
            str(tmp_path),
        ],
        check=True,
        cwd=repo_root,
    )

    compose = yaml.safe_load((tmp_path / "compose.scale.yml").read_text())
    shopping_ports = compose["services"]["shopping_0"]["ports"]
    assert "10.0.0.6:3306:3306" in shopping_ports
    assert "10.0.0.5:7770:80" in shopping_ports

    map_service_volumes = compose["services"]["map_0"]["volumes"]
    assert "webarena-verified-map-tile-db:/data/database:ro" in map_service_volumes
    assert "webarena-verified-map-tiles:/data/tiles:ro" in map_service_volumes

    volumes = compose["volumes"]
    assert volumes["webarena-verified-map-tile-db"] == {
        "external": True,
        "name": "webarena-verified-map-tile-db",
    }
    assert volumes["webarena-verified-map-tiles"] == {
        "external": True,
        "name": "webarena-verified-map-tiles",
    }
    assert volumes["webarena-verified-envs_webarena-verified-wikipedia-data"] == {
        "external": True,
        "name": "webarena-verified-envs_webarena-verified-wikipedia-data",
    }


def test_generate_compose_scale_requires_explicit_host_contract(tmp_path: Path) -> None:
    base_config = {
        "benchmark_name": "WebArena Verified",
        "benchmark_codebase": str(tmp_path),
        "instances": [
            {
                "site_name": "shopping",
                "site_url": "http://old-host:7770",
                "reset_endpoint": "http://old-host:7771/init",
            }
        ],
    }
    scale_config = {
        "sites": {
            "shopping": {
                "image": "example/shopping@sha256:abcd",
                "replicas": 1,
                "real_port_base": 7770,
                "container_web_port": 80,
            }
        }
    }

    base_path = tmp_path / "instances.base.json"
    config_path = tmp_path / "scale.yml"
    base_path.write_text(json.dumps(base_config, indent=2) + "\n")
    config_path.write_text(json.dumps(scale_config))

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "generate_compose_scale.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--config",
            str(config_path),
            "--base-config",
            str(base_path),
            "--out-dir",
            str(tmp_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "provide --host-config or --advertise-host" in completed.stderr


def test_generate_compose_scale_rejects_unapproved_public_binds_on_raw_cli(
    tmp_path: Path,
) -> None:
    base_config = {
        "benchmark_name": "WebArena Verified",
        "benchmark_codebase": str(tmp_path),
        "instances": [
            {
                "site_name": "shopping",
                "site_url": "http://old-host:7770",
                "reset_endpoint": "http://old-host:7771/init",
            }
        ],
    }
    scale_config = {
        "sites": {
            "shopping": {
                "image": "example/shopping@sha256:abcd",
                "replicas": 1,
                "real_port_base": 7770,
                "container_web_port": 80,
            }
        }
    }

    base_path = tmp_path / "instances.base.json"
    config_path = tmp_path / "scale.yml"
    base_path.write_text(json.dumps(base_config, indent=2) + "\n")
    config_path.write_text(json.dumps(scale_config))

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "generate_compose_scale.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--config",
            str(config_path),
            "--base-config",
            str(base_path),
            "--advertise-host",
            "203.0.113.10",
            "--bind-host",
            "0.0.0.0",
            "--db-bind-host",
            "0.0.0.0",
            "--out-dir",
            str(tmp_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "allow_public_web_bind=true" in completed.stderr
