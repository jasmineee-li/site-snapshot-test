"""generate_compose_scale.py - Emit docker-compose + proxy + instances config
from scripts/scale_config.yml for the r5.8xlarge 30-replica scale-out.

Reads a canonical scale_config.yml (replica counts, port bases, volumes) and
produces:
  - <out_dir>/compose.scale.yml        - all replicas per scale_config
  - <out_dir>/compose.smoke.yml        - 1-per-site smoke-test subset
  - <out_dir>/proxy_ports.conf         - nginx port map
  - <out_dir>/instances.json.fragment  - JSON array for instances.json

Port scheme:
  real_web_port    = site.real_port_base + i * port_step
  real_envctrl_port = real_web_port + 1
  proxy_port       = real_web_port + proxy_port_offset

Volume names:
  shared RO:     as declared (shared by every replica, external: true)
  per-replica:   <volume_name_prefix>_<i>

Service / container names:
  service:     <site>_<i>            (e.g. shopping_0, gitlab_3)
  container:   webarena-verified-<site>_<i>

Usage:
  python scripts/generate_compose_scale.py \
      --config scripts/scale_config.yml \
      --host-ip 1.2.3.4 \
      --proxy-token ab12... \
      --mode scale \
      --out-dir /tmp

Use --mode smoke for the Phase C one-per-site compose.

Requires PyYAML (already available in the project env).
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import pathlib
import sys
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Replica expansion
# ---------------------------------------------------------------------------


def expand_site(
    site_name: str,
    site_cfg: dict[str, Any],
    replica_count: int,
    host_ip: str,
    proxy_port_offset: int,
    ip_allocator,
) -> list[dict[str, Any]]:
    """Expand one site into a list of per-replica record dicts.

    Returned records carry everything downstream emitters need:
      - service_name / container_name
      - image / platform / mem_limit / shm_size
      - ports (list of "127.0.0.1:HOST:CONTAINER" strings)
      - environment (list of "K=V" strings)
      - volumes (list of compose volume-string entries)
      - ipv4_address (on the shared bridge network)
      - real_web_port / real_envctrl_port / proxy_port / db_port
    """
    out: list[dict[str, Any]] = []
    real_base = int(site_cfg["real_port_base"])
    port_step = int(site_cfg.get("port_step", 10))
    container_web_port = int(site_cfg["container_web_port"])
    db_port_base = site_cfg.get("db_port_base")
    db_port_step = int(site_cfg.get("db_port_step", 1))

    for i in range(replica_count):
        real_web = real_base + i * port_step
        real_envctrl = real_web + 1
        proxy_port = real_web + proxy_port_offset

        service_name = f"{site_name}_{i}"
        container_name = f"webarena-verified-{site_name}_{i}"

        # Port mappings: 127.0.0.1-only on the host; web + envctrl always.
        ports: list[str] = [
            f"127.0.0.1:{real_web}:{container_web_port}",
            f"127.0.0.1:{real_envctrl}:8877",
        ]
        db_port = None
        if db_port_base is not None:
            db_port = int(db_port_base) + i * db_port_step
            # Postgres-based images expose 5432, MySQL 3306. shopping /
            # shopping_admin are MySQL; gitlab/reddit/map are PostgreSQL.
            if site_name in ("shopping", "shopping_admin"):
                container_db_port = 3306
            else:
                container_db_port = 5432
            ports.append(f"127.0.0.1:{db_port}:{container_db_port}")

        # Environment: always inject WA_ENV_CTRL_EXTERNAL_SITE_URL per replica.
        env_list: list[str] = [
            f"WA_ENV_CTRL_EXTERNAL_SITE_URL=http://{host_ip}:{real_web}",
        ]
        # Copy any per-site extras declared in the config.
        for entry in site_cfg.get("environment", []) or []:
            if isinstance(entry, str):
                env_list.append(entry)
            elif isinstance(entry, dict):
                for k, v in entry.items():
                    env_list.append(f"{k}={v}")

        # Volumes.
        volumes: list[str] = []
        for shared in site_cfg.get("shared_ro_volumes", []) or []:
            volumes.append(f"{shared['volume_name']}:{shared['container_path']}:ro")
        for per in site_cfg.get("per_replica_volumes", []) or []:
            vol_name = f"{per['volume_name_prefix']}_{i}"
            mode = per.get("mode", "rw")
            suffix = f":{mode}" if mode != "rw" else ""
            volumes.append(f"{vol_name}:{per['container_path']}{suffix}")

        # gitlab.rb bind-mount (tuned settings for memory savings).
        if "gitlab_rb_host_path" in site_cfg:
            volumes.append(
                f"{site_cfg['gitlab_rb_host_path']}:{site_cfg['gitlab_rb_container_path']}:ro"
            )

        out.append(
            {
                "site_name": site_name,
                "replica_index": i,
                "service_name": service_name,
                "container_name": container_name,
                "image": site_cfg["image"],
                "platform": site_cfg.get("platform"),
                "mem_limit": site_cfg.get("mem_limit"),
                "shm_size": site_cfg.get("shm_size"),
                "ports": ports,
                "environment": env_list,
                "volumes": volumes,
                "ipv4_address": str(next(ip_allocator)),
                "real_web_port": real_web,
                "real_envctrl_port": real_envctrl,
                "proxy_port": proxy_port,
                "db_port": db_port,
                "container_web_port": container_web_port,
            }
        )

    return out


# ---------------------------------------------------------------------------
# Compose emission
# ---------------------------------------------------------------------------


def build_compose(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    network_name: str,
    subnet: str,
) -> dict[str, Any]:
    """Assemble the full compose document as a plain dict (ready for YAML)."""
    services: dict[str, Any] = {}

    # Collect declared shared + per-replica volume names by scanning the
    # records (which have already resolved per-replica suffixes).
    shared_names: set = set()
    per_replica_names: set = set()
    for site_name, site_cfg in config["sites"].items():
        for shared in site_cfg.get("shared_ro_volumes", []) or []:
            shared_names.add(shared["volume_name"])

    for rec in records:
        for vol_entry in rec["volumes"]:
            # Bind mounts (start with /) are not named volumes.
            if vol_entry.startswith("/"):
                continue
            vol_name = vol_entry.split(":", 1)[0]
            if vol_name in shared_names:
                continue
            # gitlab.rb bind mount uses an absolute path (handled above).
            per_replica_names.add(vol_name)

    for rec in records:
        svc: dict[str, Any] = {
            "image": rec["image"],
            "container_name": rec["container_name"],
            "ports": rec["ports"],
            "environment": rec["environment"],
            "restart": "unless-stopped",
            "networks": {
                network_name: {"ipv4_address": rec["ipv4_address"]},
            },
        }
        if rec["platform"]:
            svc["platform"] = rec["platform"]
        if rec["mem_limit"]:
            svc["mem_limit"] = rec["mem_limit"]
        if rec["shm_size"]:
            svc["shm_size"] = rec["shm_size"]
        if rec["volumes"]:
            svc["volumes"] = rec["volumes"]
        services[rec["service_name"]] = svc

    # Networks.
    networks = {
        network_name: {
            "driver": "bridge",
            "ipam": {"config": [{"subnet": subnet}]},
        }
    }

    # Volumes block. Shared RO volumes are declared external (pre-hydrated by
    # Phase B.2). Per-replica volumes are declared with explicit `name:` so
    # compose does not prefix them with the project name.
    volumes_section: dict[str, Any] = {}
    for name in sorted(shared_names):
        volumes_section[name] = {"external": True, "name": name}
    for name in sorted(per_replica_names):
        volumes_section[name] = {"name": name}

    return {
        "services": services,
        "networks": networks,
        "volumes": volumes_section,
    }


# ---------------------------------------------------------------------------
# Proxy ports + instances fragment
# ---------------------------------------------------------------------------


def build_proxy_ports(records: list[dict[str, Any]]) -> str:
    """Render proxy_ports.conf text. One line per replica.

    Format matches scripts/proxy_ports.conf: `name:real_port[:proxy_port]`.
    We emit the explicit proxy_port to avoid relying on the default offset.
    """
    lines: list[str] = [
        "# proxy_ports.conf - auto-generated by generate_compose_scale.py",
        "# Do not edit by hand; regenerate from scripts/scale_config.yml.",
        "#",
        "# Format: service_name:real_port:proxy_port",
        "",
    ]
    for rec in records:
        lines.append(f"{rec['service_name']}:{rec['real_web_port']}:{rec['proxy_port']}")
    return "\n".join(lines) + "\n"


def build_instances_fragment(
    records: list[dict[str, Any]],
    host_ip: str,
    proxy_token: str,
) -> list[dict[str, Any]]:
    """Produce BenchmarkInstance entries matching the instances.json schema.

    Each entry has:
      - site_name (site_<i> to disambiguate replicas)
      - site_url: proxy URL (the orchestrator never hits real ports directly)
      - reset_endpoint: proxy URL + /init
      - db_connection: per-site DB URL, only for sites with db_port_base
    """
    # Default credentials per-site mirror what's in instances.json today.
    creds = {
        "shopping": ("magentouser", "MyPassword", "magentodb", "mysql"),
        "shopping_admin": ("magentouser", "MyPassword", "magentodb", "mysql"),
        "gitlab": ("gitlab", "", "gitlabhq_production", "postgresql"),
        "reddit": ("postmill", "postmill", "postmill", "postgresql"),
        "map": ("renderer", "renderer", "gis", "postgresql"),
    }
    out: list[dict[str, Any]] = []
    for rec in records:
        entry: dict[str, Any] = {
            "site_name": rec["service_name"],
            "base_site_name": rec["site_name"],
            "replica_index": rec["replica_index"],
            "site_url": f"http://{host_ip}:{rec['proxy_port']}",
            "reset_endpoint": f"http://{host_ip}:{rec['proxy_port'] + 1}/init",
        }
        if rec["db_port"] and rec["site_name"] in creds:
            user, pw, db, scheme = creds[rec["site_name"]]
            auth_part = f"{user}:{pw}@" if pw else f"{user}@"
            entry["db_connection"] = f"{scheme}://{auth_part}{host_ip}:{rec['db_port']}/{db}"
        out.append(entry)
    # NOTE: reset_endpoint maps to proxy_port+1 follows plan section 2's
    # convention that envctrl = web + 1. The nginx proxy must listen on
    # proxy_port (web) AND proxy_port+1 (envctrl). build_proxy_ports emits
    # only web; nginx listens on both because the proxy config is generated
    # from real_port pairs - the env-ctrl line is added here so reviewers
    # notice. If deploy_benchmark_proxy.sh only listens on web ports, adjust.
    return out


# ---------------------------------------------------------------------------
# IP allocator
# ---------------------------------------------------------------------------


def make_ip_allocator(subnet: str, start_offset: int = 10):
    """Return an iterator that yields IPv4Address entries in the given subnet.

    Reserves the first `start_offset` addresses (network + gateway + docker
    bookkeeping). 30 replicas fit trivially in a /20.
    """
    net = ipaddress.ip_network(subnet, strict=False)
    hosts = net.hosts()
    for _ in range(start_offset):
        next(hosts)
    return hosts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument("--config", required=True, help="path to scale_config.yml")
    ap.add_argument("--host-ip", required=True, help="host IP for site URLs")
    ap.add_argument("--proxy-token", required=True, help="proxy token for instances.json")
    ap.add_argument(
        "--mode",
        choices=["scale", "smoke"],
        default="scale",
        help="scale = full replica counts; smoke = smoke_test_replicas",
    )
    ap.add_argument(
        "--out-dir",
        default="/tmp",
        help="directory to write compose + proxy_ports + instances fragment",
    )
    args = ap.parse_args()

    cfg_path = pathlib.Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2
    with cfg_path.open() as f:
        config = yaml.safe_load(f)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    network_name = config.get("network", {}).get("name", "worldsim-bench")
    subnet = config.get("network", {}).get("subnet", "172.20.0.0/20")
    proxy_offset = int(config.get("proxy_port_offset", 10000))

    # Decide replica counts for this mode.
    if args.mode == "scale":
        replica_counts = {name: int(sc["replicas"]) for name, sc in config["sites"].items()}
        compose_filename = "compose.scale.yml"
    else:
        smoke = config.get("smoke_test_replicas", {})
        replica_counts = {name: int(smoke.get(name, 1)) for name in config["sites"]}
        compose_filename = "compose.smoke.yml"

    ip_alloc = make_ip_allocator(subnet)

    records: list[dict[str, Any]] = []
    for site_name, site_cfg in config["sites"].items():
        count = replica_counts.get(site_name, 0)
        if count <= 0:
            continue
        records.extend(
            expand_site(
                site_name=site_name,
                site_cfg=site_cfg,
                replica_count=count,
                host_ip=args.host_ip,
                proxy_port_offset=proxy_offset,
                ip_allocator=ip_alloc,
            )
        )

    compose_doc = build_compose(config, records, network_name, subnet)

    compose_path = out_dir / compose_filename
    with compose_path.open("w") as f:
        yaml.safe_dump(compose_doc, f, sort_keys=False, default_flow_style=False, width=200)

    proxy_path = out_dir / "proxy_ports.conf"
    proxy_path.write_text(build_proxy_ports(records))

    fragment = build_instances_fragment(records, args.host_ip, args.proxy_token)
    frag_path = out_dir / "instances.json.fragment"
    with frag_path.open("w") as f:
        json.dump(fragment, f, indent=2)
        f.write("\n")

    print(f"wrote {compose_path} ({len(records)} services)")
    print(f"wrote {proxy_path} ({len(records)} port entries)")
    print(f"wrote {frag_path} ({len(fragment)} instance entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
