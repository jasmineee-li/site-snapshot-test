"""generate_compose_scale.py - Emit docker-compose + proxy + instances config
from scripts/scale_config.yml for the r5.8xlarge 30-replica scale-out.

Reads a canonical scale_config.yml (replica counts, port bases, volumes) plus a
canonical base instances config and produces:
  - <out_dir>/compose.scale.yml        - all replicas per scale_config
  - <out_dir>/compose.smoke.yml        - 1-per-site smoke-test subset
  - <out_dir>/proxy_ports.conf         - nginx port map for Phase 0c only
  - <out_dir>/instances.json           - full runtime config for Phases 0c/3/4
  - <out_dir>/instances.json.fragment  - instances array only

Port scheme:
  real_web_port     = site.real_port_base + i * port_step
  real_envctrl_port = real_web_port + 1
  proxy_port        = real_web_port + proxy_port_offset

Important contract:
  - runtime site_url/reset_endpoint always use the real evaluation ports
  - verification_proxy is Phase 0c-only and never replaces runtime URLs
  - site_name stays canonical (shopping, gitlab, ...) for routing
  - replica identity lives in replica_index / replica_name

Usage:
  python scripts/generate_compose_scale.py \
      --config scripts/scale_config.yml \
      --base-config instances.json \
      --host-config configs/benchmark_hosts/r5.yaml \
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
from urllib.parse import urlsplit, urlunsplit

import yaml

from worldsim.config import BenchmarkConfig
from worldsim.host_config import BenchmarkHostConfig, load_host_config

PVPO_CDP_PORT_BASE = 9222
from worldsim.placeholders import placeholder_for_site

# ---------------------------------------------------------------------------
# Replica expansion
# ---------------------------------------------------------------------------


def expand_site(
    site_name: str,
    site_cfg: dict[str, Any],
    replica_count: int,
    orchestrator_host: str,
    bind_host: str,
    db_bind_host: str,
    proxy_port_offset: int,
    ip_allocator,
) -> list[dict[str, Any]]:
    """Expand one site into a list of per-replica record dicts."""
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

        ports: list[str] = [
            f"{bind_host}:{real_web}:{container_web_port}",
            f"{bind_host}:{real_envctrl}:8877",
        ]
        db_port = None
        if db_port_base is not None:
            db_port = int(db_port_base) + i * db_port_step
            # Defensive: keep the legacy site-name → container-DB-port map
            # so generator unit tests (which use synthetic fixtures with
            # shopping/etc.) still exercise the multi-site code path. In
            # production after the 2026-04-21 WASP-aligned scope all
            # surviving sites are PostgreSQL (5432).
            container_db_port = 3306 if site_name in ("shopping", "shopping_admin") else 5432
            ports.append(f"{db_bind_host}:{db_port}:{container_db_port}")

        # Bake the PROXY port (real_web + proxy_port_offset) into
        # WA_ENV_CTRL_EXTERNAL_SITE_URL, not the raw container port. Phase 4's
        # _reset_task_environment POSTs to reset_endpoint before every task,
        # which triggers env-ctrl's _init(). Baking the proxy port makes
        # _init() idempotent with what the proxy expects. When
        # proxy_port_offset == 0 (loopback / --no-verification-proxy mode)
        # this stays on the raw port, which is correct.
        env_site_port = real_web + proxy_port_offset if proxy_port_offset else real_web
        # env-ctrl reports this URL in _init() responses. Callers (orchestrator,
        # chrome containers) navigate to it, so it must match what they can
        # reach — orchestrator_host, not the external advertise_host.
        env_list: list[str] = [
            f"WA_ENV_CTRL_EXTERNAL_SITE_URL=http://{orchestrator_host}:{env_site_port}",
        ]
        for entry in site_cfg.get("environment", []) or []:
            if isinstance(entry, str):
                env_list.append(entry)
            elif isinstance(entry, dict):
                for k, v in entry.items():
                    env_list.append(f"{k}={v}")

        volumes: list[str] = []
        for shared in site_cfg.get("shared_ro_volumes", []) or []:
            volumes.append(f"{shared['volume_name']}:{shared['container_path']}:ro")
        for per in site_cfg.get("per_replica_volumes", []) or []:
            vol_name = f"{per['volume_name_prefix']}_{i}"
            mode = per.get("mode", "rw")
            suffix = f":{mode}" if mode != "rw" else ""
            volumes.append(f"{vol_name}:{per['container_path']}{suffix}")

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

    shared_names: set[str] = set()
    per_replica_names: set[str] = set()
    for site_cfg in config["sites"].values():
        for shared in site_cfg.get("shared_ro_volumes", []) or []:
            shared_names.add(shared["volume_name"])

    for rec in records:
        for vol_entry in rec["volumes"]:
            if vol_entry.startswith("/"):
                continue
            vol_name = vol_entry.split(":", 1)[0]
            if vol_name not in shared_names:
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

    networks = {
        network_name: {
            "driver": "bridge",
            "ipam": {"config": [{"subnet": subnet}]},
        }
    }

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
# Proxy ports + instances config
# ---------------------------------------------------------------------------


def build_proxy_ports(records: list[dict[str, Any]]) -> str:
    """Render proxy_ports.conf text for the Phase 0c verification proxy.

    Emits two listener entries per replica:

    * ``{service_name}:{real_web_port}:{proxy_port}`` — the web endpoint
      the benign/adversarial agent browses (site_url).
    * ``{service_name}_envctrl:{real_envctrl_port}:{proxy_port+1}`` — the
      env-ctrl ``/init`` endpoint Phase 2c / Phase 4 POST to before each
      task (reset_endpoint). Without this listener nginx does not front
      envctrl, docker-compose binds it to loopback, and every
      feasibility test times out on the reset call.

    The envctrl proxy port follows ``proxy_port + 1`` (equivalent to
    ``real_envctrl + proxy_port_offset``), which keeps the convention
    ``proxy = real + offset`` intact across both endpoint types.
    """
    lines: list[str] = [
        "# proxy_ports.conf - auto-generated by generate_compose_scale.py",
        "# Do not edit by hand; regenerate from scripts/scale_config.yml.",
        "#",
        "# Format: service_name:real_port:proxy_port",
        "# Two entries per replica: the web endpoint and the env-ctrl /init",
        "# endpoint (service_name_envctrl) at real_envctrl_port == real_web+1.",
        "",
    ]
    for rec in records:
        lines.append(f"{rec['service_name']}:{rec['real_web_port']}:{rec['proxy_port']}")
        lines.append(
            f"{rec['service_name']}_envctrl:{rec['real_envctrl_port']}:{rec['proxy_port'] + 1}"
        )
    return "\n".join(lines) + "\n"


def _replace_url_host_port(raw_url: str, *, host_ip: str, port: int) -> str:
    parts = urlsplit(raw_url)
    username = parts.username
    password = parts.password
    userinfo = ""
    if username:
        userinfo = username
        if password:
            userinfo += f":{password}"
        userinfo += "@"
    host_display = f"[{host_ip}]" if ":" in host_ip else host_ip
    return urlunsplit(
        (
            parts.scheme,
            f"{userinfo}{host_display}:{port}",
            parts.path,
            parts.query,
            parts.fragment,
        )
    )


def _generated_placeholder_map(records: list[dict[str, Any]], host_ip: str) -> dict[str, str]:
    placeholders: dict[str, str] = {}
    for rec in records:
        if rec["replica_index"] != 0:
            continue
        token = placeholder_for_site(rec["site_name"])
        if token:
            placeholders[token] = f"http://{host_ip}:{rec['real_web_port']}"
    return placeholders


def _base_instances_by_site(base_instances: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_site: dict[str, dict[str, Any]] = {}
    for instance in base_instances:
        site_name = str(instance.get("site_name", "")).strip()
        if not site_name:
            raise ValueError("base config instance missing site_name")
        if site_name in by_site:
            raise ValueError(
                f"base config contains multiple canonical instances for site {site_name!r}"
            )
        by_site[site_name] = instance
    return by_site


def _default_validation_endpoint(site_name: str) -> str | None:
    # Defensive: shopping/shopping_admin entries are kept here for
    # backward-compat with generator unit tests that exercise multi-site
    # fixtures. Production sites after the 2026-04-21 WASP-aligned scope
    # are gitlab + reddit; only gitlab gets a validation endpoint.
    if site_name in {"shopping", "shopping_admin"}:
        return "/rest/V1/modules"
    if site_name == "gitlab":
        return "/api/v4/user"
    return None


def _clone_instance(
    base_instance: dict[str, Any],
    rec: dict[str, Any],
    *,
    orchestrator_host: str,
    placeholder_map: dict[str, str],
    pvpo_cdp_port: int,
) -> dict[str, Any]:
    instance = json.loads(json.dumps(base_instance))
    instance["site_name"] = rec["site_name"]
    # All three runtime URLs use orchestrator_host — the address the
    # orchestrator and its browser containers actually route to. For on-host
    # topologies this is a Docker bridge / loopback address; for remote
    # orchestrators it's the public IP. Never the external advertise_host
    # alone, because AWS VPCs don't hairpin EIP-to-self traffic even on
    # SG-open ports.
    instance["site_url"] = f"http://{orchestrator_host}:{rec['real_web_port']}"
    instance["reset_endpoint"] = f"http://{orchestrator_host}:{rec['real_envctrl_port']}/init"
    instance["replica_index"] = rec["replica_index"]
    instance["replica_name"] = rec["service_name"]

    db_connection = instance.get("db_connection")
    if isinstance(db_connection, str) and db_connection.strip() and rec["db_port"] is not None:
        instance["db_connection"] = _replace_url_host_port(
            db_connection,
            host_ip=orchestrator_host,
            port=int(rec["db_port"]),
        )

    existing_placeholders = instance.get("url_placeholders")
    merged_placeholders = (
        dict(existing_placeholders) if isinstance(existing_placeholders, dict) else {}
    )
    merged_placeholders.update(placeholder_map)
    if merged_placeholders:
        instance["url_placeholders"] = merged_placeholders
    instance["pvpo_cdp_url"] = f"http://127.0.0.1:{pvpo_cdp_port}"
    api_auth = instance.get("api_auth")
    if (
        isinstance(api_auth, dict)
        and str(api_auth.get("type", "")).strip() == "bearer_token"
        and isinstance(api_auth.get("token_endpoint"), str)
        and api_auth["token_endpoint"].strip()
        and not (
            isinstance(api_auth.get("validation_endpoint"), str)
            and api_auth["validation_endpoint"].strip()
        )
    ):
        validation_endpoint = _default_validation_endpoint(rec["site_name"])
        if validation_endpoint is not None:
            api_auth["validation_endpoint"] = validation_endpoint

    return instance


def build_instances_config(
    *,
    base_config: dict[str, Any],
    records: list[dict[str, Any]],
    host_access: dict[str, Any],
    orchestrator_host: str,
    proxy_token: str | None,
    proxy_port_offset: int,
    proxy_scheme: str | None,
    emit_verification_proxy: bool = True,
) -> dict[str, Any]:
    """Produce a full BenchmarkConfig with scaled runtime instances.

    When ``emit_verification_proxy=False``, the ``verification_proxy`` block
    is stripped from the output. Use this for loopback dev stacks where no
    nginx proxy is running; otherwise ``magento_health.py`` probes a
    non-existent proxy port and aborts Phase 4 startup.
    """
    output = json.loads(json.dumps(base_config))
    output["host_access"] = dict(host_access)
    base_instances = output.get("instances")
    if not isinstance(base_instances, list) or not base_instances:
        raise ValueError("base config must contain a non-empty instances array")

    by_site = _base_instances_by_site(base_instances)
    # url_placeholders feed directly into task prompts and agent navigation.
    # They must match site_url — which now uses orchestrator_host — or the
    # agent will try to navigate to an unreachable host.
    placeholder_map = _generated_placeholder_map(records, orchestrator_host)

    instances: list[dict[str, Any]] = []
    instance_benchmark = output.get("benchmark_name") or output.get("benchmark")
    for offset, rec in enumerate(records):
        site_name = rec["site_name"]
        base_instance = by_site.get(site_name)
        if base_instance is None:
            raise ValueError(f"scale config site {site_name!r} missing from base config instances")
        instance = _clone_instance(
            base_instance,
            rec,
            orchestrator_host=orchestrator_host,
            placeholder_map=placeholder_map,
            pvpo_cdp_port=PVPO_CDP_PORT_BASE + offset,
        )
        if isinstance(instance_benchmark, str) and instance_benchmark.strip():
            instance["benchmark_name"] = instance_benchmark.strip()
        instances.append(instance)

    if emit_verification_proxy:
        verification_proxy = output.get("verification_proxy")
        if not isinstance(verification_proxy, dict):
            verification_proxy = {}
        if proxy_token is not None:
            verification_proxy["token"] = proxy_token
        verification_proxy["port_offset"] = proxy_port_offset
        if proxy_scheme is not None:
            verification_proxy["scheme"] = proxy_scheme
        if verification_proxy:
            output["verification_proxy"] = verification_proxy
    else:
        output.pop("verification_proxy", None)

    top_level_placeholders = output.get("url_placeholders")
    merged_top_level = (
        dict(top_level_placeholders) if isinstance(top_level_placeholders, dict) else {}
    )
    merged_top_level.update(placeholder_map)
    output["url_placeholders"] = merged_top_level
    output["instances"] = instances

    validated = BenchmarkConfig.model_validate(output)
    return validated.model_dump(mode="json")


# ---------------------------------------------------------------------------
# IP allocator
# ---------------------------------------------------------------------------


def make_ip_allocator(subnet: str, start_offset: int = 10):
    """Return an iterator that yields IPv4Address entries in the given subnet."""
    net = ipaddress.ip_network(subnet, strict=False)
    hosts = net.hosts()
    for _ in range(start_offset):
        next(hosts)
    return hosts


def _unpinned_images(config: dict[str, Any]) -> list[str]:
    unpinned: list[str] = []
    for site_name, site_cfg in config.get("sites", {}).items():
        image = str(site_cfg.get("image", "")).strip()
        if not image:
            continue
        if "@sha256:" not in image:
            unpinned.append(f"{site_name}:{image}")
    return unpinned


def _resolve_host_contract(
    *,
    host_config_path: str | None,
    advertise_host: str | None,
    bind_host: str | None,
    db_bind_host: str | None,
    allow_public_web_bind: bool,
    allow_public_db_bind: bool,
) -> BenchmarkHostConfig:
    if host_config_path:
        host_cfg = load_host_config(host_config_path)
        if advertise_host is None:
            advertise_host = host_cfg.advertise_host
        if bind_host is None:
            bind_host = host_cfg.bind_host
        if db_bind_host is None:
            db_bind_host = host_cfg.db_bind_host
    else:
        if advertise_host is None:
            raise ValueError("provide --host-config or --advertise-host")
        if bind_host is None:
            raise ValueError("provide --host-config or --bind-host")
        if db_bind_host is None:
            db_bind_host = bind_host
        host_cfg = BenchmarkHostConfig.model_validate(
            {
                "name": "cli",
                "access_mode": "remote_direct_restricted",
                "advertise_host": advertise_host,
                "bind_host": bind_host,
                "db_bind_host": db_bind_host,
                "allow_public_web_bind": allow_public_web_bind,
                "allow_public_db_bind": allow_public_db_bind,
            }
        )
    return host_cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument("--config", required=True, help="path to scale_config.yml")
    ap.add_argument(
        "--base-config",
        required=True,
        help="canonical instances.json to fan out across replicas",
    )
    ap.add_argument("--host-config", help="checked-in benchmark host config YAML")
    ap.add_argument("--advertise-host", help="host/IP for runtime URLs and site URLs")
    ap.add_argument(
        "--bind-host", help="host/interface Docker should publish web/env-ctrl ports on"
    )
    ap.add_argument("--db-bind-host", help="host/interface Docker should publish DB ports on")
    ap.add_argument(
        "--allow-public-web-bind",
        action="store_true",
        help="explicitly allow publishing web/env-ctrl ports on all host interfaces",
    )
    ap.add_argument(
        "--allow-public-db-bind",
        action="store_true",
        help="explicitly allow publishing DB ports on all host interfaces",
    )
    ap.add_argument(
        "--proxy-token",
        default=None,
        help="optional proxy token override for instances.json",
    )
    ap.add_argument(
        "--proxy-scheme",
        choices=["http", "https"],
        default=None,
        help="optional verification_proxy.scheme override",
    )
    ap.add_argument(
        "--no-verification-proxy",
        action="store_true",
        help=(
            "omit verification_proxy block and bake raw-port URLs throughout "
            "(loopback / dev stacks without an nginx proxy). Auto-enabled when "
            "orchestrator_host resolves to a loopback address."
        ),
    )
    ap.add_argument(
        "--mode",
        choices=["scale", "smoke"],
        default="scale",
        help="scale = full replica counts; smoke = smoke_test_replicas",
    )
    ap.add_argument(
        "--require-pinned-images",
        action="store_true",
        help="fail if any site image uses a mutable ref like :latest instead of a digest",
    )
    ap.add_argument(
        "--out-dir",
        default="/tmp",
        help="directory to write compose + proxy_ports + instances config",
    )
    args = ap.parse_args()

    cfg_path = pathlib.Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2
    with cfg_path.open() as f:
        config = yaml.safe_load(f)

    unpinned = _unpinned_images(config)
    if unpinned:
        message = "Unpinned benchmark images detected: " + ", ".join(unpinned)
        if args.require_pinned_images:
            print(f"ERROR: {message}", file=sys.stderr)
            return 2
        print(f"WARNING: {message}", file=sys.stderr)

    base_config_path = pathlib.Path(args.base_config)
    if not base_config_path.exists():
        print(f"ERROR: base config not found: {base_config_path}", file=sys.stderr)
        return 2
    with base_config_path.open() as f:
        base_config = json.load(f)

    try:
        host_cfg = _resolve_host_contract(
            host_config_path=args.host_config,
            advertise_host=args.advertise_host,
            bind_host=args.bind_host,
            db_bind_host=args.db_bind_host,
            allow_public_web_bind=args.allow_public_web_bind,
            allow_public_db_bind=args.allow_public_db_bind,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    network_name = config.get("network", {}).get("name", "worldsim-bench")
    subnet = config.get("network", {}).get("subnet", "172.20.0.0/20")
    proxy_offset = int(config.get("proxy_port_offset", 10000))

    # Loopback orchestrator_host → no nginx proxy on that host. Auto-omit
    # the verification_proxy block and bake raw ports so reset_endpoint's
    # _init() and all other orchestrator traffic hit the same origin.
    loopback_hosts = {"127.0.0.1", "localhost", "0.0.0.0", "::1"}
    orchestrator_is_loopback = host_cfg.orchestrator_host.strip() in loopback_hosts
    emit_proxy = not (args.no_verification_proxy or orchestrator_is_loopback)
    if not emit_proxy:
        proxy_offset = 0
        reason = (
            "--no-verification-proxy"
            if args.no_verification_proxy
            else f"orchestrator_host={host_cfg.orchestrator_host!r} is loopback"
        )
        print(f"verification_proxy omitted: {reason}", file=sys.stderr)

    # Banner: surface the resolved hosts so the operator catches a wrong
    # value before downstream consumers see it.
    print(
        f"advertise_host={host_cfg.advertise_host} "
        f"orchestrator_host={host_cfg.orchestrator_host} "
        f"bind_host={host_cfg.bind_host} "
        f"db_bind_host={host_cfg.db_bind_host} "
        f"proxy_port_offset={proxy_offset} "
        f"emit_verification_proxy={emit_proxy}",
        file=sys.stderr,
    )

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
                orchestrator_host=host_cfg.orchestrator_host,
                bind_host=host_cfg.bind_host,
                db_bind_host=host_cfg.db_bind_host,
                proxy_port_offset=proxy_offset,
                ip_allocator=ip_alloc,
            )
        )

    compose_doc = build_compose(config, records, network_name, subnet)
    instances_config = build_instances_config(
        base_config=base_config,
        records=records,
        host_access={
            "source": str(args.host_config or "cli"),
            "advertise_host": host_cfg.advertise_host,
            "orchestrator_host": host_cfg.orchestrator_host,
            "runtime_host": host_cfg.orchestrator_host,
            "bind_host": host_cfg.bind_host,
            "db_bind_host": host_cfg.db_bind_host,
        },
        orchestrator_host=host_cfg.orchestrator_host,
        proxy_token=args.proxy_token,
        proxy_port_offset=proxy_offset,
        proxy_scheme=args.proxy_scheme,
        emit_verification_proxy=emit_proxy,
    )

    compose_path = out_dir / compose_filename
    with compose_path.open("w") as f:
        yaml.safe_dump(compose_doc, f, sort_keys=False, default_flow_style=False, width=200)

    proxy_path = out_dir / "proxy_ports.conf"
    proxy_path.write_text(build_proxy_ports(records))

    config_path = out_dir / "instances.json"
    with config_path.open("w") as f:
        json.dump(instances_config, f, indent=2)
        f.write("\n")

    fragment_path = out_dir / "instances.json.fragment"
    with fragment_path.open("w") as f:
        json.dump(instances_config["instances"], f, indent=2)
        f.write("\n")

    print(f"wrote {compose_path} ({len(records)} services)")
    print(f"wrote {proxy_path} ({len(records)} port entries)")
    print(f"wrote {config_path} ({len(instances_config['instances'])} instance entries)")
    print(f"wrote {fragment_path} ({len(instances_config['instances'])} instance entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
