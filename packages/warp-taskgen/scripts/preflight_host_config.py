#!/usr/bin/env python3
"""Fail-closed validation for benchmark host deployment contracts."""

from __future__ import annotations

import argparse
import sys

from warp_taskgen.host_config import BenchmarkHostConfig, _is_loopback_host, load_host_config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host-config", help="path to benchmark host config YAML")
    ap.add_argument("--access-mode", default="remote_direct_restricted")
    ap.add_argument("--advertise-host", help="host/IP for runtime URLs and advertised site URLs")
    ap.add_argument(
        "--bind-host", help="host/interface Docker should publish web/env-ctrl ports on"
    )
    ap.add_argument("--db-bind-host", help="host/interface Docker should publish DB ports on")
    ap.add_argument("--allow-public-web-bind", action="store_true")
    ap.add_argument("--allow-public-db-bind", action="store_true")
    ap.add_argument(
        "--require-remote-db",
        action="store_true",
        help="fail if the selected config keeps DB ports on a loopback-only bind",
    )
    args = ap.parse_args()

    try:
        if args.host_config:
            cfg = load_host_config(args.host_config)
        else:
            if not args.advertise_host or not args.bind_host:
                raise ValueError(
                    "provide --host-config or explicit --advertise-host/--bind-host values"
                )
            cfg = BenchmarkHostConfig.model_validate(
                {
                    "name": "cli",
                    "access_mode": args.access_mode,
                    "advertise_host": args.advertise_host,
                    "bind_host": args.bind_host,
                    "db_bind_host": args.db_bind_host or args.bind_host,
                    "allow_public_web_bind": args.allow_public_web_bind,
                    "allow_public_db_bind": args.allow_public_db_bind,
                }
            )
    except Exception as exc:  # pragma: no cover - surfaced to callers as CLI stderr
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.require_remote_db and _is_loopback_host(cfg.db_bind_host or ""):
        print(
            "ERROR: remote DB access is required but db_bind_host resolves to loopback-only",
            file=sys.stderr,
        )
        return 2

    print(f"host={cfg.name}")
    print(f"access_mode={cfg.access_mode}")
    print(f"advertise_host={cfg.advertise_host}")
    print(f"bind_host={cfg.bind_host}")
    print(f"db_bind_host={cfg.db_bind_host}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
