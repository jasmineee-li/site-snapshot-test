#!/usr/bin/env python3
"""Serve one configured benchmark instance's host-owned restore operation."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_restoration.daemon import RestoreDaemon


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--compose", required=True, type=Path)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument(
        "--readiness-timeout",
        type=float,
        default=300.0,
        help="Maximum seconds to wait for the recreated site (1-600).",
    )
    args = parser.parse_args()
    daemon = RestoreDaemon(
        instances_path=args.instances,
        compose_path=args.compose,
        instance_id=args.instance_id,
        socket_path=args.socket,
        state_dir=args.state_dir,
        readiness_timeout=args.readiness_timeout,
    )
    daemon.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
