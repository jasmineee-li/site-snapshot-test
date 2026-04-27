#!/usr/bin/env python3
"""Render checked-in host config as shell-compatible environment exports."""

from __future__ import annotations

import argparse
import shlex
import sys

from worldsim.host_config import load_host_config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host-config", required=True, help="path to benchmark host config YAML")
    args = ap.parse_args()

    cfg = load_host_config(args.host_config)
    values = {
        "HOST_IP": cfg.advertise_host,
        "ORCHESTRATOR_HOST": cfg.orchestrator_host,
        "SSH_USER": cfg.ssh_user,
        "COMPOSE_DIR_REMOTE": cfg.compose_dir_remote,
        "COMPOSE_FILE_REMOTE": cfg.compose_file_remote,
        "WORLDSIM_ADVERTISE_HOST": cfg.advertise_host,
        "WORLDSIM_ORCHESTRATOR_HOST": cfg.orchestrator_host,
        "WORLDSIM_BIND_HOST": cfg.bind_host,
        "WORLDSIM_DB_BIND_HOST": cfg.db_bind_host,
        "WORLDSIM_TRUSTED_OPERATOR_CIDRS": ",".join(cfg.trusted_operator_cidrs),
    }
    for key, value in values.items():
        print(f"{key}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
