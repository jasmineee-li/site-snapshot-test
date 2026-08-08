#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST_CONFIG="$REPO_ROOT/configs/benchmark_hosts/r8a.yaml"

"$REPO_ROOT/scripts/audit_r8a_control_plane.sh" --host-config "$HOST_CONFIG"
exec "$REPO_ROOT/scripts/deploy_benchmark_proxy.sh" \
  --host-config "$HOST_CONFIG" \
  --topology scale \
  --insecure-http \
  "$@"
