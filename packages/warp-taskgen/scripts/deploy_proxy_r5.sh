#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST_CONFIG="$REPO_ROOT/configs/benchmark_hosts/r5.yaml"

exec "$REPO_ROOT/scripts/deploy_benchmark_proxy.sh" \
  --host-config "$HOST_CONFIG" \
  --topology scale \
  --insecure-http \
  "$@"
