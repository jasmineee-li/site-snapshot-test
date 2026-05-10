#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST_CONFIG="$REPO_ROOT/configs/benchmark_hosts/r5.yaml"
LOCAL_COMPOSE="$REPO_ROOT/scripts/docker-compose.scale.yml"
INSTANCES_SCALE="$REPO_ROOT/instances.scale.json"

"$REPO_ROOT/scripts/generate_scale_r5.sh"
uv run python "$REPO_ROOT/scripts/preflight_security_group.py" \
  --host-config "$HOST_CONFIG" \
  --instances "$INSTANCES_SCALE"
exec "$REPO_ROOT/scripts/bootstrap_ec2.sh" \
  --host-config "$HOST_CONFIG" \
  --local-compose-file "$LOCAL_COMPOSE" \
  "$@"
