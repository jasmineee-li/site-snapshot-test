#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST_CONFIG="$REPO_ROOT/configs/benchmark_hosts/r8a.yaml"
SCALE_CONFIG="$REPO_ROOT/scripts/scale_config.r8a-24x24.yml"
LOCAL_COMPOSE="$REPO_ROOT/scripts/docker-compose.scale.yml"
INSTANCES_SCALE="$REPO_ROOT/instances.scale.json"

"$REPO_ROOT/scripts/audit_r8a_control_plane.sh" --host-config "$HOST_CONFIG"
"$REPO_ROOT/scripts/generate_scale_r5.sh" \
  --host-config "$HOST_CONFIG" \
  --scale-config "$SCALE_CONFIG"
uv run python "$REPO_ROOT/scripts/preflight_security_group.py" \
  --host-config "$HOST_CONFIG" \
  --instances "$INSTANCES_SCALE"
exec "$REPO_ROOT/scripts/bootstrap_ec2.sh" \
  --host-config "$HOST_CONFIG" \
  --local-compose-file "$LOCAL_COMPOSE" \
  "$@"
