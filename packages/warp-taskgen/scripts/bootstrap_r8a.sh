#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST_CONFIG="$REPO_ROOT/configs/benchmark_hosts/r8a.yaml"
SCALE_CONFIG="$REPO_ROOT/scripts/scale_config.r8a-24x24.yml"
LOCAL_COMPOSE="$REPO_ROOT/scripts/docker-compose.scale.yml"
INSTANCES_SCALE="$REPO_ROOT/instances.scale.json"
FORWARD_ARGS=()

# The tracked r8a config is a public template. Strip the wrapper-only
# --host-config option before forwarding the remaining bootstrap options.
source "$REPO_ROOT/scripts/lib/r8a_host_config.sh"
while (($#)); do
  case "$1" in
    -h|--help)
      cat <<'USAGE'
bootstrap_r8a.sh --host-config configs/benchmark_hosts/r8a.local.yaml [bootstrap options]

The host config must be an existing gitignored *.local.yaml overlay.
USAGE
      exit 0
      ;;
    --host-config)
      (($# >= 2)) || { printf 'ERROR: --host-config requires a path\n' >&2; exit 2; }
      HOST_CONFIG="$(r8a_resolve_host_config_path "$REPO_ROOT" "$2")"
      shift 2
      ;;
    --host-config=*)
      HOST_CONFIG="$(r8a_resolve_host_config_path "$REPO_ROOT" "${1#*=}")"
      shift
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

r8a_require_ignored_local_config "$REPO_ROOT" "$HOST_CONFIG"

"$REPO_ROOT/scripts/audit_r8a_control_plane.sh" --host-config "$HOST_CONFIG"
"$REPO_ROOT/scripts/generate_scale.sh" \
  --host-config "$HOST_CONFIG" \
  --scale-config "$SCALE_CONFIG"
uv run python "$REPO_ROOT/scripts/preflight_security_group.py" \
  --host-config "$HOST_CONFIG" \
  --instances "$INSTANCES_SCALE"
exec "$REPO_ROOT/scripts/bootstrap_ec2.sh" \
  --host-config "$HOST_CONFIG" \
  --local-compose-file "$LOCAL_COMPOSE" \
  "${FORWARD_ARGS[@]}"
