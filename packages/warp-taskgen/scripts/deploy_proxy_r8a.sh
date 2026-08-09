#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST_CONFIG="$REPO_ROOT/configs/benchmark_hosts/r8a.yaml"
FORWARD_ARGS=()

# The tracked r8a config is a public template. Strip the wrapper-only
# --host-config option before forwarding the remaining proxy options.
source "$REPO_ROOT/scripts/lib/r8a_host_config.sh"
while (($#)); do
  case "$1" in
    -h|--help)
      cat <<'USAGE'
deploy_proxy_r8a.sh --host-config configs/benchmark_hosts/r8a.local.yaml [proxy options]

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
exec "$REPO_ROOT/scripts/deploy_benchmark_proxy.sh" \
  --host-config "$HOST_CONFIG" \
  --topology scale \
  --insecure-http \
  "${FORWARD_ARGS[@]}"
