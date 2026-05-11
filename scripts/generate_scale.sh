#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST_CONFIG="$REPO_ROOT/configs/benchmark_hosts/r8a.yaml"
SCALE_CONFIG="$REPO_ROOT/scripts/scale_config.yml"
ARGS=()
while (("$#")); do
  case "$1" in
    --host-config)
      HOST_CONFIG="$2"
      shift 2
      ;;
    --host-config=*)
      HOST_CONFIG="${1#*=}"
      shift
      ;;
    --scale-config)
      SCALE_CONFIG="$2"
      shift 2
      ;;
    --scale-config=*)
      SCALE_CONFIG="${1#*=}"
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done
if [[ "$HOST_CONFIG" != /* ]]; then
  HOST_CONFIG="$REPO_ROOT/$HOST_CONFIG"
fi
if [[ "$SCALE_CONFIG" != /* ]]; then
  SCALE_CONFIG="$REPO_ROOT/$SCALE_CONFIG"
fi
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/worldsim-scale.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

uv run python "$REPO_ROOT/scripts/generate_compose_scale.py" \
  --config "$SCALE_CONFIG" \
  --base-config "$REPO_ROOT/instances.json" \
  --host-config "$HOST_CONFIG" \
  --out-dir "$OUT_DIR" \
  --final-config-dir "$REPO_ROOT" \
  ${ARGS[@]+"${ARGS[@]}"}

SMOKE_OUT_DIR="$OUT_DIR/smoke"
mkdir -p "$SMOKE_OUT_DIR"
uv run python "$REPO_ROOT/scripts/generate_compose_scale.py" \
  --config "$SCALE_CONFIG" \
  --base-config "$REPO_ROOT/instances.json" \
  --host-config "$HOST_CONFIG" \
  --mode smoke \
  --out-dir "$SMOKE_OUT_DIR" \
  --final-config-dir "$REPO_ROOT" \
  ${ARGS[@]+"${ARGS[@]}"}

cp "$OUT_DIR/compose.scale.yml" "$REPO_ROOT/scripts/docker-compose.scale.yml"
cp "$OUT_DIR/proxy_ports.conf" "$REPO_ROOT/scripts/proxy_ports.conf"
cp "$OUT_DIR/instances.json" "$REPO_ROOT/instances.scale.json"
cp "$OUT_DIR/instances.json.fragment" "$REPO_ROOT/instances.scale.json.fragment"
cp "$SMOKE_OUT_DIR/compose.smoke.yml" "$REPO_ROOT/scripts/docker-compose.smoke.yml"
cp "$SMOKE_OUT_DIR/instances.json" "$REPO_ROOT/instances.smoke.json"
cp "$SMOKE_OUT_DIR/instances.json.fragment" "$REPO_ROOT/instances.smoke.json.fragment"

printf 'wrote %s\n' "$REPO_ROOT/scripts/docker-compose.scale.yml"
printf 'wrote %s\n' "$REPO_ROOT/scripts/proxy_ports.conf"
printf 'wrote %s\n' "$REPO_ROOT/instances.scale.json"
printf 'wrote %s\n' "$REPO_ROOT/instances.scale.json.fragment"
printf 'wrote %s\n' "$REPO_ROOT/scripts/docker-compose.smoke.yml"
printf 'wrote %s\n' "$REPO_ROOT/instances.smoke.json"
printf 'wrote %s\n' "$REPO_ROOT/instances.smoke.json.fragment"
