#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST_CONFIG="$REPO_ROOT/configs/benchmark_hosts/r5.yaml"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/worldsim-scale.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

uv run python "$REPO_ROOT/scripts/generate_compose_scale.py" \
  --config "$REPO_ROOT/scripts/scale_config.yml" \
  --base-config "$REPO_ROOT/instances.json" \
  --host-config "$HOST_CONFIG" \
  --out-dir "$OUT_DIR" \
  "$@"

cp "$OUT_DIR/compose.scale.yml" "$REPO_ROOT/scripts/docker-compose.scale.yml"
cp "$OUT_DIR/proxy_ports.conf" "$REPO_ROOT/scripts/proxy_ports.conf"
cp "$OUT_DIR/instances.json" "$REPO_ROOT/instances.scale.json"
cp "$OUT_DIR/instances.json.fragment" "$REPO_ROOT/instances.scale.json.fragment"

printf 'wrote %s\n' "$REPO_ROOT/scripts/docker-compose.scale.yml"
printf 'wrote %s\n' "$REPO_ROOT/scripts/proxy_ports.conf"
printf 'wrote %s\n' "$REPO_ROOT/instances.scale.json"
printf 'wrote %s\n' "$REPO_ROOT/instances.scale.json.fragment"
