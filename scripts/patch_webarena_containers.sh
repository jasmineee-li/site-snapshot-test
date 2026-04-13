#!/usr/bin/env bash
# patch_webarena_containers.sh — Fix the base_url env-var fallback bug in
# WebArena Verified containers.
#
# Problem: The original Docker images (am1n3e/webarena-verified-*) ship
# env-ctrl Python code where _init() requires a base_url argument, but the
# HTTP server (POST /init) never passes one. Shopping's _init had an
# os.environ fallback; reddit and gitlab did not.
#
# This script:
#   1. Copies docker-compose.override.yml into the vendors directory so
#      containers get WA_ENV_CTRL_EXTERNAL_SITE_URL set as an env var.
#   2. Patches the Python _init() methods inside running containers to
#      add the env-var fallback (for images whose baked-in code lacks it).
#
# Usage:
#   # After docker compose up in vendors/webarena-verified/:
#   ./scripts/patch_webarena_containers.sh [HOST_IP]
#
#   HOST_IP defaults to the EC2 instance IP from instances.json.
#
# Idempotent: safe to run multiple times.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_DIR="$REPO_ROOT/vendors/webarena-verified"
OVERRIDE_SRC="$REPO_ROOT/scripts/webarena-compose-override.yml"

# Default to the EC2 IP from instances.json
HOST_IP="${1:-18.117.99.179}"

# --- Step 1: Deploy docker-compose.override.yml ---

if [[ ! -d "$COMPOSE_DIR" ]]; then
    echo "ERROR: $COMPOSE_DIR does not exist."
    echo "Clone the webarena-verified repo first:"
    echo "  git clone <webarena-verified-repo> $COMPOSE_DIR"
    exit 1
fi

echo "==> Deploying docker-compose.override.yml to $COMPOSE_DIR"
cp "$OVERRIDE_SRC" "$COMPOSE_DIR/docker-compose.override.yml"
echo "    Done. Override sets WA_ENV_CTRL_EXTERNAL_SITE_URL for all sites."

# --- Step 2: Patch running containers ---
# The env var from step 1 only takes effect on next `docker compose up`.
# For already-running containers, we patch the Python code in-place.

# Container name -> site file, port
declare -A SITE_FILES=(
    ["webarena-verified-shopping"]="shopping.py"
    ["webarena-verified-shopping_admin"]="shopping_admin.py"
    ["webarena-verified-reddit"]="reddit.py"
    ["webarena-verified-gitlab"]="gitlab.py"
)

declare -A SITE_PORTS=(
    ["webarena-verified-shopping"]="7770"
    ["webarena-verified-shopping_admin"]="7780"
    ["webarena-verified-reddit"]="9999"
    ["webarena-verified-gitlab"]="8023"
)

echo ""
echo "==> Patching running containers (adds env-var fallback to _init if missing)"

for container in "${!SITE_FILES[@]}"; do
    site_file="${SITE_FILES[$container]}"
    port="${SITE_PORTS[$container]}"
    site_url="http://${HOST_IP}:${port}"

    # Check if container is running
    if ! docker inspect --format='{{.State.Running}}' "$container" 2>/dev/null | grep -q true; then
        echo "    SKIP $container (not running)"
        continue
    fi

    echo "    Patching $container ($site_file)..."

    # Find the actual path to the site ops file inside the container.
    # The Python version may vary across images, so we locate it dynamically.
    py_path=$(docker exec "$container" python3 -c "
import importlib.util, pathlib
spec = importlib.util.find_spec('environment_control.ops.sites.${site_file%.py}')
if spec and spec.origin:
    print(spec.origin)
" 2>/dev/null)

    if [[ -z "$py_path" ]]; then
        echo "      WARN: could not locate $site_file inside container"
        continue
    fi

    # Check if the env-var fallback already exists in the container's code
    if docker exec "$container" grep -q 'WA_ENV_CTRL_EXTERNAL_SITE_URL' "$py_path" 2>/dev/null; then
        echo "      Already patched (env-var fallback present)"
    else
        echo "      Adding env-var fallback..."

        # The original code has:
        #   if not base_url:
        #       raise ValueError("base_url is required")
        #
        # We need to insert the env-var lookup before the raise.
        # Use Python to do the patching inside the container since sed with
        # complex multi-line replacements is fragile.
        docker exec "$container" python3 -c "
import pathlib

p = pathlib.Path('$py_path')
src = p.read_text()

# Check if 'import os' is present; add it if not
if 'import os' not in src:
    src = src.replace('from __future__ import annotations', 'from __future__ import annotations\n\nimport os', 1)

# Pattern: find 'if not base_url:' followed by 'raise ValueError'
# and insert the env-var fallback between them
old = '''        if not base_url:
            raise ValueError'''
new = '''        if not base_url:
            base_url = os.environ.get(\"WA_ENV_CTRL_EXTERNAL_SITE_URL\", \"\")
        if not base_url:
            raise ValueError'''

if old in src:
    src = src.replace(old, new, 1)
    p.write_text(src)
    print('  Patched successfully')
else:
    print('  Pattern not found (code may have different structure)')
"
    fi

    # Set the env var in the running container's env-ctrl process
    # This is a belt-and-suspenders approach: even if the Python patch
    # didn't work, setting the env var will make the fallback work.
    echo "      Setting WA_ENV_CTRL_EXTERNAL_SITE_URL=$site_url"
    docker exec "$container" bash -c "
        # Write env var for any future process
        echo 'export WA_ENV_CTRL_EXTERNAL_SITE_URL=$site_url' >> /etc/environment

        # Try to restart the env-ctrl process so it picks up changes.
        # env-ctrl might run under supervisord or as a standalone process.
        if command -v supervisorctl &>/dev/null && supervisorctl status env-ctrl &>/dev/null 2>&1; then
            supervisorctl restart env-ctrl 2>/dev/null || true
        else
            # For gitlab which uses gitlab-ctl
            pkill -f 'env-ctrl serve' 2>/dev/null || true
            # The process manager should restart it automatically
        fi
    " 2>/dev/null || echo "      Warning: could not restart env-ctrl (may need manual restart)"
done

echo ""
echo "==> Done."
echo ""
echo "For already-running containers, restart env-ctrl to pick up changes:"
echo "  docker compose -f $COMPOSE_DIR/docker-compose.yml -f $COMPOSE_DIR/docker-compose.override.yml restart"
echo ""
echo "For new containers (fresh docker compose up), the override file handles everything:"
echo "  cd $COMPOSE_DIR && docker compose up -d"
