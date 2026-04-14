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
#   1. On a workstation (no --on-ec2): copies docker-compose.override.yml
#      into the vendors directory so containers get
#      WA_ENV_CTRL_EXTERNAL_SITE_URL set as an env var. The container-level
#      patch step is skipped because `docker exec` targets the local daemon.
#   2. On the EC2 host (with --on-ec2): patches the Python _init() methods
#      inside running containers to add the env-var fallback (for images
#      whose baked-in code lacks it). This mode is invoked by
#      scripts/bootstrap_ec2.sh after SCP-ing the patcher helper up.
#
# Usage:
#   # Local workstation -- only deploys the override file into vendors/.
#   ./scripts/patch_webarena_containers.sh [HOST_IP]
#
#   # EC2 host (invoked by bootstrap_ec2.sh). Requires
#   # /home/ubuntu/wa_envctrl_patcher.py to be present (scp'd first).
#   ./scripts/patch_webarena_containers.sh --on-ec2 [HOST_IP]
#
#   HOST_IP defaults to the EC2 instance IP from instances.json.
#
# Idempotent: safe to run multiple times. The Python patcher invoked inside
# each container repairs any prior broken run where `import os\n` was
# prepended above `from __future__ import annotations` (SyntaxError) by
# stripping the bad prepend before re-inserting `import os` after the
# __future__ import.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_DIR="$REPO_ROOT/vendors/webarena-verified"
OVERRIDE_SRC="$REPO_ROOT/scripts/webarena-compose-override.yml"

ON_EC2=0
if [[ "${1:-}" == "--on-ec2" ]]; then
    ON_EC2=1
    shift
fi

# Default to the EC2 IP from instances.json
HOST_IP="${1:-18.117.99.179}"

# --- Step 1: Deploy docker-compose.override.yml (only off-EC2 workflow) ---

if [[ "$ON_EC2" -eq 0 ]]; then
    if [[ ! -d "$COMPOSE_DIR" ]]; then
        echo "ERROR: $COMPOSE_DIR does not exist."
        echo "Clone the webarena-verified repo first:"
        echo "  git clone <webarena-verified-repo> $COMPOSE_DIR"
        exit 1
    fi

    echo "==> Deploying docker-compose.override.yml to $COMPOSE_DIR"
    cp "$OVERRIDE_SRC" "$COMPOSE_DIR/docker-compose.override.yml"
    echo "    Done. Override sets WA_ENV_CTRL_EXTERNAL_SITE_URL for all sites."
    echo ""
    echo "The container-level env-ctrl patch runs against the LOCAL docker daemon."
    echo "To patch containers running on the EC2 host, use:"
    echo "    ./scripts/bootstrap_ec2.sh"
    echo "which scp's this script and the patcher helper up and re-invokes"
    echo "this script with --on-ec2 remotely."
    exit 0
fi

# --- On-EC2 mode: verify the Python patcher helper is present ----------------
# bootstrap_ec2.sh scp's scripts/wa_envctrl_patcher.py to /home/ubuntu/ before
# invoking this script remotely. We also copy it into each container before
# running it with docker exec (rather than trying to shell-quote a multiline
# Python snippet, which is a nightmare with `from __future__` + f-strings).

PATCHER_HOST_PATH="/home/ubuntu/wa_envctrl_patcher.py"
if [[ ! -f "$PATCHER_HOST_PATH" ]]; then
    echo "ERROR: $PATCHER_HOST_PATH missing on EC2 host."
    echo "       scp scripts/wa_envctrl_patcher.py up first (bootstrap_ec2.sh handles this)."
    exit 1
fi

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
echo "    Patcher helper: $PATCHER_HOST_PATH"

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
" 2>/dev/null || true)

    if [[ -z "$py_path" ]]; then
        echo "      WARN: could not locate $site_file inside container"
        continue
    fi

    # Copy the patcher helper into the container, then run it. We avoid
    # `python3 -c '<big literal>'` shell quoting entirely. The helper is
    # idempotent and self-repairs any prior broken patch that prepended
    # `import os\n` above a `from __future__` line.
    if ! docker cp "$PATCHER_HOST_PATH" "$container:/tmp/wa_envctrl_patcher.py"; then
        echo "      WARN: docker cp of patcher helper into $container failed"
        continue
    fi

    if ! docker exec "$container" python3 /tmp/wa_envctrl_patcher.py "$py_path"; then
        echo "      WARN: patcher exited non-zero for $container ($py_path)"
    fi

    # Set the env var in the running container's env-ctrl process
    # This is a belt-and-suspenders approach: even if the Python patch
    # didn't work, setting the env var will make the fallback work.
    echo "      Setting WA_ENV_CTRL_EXTERNAL_SITE_URL=$site_url (for future processes)"
    docker exec "$container" bash -c "
        # Write env var for any future process
        grep -q '^export WA_ENV_CTRL_EXTERNAL_SITE_URL=' /etc/environment 2>/dev/null || \
            echo 'export WA_ENV_CTRL_EXTERNAL_SITE_URL=$site_url' >> /etc/environment

        # Try to restart the env-ctrl process so it picks up changes.
        # env-ctrl might run under supervisord or as a standalone process.
        if command -v supervisorctl >/dev/null 2>&1 && supervisorctl status env-ctrl >/dev/null 2>&1; then
            supervisorctl restart env-ctrl 2>/dev/null || true
        else
            pkill -f 'env-ctrl serve' 2>/dev/null || true
            # NOTE: gitlab has no process manager for env-ctrl; bootstrap_ec2.sh
            # respawns it via 'docker exec -d' after killing it.
        fi
    " 2>/dev/null || echo "      Warning: could not restart env-ctrl (may need manual respawn)"
done

echo ""
echo "==> Done."
echo ""
echo "If gitlab's env-ctrl does not respawn automatically, run:"
echo "  docker exec -d webarena-verified-gitlab sh -c 'setsid /usr/local/bin/env-ctrl serve --port 8877 >>/tmp/env-ctrl.log 2>&1 </dev/null'"
echo ""
echo "For new containers (fresh docker compose up), the override file handles everything:"
echo "  cd $COMPOSE_DIR && docker compose up -d"
