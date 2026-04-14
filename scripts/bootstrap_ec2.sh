#!/usr/bin/env bash
# bootstrap_ec2.sh — Single-entrypoint host-side orchestrator for the
# WebArena Verified stack on the EC2 instance.
#
# Takes a fresh (or partially bootstrapped) EC2 host and drives it to a
# state where all 6 sites (map / wikipedia / shopping / shopping_admin /
# gitlab / reddit) are running with their env-ctrl endpoints returning
# 200 on /init.
#
# Consolidates today's live-run fixes:
#   * amd64 wikipedia image rebuild (upstream arm64-only image crash-loops)
#   * resume-safe, parallel-mirror ZIM download + atomic volume replace
#   * resume-safe map data download + strict sentinel-gated extraction
#   * correct Docker Compose override placement at /home/ubuntu/
#   * in-container Python env-ctrl patcher that inserts `import os` AFTER
#     `from __future__` (the prior script inserted it at line 0 and
#     triggered a SyntaxError)
#   * gitlab env-ctrl respawn recipe (gitlab has no process manager for
#     env-ctrl; must use `docker exec -d ... setsid` so the respawned
#     process isn't SIGHUP'd when the SSH exec returns)
#
# Idempotent: safe to run N times. Every step short-circuits on the
# existing work's sentinel / already-running state.
#
# Usage (from the repo root, on your workstation):
#   ./scripts/bootstrap_ec2.sh
#
#   # Optional overrides:
#   HOST_IP=1.2.3.4 SSH_KEY=~/.ssh/other-key.pem ./scripts/bootstrap_ec2.sh
#
# Intentionally NO `set -euo pipefail` at top level. Individual steps may
# fail (e.g. aria2 transient error); we report and keep going so a
# partially-bootstrapped host can make forward progress on re-run.

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HOST_IP="${HOST_IP:-18.117.99.179}"
SSH_KEY_RAW="${SSH_KEY:-$HOME/.ssh/webarena-key.pem}"
# Expand ~ in SSH_KEY if present (SSH_KEY=~/.ssh/foo from env is literal ~).
SSH_KEY="${SSH_KEY_RAW/#\~/$HOME}"
SSH_USER="${SSH_USER:-ubuntu}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/scripts"

SSH_OPTS=(
    -i "$SSH_KEY"
    -o StrictHostKeyChecking=accept-new
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=120
    -o ConnectTimeout=15
)

SCP_OPTS=(
    -i "$SSH_KEY"
    -o StrictHostKeyChecking=accept-new
    -o ConnectTimeout=15
)

# Sites / ports used for env-ctrl verification.
#
# env-ctrl listens on port 8877 inside every container; the compose file
# maps it to host ports listed as "envctrl" below. The "site" port is the
# user-facing port (stamped into /etc/environment inside each container).
#
# If these disagree with the compose file, update the SITE_ROWS array;
# the verification in step 8 is what bootstrap_ec2.sh blocks on.
#
# Parallel-array layout (rather than `declare -A`) so this script also
# works under macOS's default bash 3.2.
#
#   SITE_ROWS[i] = "site_name:envctrl_port:site_port"
SITE_ROWS=(
    "shopping:7771:7770"
    "shopping_admin:7781:7780"
    "gitlab:8024:8023"
    "reddit:9998:9999"
    "wikipedia:8889:8888"
    "map:3031:3030"
)

split_row() {
    # Splits "site:envctrl:site_port" into three global vars. Bash 3.2
    # friendly. Sets _S_NAME, _S_ENVCTRL, _S_PUBLIC.
    local row="$1"
    _S_NAME="${row%%:*}"
    local rest="${row#*:}"
    _S_ENVCTRL="${rest%%:*}"
    _S_PUBLIC="${rest#*:}"
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() {
    printf '\n==> %s\n' "$*"
}

ssh_host() {
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$HOST_IP" "$@"
}

scp_up() {
    # scp_up <local-path> <remote-path>
    scp "${SCP_OPTS[@]}" "$1" "$SSH_USER@$HOST_IP:$2"
}

sanity_check_local() {
    local missing=0
    for required in \
        "$SCRIPTS_DIR/setup-map-robust.sh" \
        "$SCRIPTS_DIR/setup-wikipedia-robust.sh" \
        "$SCRIPTS_DIR/build-wikipedia-amd64.sh" \
        "$SCRIPTS_DIR/webarena-compose-override.yml" \
        "$SCRIPTS_DIR/patch_webarena_containers.sh" \
        "$SCRIPTS_DIR/wa_envctrl_patcher.py"; do
        if [[ ! -f "$required" ]]; then
            echo "ERROR: required local file missing: $required"
            missing=1
        fi
    done
    if [[ ! -f "$SSH_KEY" ]]; then
        echo "ERROR: SSH key missing: $SSH_KEY"
        missing=1
    fi
    if [[ "$missing" -ne 0 ]]; then
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Step 1 & 2: scp all helper scripts up to /home/ubuntu/
# ---------------------------------------------------------------------------

step_upload_scripts() {
    log "Step 1/2: scp helper scripts to $SSH_USER@$HOST_IP:/home/ubuntu/"

    local ok=1
    for f in \
        "$SCRIPTS_DIR/setup-map-robust.sh" \
        "$SCRIPTS_DIR/setup-wikipedia-robust.sh" \
        "$SCRIPTS_DIR/build-wikipedia-amd64.sh" \
        "$SCRIPTS_DIR/webarena-compose-override.yml" \
        "$SCRIPTS_DIR/patch_webarena_containers.sh" \
        "$SCRIPTS_DIR/wa_envctrl_patcher.py"; do
        printf '    scp %s\n' "$(basename "$f")"
        if ! scp_up "$f" "/home/ubuntu/$(basename "$f")"; then
            echo "    WARN: scp failed for $f"
            ok=0
        fi
    done

    # Replace any stale docker-compose.override.yml at /home/ubuntu/ with
    # the canonical one from this repo. docker compose at /home/ubuntu/
    # picks this up automatically.
    printf '    scp webarena-compose-override.yml -> docker-compose.override.yml\n'
    if ! scp_up "$SCRIPTS_DIR/webarena-compose-override.yml" "/home/ubuntu/docker-compose.override.yml"; then
        echo "    WARN: scp of docker-compose.override.yml failed"
        ok=0
    fi

    # Make the shell scripts executable on the remote side.
    ssh_host 'chmod +x /home/ubuntu/setup-map-robust.sh /home/ubuntu/setup-wikipedia-robust.sh /home/ubuntu/build-wikipedia-amd64.sh /home/ubuntu/patch_webarena_containers.sh 2>/dev/null || true'

    if [[ "$ok" -eq 1 ]]; then
        echo "    - uploaded all scripts"
    fi
    return $((1 - ok))
}

# ---------------------------------------------------------------------------
# Step 3: build amd64 wikipedia image on the EC2 host
# ---------------------------------------------------------------------------

step_build_wikipedia_image() {
    log "Step 3: build amd64 wikipedia image (idempotent; skips if present)"
    if ! ssh_host 'bash /home/ubuntu/build-wikipedia-amd64.sh'; then
        echo "    - WARN: build-wikipedia-amd64.sh returned non-zero"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Step 4: download + extract map data
# ---------------------------------------------------------------------------

step_setup_map_data() {
    log "Step 4: download + extract map data (resume-safe; per-volume sentinels)"
    if ! ssh_host 'bash /home/ubuntu/setup-map-robust.sh'; then
        echo "    - WARN: setup-map-robust.sh returned non-zero (may still have made forward progress)"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Step 5: download + replace-in-volume the wiki ZIM
# ---------------------------------------------------------------------------

step_setup_wikipedia_zim() {
    log "Step 5: download + verify + replace wiki ZIM (only restarts if replaced)"
    if ! ssh_host 'bash /home/ubuntu/setup-wikipedia-robust.sh'; then
        echo "    - WARN: setup-wikipedia-robust.sh returned non-zero"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Step 6: bring up all 6 sites
# ---------------------------------------------------------------------------

step_compose_up_all() {
    log "Step 6: docker compose up -d (all 6 sites)"
    if ! ssh_host 'cd /home/ubuntu && sudo docker compose up -d'; then
        echo "    - WARN: docker compose up -d returned non-zero"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Step 7: patch env-ctrl inside the running containers (on-EC2 mode)
# ---------------------------------------------------------------------------

step_patch_containers() {
    log "Step 7: env-ctrl base_url Python patch inside running containers"
    # Run the patcher on the EC2 in --on-ec2 mode so it targets the
    # remote docker daemon. The --on-ec2 mode of patch_webarena_containers.sh
    # expects wa_envctrl_patcher.py at /home/ubuntu/ (uploaded in step 1).
    if ! ssh_host "bash /home/ubuntu/patch_webarena_containers.sh --on-ec2 $HOST_IP"; then
        echo "    - WARN: patch_webarena_containers.sh --on-ec2 returned non-zero"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Step 8: verify each site's env-ctrl /init returns 200
# ---------------------------------------------------------------------------

# Prints per-site status lines of the form:
#   SITE  HTTP_CODE  URL
probe_envctrl() {
    # probe_envctrl <url>; echoes HTTP code (or 000 on network error).
    local url="$1"
    curl -sS -o /dev/null -w '%{http_code}' -X POST \
        --max-time 30 \
        -H 'Content-Type: application/json' \
        --data '{}' \
        "$url" 2>/dev/null || echo "000"
}

step_verify_envctrl() {
    log "Step 8: verify env-ctrl /init endpoints return 200"

    local any_failed=0
    local gitlab_failed=0

    for row in "${SITE_ROWS[@]}"; do
        split_row "$row"
        local url="http://$HOST_IP:$_S_ENVCTRL/init"
        local code
        code=$(probe_envctrl "$url")
        printf '    %-16s  HTTP %s  %s\n' "$_S_NAME" "$code" "$url"
        if [[ "$code" != "200" ]]; then
            any_failed=1
            if [[ "$_S_NAME" == "gitlab" ]]; then
                gitlab_failed=1
            fi
        fi
    done

    echo ""
    if [[ "$any_failed" -eq 0 ]]; then
        echo "    all env-ctrl endpoints healthy"
        return 0
    fi

    # Step 9: gitlab-specific respawn. gitlab has no process manager for
    # env-ctrl; `pkill -f environment_control.cli` leaves nothing to
    # restart it. After killing, we must respawn via `docker exec -d`
    # with setsid so the process detaches from the SSH session (plain
    # background via `&` gets SIGHUP'd when the ssh exec returns).
    if [[ "$gitlab_failed" -eq 1 ]]; then
        log "Step 9: gitlab env-ctrl respawn (detached via docker exec -d + setsid)"
        ssh_host "docker exec -d webarena-verified-gitlab sh -c 'setsid /usr/local/bin/env-ctrl serve --port 8877 >>/tmp/env-ctrl.log 2>&1 </dev/null' || true"

        # Give it a moment and re-check gitlab only.
        sleep 1
        local gitlab_envctrl=""
        for row in "${SITE_ROWS[@]}"; do
            split_row "$row"
            if [[ "$_S_NAME" == "gitlab" ]]; then
                gitlab_envctrl="$_S_ENVCTRL"
                break
            fi
        done
        local url="http://$HOST_IP:$gitlab_envctrl/init"
        local code
        code=$(probe_envctrl "$url")
        printf '    re-check gitlab: HTTP %s  %s\n' "$code" "$url"
    fi

    return 1
}

# ---------------------------------------------------------------------------
# Step 10: final summary
# ---------------------------------------------------------------------------

step_print_summary() {
    log "Step 10: summary"

    printf '    HOST_IP = %s\n\n' "$HOST_IP"

    printf '    %-16s  %-10s  %-28s  %s\n' site envctrl_http site_url envctrl_url
    for row in "${SITE_ROWS[@]}"; do
        split_row "$row"
        local envctrl_url="http://$HOST_IP:$_S_ENVCTRL"
        local site_url="http://$HOST_IP:$_S_PUBLIC"
        local code
        code=$(probe_envctrl "$envctrl_url/init")
        printf '    %-16s  HTTP %-5s  %-28s  %s/init\n' \
            "$_S_NAME" "$code" "$site_url" "$envctrl_url"
    done
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    log "bootstrap_ec2.sh — WebArena Verified stack driver"
    printf '    HOST_IP = %s\n' "$HOST_IP"
    printf '    SSH_KEY = %s\n' "$SSH_KEY"
    printf '    SSH_USER = %s\n' "$SSH_USER"

    if ! sanity_check_local; then
        return 1
    fi

    step_upload_scripts              || echo "    (continuing past step 1-2 warning)"
    step_build_wikipedia_image       || echo "    (continuing past step 3 warning)"
    step_setup_map_data              || echo "    (continuing past step 4 warning)"
    step_setup_wikipedia_zim         || echo "    (continuing past step 5 warning)"
    step_compose_up_all              || echo "    (continuing past step 6 warning)"
    step_patch_containers            || echo "    (continuing past step 7 warning)"
    local verify_rc=0
    step_verify_envctrl              || verify_rc=$?
    step_print_summary

    if [[ "$verify_rc" -ne 0 ]]; then
        echo "==> bootstrap completed with one or more env-ctrl endpoints NOT 200."
        echo "    Inspect the per-site lines above. Re-run bootstrap_ec2.sh to retry."
        return 1
    fi
    echo "==> bootstrap completed. All env-ctrl endpoints are healthy."
    return 0
}

main "$@"
exit $?
