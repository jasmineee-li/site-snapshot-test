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
#   * local amd64 image builds for shopping/shopping_admin/gitlab/reddit with
#     the env-ctrl fallback baked in from git (no per-run patch dependency)
#   * amd64 wikipedia image rebuild (upstream arm64-only image crash-loops)
#   * resume-safe, parallel-mirror ZIM download + atomic volume replace
#   * resume-safe map data download + strict sentinel-gated extraction
#   * correct Docker Compose override placement at /home/ubuntu/ plus a
#     generated `.env` that stamps the selected advertised/bind hosts into
#     the smoke path
#   * optional legacy env-ctrl patcher kept only as a manual fallback
#
# Idempotent: safe to run N times. Every step short-circuits on the
# existing work's sentinel / already-running state.
#
# Usage (from the repo root, on your workstation):
#   ./scripts/bootstrap_ec2.sh --host-config configs/benchmark_hosts/r5.yaml
#
#   # Explicit overrides (generic path):
#   HOST_IP=1.2.3.4 WORLDSIM_BIND_HOST=0.0.0.0 \
#   WORLDSIM_DB_BIND_HOST=0.0.0.0 SSH_KEY=~/.ssh/other-key.pem \
#   ./scripts/bootstrap_ec2.sh
#
# Intentionally NO `set -euo pipefail` at top level. Individual steps may
# fail (e.g. aria2 transient error); we report and keep going so a
# partially-bootstrapped host can make forward progress on re-run.

# ---------------------------------------------------------------------------
# Config / args
# ---------------------------------------------------------------------------

HOST_CONFIG=""
HOST_IP="${HOST_IP:-}"
LOCAL_COMPOSE_FILE=""
SSH_KEY_RAW="${SSH_KEY:-$HOME/.ssh/webarena-key.pem}"
# Expand ~ in SSH_KEY if present (SSH_KEY=~/.ssh/foo from env is literal ~).
SSH_KEY="${SSH_KEY_RAW/#\~/$HOME}"
SSH_USER="${SSH_USER:-ubuntu}"
COMPOSE_DIR_REMOTE="${COMPOSE_DIR_REMOTE:-}"
COMPOSE_FILE_REMOTE="${COMPOSE_FILE_REMOTE:-}"
WORLDSIM_ADVERTISE_HOST="${WORLDSIM_ADVERTISE_HOST:-}"
WORLDSIM_BIND_HOST="${WORLDSIM_BIND_HOST:-}"
WORLDSIM_DB_BIND_HOST="${WORLDSIM_DB_BIND_HOST:-}"
ALLOW_PUBLIC_WEB_BIND="${ALLOW_PUBLIC_WEB_BIND:-0}"
ALLOW_PUBLIC_DB_BIND="${ALLOW_PUBLIC_DB_BIND:-0}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/scripts"
LOCAL_WEBARENA_SOURCE="$REPO_ROOT/vendors/webarena-verified"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host-config)
            HOST_CONFIG="$2"
            shift 2
            ;;
        --host)
            HOST_IP="$2"
            shift 2
            ;;
        --ssh-key)
            SSH_KEY="$2"
            shift 2
            ;;
        --ssh-user)
            SSH_USER="$2"
            shift 2
            ;;
        --compose-dir-remote)
            COMPOSE_DIR_REMOTE="$2"
            shift 2
            ;;
        --compose-file-remote)
            COMPOSE_FILE_REMOTE="$2"
            shift 2
            ;;
        --local-compose-file)
            LOCAL_COMPOSE_FILE="$2"
            shift 2
            ;;
        --bind-host)
            WORLDSIM_BIND_HOST="$2"
            shift 2
            ;;
        --db-bind-host)
            WORLDSIM_DB_BIND_HOST="$2"
            shift 2
            ;;
        --allow-public-web-bind)
            ALLOW_PUBLIC_WEB_BIND=1
            shift
            ;;
        --allow-public-db-bind)
            ALLOW_PUBLIC_DB_BIND=1
            shift
            ;;
        -h|--help)
            sed -n '1,35p' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

load_host_config_defaults() {
    local config_path="$1"
    if ! uv run python "$SCRIPTS_DIR/preflight_host_config.py" \
        --host-config "$config_path" \
        --require-remote-db >/dev/null; then
        return 1
    fi
    while IFS='=' read -r key quoted_value; do
        eval "local value=$quoted_value"
        case "$key" in
            HOST_IP)
                [[ -n "$HOST_IP" ]] || HOST_IP="$value"
                [[ -n "$WORLDSIM_ADVERTISE_HOST" ]] || WORLDSIM_ADVERTISE_HOST="$value"
                ;;
            SSH_USER)
                [[ "$SSH_USER" != "ubuntu" ]] || SSH_USER="$value"
                ;;
            COMPOSE_DIR_REMOTE)
                [[ -n "$COMPOSE_DIR_REMOTE" ]] || COMPOSE_DIR_REMOTE="$value"
                ;;
            COMPOSE_FILE_REMOTE)
                [[ -n "$COMPOSE_FILE_REMOTE" ]] || COMPOSE_FILE_REMOTE="$value"
                ;;
            WORLDSIM_ADVERTISE_HOST)
                [[ -n "$WORLDSIM_ADVERTISE_HOST" ]] || WORLDSIM_ADVERTISE_HOST="$value"
                ;;
            WORLDSIM_BIND_HOST)
                [[ -n "$WORLDSIM_BIND_HOST" ]] || WORLDSIM_BIND_HOST="$value"
                ;;
            WORLDSIM_DB_BIND_HOST)
                [[ -n "$WORLDSIM_DB_BIND_HOST" ]] || WORLDSIM_DB_BIND_HOST="$value"
                ;;
        esac
    done < <(uv run python "$SCRIPTS_DIR/export_host_config_env.py" --host-config "$config_path")
}

if [[ -n "$HOST_CONFIG" ]]; then
    load_host_config_defaults "$HOST_CONFIG" || exit 1
fi

if [[ -z "$HOST_IP" ]]; then
    echo "ERROR: missing benchmark host. Pass --host-config or set HOST_IP/--host." >&2
    exit 1
fi
if [[ -z "$WORLDSIM_ADVERTISE_HOST" ]]; then
    WORLDSIM_ADVERTISE_HOST="$HOST_IP"
fi
if [[ -z "$WORLDSIM_BIND_HOST" || -z "$WORLDSIM_DB_BIND_HOST" ]]; then
    echo "ERROR: missing WORLDSIM_BIND_HOST or WORLDSIM_DB_BIND_HOST. Use --host-config or set them explicitly." >&2
    exit 1
fi
if [[ -z "$HOST_CONFIG" ]]; then
    preflight_args=(
        --advertise-host "$WORLDSIM_ADVERTISE_HOST"
        --bind-host "$WORLDSIM_BIND_HOST"
        --db-bind-host "$WORLDSIM_DB_BIND_HOST"
        --require-remote-db
    )
    if [[ "$ALLOW_PUBLIC_WEB_BIND" == "1" ]]; then
        preflight_args+=(--allow-public-web-bind)
    fi
    if [[ "$ALLOW_PUBLIC_DB_BIND" == "1" ]]; then
        preflight_args+=(--allow-public-db-bind)
    fi
    if ! uv run python "$SCRIPTS_DIR/preflight_host_config.py" \
        "${preflight_args[@]}" >/dev/null; then
        exit 1
    fi
fi
if [[ -z "$COMPOSE_DIR_REMOTE" ]]; then
    COMPOSE_DIR_REMOTE="/home/ubuntu"
fi
if [[ -z "$COMPOSE_FILE_REMOTE" ]]; then
    COMPOSE_FILE_REMOTE="$COMPOSE_DIR_REMOTE/docker-compose.yml"
fi
SSH_KEY="${SSH_KEY/#\~/$HOME}"

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
        "$SCRIPTS_DIR/build-webarena-amd64-images.sh" \
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
    if [[ ! -d "$LOCAL_WEBARENA_SOURCE" ]]; then
        echo "ERROR: required pinned vendor checkout missing: $LOCAL_WEBARENA_SOURCE"
        missing=1
    fi
    if [[ -n "$LOCAL_COMPOSE_FILE" && ! -f "$LOCAL_COMPOSE_FILE" ]]; then
        echo "ERROR: local compose file missing: $LOCAL_COMPOSE_FILE"
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
    log "Step 1/2: scp helper scripts to $SSH_USER@$HOST_IP:$COMPOSE_DIR_REMOTE/"

    local ok=1
    for f in \
        "$SCRIPTS_DIR/setup-map-robust.sh" \
        "$SCRIPTS_DIR/setup-wikipedia-robust.sh" \
        "$SCRIPTS_DIR/build-webarena-amd64-images.sh" \
        "$SCRIPTS_DIR/build-wikipedia-amd64.sh" \
        "$SCRIPTS_DIR/webarena-compose-override.yml" \
        "$SCRIPTS_DIR/patch_webarena_containers.sh" \
        "$SCRIPTS_DIR/wa_envctrl_patcher.py"; do
        printf '    scp %s\n' "$(basename "$f")"
        if ! scp_up "$f" "$COMPOSE_DIR_REMOTE/$(basename "$f")"; then
            echo "    WARN: scp failed for $f"
            ok=0
        fi
    done

    # Replace any stale docker-compose.override.yml at the compose dir with
    # the canonical one from this repo. docker compose at /home/ubuntu/
    # picks this up automatically.
    printf '    scp webarena-compose-override.yml -> docker-compose.override.yml\n'
    if ! scp_up "$SCRIPTS_DIR/webarena-compose-override.yml" "$COMPOSE_DIR_REMOTE/docker-compose.override.yml"; then
        echo "    WARN: scp of docker-compose.override.yml failed"
        ok=0
    fi

    if [[ -n "$LOCAL_COMPOSE_FILE" ]]; then
        printf '    scp %s -> %s\n' "$(basename "$LOCAL_COMPOSE_FILE")" "$COMPOSE_FILE_REMOTE"
        if ! scp_up "$LOCAL_COMPOSE_FILE" "$COMPOSE_FILE_REMOTE"; then
            echo "    WARN: scp of compose file failed"
            ok=0
        fi
    fi

    printf '    ship pinned webarena-verified source snapshot\n'
    if ! ssh_host 'mkdir -p /home/ubuntu/vendors'; then
        echo "    WARN: failed to create /home/ubuntu/vendors on host"
        ok=0
    else
        local archive
        archive="$(mktemp /tmp/webarena-verified.XXXXXX.tgz)"
        # COPYFILE_DISABLE=1 stops macOS BSD tar from embedding AppleDouble
        # `._*` resource-fork siblings. Without it, extraction on Linux
        # produces ._webarena-verified next to webarena-verified/ which
        # (a) litters the vendor tree and (b) prevents the staging rmdir
        # from succeeding. Harmless no-op on Linux callers.
        # rm -rf on $staging (not rmdir) is defence-in-depth for any
        # future unexpected siblings.
        if COPYFILE_DISABLE=1 tar -czf "$archive" --exclude='.git' -C "$REPO_ROOT/vendors" webarena-verified \
            && scp_up "$archive" "/home/ubuntu/webarena-verified-src.tgz" \
            && ssh_host 'set -e; staging=$(mktemp -d /home/ubuntu/webarena-vendors.XXXXXX); tar -xzf /home/ubuntu/webarena-verified-src.tgz -C "$staging"; rm -rf /home/ubuntu/vendors/webarena-verified; mv "$staging/webarena-verified" /home/ubuntu/vendors/webarena-verified; rm -rf "$staging"'; then
            :
        else
            echo "    WARN: failed to upload pinned webarena-verified source snapshot"
            ok=0
        fi
        rm -f "$archive"
    fi

    # Make the shell scripts executable on the remote side.
    ssh_host "chmod +x $COMPOSE_DIR_REMOTE/setup-map-robust.sh $COMPOSE_DIR_REMOTE/setup-wikipedia-robust.sh $COMPOSE_DIR_REMOTE/build-webarena-amd64-images.sh $COMPOSE_DIR_REMOTE/build-wikipedia-amd64.sh $COMPOSE_DIR_REMOTE/patch_webarena_containers.sh 2>/dev/null || true"

    if [[ "$ok" -eq 1 ]]; then
        echo "    - uploaded all scripts"
    fi
    return $((1 - ok))
}

# ---------------------------------------------------------------------------
# Step 3: build patched amd64 images on the EC2 host
# ---------------------------------------------------------------------------

step_build_site_images() {
    log "Step 3: build patched amd64 site images (idempotent; skips if present)"
    if ! ssh_host "WEBARENA_SOURCE_DIR=/home/ubuntu/vendors/webarena-verified bash $COMPOSE_DIR_REMOTE/build-webarena-amd64-images.sh"; then
        echo "    - WARN: build-webarena-amd64-images.sh returned non-zero"
        return 1
    fi
    if ! ssh_host "bash $COMPOSE_DIR_REMOTE/build-wikipedia-amd64.sh"; then
        echo "    - WARN: build-wikipedia-amd64.sh returned non-zero"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Step 3b: write compose env on the host
# ---------------------------------------------------------------------------

step_write_compose_env() {
    log "Step 3b: write $COMPOSE_DIR_REMOTE/.env from the selected host contract"
    if ! ssh_host "tmp=$COMPOSE_DIR_REMOTE/.env.tmp; if [ -f $COMPOSE_DIR_REMOTE/.env ]; then grep -Ev '^(WORLDSIM_HOST_IP|WORLDSIM_ADVERTISE_HOST|WORLDSIM_BIND_HOST|WORLDSIM_DB_BIND_HOST)=' $COMPOSE_DIR_REMOTE/.env > \"\$tmp\" || true; else : > \"\$tmp\"; fi; printf 'WORLDSIM_HOST_IP=%s\nWORLDSIM_ADVERTISE_HOST=%s\nWORLDSIM_BIND_HOST=%s\nWORLDSIM_DB_BIND_HOST=%s\n' '$HOST_IP' '$WORLDSIM_ADVERTISE_HOST' '$WORLDSIM_BIND_HOST' '$WORLDSIM_DB_BIND_HOST' >> \"\$tmp\"; mv \"\$tmp\" $COMPOSE_DIR_REMOTE/.env"; then
        echo "    - WARN: failed to write $COMPOSE_DIR_REMOTE/.env"
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
    log "Step 6: docker compose up -d --force-recreate (all 6 sites)"
    # --force-recreate ensures containers land on the final merged config
    # (override.yml image pins, env vars, port bindings). Without it,
    # intermediate recreates during Step 4/5 (map setup, ZIM replacement
    # helper containers) can leave compose's per-container metadata
    # referencing stale image tags, and Step 6's plain `up -d` treats
    # them as up-to-date and skips recreation — leaving am1n3e/*:latest
    # containers running while the merged config says worldsim/*:amd64.
    # This bit us on the 2026-04-17 r5 bring-up: env-ctrl endpoints
    # returned HTTP 000 because the base_url fallback wasn't baked in
    # and wikipedia crash-looped on the arm64 manifest. Forcing recreate
    # costs ~90s per bring-up; worth it for idempotent correctness.
    if ! ssh_host "cd $COMPOSE_DIR_REMOTE && sudo docker compose -f $COMPOSE_FILE_REMOTE up -d --force-recreate"; then
        echo "    - WARN: docker compose up -d --force-recreate returned non-zero"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Step 6b: configure database access inside running containers
# ---------------------------------------------------------------------------
#
# Container (re)creation resets in-container state. This step grants
# remote MySQL access (shopping, shopping_admin) and enables PostgreSQL
# TCP access (GitLab). Reddit and Map already accept TCP by default.
# Wikipedia has no database (Kiwix ZIM). This is for read-only reward
# evaluation and postcondition verification, not SQL seeding.

step_configure_db_access() {
    log "Step 6b: configure database access for reward evaluation"
    if [[ -x "$SCRIPTS_DIR/configure_db_access.sh" ]]; then
        HOST_IP="$HOST_IP" SSH_KEY="$SSH_KEY" SSH_USER="$SSH_USER" \
            "$SCRIPTS_DIR/configure_db_access.sh"
    else
        echo "    WARN: scripts/configure_db_access.sh not found or not executable"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Step 7: optional legacy env-ctrl patch inside running containers
# ---------------------------------------------------------------------------

step_patch_containers() {
    log "Step 7: optional legacy env-ctrl patch inside running containers"
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
    if [[ -n "$HOST_CONFIG" ]]; then
        printf '    HOST_CONFIG = %s\n' "$HOST_CONFIG"
    fi
    if [[ -n "$LOCAL_COMPOSE_FILE" ]]; then
        printf '    LOCAL_COMPOSE_FILE = %s\n' "$LOCAL_COMPOSE_FILE"
    fi
    printf '    HOST_IP = %s\n' "$HOST_IP"
    printf '    WORLDSIM_ADVERTISE_HOST = %s\n' "$WORLDSIM_ADVERTISE_HOST"
    printf '    WORLDSIM_BIND_HOST = %s\n' "$WORLDSIM_BIND_HOST"
    printf '    WORLDSIM_DB_BIND_HOST = %s\n' "$WORLDSIM_DB_BIND_HOST"
    printf '    SSH_KEY = %s\n' "$SSH_KEY"
    printf '    SSH_USER = %s\n' "$SSH_USER"

    if ! sanity_check_local; then
        return 1
    fi

    step_upload_scripts              || return 1
    step_build_site_images           || return 1
    step_write_compose_env           || return 1
    step_setup_map_data              || echo "    (continuing past step 4 warning)"
    step_setup_wikipedia_zim         || echo "    (continuing past step 5 warning)"
    step_compose_up_all              || echo "    (continuing past step 6 warning)"
    step_configure_db_access         || echo "    (continuing past step 6b warning)"
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
