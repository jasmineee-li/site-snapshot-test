#!/usr/bin/env bash
# deploy_benchmark_proxy.sh — Deploy an authenticated nginx reverse proxy
# on a benchmark EC2 instance so that Modal sandboxes (Phase 0c) can reach
# live sites for verification probing.
#
# Problem: Modal sandboxes exit from dynamic IPs that the EC2 security group
# blocks. Opening 0.0.0.0/0 is insecure (benchmark instances have default
# credentials, known-vulnerable software). This script deploys an nginx
# reverse proxy on offset ports (e.g. 17770 for 7770) that requires a
# secret token header (X-Worldsim-Token) on every request.
#
# Benchmark-agnostic: reads site-to-port mappings from a config file, not
# hardcoded to WebArena. The config is a simple text file with one line per
# site: "name:real_port:proxy_port". For convenience, you can omit
# proxy_port and it defaults to real_port + PORT_OFFSET (default 10000).
#
# Idempotent: safe to re-run. Overwrites the nginx config and restarts.
# If a token already exists on disk, it is reused unless --new-token is
# passed.
#
# Usage (from the repo root, on your workstation):
#
#   ./scripts/deploy_benchmark_proxy.sh
#
#   # With explicit arguments:
#   ./scripts/deploy_benchmark_proxy.sh \
#       --host 18.117.99.179 \
#       --ssh-key ~/.ssh/webarena-key.pem \
#       --port-map scripts/proxy_ports.conf \
#       --token-file .proxy_token
#
#   # Force a new token:
#   ./scripts/deploy_benchmark_proxy.sh --new-token
#
# After running, open the proxy ports in the EC2 security group for
# 0.0.0.0/0 (they are token-protected). The script outputs the exact
# ports to open and a JSON snippet for instances.json.

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

HOST_IP="${HOST_IP:-18.117.99.179}"
SSH_KEY_RAW="${SSH_KEY:-$HOME/.ssh/webarena-key.pem}"
SSH_KEY="${SSH_KEY_RAW/#\~/$HOME}"
SSH_USER="${SSH_USER:-ubuntu}"
PORT_OFFSET="${PORT_OFFSET:-10000}"
TOKEN_FILE="${TOKEN_FILE:-.proxy_token}"
PORT_MAP_FILE="${PORT_MAP_FILE:-}"
NEW_TOKEN=0

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)       HOST_IP="$2";       shift 2 ;;
        --ssh-key)    SSH_KEY="$2";        shift 2 ;;
        --ssh-user)   SSH_USER="$2";       shift 2 ;;
        --port-map)   PORT_MAP_FILE="$2";  shift 2 ;;
        --token-file) TOKEN_FILE="$2";     shift 2 ;;
        --port-offset) PORT_OFFSET="$2";   shift 2 ;;
        --new-token)  NEW_TOKEN=1;         shift ;;
        --help|-h)
            echo "Usage: $0 [--host IP] [--ssh-key PATH] [--ssh-user USER]"
            echo "          [--port-map FILE] [--token-file FILE] [--port-offset N]"
            echo "          [--new-token]"
            echo ""
            echo "Environment variables: HOST_IP, SSH_KEY, SSH_USER, PORT_OFFSET,"
            echo "                       TOKEN_FILE, PORT_MAP_FILE"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# SSH helpers (same pattern as bootstrap_ec2.sh)
# ---------------------------------------------------------------------------

SSH_OPTS=(
    -i "$SSH_KEY"
    -o StrictHostKeyChecking=accept-new
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=120
    -o ConnectTimeout=15
)

log() {
    printf '\n==> %s\n' "$*"
}

ssh_host() {
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$HOST_IP" "$@"
}

# ---------------------------------------------------------------------------
# Port map loading
# ---------------------------------------------------------------------------
# Format: one line per site, "name:real_port" or "name:real_port:proxy_port".
# Lines starting with # and blank lines are ignored.
#
# If no port map file is provided, fall back to a default WebArena mapping.

DEFAULT_PORT_MAP="shopping:7770
shopping_admin:7780
gitlab:8023
reddit:9999
wikipedia:8888
map:3030"

load_port_map() {
    # Populates parallel arrays: SITE_NAMES, REAL_PORTS, PROXY_PORTS
    SITE_NAMES=()
    REAL_PORTS=()
    PROXY_PORTS=()

    local input
    if [[ -n "$PORT_MAP_FILE" && -f "$PORT_MAP_FILE" ]]; then
        input=$(cat "$PORT_MAP_FILE")
    elif [[ -n "$PORT_MAP_FILE" ]]; then
        echo "ERROR: port map file not found: $PORT_MAP_FILE" >&2
        return 1
    else
        input="$DEFAULT_PORT_MAP"
    fi

    while IFS= read -r line; do
        # Skip comments and blanks.
        line="${line%%#*}"
        line="$(echo "$line" | xargs)"  # trim whitespace
        [[ -z "$line" ]] && continue

        local name real_port proxy_port
        IFS=: read -r name real_port proxy_port <<< "$line"

        if [[ -z "$name" || -z "$real_port" ]]; then
            echo "WARN: skipping malformed line: $line" >&2
            continue
        fi

        if [[ -z "$proxy_port" ]]; then
            proxy_port=$((real_port + PORT_OFFSET))
        fi

        SITE_NAMES+=("$name")
        REAL_PORTS+=("$real_port")
        PROXY_PORTS+=("$proxy_port")
    done <<< "$input"

    if [[ ${#SITE_NAMES[@]} -eq 0 ]]; then
        echo "ERROR: no sites found in port map" >&2
        return 1
    fi

    printf '    loaded %d site(s) from port map\n' "${#SITE_NAMES[@]}"
    for i in "${!SITE_NAMES[@]}"; do
        printf '      %-20s  %s -> %s\n' "${SITE_NAMES[$i]}" "${REAL_PORTS[$i]}" "${PROXY_PORTS[$i]}"
    done
}

# ---------------------------------------------------------------------------
# Token management
# ---------------------------------------------------------------------------

ensure_token() {
    if [[ "$NEW_TOKEN" -eq 1 ]] || [[ ! -f "$TOKEN_FILE" ]]; then
        TOKEN=$(openssl rand -hex 32)
        echo "$TOKEN" > "$TOKEN_FILE"
        chmod 600 "$TOKEN_FILE"
        printf '    generated new token -> %s\n' "$TOKEN_FILE"
    else
        TOKEN=$(cat "$TOKEN_FILE")
        printf '    reusing existing token from %s\n' "$TOKEN_FILE"
    fi

    if [[ -z "$TOKEN" ]]; then
        echo "ERROR: token is empty" >&2
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Nginx config generation
# ---------------------------------------------------------------------------

generate_nginx_config() {
    # Generates the nginx config as a string. Caller writes it to the host.
    local config=""

    config+="# worldsim-proxy.conf — auto-generated by deploy_benchmark_proxy.sh
# Authenticated reverse proxy for Phase 0c live instance verification.
# Do not edit manually; re-run the deploy script to update.

"

    for i in "${!SITE_NAMES[@]}"; do
        local name="${SITE_NAMES[$i]}"
        local real_port="${REAL_PORTS[$i]}"
        local proxy_port="${PROXY_PORTS[$i]}"

        config+="# ${name}: proxy ${proxy_port} -> real ${real_port}
server {
    listen ${proxy_port};
    server_name _;

    # Require X-Worldsim-Token header on every request.
    if (\$http_x_worldsim_token != \"${TOKEN}\") {
        return 403;
    }

    location / {
        proxy_pass http://127.0.0.1:${real_port};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # Pass through large bodies (form submissions, API calls).
        client_max_body_size 50m;
        proxy_read_timeout 120s;
        proxy_connect_timeout 10s;
    }
}

"
    done

    NGINX_CONFIG="$config"
}

# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------

step_install_nginx() {
    log "Step 1: ensure nginx is installed on $HOST_IP"
    ssh_host 'command -v nginx >/dev/null 2>&1 && echo "nginx already installed" && exit 0; \
        if command -v apt-get >/dev/null 2>&1; then \
            sudo apt-get update -qq && sudo apt-get install -y -qq nginx; \
        elif command -v yum >/dev/null 2>&1; then \
            sudo yum install -y nginx; \
        elif command -v dnf >/dev/null 2>&1; then \
            sudo dnf install -y nginx; \
        else \
            echo "ERROR: no supported package manager (apt-get, yum, dnf)" >&2; \
            exit 1; \
        fi'
}

step_deploy_config() {
    log "Step 2: deploy nginx proxy config"

    generate_nginx_config

    # Write the config to a temp file locally, scp it up, move into place.
    local tmp_config
    tmp_config=$(mktemp /tmp/worldsim-proxy-XXXXXX.conf)
    echo "$NGINX_CONFIG" > "$tmp_config"

    scp "${SSH_OPTS[@]}" "$tmp_config" "$SSH_USER@$HOST_IP:/tmp/worldsim-proxy.conf"
    rm -f "$tmp_config"

    # Move into nginx config dir and enable.
    ssh_host 'sudo mkdir -p /etc/nginx/sites-enabled /etc/nginx/conf.d && \
        sudo mv /tmp/worldsim-proxy.conf /etc/nginx/conf.d/worldsim-proxy.conf && \
        echo "    config written to /etc/nginx/conf.d/worldsim-proxy.conf"'
}

step_test_and_restart_nginx() {
    log "Step 3: test nginx config and restart"
    ssh_host 'sudo nginx -t && sudo systemctl restart nginx && sudo systemctl enable nginx && \
        echo "    nginx restarted and enabled"'
}

step_verify_proxy() {
    log "Step 4: verify proxy ports are listening"

    local any_failed=0
    for i in "${!SITE_NAMES[@]}"; do
        local name="${SITE_NAMES[$i]}"
        local proxy_port="${PROXY_PORTS[$i]}"

        # Check that the proxy port is listening on the remote host.
        local listening
        listening=$(ssh_host "ss -tln | grep -c ':${proxy_port} ' || true")

        if [[ "$listening" -gt 0 ]]; then
            printf '    %-20s  port %s  LISTENING\n' "$name" "$proxy_port"
        else
            printf '    %-20s  port %s  NOT LISTENING\n' "$name" "$proxy_port"
            any_failed=1
        fi
    done

    if [[ "$any_failed" -ne 0 ]]; then
        echo ""
        echo "    WARN: some proxy ports are not listening."
        echo "    Check nginx logs: ssh $SSH_USER@$HOST_IP 'sudo journalctl -u nginx --no-pager -n 30'"
        return 1
    fi
}

step_print_summary() {
    log "Step 5: summary"

    echo ""
    echo "    Token: $TOKEN"
    echo "    Token file: $TOKEN_FILE"
    echo ""

    echo "    === Security Group Ports to Open ==="
    echo "    Open these TCP ports in the EC2 security group for 0.0.0.0/0:"
    echo ""
    for i in "${!SITE_NAMES[@]}"; do
        printf '      %s  (proxy for %s)\n' "${PROXY_PORTS[$i]}" "${SITE_NAMES[$i]}"
    done

    echo ""
    echo "    === instances.json verification_proxy block ==="
    echo "    Add this to your instances.json:"
    echo ""
    echo '    "verification_proxy": {'
    printf '      "token": "%s",\n' "$TOKEN"
    printf '      "port_offset": %s\n' "$PORT_OFFSET"
    echo '    }'
    echo ""

    echo "    === Proxy URL examples ==="
    for i in "${!SITE_NAMES[@]}"; do
        printf '      %s: http://%s:%s\n' "${SITE_NAMES[$i]}" "$HOST_IP" "${PROXY_PORTS[$i]}"
    done
    echo ""

    echo "    === Test command ==="
    echo "    Verify with (after opening security group):"
    printf '    curl -H "X-Worldsim-Token: %s" http://%s:%s/\n' \
        "$TOKEN" "$HOST_IP" "${PROXY_PORTS[0]}"
    echo ""

    echo "    Without the token (should return 403):"
    printf '    curl http://%s:%s/\n' "$HOST_IP" "${PROXY_PORTS[0]}"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    log "deploy_benchmark_proxy.sh — authenticated reverse proxy for Phase 0c"
    printf '    HOST_IP     = %s\n' "$HOST_IP"
    printf '    SSH_KEY     = %s\n' "$SSH_KEY"
    printf '    SSH_USER    = %s\n' "$SSH_USER"
    printf '    PORT_OFFSET = %s\n' "$PORT_OFFSET"

    if [[ ! -f "$SSH_KEY" ]]; then
        echo "ERROR: SSH key not found: $SSH_KEY" >&2
        exit 1
    fi

    load_port_map || exit 1
    ensure_token  || exit 1

    step_install_nginx        || echo "    (continuing past step 1 warning)"
    step_deploy_config        || { echo "ERROR: failed to deploy config" >&2; exit 1; }
    step_test_and_restart_nginx || { echo "ERROR: nginx config test/restart failed" >&2; exit 1; }
    step_verify_proxy         || echo "    (continuing past step 4 warning)"
    step_print_summary

    echo "==> Proxy deployed. Open the security group ports above, then test."
}

main "$@"
exit $?
