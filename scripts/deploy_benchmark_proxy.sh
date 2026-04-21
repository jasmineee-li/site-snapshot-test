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
#       --host-config configs/benchmark_hosts/r5.yaml \
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

HOST_CONFIG=""
HOST_IP="${HOST_IP:-}"
SSH_KEY_RAW="${SSH_KEY:-$HOME/.ssh/webarena-key.pem}"
SSH_KEY="${SSH_KEY_RAW/#\~/$HOME}"
SSH_USER="${SSH_USER:-ubuntu}"
PORT_OFFSET="${PORT_OFFSET:-10000}"
TLS_CERT_FILE="${TLS_CERT_FILE:-}"
TLS_KEY_FILE="${TLS_KEY_FILE:-}"
TLS_VERIFY_HOST="${TLS_VERIFY_HOST:-}"
ALLOW_VERIFY_FAILURE="${ALLOW_VERIFY_FAILURE:-0}"
ALLOW_INSECURE_HTTP="${ALLOW_INSECURE_HTTP:-0}"
BENCHMARK_TOPOLOGY="${BENCHMARK_TOPOLOGY:-}"
USE_LEGACY_DEFAULT_MAP=0
NEW_TOKEN=0
VIA_SSM=0
SSM_INSTANCE_ID="${SSM_INSTANCE_ID:-}"
SSM_REGION="${SSM_REGION:-}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOKEN_FILE="${TOKEN_FILE:-$REPO_ROOT/.proxy_token}"
PROXY_METADATA_FILE="${PROXY_METADATA_FILE:-$REPO_ROOT/.benchmark_proxy_metadata}"
PROXY_PORT_MAP_FILE="${PROXY_PORT_MAP_FILE:-$REPO_ROOT/.benchmark_proxy_ports.conf}"
PORT_MAP_FILE="${PORT_MAP_FILE:-}"
SCALE_PORT_MAP_FILE="${SCALE_PORT_MAP_FILE:-$REPO_ROOT/scripts/proxy_ports.conf}"
TOPOLOGY_FILE="${TOPOLOGY_FILE:-$REPO_ROOT/.benchmark_topology}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host-config) HOST_CONFIG="$2"; shift 2 ;;
        --host)       HOST_IP="$2";       shift 2 ;;
        --ssh-key)    SSH_KEY="$2";        shift 2 ;;
        --ssh-user)   SSH_USER="$2";       shift 2 ;;
        --port-map)   PORT_MAP_FILE="$2";  shift 2 ;;
        --token-file) TOKEN_FILE="$2";     shift 2 ;;
        --port-offset) PORT_OFFSET="$2";   shift 2 ;;
        --tls-cert)   TLS_CERT_FILE="$2";  shift 2 ;;
        --tls-key)    TLS_KEY_FILE="$2";   shift 2 ;;
        --topology)   BENCHMARK_TOPOLOGY="$2"; shift 2 ;;
        --tls-verify-host) TLS_VERIFY_HOST="$2"; shift 2 ;;
        --use-legacy-default-map) USE_LEGACY_DEFAULT_MAP=1; shift ;;
        --allow-verify-failure) ALLOW_VERIFY_FAILURE=1; shift ;;
        --insecure-http) ALLOW_INSECURE_HTTP=1; shift ;;
        --new-token)  NEW_TOKEN=1;         shift ;;
        --via-ssm)    VIA_SSM=1;           shift ;;
        --ssm-instance-id) SSM_INSTANCE_ID="$2"; shift 2 ;;
        --ssm-region) SSM_REGION="$2";      shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--host IP] [--ssh-key PATH] [--ssh-user USER]"
            echo "          [--port-map FILE] [--token-file FILE] [--port-offset N]"
            echo "          [--tls-cert PATH] [--tls-key PATH] [--tls-verify-host HOST]"
            echo "          [--topology scale|legacy] [--use-legacy-default-map]"
            echo "          [--allow-verify-failure] [--insecure-http] [--new-token]"
            echo ""
            echo "          [--via-ssm] [--ssm-instance-id ID] [--ssm-region REGION]"
            echo ""
            echo "Environment variables: HOST_IP, SSH_KEY, SSH_USER, PORT_OFFSET,"
            echo "                       TOKEN_FILE, PORT_MAP_FILE, TLS_CERT_FILE, TLS_KEY_FILE,"
            echo "                       TLS_VERIFY_HOST, ALLOW_VERIFY_FAILURE,"
            echo "                       ALLOW_INSECURE_HTTP, BENCHMARK_TOPOLOGY, TOPOLOGY_FILE,"
            echo "                       PROXY_METADATA_FILE, PROXY_PORT_MAP_FILE,"
            echo "                       SSM_INSTANCE_ID, SSM_REGION"
            echo ""
            echo "With --via-ssm, the script uses AWS SSM send-command instead of SSH."
            echo "Required when the EC2 security group blocks SSH ingress."
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -n "$HOST_CONFIG" ]]; then
    while IFS='=' read -r key quoted_value; do
        eval "value=$quoted_value"
        case "$key" in
            HOST_IP)
                [[ -n "$HOST_IP" ]] || HOST_IP="$value"
                ;;
            SSH_USER)
                [[ "$SSH_USER" != "ubuntu" ]] || SSH_USER="$value"
                ;;
        esac
    done < <(uv run python "$REPO_ROOT/scripts/export_host_config_env.py" --host-config "$HOST_CONFIG")
fi

if [[ "${BASH_SOURCE[0]}" == "$0" && -z "$HOST_IP" ]]; then
    echo "ERROR: missing benchmark host. Pass --host-config or set HOST_IP/--host." >&2
    exit 1
fi
SSH_KEY="${SSH_KEY/#\~/$HOME}"

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
    if [[ "$VIA_SSM" == "1" ]]; then
        ssm_run "$@"
    else
        ssh "${SSH_OPTS[@]}" "$SSH_USER@$HOST_IP" "$@"
    fi
}

# ---------------------------------------------------------------------------
# SSM helpers (used when --via-ssm is set; SG typically blocks SSH ingress)
# ---------------------------------------------------------------------------
# ssm_run: runs a shell command on the target instance. The command is
# base64-encoded so heredocs / quoting survive JSON transport through
# `aws ssm send-command`. Blocks until the command completes (up to 180s)
# and emits StandardOutputContent; exits non-zero on Status != Success.
ssm_run() {
    local script cmd_id status stdout stderr
    script="$*"
    local b64
    b64="$(printf '%s' "$script" | base64 | tr -d '\n')"
    local region_arg=""
    [[ -n "$SSM_REGION" ]] && region_arg="--region $SSM_REGION"
    local wrapped
    wrapped="echo '$b64' | base64 -d | bash"
    cmd_id=$(aws ssm send-command \
        --instance-ids "$SSM_INSTANCE_ID" \
        --document-name "AWS-RunShellScript" \
        --parameters "commands=[\"$wrapped\"]" \
        --query "Command.CommandId" \
        --output text \
        $region_arg 2>&1)
    if [[ -z "$cmd_id" || "$cmd_id" == None ]]; then
        echo "ERROR: ssm send-command failed: $cmd_id" >&2
        return 1
    fi
    local deadline=$(( SECONDS + 180 ))
    while (( SECONDS < deadline )); do
        status=$(aws ssm get-command-invocation \
            --command-id "$cmd_id" \
            --instance-id "$SSM_INSTANCE_ID" \
            --query "Status" \
            --output text \
            $region_arg 2>/dev/null)
        case "$status" in
            Success|Failed|Cancelled|TimedOut) break ;;
        esac
        sleep 3
    done
    stdout=$(aws ssm get-command-invocation \
        --command-id "$cmd_id" \
        --instance-id "$SSM_INSTANCE_ID" \
        --query "StandardOutputContent" \
        --output text \
        $region_arg 2>/dev/null)
    stderr=$(aws ssm get-command-invocation \
        --command-id "$cmd_id" \
        --instance-id "$SSM_INSTANCE_ID" \
        --query "StandardErrorContent" \
        --output text \
        $region_arg 2>/dev/null)
    [[ -n "$stdout" && "$stdout" != None ]] && printf '%s\n' "$stdout"
    [[ -n "$stderr" && "$stderr" != None ]] && printf '%s\n' "$stderr" >&2
    [[ "$status" == "Success" ]]
}

# ssm_put_file: writes $1 (local path) to $2 (remote path). Base64-encoded
# heredoc; idempotent overwrite.
ssm_put_file() {
    local local_path="$1" remote_path="$2"
    if [[ ! -f "$local_path" ]]; then
        echo "ERROR: local file not found: $local_path" >&2
        return 1
    fi
    local b64
    b64="$(base64 -i "$local_path" | tr -d '\n')"
    ssm_run "echo '$b64' | base64 -d | sudo tee '$remote_path' > /dev/null"
}

# ---------------------------------------------------------------------------
# Port map loading
# ---------------------------------------------------------------------------
# Format: one line per site, "name:real_port" or "name:real_port:proxy_port".
# Lines starting with # and blank lines are ignored.
#
DEFAULT_PORT_MAP="gitlab:8023
reddit:9999"

validate_topology() {
    local topology="$1"
    [[ "$topology" == "scale" || "$topology" == "legacy" ]]
}

validate_verify_host() {
    local host="$1"
    [[ "$host" =~ ^[A-Za-z0-9.-]+$ ]] || return 1
    [[ "$host" != .* && "$host" != *..* && "$host" != *- && "$host" != -* ]]
}

validate_proxy_token() {
    local token="$1"
    [[ "$token" =~ ^[A-Fa-f0-9]{64}$ ]]
}

validate_port_number() {
    local port="$1"
    [[ "$port" =~ ^[0-9]+$ ]] || return 1
    (( port >= 1 && port <= 65535 ))
}

resolve_benchmark_topology() {
    if [[ "$USE_LEGACY_DEFAULT_MAP" == "1" ]]; then
        echo "legacy"
        return 0
    fi
    if [[ -n "$BENCHMARK_TOPOLOGY" ]]; then
        if ! validate_topology "$BENCHMARK_TOPOLOGY"; then
            echo "ERROR: invalid BENCHMARK_TOPOLOGY: $BENCHMARK_TOPOLOGY" >&2
            return 1
        fi
        echo "$BENCHMARK_TOPOLOGY"
        return 0
    fi
    if [[ -f "$TOPOLOGY_FILE" ]]; then
        local topology
        topology="$(tr -d '[:space:]' < "$TOPOLOGY_FILE")"
        if ! validate_topology "$topology"; then
            echo "ERROR: invalid topology in $TOPOLOGY_FILE: $topology" >&2
            return 1
        fi
        echo "$topology"
        return 0
    fi
    echo "ERROR: no topology configured; pass --topology scale|legacy, --port-map FILE, or write scale/legacy to $TOPOLOGY_FILE" >&2
    return 1
}

load_port_map() {
    # Populates parallel arrays: SITE_NAMES, REAL_PORTS, PROXY_PORTS
    SITE_NAMES=()
    REAL_PORTS=()
    PROXY_PORTS=()

    local input source
    if [[ -n "$PORT_MAP_FILE" && -f "$PORT_MAP_FILE" ]]; then
        input=$(cat "$PORT_MAP_FILE")
        source="$PORT_MAP_FILE"
    elif [[ -n "$PORT_MAP_FILE" ]]; then
        echo "ERROR: port map file not found: $PORT_MAP_FILE" >&2
        return 1
    else
        local topology
        topology=$(resolve_benchmark_topology) || return 1
        if [[ "$topology" == "scale" ]]; then
            if [[ ! -f "$SCALE_PORT_MAP_FILE" ]]; then
                echo "ERROR: scale port map file not found: $SCALE_PORT_MAP_FILE" >&2
                return 1
            fi
            input=$(cat "$SCALE_PORT_MAP_FILE")
            source="$SCALE_PORT_MAP_FILE (topology=$topology)"
        else
            input="$DEFAULT_PORT_MAP"
            source="built-in legacy WebArena map (topology=$topology)"
        fi
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
        if ! validate_port_number "$real_port" || ! validate_port_number "$proxy_port"; then
            echo "WARN: skipping line with invalid port values: $line" >&2
            continue
        fi

        SITE_NAMES+=("$name")
        REAL_PORTS+=("$real_port")
        PROXY_PORTS+=("$proxy_port")
    done <<< "$input"

    if [[ ${#SITE_NAMES[@]} -eq 0 ]]; then
        echo "ERROR: no sites found in port map" >&2
        return 1
    fi

    printf '    port map source: %s\n' "$source"
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
    if ! validate_proxy_token "$TOKEN"; then
        echo "ERROR: token must be a 64-character hex string: $TOKEN_FILE" >&2
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

        local listen_directive="listen ${proxy_port};"
        local tls_block=""
        if [[ -n "$TLS_CERT_FILE" && -n "$TLS_KEY_FILE" ]]; then
            listen_directive="listen ${proxy_port} ssl;"
            tls_block="    ssl_certificate ${TLS_CERT_FILE};
    ssl_certificate_key ${TLS_KEY_FILE};
"
        fi

        # Magento-specific buffer + proxy_redirect block was removed
        # 2026-04-21 with the WASP-aligned scoping decision.

        config+="# ${name}: proxy ${proxy_port} -> real ${real_port}
server {
    ${listen_directive}
    server_name _;
${tls_block}

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

    # Write the config to a temp file locally.
    local tmp_config
    tmp_config=$(mktemp /tmp/worldsim-proxy-XXXXXX.conf)
    echo "$NGINX_CONFIG" > "$tmp_config"

    # Back up the current config first so mistakes are recoverable.
    local timestamp
    timestamp=$(date -u +%Y%m%dT%H%M%SZ)
    ssh_host "sudo mkdir -p /etc/nginx/sites-enabled /etc/nginx/conf.d && \
        if [ -f /etc/nginx/conf.d/worldsim-proxy.conf ]; then \
            sudo cp /etc/nginx/conf.d/worldsim-proxy.conf /etc/nginx/conf.d/worldsim-proxy.conf.bak.${timestamp}; \
        fi"

    if [[ "$VIA_SSM" == "1" ]]; then
        ssm_put_file "$tmp_config" /etc/nginx/conf.d/worldsim-proxy.conf
    else
        scp "${SSH_OPTS[@]}" "$tmp_config" "$SSH_USER@$HOST_IP:/tmp/worldsim-proxy.conf"
        ssh_host 'sudo mv /tmp/worldsim-proxy.conf /etc/nginx/conf.d/worldsim-proxy.conf'
    fi
    rm -f "$tmp_config"
    ssh_host 'echo "    config written to /etc/nginx/conf.d/worldsim-proxy.conf"'
}

step_test_and_restart_nginx() {
    log "Step 3: test nginx config and reload"
    if [[ -n "$TLS_CERT_FILE" || -n "$TLS_KEY_FILE" ]]; then
        ssh_host "test -f '$TLS_CERT_FILE' && test -f '$TLS_KEY_FILE'" || {
            echo "ERROR: TLS enabled but cert/key missing on host: $TLS_CERT_FILE $TLS_KEY_FILE" >&2
            return 1
        }
    fi
    # Prefer reload over restart: zero-downtime, and if nginx -t fails the
    # old config keeps serving.
    ssh_host 'sudo nginx -t && sudo systemctl reload nginx && sudo systemctl enable nginx && \
        echo "    nginx reloaded and enabled"'
}

proxy_scheme() {
    if [[ -n "$TLS_CERT_FILE" && -n "$TLS_KEY_FILE" ]]; then
        echo "https"
    else
        echo "http"
    fi
}

proxy_http_code_is_healthy() {
    local code="$1"
    [[ "$code" =~ ^[23][0-9][0-9]$ ]]
}

remote_authed_http_code() {
    local scheme="$1" proxy_port="$2" verify_host="$3"
    # Inline the token into the curl header directly. Tried env-var
    # indirection (TOKEN='...' curl -H "X-Worldsim-Token: \$TOKEN") but
    # under the SSM base64 → bash pipe path the shell expands \$TOKEN
    # before the TOKEN= assignment takes effect, so the header arrives
    # empty. Inlining the 64-char token lands it in argv briefly on the
    # remote host (visible in `ps` during the curl window ~milliseconds);
    # acceptable for a deploy-time verification step running on a host
    # we just configured.
    local remote_cmd code
    if [[ "$scheme" == "https" ]]; then
        remote_cmd="curl -sS -o /dev/null -w '%{http_code}' --max-time 10 --resolve '${verify_host}:${proxy_port}:127.0.0.1' -H 'X-Worldsim-Token: ${TOKEN}' 'https://${verify_host}:${proxy_port}/' 2>/dev/null || echo 000"
    else
        remote_cmd="curl -sS -o /dev/null -w '%{http_code}' --max-time 10 -H 'X-Worldsim-Token: ${TOKEN}' 'http://127.0.0.1:${proxy_port}/' 2>/dev/null || echo 000"
    fi
    code=$(ssh_host "$remote_cmd")
    echo "$code"
}

write_proxy_metadata() {
    local current_proxy_scheme metadata_verify_host
    current_proxy_scheme="$(proxy_scheme)"
    metadata_verify_host=""
    if [[ "$current_proxy_scheme" == "https" ]]; then
        metadata_verify_host="$TLS_VERIFY_HOST"
    fi
    cat > "$PROXY_METADATA_FILE" <<EOF
PROXY_SCHEME=$current_proxy_scheme
PROXY_VERIFY_HOST=$metadata_verify_host
PROXY_PORT_MAP_FILE=$PROXY_PORT_MAP_FILE
EOF
    chmod 600 "$PROXY_METADATA_FILE"
}

write_proxy_port_map() {
    : > "$PROXY_PORT_MAP_FILE"
    for i in "${!SITE_NAMES[@]}"; do
        printf '%s:%s:%s\n' "${SITE_NAMES[$i]}" "${REAL_PORTS[$i]}" "${PROXY_PORTS[$i]}" >> "$PROXY_PORT_MAP_FILE"
    done
    chmod 600 "$PROXY_PORT_MAP_FILE"
}

step_verify_proxy() {
    log "Step 4: verify authenticated proxy path"

    local scheme
    scheme="$(proxy_scheme)"
    local verify_host
    verify_host="${TLS_VERIFY_HOST:-$HOST_IP}"
    local any_failed=0
    for i in "${!SITE_NAMES[@]}"; do
        local name="${SITE_NAMES[$i]}"
        local proxy_port="${PROXY_PORTS[$i]}"
        local authed_code unauthed_code
        if [[ "$scheme" == "https" ]]; then
            authed_code=$(remote_authed_http_code "$scheme" "$proxy_port" "$verify_host")
            unauthed_code=$(ssh_host "curl -sS -o /dev/null -w '%{http_code}' \
                --max-time 10 \
                --resolve '${verify_host}:${proxy_port}:127.0.0.1' \
                'https://${verify_host}:${proxy_port}/' 2>/dev/null || echo 000")
        else
            authed_code=$(remote_authed_http_code "$scheme" "$proxy_port" "$verify_host")
            unauthed_code=$(ssh_host "curl -sS -o /dev/null -w '%{http_code}' \
                --max-time 10 \
                'http://127.0.0.1:${proxy_port}/' 2>/dev/null || echo 000")
        fi

        if proxy_http_code_is_healthy "$authed_code" && [[ "$unauthed_code" == "403" ]]; then
            printf '    %-20s  auth=%s unauth=%s  %s://%s:%s/\n' \
                "$name" "$authed_code" "$unauthed_code" "$scheme" "$([[ "$scheme" == "https" ]] && echo "$verify_host" || echo "127.0.0.1")" "$proxy_port"
        else
            printf '    %-20s  auth=%s unauth=%s  %s://%s:%s/\n' \
                "$name" "$authed_code" "$unauthed_code" "$scheme" "$([[ "$scheme" == "https" ]] && echo "$verify_host" || echo "127.0.0.1")" "$proxy_port"
            any_failed=1
        fi
    done

    if [[ "$any_failed" -ne 0 ]]; then
        echo ""
        echo "    WARN: some proxy ports did not pass authenticated HTTP verification."
        echo "    Check nginx logs: ssh $SSH_USER@$HOST_IP 'sudo journalctl -u nginx --no-pager -n 30'"
        return 1
    fi
}

step_print_summary() {
    log "Step 5: summary"
    local current_proxy_scheme token_fingerprint
    current_proxy_scheme="$(proxy_scheme)"
    token_fingerprint="$(printf '%s' "$TOKEN" | openssl dgst -sha256 | awk '{print $2}')"
    write_proxy_port_map
    write_proxy_metadata

    echo ""
    echo "    Token file: $TOKEN_FILE"
    echo "    Proxy metadata file: $PROXY_METADATA_FILE"
    echo "    Proxy port map file: $PROXY_PORT_MAP_FILE"
    echo "    Token fingerprint (sha256): ${token_fingerprint:0:16}..."
    echo "    Raw token is not echoed; read it from the token file when updating instances.json."
    if [[ "$current_proxy_scheme" == "http" ]]; then
        echo "    WARNING: proxy TLS is disabled, so the token travels over plaintext HTTP."
        echo "    Prefer --tls-cert/--tls-key before exposing proxy ports publicly."
    else
        echo "    TLS verify host: $TLS_VERIFY_HOST"
    fi
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
    printf '      "token": "<contents of %s>",\n' "$TOKEN_FILE"
    printf '      "scheme": "%s",\n' "$current_proxy_scheme"
    if [[ "$current_proxy_scheme" == "https" ]]; then
        printf '      "verify_host": "%s",\n' "$TLS_VERIFY_HOST"
    fi
    printf '      "port_offset": %s\n' "$PORT_OFFSET"
    echo '    }'
    echo ""

    echo "    === Proxy URL examples ==="
    for i in "${!SITE_NAMES[@]}"; do
        if [[ "$current_proxy_scheme" == "https" ]]; then
            printf '      %s: https://%s:%s  (resolve to %s)\n' \
                "${SITE_NAMES[$i]}" "$TLS_VERIFY_HOST" "${PROXY_PORTS[$i]}" "$HOST_IP"
        else
            printf '      %s: http://%s:%s\n' "${SITE_NAMES[$i]}" "$HOST_IP" "${PROXY_PORTS[$i]}"
        fi
    done
    echo ""

    echo "    === Test command ==="
    echo "    Verify with (after opening security group):"
    if [[ "$current_proxy_scheme" == "https" ]]; then
        printf '    curl --resolve "%s:%s:%s" -H "X-Worldsim-Token: $(cat %q)" https://%s:%s/\n' \
            "$TLS_VERIFY_HOST" "${PROXY_PORTS[0]}" "$HOST_IP" "$TOKEN_FILE" "$TLS_VERIFY_HOST" "${PROXY_PORTS[0]}"
    else
        printf '    curl -H "X-Worldsim-Token: $(cat %q)" http://%s:%s/\n' \
            "$TOKEN_FILE" "$HOST_IP" "${PROXY_PORTS[0]}"
    fi
    echo ""

    echo "    Without the token (should return 403):"
    if [[ "$current_proxy_scheme" == "https" ]]; then
        printf '    curl --resolve "%s:%s:%s" https://%s:%s/\n' \
            "$TLS_VERIFY_HOST" "${PROXY_PORTS[0]}" "$HOST_IP" "$TLS_VERIFY_HOST" "${PROXY_PORTS[0]}"
    else
        printf '    curl http://%s:%s/\n' "$HOST_IP" "${PROXY_PORTS[0]}"
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    log "deploy_benchmark_proxy.sh — authenticated reverse proxy for Phase 0c"
    printf '    HOST_IP     = %s\n' "$HOST_IP"
    printf '    SSH_USER    = %s\n' "$SSH_USER"
    printf '    PORT_OFFSET = %s\n' "$PORT_OFFSET"
    if [[ -n "$TLS_CERT_FILE" || -n "$TLS_KEY_FILE" ]]; then
        printf '    TLS         = %s\n' "enabled"
    else
        printf '    TLS         = %s\n' "disabled"
    fi

    if [[ "$VIA_SSM" == "1" ]]; then
        if [[ -z "$SSM_INSTANCE_ID" ]]; then
            echo "ERROR: --via-ssm requires --ssm-instance-id (or \$SSM_INSTANCE_ID)" >&2
            exit 1
        fi
        if ! command -v aws >/dev/null 2>&1; then
            echo "ERROR: --via-ssm requires the aws CLI" >&2
            exit 1
        fi
        printf '    TRANSPORT   = SSM (%s)\n' "$SSM_INSTANCE_ID"
    else
        if [[ ! -f "$SSH_KEY" ]]; then
            echo "ERROR: SSH key not found: $SSH_KEY (or pass --via-ssm)" >&2
            exit 1
        fi
        printf '    TRANSPORT   = SSH (%s)\n' "$SSH_KEY"
    fi

    if [[ -n "$TLS_CERT_FILE" && -z "$TLS_KEY_FILE" ]] || [[ -z "$TLS_CERT_FILE" && -n "$TLS_KEY_FILE" ]]; then
        echo "ERROR: --tls-cert and --tls-key must be provided together" >&2
        exit 1
    fi
    if [[ -n "$TLS_CERT_FILE" && -z "$TLS_VERIFY_HOST" ]]; then
        echo "ERROR: --tls-verify-host is required when TLS is enabled" >&2
        exit 1
    fi
    if [[ -n "$TLS_VERIFY_HOST" ]] && ! validate_verify_host "$TLS_VERIFY_HOST"; then
        echo "ERROR: invalid --tls-verify-host: $TLS_VERIFY_HOST" >&2
        exit 1
    fi
    if [[ -z "$TLS_CERT_FILE" && "$ALLOW_INSECURE_HTTP" != "1" ]]; then
        echo "ERROR: public proxy deployment requires TLS by default; provide --tls-cert/--tls-key/--tls-verify-host or opt in with --insecure-http" >&2
        exit 1
    fi

    load_port_map || exit 1
    ensure_token  || exit 1

    step_install_nginx        || echo "    (continuing past step 1 warning)"
    step_deploy_config        || { echo "ERROR: failed to deploy config" >&2; exit 1; }
    step_test_and_restart_nginx || { echo "ERROR: nginx config test/restart failed" >&2; exit 1; }
    if ! step_verify_proxy; then
        if [[ "$ALLOW_VERIFY_FAILURE" == "1" ]]; then
            echo "    (continuing past step 4 warning due to --allow-verify-failure)"
        else
            echo "ERROR: proxy verification failed" >&2
            exit 1
        fi
    fi
    step_print_summary

    echo "==> Proxy deployed. Open the security group ports above, then test."
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
    exit $?
fi
