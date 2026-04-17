#!/usr/bin/env bash
# benchmark_host.sh - start / stop / status CLI for the r5.8xlarge bench host.
#
# Keeps the benchmark EC2 instance stopped between runs ($80/mo EBS keeps
# the data, $3.60/mo keeps the Elastic IP attached). Reads the instance id
# from .benchmark_host_id at the repo root (gitignored).
#
# Usage:
#   ./scripts/benchmark_host.sh start        # ec2 start + SSH wait + compose start + health check
#   ./scripts/benchmark_host.sh stop         # compose stop + ec2 stop
#   ./scripts/benchmark_host.sh status       # print instance state + per-replica /init health
#
# Environment:
#   AWS_REGION           (default us-east-2)
#   BENCHMARK_HOST_ID    (instance id; fallback to .benchmark_host_id file)
#   BENCHMARK_TOPOLOGY   (scale or legacy; fallback .benchmark_topology file)
#   SSH_KEY              (default ~/.ssh/webarena-key.pem)
#   SSH_USER             (default ubuntu)
#   COMPOSE_FILE_REMOTE  (path on the host; default /home/ubuntu/docker-compose.yml)
#   PORT_MAP_FILE        (explicit port map; else chosen from topology)
#   PROXY_TOKEN_FILE     (default .proxy_token)
#   PROXY_METADATA_FILE  (default .benchmark_proxy_metadata written by deploy script)
#   PROXY_PORT_MAP_FILE  (default .benchmark_proxy_ports.conf written by deploy script)
#   PROXY_PORT_OFFSET    (default 10000 when proxy_port is omitted)
#   PROXY_SCHEME         (optional override; otherwise read from metadata file)
#   PROXY_VERIFY_HOST    (optional override; otherwise read from metadata file)
#   ALLOW_PROXY_SKIP     (default 0; set to 1 to skip proxy verification when config is absent)
#   VERIFY_PROXY_ON_START (default 0; set to 1 to fail start on proxy errors)
#   HEALTH_TIMEOUT       (default 300 seconds)
#
# Requires: aws cli (logged in), jq, ssh, curl. The script never creates or
# destroys instances - that happens once, by hand, via `aws ec2 run-instances`.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AWS_REGION="${AWS_REGION:-us-east-2}"
SSH_KEY_RAW="${SSH_KEY:-$HOME/.ssh/webarena-key.pem}"
SSH_KEY="${SSH_KEY_RAW/#\~/$HOME}"
SSH_USER="${SSH_USER:-ubuntu}"
COMPOSE_FILE_REMOTE="${COMPOSE_FILE_REMOTE:-/home/ubuntu/docker-compose.yml}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"
BENCHMARK_TOPOLOGY="${BENCHMARK_TOPOLOGY:-}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INSTANCE_ID_FILE="$REPO_ROOT/.benchmark_host_id"
TOPOLOGY_FILE="${TOPOLOGY_FILE:-$REPO_ROOT/.benchmark_topology}"
PORT_MAP_FILE="${PORT_MAP_FILE:-}"
SCALE_PORT_MAP_FILE="${SCALE_PORT_MAP_FILE:-$REPO_ROOT/scripts/proxy_ports.conf}"
PROXY_TOKEN_FILE="${PROXY_TOKEN_FILE:-$REPO_ROOT/.proxy_token}"
PROXY_METADATA_FILE="${PROXY_METADATA_FILE:-$REPO_ROOT/.benchmark_proxy_metadata}"
PROXY_PORT_MAP_FILE="${PROXY_PORT_MAP_FILE:-$REPO_ROOT/.benchmark_proxy_ports.conf}"
PROXY_PORT_OFFSET="${PROXY_PORT_OFFSET:-10000}"
PROXY_SCHEME="${PROXY_SCHEME:-}"
PROXY_VERIFY_HOST="${PROXY_VERIFY_HOST:-}"
ALLOW_PROXY_SKIP="${ALLOW_PROXY_SKIP:-0}"
VERIFY_PROXY_ON_START="${VERIFY_PROXY_ON_START:-0}"
PROXY_METADATA_LOADED=0

DEFAULT_PORT_MAP="shopping:7770
shopping_admin:7780
gitlab:8023
reddit:9999
wikipedia:8888
map:3030"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() {
    printf '\n==> %s\n' "$*"
}

err() {
    printf 'ERROR: %s\n' "$*" >&2
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        err "required command not found: $1"
        exit 1
    fi
}

resolve_instance_id() {
    if [[ -n "${BENCHMARK_HOST_ID:-}" ]]; then
        echo "$BENCHMARK_HOST_ID"
        return
    fi
    if [[ -f "$INSTANCE_ID_FILE" ]]; then
        cat "$INSTANCE_ID_FILE"
        return
    fi
    err "no instance id. Set BENCHMARK_HOST_ID or write to $INSTANCE_ID_FILE"
    exit 2
}

aws_describe() {
    # echo JSON for the single instance
    local id="$1"
    aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$id" \
        --output json
}

ssh_opts=(
    -i "$SSH_KEY"
    -o StrictHostKeyChecking=accept-new
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=120
    -o ConnectTimeout=15
)

ssh_host() {
    # ssh_host <ip> <command...>
    local ip="$1"; shift
    ssh "${ssh_opts[@]}" "$SSH_USER@$ip" "$@"
}

# ---------------------------------------------------------------------------
# Instance state helpers
# ---------------------------------------------------------------------------

get_state_and_ip() {
    # get_state_and_ip <id>; prints "<state> <public_ip>" (ip may be empty)
    local id="$1"
    local json
    json=$(aws_describe "$id")
    local state ip
    state=$(echo "$json" | jq -r '.Reservations[0].Instances[0].State.Name')
    ip=$(echo "$json" | jq -r '.Reservations[0].Instances[0].PublicIpAddress // ""')
    printf '%s %s\n' "$state" "$ip"
}

wait_for_state() {
    # wait_for_state <id> <target-state> <timeout-seconds>
    local id="$1" target="$2" timeout="$3"
    local start
    start=$(date +%s)
    while :; do
        local s ip
        read -r s ip <<< "$(get_state_and_ip "$id")"
        if [[ "$s" == "$target" ]]; then
            printf '    reached state=%s ip=%s\n' "$s" "$ip"
            return 0
        fi
        local now
        now=$(date +%s)
        if (( now - start > timeout )); then
            err "timed out waiting for state=$target (current=$s)"
            return 1
        fi
        printf '    state=%s (waiting for %s)...\n' "$s" "$target"
        sleep 5
    done
}

wait_for_ssh() {
    # wait_for_ssh <ip> <timeout>
    local ip="$1" timeout="$2"
    local start
    start=$(date +%s)
    while :; do
        if ssh -o BatchMode=yes -o ConnectTimeout=5 "${ssh_opts[@]}" \
                "$SSH_USER@$ip" 'echo ok' >/dev/null 2>&1; then
            printf '    ssh ready\n'
            return 0
        fi
        local now
        now=$(date +%s)
        if (( now - start > timeout )); then
            err "timed out waiting for ssh on $ip"
            return 1
        fi
        sleep 5
    done
}

# ---------------------------------------------------------------------------
# Container health and env-ctrl readiness via SSH.
# ---------------------------------------------------------------------------

wait_for_containers_healthy() {
    # wait_for_containers_healthy <ip>
    # Polls docker compose status via SSH, requires all services to be
    # `running`. Delegates /init probing to the caller (or a later phase).
    local ip="$1"
    local start
    start=$(date +%s)
    while :; do
        local running_count total
        running_count=$(ssh_host "$ip" "sudo docker compose -f $COMPOSE_FILE_REMOTE ps --format '{{.State}}' 2>/dev/null | grep -c '^running' || true") || true
        total=$(ssh_host "$ip" "sudo docker compose -f $COMPOSE_FILE_REMOTE ps --services 2>/dev/null | wc -l" | tr -d ' ')
        running_count=$(echo "$running_count" | tr -d ' ')
        printf '    %s/%s containers running\n' "$running_count" "$total"
        if [[ -n "$total" && "$running_count" == "$total" && "$total" != "0" ]]; then
            return 0
        fi
        local now
        now=$(date +%s)
        if (( now - start > HEALTH_TIMEOUT )); then
            err "timed out waiting for all containers to run ($running_count/$total)"
            return 1
        fi
        sleep 5
    done
}

port_map_file() {
    echo "$PORT_MAP_FILE"
}

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
    if [[ -n "$BENCHMARK_TOPOLOGY" ]]; then
        if ! validate_topology "$BENCHMARK_TOPOLOGY"; then
            err "invalid BENCHMARK_TOPOLOGY: $BENCHMARK_TOPOLOGY"
            return 1
        fi
        echo "$BENCHMARK_TOPOLOGY"
        return 0
    fi
    if [[ -f "$TOPOLOGY_FILE" ]]; then
        local topology
        topology="$(tr -d '[:space:]' < "$TOPOLOGY_FILE")"
        if ! validate_topology "$topology"; then
            err "invalid topology in $TOPOLOGY_FILE: $topology"
            return 1
        fi
        echo "$topology"
        return 0
    fi
    err "no topology configured. Set BENCHMARK_TOPOLOGY, PORT_MAP_FILE, or write scale/legacy to $TOPOLOGY_FILE"
    return 1
}

load_port_map_text() {
    local source input
    if [[ -n "$PORT_MAP_FILE" ]]; then
        if [[ ! -f "$PORT_MAP_FILE" ]]; then
            err "port map file not found: $PORT_MAP_FILE"
            return 1
        fi
        input="$(cat "$PORT_MAP_FILE")"
        source="$PORT_MAP_FILE"
    else
        local topology
        topology="$(resolve_benchmark_topology)" || return 1
        if [[ "$topology" == "scale" ]]; then
            if [[ ! -f "$SCALE_PORT_MAP_FILE" ]]; then
                err "scale port map file not found: $SCALE_PORT_MAP_FILE"
                return 1
            fi
            input="$(cat "$SCALE_PORT_MAP_FILE")"
            source="$SCALE_PORT_MAP_FILE (topology=$topology)"
        else
            input="$DEFAULT_PORT_MAP"
            source="built-in legacy WebArena map (topology=$topology)"
        fi
    fi

    PORT_MAP_TEXT="$input"
    PORT_MAP_SOURCE="$source"
}

load_proxy_port_map_text() {
    if [[ -f "$PROXY_PORT_MAP_FILE" ]]; then
        PORT_MAP_TEXT="$(cat "$PROXY_PORT_MAP_FILE")"
        PORT_MAP_SOURCE="$PROXY_PORT_MAP_FILE (proxy metadata)"
        return 0
    fi
    load_port_map_text
}

proxy_token_file() {
    echo "$PROXY_TOKEN_FILE"
}

load_proxy_metadata() {
    if [[ "$PROXY_METADATA_LOADED" == "1" ]]; then
        return 0
    fi
    if [[ -f "$PROXY_METADATA_FILE" ]]; then
        while IFS='=' read -r key value; do
            case "$key" in
                PROXY_SCHEME)
                    if [[ -z "$PROXY_SCHEME" ]]; then
                        PROXY_SCHEME="$value"
                    fi
                    ;;
                PROXY_VERIFY_HOST)
                    if [[ -z "$PROXY_VERIFY_HOST" ]]; then
                        PROXY_VERIFY_HOST="$value"
                    fi
                    ;;
                PROXY_PORT_MAP_FILE)
                    PROXY_PORT_MAP_FILE="$value"
                    ;;
            esac
        done < "$PROXY_METADATA_FILE"
    fi
    if [[ -n "$PROXY_SCHEME" && "$PROXY_SCHEME" != "http" && "$PROXY_SCHEME" != "https" ]]; then
        PROXY_SCHEME=""
    fi
    if [[ -n "$PROXY_VERIFY_HOST" ]] && ! validate_verify_host "$PROXY_VERIFY_HOST"; then
        PROXY_VERIFY_HOST=""
    fi
    PROXY_METADATA_LOADED=1
}

proxy_http_code_is_healthy() {
    local code="$1"
    [[ "$code" =~ ^[23][0-9][0-9]$ ]]
}

should_allow_proxy_skip() {
    [[ "$ALLOW_PROXY_SKIP" == "1" ]]
}

probe_init_once() {
    # probe_init_once <ip>
    local ip="$1"
    load_port_map_text || return 1

    local ok=0 total=0
    while IFS= read -r line; do
        line="${line%%#*}"
        line=$(echo "$line" | xargs)
        [[ -z "$line" ]] && continue
        local name real proxy
        IFS=: read -r name real proxy <<< "$line"
        if ! validate_port_number "$real"; then
            printf '    %-24s  INVALID PORT  %s\n' "$name" "$real"
            total=$((total + 1))
            continue
        fi
        local envctrl_port=$((real + 1))
        local code
        code=$(ssh_host "$ip" "curl -sS -o /dev/null -w '%{http_code}' -X POST --max-time 10 -H 'Content-Type: application/json' --data '{}' http://127.0.0.1:${envctrl_port}/init 2>/dev/null || echo 000")
        code=$(echo "$code" | tr -d '[:space:]')
        printf '    %-24s  HTTP %s  http://127.0.0.1:%s/init (via SSH)\n' "$name" "$code" "$envctrl_port"
        total=$((total + 1))
        [[ "$code" == "200" ]] && ok=$((ok + 1))
    done <<< "$PORT_MAP_TEXT"
    printf '    source: %s\n' "$PORT_MAP_SOURCE"
    printf '    %s / %s endpoints healthy\n' "$ok" "$total"
    [[ "$total" -gt 0 && "$ok" -eq "$total" ]]
}

should_verify_proxy_on_start() {
    [[ "$VERIFY_PROXY_ON_START" == "1" ]]
}

proxy_verification_configured() {
    local token_file
    token_file="$(proxy_token_file)"
    load_proxy_metadata
    if [[ ! -s "$token_file" ]]; then
        return 1
    fi
    local token
    token="$(cat "$token_file")"
    if ! validate_proxy_token "$token"; then
        return 1
    fi
    if [[ -z "$PROXY_SCHEME" ]]; then
        return 1
    fi
    if [[ "$PROXY_SCHEME" == "https" && -z "$PROXY_VERIFY_HOST" ]]; then
        return 1
    fi
    return 0
}

probe_proxy_once() {
    # probe_proxy_once <ip>
    local ip="$1"
    local token_file
    token_file="$(proxy_token_file)"
    load_proxy_metadata
    if ! load_proxy_port_map_text; then
        if should_allow_proxy_skip; then
            printf '    (no port map configuration; skipping proxy probe due to ALLOW_PROXY_SKIP=1)\n'
            return 0
        fi
        return 1
    fi
    if [[ ! -f "$token_file" ]]; then
        if should_allow_proxy_skip; then
            printf '    (no %s; skipping proxy probe due to ALLOW_PROXY_SKIP=1)\n' "$token_file"
            return 0
        fi
        printf '    (no %s; proxy probe cannot run)\n' "$token_file"
        return 1
    fi

    local token
    token="$(cat "$token_file")"
    if [[ -z "$token" ]]; then
        if should_allow_proxy_skip; then
            printf '    (%s is empty; skipping proxy probe due to ALLOW_PROXY_SKIP=1)\n' "$token_file"
            return 0
        fi
        printf '    (%s is empty; proxy probe cannot run)\n' "$token_file"
        return 1
    fi
    if ! validate_proxy_token "$token"; then
        printf '    (%s contains an invalid proxy token; expected 64 hex chars)\n' "$token_file"
        return 1
    fi

    local ok=0 total=0
    local scheme
    scheme="${PROXY_SCHEME:-https}"
    while IFS= read -r line; do
        line="${line%%#*}"
        line=$(echo "$line" | xargs)
        [[ -z "$line" ]] && continue
        local name real proxy
        IFS=: read -r name real proxy <<< "$line"
        [[ -z "$proxy" ]] && proxy=$((real + PROXY_PORT_OFFSET))
        if ! validate_port_number "$real" || ! validate_port_number "$proxy"; then
            printf '    %-24s  INVALID PORT MAP  real=%s proxy=%s\n' "$name" "$real" "$proxy"
            total=$((total + 1))
            continue
        fi
        local authed_code unauthed_code
        if [[ "$scheme" == "https" ]]; then
            local header_file remote_cmd
            header_file=$(mktemp /tmp/worldsim-proxy-header-XXXXXX)
            printf 'X-Worldsim-Token: %s\n' "$token" > "$header_file"
            remote_cmd="header_file=\$(mktemp); trap 'rm -f \"\$header_file\"' EXIT; cat > \"\$header_file\"; curl -sS -o /dev/null -w '%{http_code}' --max-time 10 --resolve '${PROXY_VERIFY_HOST}:${proxy}:127.0.0.1' -H @\"\$header_file\" 'https://${PROXY_VERIFY_HOST}:${proxy}/' 2>/dev/null || echo 000"
            authed_code=$(ssh_host "$ip" "$remote_cmd" < "$header_file")
            rm -f "$header_file"
            unauthed_code=$(ssh_host "$ip" "curl -sS -o /dev/null -w '%{http_code}' \
                --max-time 10 \
                --resolve '${PROXY_VERIFY_HOST}:${proxy}:127.0.0.1' \
                'https://${PROXY_VERIFY_HOST}:${proxy}/' 2>/dev/null || echo 000")
            printf '    %-24s  auth=%s unauth=%s  https://%s:%s/ (via SSH)\n' \
                "$name" "$authed_code" "$unauthed_code" "$PROXY_VERIFY_HOST" "$proxy"
        else
            local header_file remote_cmd
            header_file=$(mktemp /tmp/worldsim-proxy-header-XXXXXX)
            printf 'X-Worldsim-Token: %s\n' "$token" > "$header_file"
            remote_cmd="header_file=\$(mktemp); trap 'rm -f \"\$header_file\"' EXIT; cat > \"\$header_file\"; curl -sS -o /dev/null -w '%{http_code}' --max-time 10 -H @\"\$header_file\" 'http://127.0.0.1:${proxy}/' 2>/dev/null || echo 000"
            authed_code=$(ssh_host "$ip" "$remote_cmd" < "$header_file")
            rm -f "$header_file"
            unauthed_code=$(ssh_host "$ip" "curl -sS -o /dev/null -w '%{http_code}' \
                --max-time 10 \
                'http://127.0.0.1:${proxy}/' 2>/dev/null || echo 000")
            printf '    %-24s  auth=%s unauth=%s  http://127.0.0.1:%s/ (via SSH)\n' \
                "$name" "$authed_code" "$unauthed_code" "$proxy"
        fi
        total=$((total + 1))
        authed_code=$(echo "$authed_code" | tr -d '[:space:]')
        unauthed_code=$(echo "$unauthed_code" | tr -d '[:space:]')
        if proxy_http_code_is_healthy "$authed_code" && [[ "$unauthed_code" == "403" ]]; then
            ok=$((ok + 1))
        fi
    done <<< "$PORT_MAP_TEXT"
    if [[ "$total" -eq 0 ]]; then
        if should_allow_proxy_skip; then
            printf '    (no proxy rows found; skipping proxy probe due to ALLOW_PROXY_SKIP=1)\n'
            return 0
        fi
        printf '    (no proxy rows found; proxy probe cannot run)\n'
        return 1
    fi
    printf '    source: %s\n' "$PORT_MAP_SOURCE"
    printf '    %s / %s proxy endpoints reachable\n' "$ok" "$total"
    [[ "$ok" -eq "$total" ]]
}

wait_for_init_healthy() {
    # wait_for_init_healthy <ip>
    local ip="$1"
    local start
    start=$(date +%s)
    while :; do
        if probe_init_once "$ip"; then
            return 0
        fi
        local now
        now=$(date +%s)
        if (( now - start > HEALTH_TIMEOUT )); then
            err "timed out waiting for /init readiness"
            return 1
        fi
        sleep 5
    done
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd_start() {
    require_cmd aws
    require_cmd jq
    require_cmd ssh
    require_cmd curl

    local id
    id=$(resolve_instance_id)

    log "Starting EC2 instance $id in $AWS_REGION"
    aws ec2 start-instances --region "$AWS_REGION" --instance-ids "$id" >/dev/null

    wait_for_state "$id" running 180

    local state ip
    read -r state ip <<< "$(get_state_and_ip "$id")"
    if [[ -z "$ip" ]]; then
        err "instance is running but no public IP. If an Elastic IP should be attached, check the console."
        exit 1
    fi

    log "Waiting for SSH on $ip"
    wait_for_ssh "$ip" 120

    log "Starting docker compose stack on $ip"
    ssh_host "$ip" "sudo docker compose -f $COMPOSE_FILE_REMOTE start"

    log "Waiting for containers to reach running state"
    wait_for_containers_healthy "$ip" || true

    log "Waiting for /init readiness on real env-ctrl ports"
    wait_for_init_healthy "$ip"

    if should_verify_proxy_on_start; then
        if proxy_verification_configured; then
            log "Probing authenticated proxy path"
            probe_proxy_once "$ip"
        else
            err "VERIFY_PROXY_ON_START=1 requires a valid token and proxy metadata"
            err "Expected token plus metadata in $PROXY_METADATA_FILE"
            exit 1
        fi
    else
        log "Skipping proxy verification during start"
        echo "    Proxy is for Phase 0c verification only; set VERIFY_PROXY_ON_START=1 to enforce it here."
    fi

    log "Start complete. Instance $id is up at $ip."
    echo "    Next: run Phase 0c or Phase 3."
}

cmd_stop() {
    require_cmd aws
    require_cmd jq
    require_cmd curl

    local id
    id=$(resolve_instance_id)

    local state ip
    read -r state ip <<< "$(get_state_and_ip "$id")"
    if [[ "$state" != "running" ]]; then
        log "instance is already $state; skipping compose stop"
    else
        log "Stopping docker compose containers (preserves volumes)"
        if [[ -n "$ip" ]]; then
            # `stop` not `down`, so volumes and networks survive.
            ssh_host "$ip" "sudo docker compose -f $COMPOSE_FILE_REMOTE stop" || \
                err "compose stop returned non-zero; continuing to ec2 stop"
        fi
    fi

    log "Stopping EC2 instance $id"
    aws ec2 stop-instances --region "$AWS_REGION" --instance-ids "$id" >/dev/null

    wait_for_state "$id" stopped 300
    log "Instance $id is stopped. EBS + Elastic IP retained."
}

cmd_status() {
    require_cmd aws
    require_cmd jq
    require_cmd ssh
    require_cmd curl

    local id
    id=$(resolve_instance_id)

    local state ip
    read -r state ip <<< "$(get_state_and_ip "$id")"
    log "Instance $id"
    printf '    region : %s\n' "$AWS_REGION"
    printf '    state  : %s\n' "$state"
    printf '    ip     : %s\n' "${ip:-<none>}"

    if [[ "$state" != "running" || -z "$ip" ]]; then
        return 0
    fi

    log "Container state"
    ssh_host "$ip" "sudo docker compose -f $COMPOSE_FILE_REMOTE ps --format 'table {{.Service}}\t{{.State}}\t{{.Status}}' 2>/dev/null" || true

    log "/init health probe"
    local init_ok=0
    if ! probe_init_once "$ip"; then
        init_ok=1
    fi

    local proxy_ok=0
    if proxy_verification_configured; then
        log "Authenticated proxy probe"
        if ! probe_proxy_once "$ip"; then
            proxy_ok=1
        fi
    else
        log "Authenticated proxy probe"
        printf '    skipped (need non-empty %s and proxy metadata in %s; proxy verification is optional for host status)\n' "$PROXY_TOKEN_FILE" "$PROXY_METADATA_FILE"
    fi

    return $(( init_ok || proxy_ok ))
}

usage() {
    cat <<EOF
Usage: $0 <start|stop|status>

  start   aws ec2 start + SSH wait + docker compose start + health poll
  stop    docker compose stop + aws ec2 stop (retains volumes and IP)
  status  print instance state, container state, and /init probe results

Instance id read from \$BENCHMARK_HOST_ID or $INSTANCE_ID_FILE.
Port map comes from \$PORT_MAP_FILE or BENCHMARK_TOPOLOGY/$TOPOLOGY_FILE.
EOF
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    cmd="${1:-}"
    case "$cmd" in
        start)   cmd_start ;;
        stop)    cmd_stop ;;
        status)  cmd_status ;;
        ""|-h|--help) usage ;;
        *) err "unknown command: $cmd"; usage; exit 2 ;;
    esac
fi
