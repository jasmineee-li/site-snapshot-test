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
#   SSH_KEY              (default ~/.ssh/webarena-key.pem)
#   SSH_USER             (default ubuntu)
#   COMPOSE_FILE_REMOTE  (path on the host; default /home/ubuntu/docker-compose.yml)
#   PROXY_TOKEN_FILE     (default .proxy_token)
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
PROXY_TOKEN_FILE="${PROXY_TOKEN_FILE:-.proxy_token}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INSTANCE_ID_FILE="$REPO_ROOT/.benchmark_host_id"

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
# Container health via nginx proxy (avoids reconfiguring SSH to reach
# 127.0.0.1 ports on the host).
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

    log "Start complete. Instance $id is up at $ip."
    echo "    Next: verify /init endpoints and run Phase 0c."
}

cmd_stop() {
    require_cmd aws
    require_cmd jq

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
    if [[ ! -f "$PROXY_TOKEN_FILE" ]]; then
        printf '    (no %s; skipping authenticated /init probe)\n' "$PROXY_TOKEN_FILE"
        return 0
    fi
    local token
    token=$(cat "$PROXY_TOKEN_FILE")

    # Scrape proxy ports from scripts/proxy_ports.conf if present.
    local ports_file="$REPO_ROOT/scripts/proxy_ports.conf"
    if [[ ! -f "$ports_file" ]]; then
        printf '    (no %s; skipping /init probe)\n' "$ports_file"
        return 0
    fi

    local ok=0 total=0
    while IFS= read -r line; do
        line="${line%%#*}"
        line=$(echo "$line" | xargs)
        [[ -z "$line" ]] && continue
        local name real proxy
        IFS=: read -r name real proxy <<< "$line"
        [[ -z "$proxy" ]] && proxy=$((real + 10000))
        local envctrl_proxy=$((proxy + 1))
        local code
        code=$(curl -sS -o /dev/null -w '%{http_code}' -X POST \
            --max-time 10 \
            -H "X-Worldsim-Token: $token" \
            -H 'Content-Type: application/json' \
            --data '{}' \
            "http://$ip:$envctrl_proxy/init" 2>/dev/null || echo 000)
        printf '    %-24s  HTTP %s  http://%s:%s/init\n' "$name" "$code" "$ip" "$envctrl_proxy"
        total=$((total + 1))
        [[ "$code" == "200" ]] && ok=$((ok + 1))
    done < "$ports_file"
    printf '    %s / %s endpoints healthy\n' "$ok" "$total"
}

usage() {
    cat <<EOF
Usage: $0 <start|stop|status>

  start   aws ec2 start + SSH wait + docker compose start + health poll
  stop    docker compose stop + aws ec2 stop (retains volumes and IP)
  status  print instance state, container state, and /init probe results

Instance id read from \$BENCHMARK_HOST_ID or $INSTANCE_ID_FILE.
EOF
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

cmd="${1:-}"
case "$cmd" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    status)  cmd_status ;;
    ""|-h|--help) usage ;;
    *) err "unknown command: $cmd"; usage; exit 2 ;;
esac
