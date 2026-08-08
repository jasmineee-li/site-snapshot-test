#!/usr/bin/env bash
# check_proxy_drift.sh — Diff the live /etc/nginx/conf.d/worldsim-proxy.conf
# on a benchmark EC2 instance against what `deploy_benchmark_proxy.sh` would
# render from the checked-in template + port map + token.
#
# Source of truth: this repo. If they disagree, re-run deploy_benchmark_proxy
# (or explain the drift in a commit message and update the script if a new
# directive is warranted).
#
# Two layers of check:
#   1. File-on-disk vs template (always runs).
#   2. Runtime verification (opt in via ``--verify-runtime``):
#      a. ``nginx -t`` on the host — confirms the on-disk config is loadable.
#      b. ``systemctl is-active nginx`` — confirms the daemon is still up
#         (a bad prior reload through ``systemctl reload`` can leave it
#         stopped on some distros; see the Arch forum thread cited in
#         the research brief).
#      c. Worker-PID start time vs config-file mtime — if the oldest
#         worker started BEFORE the config was last modified, nginx has
#         not reloaded since the edit and the file is on disk but not
#         in memory. Non-destructive — no reload is forced.
#      d. Recent ``[emerg]`` lines in /var/log/nginx/error.log — flags
#         a prior failed reload.
#
# The runtime verify does NOT introspect nginx's loaded-in-memory config
# directly (open-source nginx exposes no such API; ``nginx -T`` re-parses
# from disk and would give a false positive when the running process is
# holding stale config). The combination above approximates it.
#
# Usage:
#   ./scripts/check_proxy_drift.sh --host-config configs/benchmark_hosts/r8a.yaml \
#       --via-ssm --ssm-instance-id i-0abc... --insecure-http --topology legacy \
#       --verify-runtime
#
# All flags not listed below are passed through to deploy_benchmark_proxy.sh
# in --print-only mode (not implemented — see note below; for now the
# check generates the expected config by sourcing deploy_benchmark_proxy.sh
# and calling `generate_nginx_config`).

set -euo pipefail

VIA_SSM=0
SSM_INSTANCE_ID="${SSM_INSTANCE_ID:-}"
SSM_REGION="${SSM_REGION:-us-east-2}"
HOST_IP="${HOST_IP:-}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/webarena-key.pem}"
SSH_USER="${SSH_USER:-ubuntu}"

# Args we forward to generate_nginx_config.
HOST_CONFIG=""
TOKEN_FILE=""
PORT_MAP_FILE=""
PORT_OFFSET="${PORT_OFFSET:-10000}"
TLS_CERT_FILE="${TLS_CERT_FILE:-}"
TLS_KEY_FILE="${TLS_KEY_FILE:-}"
ALLOW_INSECURE_HTTP="${ALLOW_INSECURE_HTTP:-0}"
BENCHMARK_TOPOLOGY="${BENCHMARK_TOPOLOGY:-}"
USE_LEGACY_DEFAULT_MAP=0
VERIFY_RUNTIME=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --via-ssm) VIA_SSM=1; shift ;;
        --ssm-instance-id) SSM_INSTANCE_ID="$2"; shift 2 ;;
        --ssm-region) SSM_REGION="$2"; shift 2 ;;
        --host) HOST_IP="$2"; shift 2 ;;
        --ssh-key) SSH_KEY="$2"; shift 2 ;;
        --ssh-user) SSH_USER="$2"; shift 2 ;;
        --host-config) HOST_CONFIG="$2"; shift 2 ;;
        --token-file) TOKEN_FILE="$2"; shift 2 ;;
        --port-map) PORT_MAP_FILE="$2"; shift 2 ;;
        --port-offset) PORT_OFFSET="$2"; shift 2 ;;
        --tls-cert) TLS_CERT_FILE="$2"; shift 2 ;;
        --tls-key) TLS_KEY_FILE="$2"; shift 2 ;;
        --topology) BENCHMARK_TOPOLOGY="$2"; shift 2 ;;
        --use-legacy-default-map) USE_LEGACY_DEFAULT_MAP=1; shift ;;
        --insecure-http) ALLOW_INSECURE_HTTP=1; shift ;;
        --verify-runtime) VERIFY_RUNTIME=1; shift ;;
        --help|-h)
            sed -n '1,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Fetch the live config.
fetch_live() {
    if [[ "$VIA_SSM" == "1" ]]; then
        [[ -z "$SSM_INSTANCE_ID" ]] && { echo "ERROR: --ssm-instance-id required with --via-ssm" >&2; return 1; }
        local cmd_id status
        cmd_id=$(aws ssm send-command \
            --instance-ids "$SSM_INSTANCE_ID" \
            --document-name AWS-RunShellScript \
            --parameters 'commands=["cat /etc/nginx/conf.d/worldsim-proxy.conf"]' \
            --region "$SSM_REGION" \
            --query "Command.CommandId" --output text)
        local deadline=$(( SECONDS + 60 ))
        while (( SECONDS < deadline )); do
            status=$(aws ssm get-command-invocation \
                --command-id "$cmd_id" \
                --instance-id "$SSM_INSTANCE_ID" \
                --region "$SSM_REGION" \
                --query "Status" --output text 2>/dev/null || echo "")
            case "$status" in Success|Failed|Cancelled|TimedOut) break ;; esac
            sleep 2
        done
        aws ssm get-command-invocation \
            --command-id "$cmd_id" \
            --instance-id "$SSM_INSTANCE_ID" \
            --region "$SSM_REGION" \
            --query "StandardOutputContent" --output text
    else
        [[ -z "$HOST_IP" ]] && { echo "ERROR: --host required without --via-ssm" >&2; return 1; }
        ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "$SSH_USER@$HOST_IP" \
            'cat /etc/nginx/conf.d/worldsim-proxy.conf'
    fi
}

# Render the expected config by sourcing deploy_benchmark_proxy.sh and
# calling its helpers — single source of truth.
render_expected() {
    # Snapshot our args before sourcing (the sourced file resets them to
    # its defaults at the top — USE_LEGACY_DEFAULT_MAP=0, etc.).
    local saved_legacy="$USE_LEGACY_DEFAULT_MAP"
    local saved_topology="$BENCHMARK_TOPOLOGY"
    local saved_port_map="$PORT_MAP_FILE"
    local saved_port_offset="$PORT_OFFSET"
    local saved_tls_cert="$TLS_CERT_FILE"
    local saved_tls_key="$TLS_KEY_FILE"

    # shellcheck disable=SC1091
    source "$REPO_ROOT/scripts/deploy_benchmark_proxy.sh" || true

    # Re-apply our CLI-derived values after the source.
    USE_LEGACY_DEFAULT_MAP="$saved_legacy"
    BENCHMARK_TOPOLOGY="$saved_topology"
    PORT_MAP_FILE="$saved_port_map"
    PORT_OFFSET="$saved_port_offset"
    TLS_CERT_FILE="$saved_tls_cert"
    TLS_KEY_FILE="$saved_tls_key"
    TOKEN_FILE="${TOKEN_FILE:-$REPO_ROOT/.proxy_token}"
    NEW_TOKEN=0

    load_port_map >/dev/null || { echo "ERROR: load_port_map failed" >&2; return 1; }
    ensure_token >/dev/null || { echo "ERROR: ensure_token failed (token file required)" >&2; return 1; }
    generate_nginx_config
    printf '%s' "$NGINX_CONFIG"
}

# The sourcing trick above relies on deploy_benchmark_proxy.sh's
# `if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi` guard —
# that guard evaluates false when we source the file from here, so
# main() doesn't execute.

# Run a single shell script on the host (SSM or SSH). Emits stdout; returns
# whatever the shell exits with so callers can gate on success.
run_on_host() {
    local script="$1"
    if [[ "$VIA_SSM" == "1" ]]; then
        [[ -z "$SSM_INSTANCE_ID" ]] && { echo "ERROR: --ssm-instance-id required with --via-ssm" >&2; return 1; }
        local b64 cmd_id status
        b64="$(printf '%s' "$script" | base64 | tr -d '\n')"
        cmd_id=$(aws ssm send-command \
            --instance-ids "$SSM_INSTANCE_ID" \
            --document-name AWS-RunShellScript \
            --parameters "commands=[\"echo '$b64' | base64 -d | bash\"]" \
            --region "$SSM_REGION" \
            --query "Command.CommandId" --output text)
        local deadline=$(( SECONDS + 60 ))
        while (( SECONDS < deadline )); do
            status=$(aws ssm get-command-invocation \
                --command-id "$cmd_id" \
                --instance-id "$SSM_INSTANCE_ID" \
                --region "$SSM_REGION" \
                --query "Status" --output text 2>/dev/null || echo "")
            case "$status" in Success|Failed|Cancelled|TimedOut) break ;; esac
            sleep 2
        done
        aws ssm get-command-invocation \
            --command-id "$cmd_id" \
            --instance-id "$SSM_INSTANCE_ID" \
            --region "$SSM_REGION" \
            --query "StandardOutputContent" --output text
        [[ "$status" == "Success" ]]
    else
        ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "$SSH_USER@$HOST_IP" "$script"
    fi
}

verify_runtime() {
    local script='
set -e
echo "=== nginx -t ==="
if ! sudo nginx -t 2>&1; then
    echo "STATUS=NGINX_T_FAILED"
    exit 1
fi
echo "=== systemctl is-active nginx ==="
ACTIVE=$(systemctl is-active nginx 2>/dev/null || echo unknown)
echo "STATUS_ACTIVE=$ACTIVE"
echo "=== file mtime vs worker start time ==="
CONF=/etc/nginx/conf.d/worldsim-proxy.conf
if [[ -f "$CONF" ]]; then
    FILE_MTIME=$(stat -c %Y "$CONF")
    echo "FILE_MTIME=$FILE_MTIME"
else
    echo "FILE_MTIME=NONE"
fi
# Oldest nginx worker stat-time (the /proc/<pid> ctime), in seconds since epoch.
# Workers are respawned on reload, so the oldest one bounds when nginx last
# applied config. A master-only stat would only tell us when nginx was
# originally started — SIGHUPs do not replace the master.
NGINX_MAIN_PID=$(systemctl show -p MainPID --value nginx 2>/dev/null || echo 0)
OLDEST_WORKER_PID=""
if [[ "$NGINX_MAIN_PID" =~ ^[0-9]+$ && "$NGINX_MAIN_PID" -gt 0 ]]; then
    OLDEST_WORKER_PID=$(pgrep -P "$NGINX_MAIN_PID" -f "nginx: worker" | head -1 || true)
fi
if [[ -n "$OLDEST_WORKER_PID" ]]; then
    WORKER_START=$(stat -c %Y "/proc/$OLDEST_WORKER_PID" 2>/dev/null || echo 0)
    echo "WORKER_START=$WORKER_START"
else
    echo "WORKER_START=0"
fi
echo "=== recent [emerg] in error.log (last 200 lines) ==="
sudo tail -n 200 /var/log/nginx/error.log 2>/dev/null | grep "\[emerg\]" | tail -n 5 || echo "NO_EMERG"
'
    run_on_host "$script"
}

parse_runtime_output() {
    # Extract numeric/string signals from the runtime blob.
    local output="$1"
    RUNTIME_ACTIVE=$(printf "%s" "$output" | sed -n "s/^STATUS_ACTIVE=//p" | tail -n 1)
    RUNTIME_FILE_MTIME=$(printf "%s" "$output" | sed -n "s/^FILE_MTIME=//p" | tail -n 1)
    RUNTIME_WORKER_START=$(printf "%s" "$output" | sed -n "s/^WORKER_START=//p" | tail -n 1)
    # Only care about the emerg block between the marker and EOF.
    RUNTIME_EMERG=$(printf "%s" "$output" | awk '/recent \[emerg\]/ {flag=1; next} flag')
}

LIVE=$(fetch_live)
EXPECTED=$(render_expected)

DRIFT=0
if [[ "$LIVE" != "$EXPECTED" ]]; then
    echo "DRIFT: live nginx config differs from repo template."
    echo "--- live (on host)"
    echo "+++ expected (from scripts/deploy_benchmark_proxy.sh)"
    diff <(printf '%s' "$LIVE") <(printf '%s' "$EXPECTED") || true
    DRIFT=1
else
    echo "OK: live nginx config matches repo template."
fi

if [[ "$VERIFY_RUNTIME" == "1" ]]; then
    echo
    echo "=== runtime verification ==="
    RUNTIME_OUT=$(verify_runtime) || {
        echo "ERROR: runtime verification failed — see output above."
        exit 2
    }
    printf '%s\n' "$RUNTIME_OUT"
    parse_runtime_output "$RUNTIME_OUT"

    RUNTIME_FAIL=0

    if [[ "$RUNTIME_ACTIVE" != "active" ]]; then
        echo "ERROR: systemctl reports nginx status='$RUNTIME_ACTIVE' (expected 'active')"
        RUNTIME_FAIL=1
    fi

    if [[ -n "$RUNTIME_FILE_MTIME" && -n "$RUNTIME_WORKER_START" \
       && "$RUNTIME_FILE_MTIME" != "NONE" && "$RUNTIME_WORKER_START" != "0" ]]; then
        if (( RUNTIME_WORKER_START < RUNTIME_FILE_MTIME )); then
            echo "ERROR: nginx workers started at $RUNTIME_WORKER_START but config was edited at $RUNTIME_FILE_MTIME"
            echo "       → the on-disk file is NOT what nginx currently has loaded."
            echo "       Reload with: sudo nginx -s reload (or re-run deploy_benchmark_proxy.sh)"
            RUNTIME_FAIL=1
        else
            echo "OK: nginx workers started at/after config mtime (loaded config is current)"
        fi
    else
        echo "WARN: could not compare file mtime to worker start (FILE_MTIME=$RUNTIME_FILE_MTIME WORKER_START=$RUNTIME_WORKER_START)"
    fi

    if [[ -n "$RUNTIME_EMERG" && "$RUNTIME_EMERG" != *"NO_EMERG"* \
       && "$RUNTIME_EMERG" != *"no lines"* ]]; then
        # Show content only — the [emerg] grep tail from the script.
        if printf "%s" "$RUNTIME_EMERG" | grep -q "\[emerg\]"; then
            echo "WARN: recent [emerg] lines in error.log — a prior reload may have failed:"
            printf "%s\n" "$RUNTIME_EMERG" | grep "\[emerg\]" | sed "s/^/  /"
            # Not a hard exit because a past [emerg] can be stale — but the
            # operator should see it.
        fi
    fi

    if (( RUNTIME_FAIL > 0 )); then
        exit 3
    fi
fi

if (( DRIFT > 0 )); then
    exit 1
fi
exit 0
