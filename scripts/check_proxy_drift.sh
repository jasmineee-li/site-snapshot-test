#!/usr/bin/env bash
# check_proxy_drift.sh — Diff the live /etc/nginx/conf.d/worldsim-proxy.conf
# on a benchmark EC2 instance against what `deploy_benchmark_proxy.sh` would
# render from the checked-in template + port map + token.
#
# Source of truth: this repo. If they disagree, re-run deploy_benchmark_proxy
# (or explain the drift in a commit message and update the script if a new
# directive is warranted).
#
# Usage:
#   ./scripts/check_proxy_drift.sh --host-config configs/benchmark_hosts/r5.yaml \
#       --via-ssm --ssm-instance-id i-0abc... --insecure-http --topology legacy
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

LIVE=$(fetch_live)
EXPECTED=$(render_expected)

if [[ "$LIVE" == "$EXPECTED" ]]; then
    echo "OK: live nginx config matches repo template."
    exit 0
fi

echo "DRIFT: live nginx config differs from repo template."
echo "--- live (on host)"
echo "+++ expected (from scripts/deploy_benchmark_proxy.sh)"
diff <(printf '%s' "$LIVE") <(printf '%s' "$EXPECTED") || true
exit 1
