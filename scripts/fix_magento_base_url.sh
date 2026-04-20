#!/usr/bin/env bash
# fix_magento_base_url.sh — Update Magento's web/{unsecure,secure}/base_url
# to point at the worldsim-proxy port (e.g. 17770) instead of the raw
# container-backend port (e.g. 7770).
#
# Why this exists
# ---------------
# WebArena's Magento images ship with base_url pointing at the raw backend
# port (`http://<ip>:7770/`). That port is intentionally NOT exposed in the
# EC2 security group — all external traffic goes through worldsim-proxy on
# the offset port. Magento uses base_url for three things that bite us:
#   1. Host-validation redirect: if the incoming Host doesn't match the
#      configured base_url host, Magento 302s to base_url.
#   2. HTML form actions: `<form action="http://<ip>:7770/...">` is baked
#      into every checkout/admin page.
#   3. Inline JS BASE_URL: JS-constructed fetch/XHR targets embed the
#      literal backend port.
#
# `proxy_redirect` on nginx fixes (1) for Location headers only. (2) and
# (3) are untouched by nginx because nginx doesn't rewrite HTML/JS
# bodies. Clients that follow absolute URLs out of HTML/JS (browsers, in
# particular — Phase 4 Browser-Use agents) will escape the proxy and hit
# the SG-closed bare port.
#
# The proper fix is to rewrite base_url inside Magento's DB to the proxy
# origin. One MySQL UPDATE + one cache flush per container, idempotent.
#
# Scope
# -----
# Affects only Magento (shopping + shopping_admin) — other WebArena sites
# either have no base_url concept (reddit, map) or handle it differently
# (wikipedia relative, gitlab external_url in gitlab.rb — see
# docs/handoffs if gitlab ever needs the same treatment).
#
# Usage
# -----
#   ./scripts/fix_magento_base_url.sh --via-ssm \
#       --ssm-instance-id i-0abc... \
#       --advertise-host 3.12.221.9 \
#       --port-offset 10000 \
#       --shopping-port 7770 --shopping-admin-port 7780 \
#       --shopping-container webarena-verified-shopping \
#       --shopping-admin-container webarena-verified-shopping_admin
#
#   # SSH mode (SG must allow 22):
#   ./scripts/fix_magento_base_url.sh --host 3.12.221.9 --ssh-key ~/.ssh/k.pem ...
#
# The script is idempotent: if base_url is already correct, it logs a no-op
# and skips the cache flush.

set -euo pipefail

VIA_SSM=0
SSM_INSTANCE_ID="${SSM_INSTANCE_ID:-}"
SSM_REGION="${SSM_REGION:-us-east-2}"
HOST_IP="${HOST_IP:-}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/webarena-key.pem}"
SSH_USER="${SSH_USER:-ubuntu}"
ADVERTISE_HOST="${ADVERTISE_HOST:-}"
PORT_OFFSET="${PORT_OFFSET:-10000}"
SHOPPING_PORT="${SHOPPING_PORT:-7770}"
SHOPPING_ADMIN_PORT="${SHOPPING_ADMIN_PORT:-7780}"
SHOPPING_CONTAINER="${SHOPPING_CONTAINER:-webarena-verified-shopping}"
SHOPPING_ADMIN_CONTAINER="${SHOPPING_ADMIN_CONTAINER:-webarena-verified-shopping_admin}"
MYSQL_USER="${MYSQL_USER:-magentouser}"
MYSQL_PASS="${MYSQL_PASS:-MyPassword}"
MYSQL_DB="${MYSQL_DB:-magentodb}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --via-ssm) VIA_SSM=1; shift ;;
        --ssm-instance-id) SSM_INSTANCE_ID="$2"; shift 2 ;;
        --ssm-region) SSM_REGION="$2"; shift 2 ;;
        --host) HOST_IP="$2"; shift 2 ;;
        --ssh-key) SSH_KEY="$2"; shift 2 ;;
        --ssh-user) SSH_USER="$2"; shift 2 ;;
        --advertise-host) ADVERTISE_HOST="$2"; shift 2 ;;
        --port-offset) PORT_OFFSET="$2"; shift 2 ;;
        --shopping-port) SHOPPING_PORT="$2"; shift 2 ;;
        --shopping-admin-port) SHOPPING_ADMIN_PORT="$2"; shift 2 ;;
        --shopping-container) SHOPPING_CONTAINER="$2"; shift 2 ;;
        --shopping-admin-container) SHOPPING_ADMIN_CONTAINER="$2"; shift 2 ;;
        --mysql-user) MYSQL_USER="$2"; shift 2 ;;
        --mysql-pass) MYSQL_PASS="$2"; shift 2 ;;
        --mysql-db) MYSQL_DB="$2"; shift 2 ;;
        --help|-h)
            sed -n '1,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$ADVERTISE_HOST" ]] && { echo "ERROR: --advertise-host required" >&2; exit 1; }

if [[ "$VIA_SSM" == "1" ]]; then
    [[ -z "$SSM_INSTANCE_ID" ]] && { echo "ERROR: --ssm-instance-id required with --via-ssm" >&2; exit 1; }
    command -v aws >/dev/null 2>&1 || { echo "ERROR: aws CLI required for --via-ssm" >&2; exit 1; }
else
    [[ -z "$HOST_IP" ]] && { echo "ERROR: --host required without --via-ssm" >&2; exit 1; }
    [[ ! -f "$SSH_KEY" ]] && { echo "ERROR: SSH key not found: $SSH_KEY" >&2; exit 1; }
fi

run_remote() {
    local script="$*"
    if [[ "$VIA_SSM" == "1" ]]; then
        local b64
        b64="$(printf '%s' "$script" | base64 | tr -d '\n')"
        local cmd_id
        cmd_id=$(aws ssm send-command \
            --instance-ids "$SSM_INSTANCE_ID" \
            --document-name AWS-RunShellScript \
            --parameters "commands=[\"echo '$b64' | base64 -d | bash\"]" \
            --region "$SSM_REGION" \
            --query "Command.CommandId" --output text)
        local status deadline=$(( SECONDS + 180 ))
        while (( SECONDS < deadline )); do
            status=$(aws ssm get-command-invocation \
                --command-id "$cmd_id" \
                --instance-id "$SSM_INSTANCE_ID" \
                --region "$SSM_REGION" \
                --query "Status" --output text 2>/dev/null || echo "")
            case "$status" in Success|Failed|Cancelled|TimedOut) break ;; esac
            sleep 3
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

verify_sql_scopes() {
    # Re-read every base_url row and assert each value matches ``desired``.
    # The idempotence check at the top reads the same rows but tolerates
    # transient state; this one runs AFTER update + cache flush and is
    # strict: any stray row with the wrong value is a hard failure.
    local container="$1" desired="$2"
    local rows
    rows=$(run_remote "docker exec ${container} mysql -u${MYSQL_USER} -p${MYSQL_PASS} -D ${MYSQL_DB} -sN -e \"SELECT scope, scope_id, path, value FROM core_config_data WHERE path IN ('web/unsecure/base_url','web/secure/base_url')\" 2>/dev/null")
    local seen=0 bad=0
    while IFS=$'\t' read -r scope scope_id path value; do
        [[ -z "$path" ]] && continue
        seen=$((seen + 1))
        if [[ "$value" != "$desired" ]]; then
            echo "    ERROR: post-update DB mismatch at scope=$scope scope_id=$scope_id path=$path value=$value"
            bad=$((bad + 1))
        fi
    done <<< "$rows"
    if (( seen == 0 )); then
        echo "    ERROR: no base_url rows found after update — DB read failed"
        return 1
    fi
    if (( bad > 0 )); then
        echo "    ERROR: $bad of $seen rows did not match desired=${desired}"
        return 1
    fi
    echo "    verified: $seen DB rows all match desired=${desired}"
    return 0
}

verify_http_probe() {
    # Fetch the themed storefront root from INSIDE the EC2 host, hitting
    # the raw backend port directly on the loopback (no proxy / token
    # needed because we're on the host). Parse Magento's rendered
    # ``var BASE_URL = '...'`` and assert it equals the proxy origin.
    # Per Magento's require_js.phtml template, this global is emitted in
    # every themed page's <head>; its value reflects the merged config
    # (DB + env.php + env vars), which is the ground truth for what
    # Magento will hand to browsers.
    local real_port="$1" desired="$2" label="$3"
    local probe="http://127.0.0.1:${real_port}/"
    local output
    output=$(run_remote "curl -sS --max-time 15 '${probe}' 2>/dev/null | grep -oE \"var BASE_URL = '[^']+'\" | head -1" || true)
    if [[ -z "$output" ]]; then
        echo "    ERROR: HTTP probe at ${probe} returned no BASE_URL declaration"
        echo "           storefront may not be reachable yet, or page skipped the require_js template."
        return 1
    fi
    local expected_decl="var BASE_URL = '${desired}'"
    if [[ "$output" != "$expected_decl" ]]; then
        echo "    ERROR: HTTP probe at ${probe} reports ${output}"
        echo "           but expected ${expected_decl}"
        return 1
    fi
    echo "    verified: HTTP probe confirms BASE_URL = ${desired}"
    return 0
}

fix_one() {
    local container="$1" real_port="$2" label="$3"
    local proxy_port=$((real_port + PORT_OFFSET))
    local desired="http://${ADVERTISE_HOST}:${proxy_port}/"

    echo "==> ${label}: container=${container} real=${real_port} proxy=${proxy_port}"
    echo "    desired base_url = ${desired}"

    # Read all base_url rows regardless of scope. WebArena Magento images
    # ship with rows at multiple scopes (e.g. shopping has
    # scope='websites', scope_id=1 for unsecure and scope='default',
    # scope_id=0 for secure; shopping_admin has only default/0). Filtering
    # to default/0 silently misses the websites-scoped row.
    local current
    current=$(run_remote "docker exec ${container} mysql -u${MYSQL_USER} -p${MYSQL_PASS} -D ${MYSQL_DB} -sN -e \"SELECT scope, scope_id, path, value FROM core_config_data WHERE path IN ('web/unsecure/base_url','web/secure/base_url')\" 2>/dev/null")
    echo "    current rows:"
    printf '%s\n' "$current" | sed 's/^/      /'

    # Idempotence: every row's value must equal the desired URL.
    local needs_update=0
    while IFS=$'\t' read -r scope scope_id path value; do
        [[ -z "$path" ]] && continue
        if [[ "$value" != "$desired" ]]; then needs_update=1; fi
    done <<< "$current"

    if [[ "$needs_update" == "0" ]]; then
        echo "    DB already correct — skipping update + cache flush"
        # Still run HTTP probe so the operator gets a positive confirmation
        # that what's in the DB actually renders on the wire. A stale PHP-FPM
        # worker holding an old cached config would show up here.
        verify_http_probe "$real_port" "$desired" "$label" || return 1
        return 0
    fi

    # Update every matching row. No scope filter: we want base_url to be
    # the proxy origin at every scope Magento stores.
    local sql="UPDATE core_config_data SET value='${desired}' WHERE path IN ('web/unsecure/base_url','web/secure/base_url');"
    run_remote "docker exec ${container} mysql -u${MYSQL_USER} -p${MYSQL_PASS} -D ${MYSQL_DB} -e \"${sql}\""

    # Flush Magento's config cache so the new base_url takes effect
    # immediately. `cache:flush` is a superset of `cache:clean config` and
    # also drops any pages/blocks FPC has rendered with the old origin;
    # for a one-shot config repair that's what we want.
    run_remote "docker exec ${container} bash -lc 'cd /var/www/magento2 && php bin/magento cache:flush'" || {
        echo "    WARN: cache:flush returned non-zero; config change persisted but may lag"
    }

    echo "    updated — running post-update verification"

    # Strict post-update verification: every scope's row must match
    # ``desired`` AND the rendered page must report the same value.
    verify_sql_scopes "$container" "$desired" || return 1
    verify_http_probe "$real_port" "$desired" "$label" || return 1
}

FAILED=0
fix_one "$SHOPPING_CONTAINER" "$SHOPPING_PORT" "shopping" || FAILED=$((FAILED + 1))
fix_one "$SHOPPING_ADMIN_CONTAINER" "$SHOPPING_ADMIN_PORT" "shopping_admin" || FAILED=$((FAILED + 1))

if (( FAILED > 0 )); then
    echo "==> FAILED: $FAILED container(s) did not verify clean — see errors above"
    exit 1
fi

echo "==> done"
