#!/usr/bin/env bash
# enable_r8a_termination_protection.sh — turn on DisableApiTermination for
# the r8a benchmark instance.
#
# Termination protection is orthogonal to stop logic but addresses the same
# "operator mistake" failure class: it makes `aws ec2 terminate-instances`
# fail until the protection is explicitly removed. CloudFormation cannot
# manage this attribute on imported instances, so we apply it via the CLI.
#
# Usage:
#   scripts/enable_r8a_termination_protection.sh
#   scripts/enable_r8a_termination_protection.sh --host-config <path>
#   scripts/enable_r8a_termination_protection.sh --disable  # rare, e.g. before
#                                                           # a deliberate replace
#
# Idempotent: reports current state before and after.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST_CONFIG="$REPO_ROOT/configs/benchmark_hosts/r8a.yaml"
ACTION="enable"

usage() {
    cat <<'USAGE'
enable_r8a_termination_protection.sh [--host-config <path>] [--disable]

Toggle DisableApiTermination for the EC2 instance referenced by the host config.

Options:
  --host-config <path>   default: configs/benchmark_hosts/r8a.yaml
  --disable              clear protection (use only when deliberately
                         replacing the instance; re-enable on the new one)
  -h, --help             this help
USAGE
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

while (($#)); do
    case "$1" in
        --host-config) HOST_CONFIG="$2"; shift 2 ;;
        --disable) ACTION="disable"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown arg: $1" ;;
    esac
done

[[ -f "$HOST_CONFIG" ]] || die "host config not found: $HOST_CONFIG"
command -v aws >/dev/null 2>&1 || die "required command not found: aws"
command -v uv >/dev/null 2>&1 || die "required command not found: uv"

eval "$(
    uv run python - "$HOST_CONFIG" <<'PY'
import shlex
import sys
from worldsim.host_config import load_host_config

cfg = load_host_config(sys.argv[1])
for key, value in {
    "CFG_NAME": cfg.name,
    "CFG_REGION": cfg.region or "",
    "CFG_INSTANCE_ID": cfg.instance_id or "",
}.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

[[ -n "$CFG_INSTANCE_ID" ]] || die "host config $HOST_CONFIG has no instance_id"
[[ -n "$CFG_REGION" ]] || die "host config $HOST_CONFIG has no region"

read_protection() {
    aws ec2 describe-instance-attribute \
        --region "$CFG_REGION" \
        --instance-id "$CFG_INSTANCE_ID" \
        --attribute disableApiTermination \
        --query 'DisableApiTermination.Value' \
        --output text
}

BEFORE="$(read_protection)"
log "host=$CFG_NAME instance=$CFG_INSTANCE_ID region=$CFG_REGION"
log "DisableApiTermination before: $BEFORE"

if [[ "$ACTION" == "enable" ]]; then
    if [[ "$BEFORE" == "True" ]]; then
        log "already enabled; nothing to do"
        exit 0
    fi
    aws ec2 modify-instance-attribute \
        --region "$CFG_REGION" \
        --instance-id "$CFG_INSTANCE_ID" \
        --disable-api-termination
else
    if [[ "$BEFORE" == "False" ]]; then
        log "already disabled; nothing to do"
        exit 0
    fi
    cat >&2 <<MSG
WARNING: clearing termination protection on $CFG_INSTANCE_ID.
  Only do this when you intend to actually terminate the instance
  (e.g. AMI-replace dance). Re-enable on the replacement.
MSG
    caller_arn="$(aws sts get-caller-identity --query Arn --output text 2>/dev/null || echo unknown)"
    printf 'operator=%s\n' "$caller_arn" >&2
    aws ec2 modify-instance-attribute \
        --region "$CFG_REGION" \
        --instance-id "$CFG_INSTANCE_ID" \
        --no-disable-api-termination
fi

AFTER="$(read_protection)"
log "DisableApiTermination after: $AFTER"
