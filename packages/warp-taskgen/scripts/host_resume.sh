#!/usr/bin/env bash
# host_resume.sh — start a parked benchmark host and wait for status checks.
#
# Reads instance_id + region from the checked-in host config. Sets the sweep
# tag BEFORE start to close the race with auto-stop automation. Polls
# describe-instance-status until both SystemStatus=ok and InstanceStatus=ok,
# then runs the control-plane audit if one exists for this host.
#
# Usage:
#   scripts/host_resume.sh --host-config configs/benchmark_hosts/r8a.yaml
#   scripts/host_resume.sh --host-config <path> --no-tag    # quick restart, not a sweep
#   scripts/host_resume.sh --host-config <path> --regen-topology

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST_CONFIG=""
SET_TAG=1
REGEN_TOPOLOGY=0
STATUS_TIMEOUT_SECS=600
SWEEP_TAG_KEY="worldsim:sweep-in-progress"
SWEEP_TAG_TRUE="true"

usage() {
    cat <<'USAGE'
host_resume.sh --host-config <path> [--no-tag] [--regen-topology]

Start the EC2 instance referenced by the host config, set the sweep tag,
wait for status checks to pass, then run the control-plane audit.

Options:
  --host-config <path>     required; e.g. configs/benchmark_hosts/r8a.yaml
  --no-tag                 do not set worldsim:sweep-in-progress=true
                           (for quick restarts that are not running a sweep)
  --regen-topology         run scripts/setup_phase4_on_host.sh after status
                           checks pass (regenerates storage_state cookies)
  -h, --help               this help
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
        --no-tag) SET_TAG=0; shift ;;
        --regen-topology) REGEN_TOPOLOGY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown arg: $1" ;;
    esac
done

[[ -n "$HOST_CONFIG" ]] || { usage >&2; die "--host-config is required"; }
[[ -f "$HOST_CONFIG" ]] || die "host config not found: $HOST_CONFIG"
command -v aws >/dev/null 2>&1 || die "required command not found: aws"
command -v uv >/dev/null 2>&1 || die "required command not found: uv"

eval "$(
    uv run python - "$HOST_CONFIG" <<'PY'
import shlex
import sys
from warp_taskgen.host_config import load_host_config

cfg = load_host_config(sys.argv[1])
for key, value in {
    "CFG_NAME": cfg.name,
    "CFG_REGION": cfg.region or "",
    "CFG_INSTANCE_ID": cfg.instance_id or "",
}.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

[[ -n "$CFG_INSTANCE_ID" ]] || die "host config $HOST_CONFIG has no instance_id; lifecycle tooling requires it"
[[ -n "$CFG_REGION" ]] || die "host config $HOST_CONFIG has no region"

log "host=$CFG_NAME instance=$CFG_INSTANCE_ID region=$CFG_REGION"

STATE="$(
    aws ec2 describe-instances \
        --region "$CFG_REGION" \
        --instance-ids "$CFG_INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text
)"
[[ -n "$STATE" && "$STATE" != "None" ]] || die "could not describe instance $CFG_INSTANCE_ID"
log "current state=$STATE"

if [[ "$SET_TAG" -eq 1 ]]; then
    log "setting $SWEEP_TAG_KEY=$SWEEP_TAG_TRUE before start (closes race with auto-stop)"
    aws ec2 create-tags \
        --region "$CFG_REGION" \
        --resources "$CFG_INSTANCE_ID" \
        --tags "Key=$SWEEP_TAG_KEY,Value=$SWEEP_TAG_TRUE"
fi

if [[ "$STATE" == "running" || "$STATE" == "pending" ]]; then
    log "instance is already $STATE; skipping start-instances"
else
    log "starting $CFG_INSTANCE_ID"
    aws ec2 start-instances \
        --region "$CFG_REGION" \
        --instance-ids "$CFG_INSTANCE_ID" \
        --query 'StartingInstances[0].{Prev:PreviousState.Name,Cur:CurrentState.Name}' \
        --output text >/dev/null
    log "waiting for instance to enter running state"
    aws ec2 wait instance-running --region "$CFG_REGION" --instance-ids "$CFG_INSTANCE_ID"
fi

log "polling instance status checks (timeout ${STATUS_TIMEOUT_SECS}s)"
deadline=$(( $(date +%s) + STATUS_TIMEOUT_SECS ))
while :; do
    status_json="$(
        aws ec2 describe-instance-status \
            --region "$CFG_REGION" \
            --instance-ids "$CFG_INSTANCE_ID" \
            --query 'InstanceStatuses[0].{System:SystemStatus.Status,Instance:InstanceStatus.Status}' \
            --output json
    )"
    sys_status="$(printf '%s' "$status_json" | uv run python -c 'import json,sys; d=json.load(sys.stdin) or {}; print(d.get("System") or "")')"
    ins_status="$(printf '%s' "$status_json" | uv run python -c 'import json,sys; d=json.load(sys.stdin) or {}; print(d.get("Instance") or "")')"
    log "  system=$sys_status instance=$ins_status"
    if [[ "$sys_status" == "ok" && "$ins_status" == "ok" ]]; then
        log "status checks ok"
        break
    fi
    if (( $(date +%s) >= deadline )); then
        die "status checks did not reach ok within ${STATUS_TIMEOUT_SECS}s (system=$sys_status instance=$ins_status)"
    fi
    sleep 15
done

audit_script="$REPO_ROOT/scripts/audit_${CFG_NAME}_control_plane.sh"
if [[ -x "$audit_script" ]]; then
    log "running control-plane audit: $audit_script"
    "$audit_script" --host-config "$HOST_CONFIG"
else
    log "no control-plane audit found for host $CFG_NAME (looked for $audit_script)"
fi

if [[ "$REGEN_TOPOLOGY" -eq 1 ]]; then
    setup_script="$REPO_ROOT/scripts/setup_phase4_on_host.sh"
    if [[ -x "$setup_script" ]]; then
        log "regenerating topology: $setup_script --host-config $HOST_CONFIG"
        "$setup_script" --host-config "$HOST_CONFIG"
    else
        log "WARNING: --regen-topology requested but $setup_script not found"
    fi
fi

cat <<MSG

[host_resume] $CFG_NAME is up.
  Sweep tag: $SWEEP_TAG_KEY=$( [[ "$SET_TAG" -eq 1 ]] && printf '%s' "$SWEEP_TAG_TRUE" || printf '<not set>' )
  Next steps:
    * Rerun scripts/setup_phase4_on_host.sh if storage_state cookies are >24h old.
    * After the sweep + archive completes, clear the sweep tag:
        aws ec2 delete-tags --region $CFG_REGION --resources $CFG_INSTANCE_ID \\
          --tags Key=$SWEEP_TAG_KEY
MSG
