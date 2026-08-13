#!/usr/bin/env bash
# host_park.sh — stop a benchmark host EC2 instance to avoid idle compute billing.
#
# Reads instance_id + region from the checked-in host config. Refuses to stop
# while a sweep tag indicates a run is in progress (override with --force).
# Refuses instance-store-backed families (stop is only available for EBS-root).
# Idempotent: a no-op note is printed when the instance is already
# stopping/stopped.
#
# Usage:
#   scripts/host_park.sh --host-config configs/benchmark_hosts/r8a.yaml
#   scripts/host_park.sh --host-config <path> --dry-run
#   scripts/host_park.sh --host-config <path> --force   # override sweep tag

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST_CONFIG=""
FORCE=0
DRY_RUN=0
SWEEP_TAG_KEY="worldsim:sweep-in-progress"
SWEEP_TAG_TRUE="true"

usage() {
    cat <<'USAGE'
host_park.sh --host-config <path> [--force] [--dry-run]

Stop the EC2 instance referenced by the host config. Refuses to stop while a
sweep is marked in progress unless --force is passed.

Options:
  --host-config <path>   required; e.g. configs/benchmark_hosts/r8a.yaml
  --force                stop even if worldsim:sweep-in-progress=true
                         (operator identity is logged to stderr)
  --dry-run              print the planned action, do not call AWS
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
        --force) FORCE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
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

# Describe instance in one call: state, root device type, sweep tag.
inst_json="$(
    aws ec2 describe-instances \
        --region "$CFG_REGION" \
        --instance-ids "$CFG_INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].{State:State.Name,Root:RootDeviceType,InstanceType:InstanceType,Tags:Tags}' \
        --output json
)"
[[ -n "$inst_json" && "$inst_json" != "null" ]] || die "could not describe instance $CFG_INSTANCE_ID in $CFG_REGION"

STATE="$(printf '%s' "$inst_json" | uv run python -c 'import json,sys; print(json.load(sys.stdin)["State"])')"
ROOT="$(printf '%s' "$inst_json" | uv run python -c 'import json,sys; print(json.load(sys.stdin)["Root"])')"
ITYPE="$(printf '%s' "$inst_json" | uv run python -c 'import json,sys; print(json.load(sys.stdin)["InstanceType"])')"
SWEEP_TAG_VALUE="$(
    uv run python -c '
import json, sys
key, raw = sys.argv[1], sys.argv[2]
data = json.loads(raw) if raw else {}
for tag in data.get("Tags") or []:
    if tag.get("Key") == key:
        print(tag.get("Value", ""))
        break
' "$SWEEP_TAG_KEY" "$inst_json"
)"

log "state=$STATE root=$ROOT type=$ITYPE sweep_tag=${SWEEP_TAG_VALUE:-<unset>}"

[[ "$ROOT" == "ebs" ]] || die "RootDeviceType=$ROOT; only EBS-backed instances support stop"

if [[ "$STATE" == "stopped" || "$STATE" == "stopping" ]]; then
    log "instance is already $STATE; nothing to do"
    exit 0
fi

if [[ "$SWEEP_TAG_VALUE" == "$SWEEP_TAG_TRUE" ]]; then
    if [[ "$FORCE" -ne 1 ]]; then
        cat >&2 <<MSG
ERROR: $SWEEP_TAG_KEY=$SWEEP_TAG_TRUE on $CFG_INSTANCE_ID; refusing to stop.

  If a sweep is still running, wait for it to finish or kill the runner.
  If the tag is stale (sweep already done):
    aws ec2 delete-tags --region $CFG_REGION --resources $CFG_INSTANCE_ID \\
      --tags Key=$SWEEP_TAG_KEY
  Or override with --force (operator identity will be logged).
MSG
        exit 1
    fi
    caller_arn="$(aws sts get-caller-identity --query Arn --output text 2>/dev/null || echo unknown)"
    printf 'WARNING: --force override; sweep tag was set; operator=%s\n' "$caller_arn" >&2
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DRY RUN: would stop $CFG_INSTANCE_ID in $CFG_REGION"
    exit 0
fi

log "stopping $CFG_INSTANCE_ID"
aws ec2 stop-instances \
    --region "$CFG_REGION" \
    --instance-ids "$CFG_INSTANCE_ID" \
    --query 'StoppingInstances[0].{Prev:PreviousState.Name,Cur:CurrentState.Name}' \
    --output text >/dev/null

log "waiting for instance to reach stopped state"
aws ec2 wait instance-stopped --region "$CFG_REGION" --instance-ids "$CFG_INSTANCE_ID"

log "$CFG_INSTANCE_ID stopped (compute now $0/hr; EBS and EIP charges continue)"
