#!/usr/bin/env bash
# Audit the canonical r8a AWS control-plane state against the checked-in host config.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST_CONFIG="$REPO_ROOT/configs/benchmark_hosts/r8a.yaml"
STACK_NAME="worldsim-r8a-control-plane"
INSTANCE_ID=""
REGION=""

usage() {
    cat <<'USAGE'
audit_r8a_control_plane.sh

Options:
  --host-config <path>       default: configs/benchmark_hosts/r8a.yaml
  --stack-name <name>        default: worldsim-r8a-control-plane
  --instance-id <id>         default: discover Name=worldsim-r8a-benchmark-green
  --region <region>          default: host config region
USAGE
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

while (($#)); do
    case "$1" in
        --host-config) HOST_CONFIG="$2"; shift 2 ;;
        --stack-name) STACK_NAME="$2"; shift 2 ;;
        --instance-id) INSTANCE_ID="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown arg: $1" ;;
    esac
done

command -v aws >/dev/null 2>&1 || die "required command not found: aws"
command -v uv >/dev/null 2>&1 || die "required command not found: uv"
[[ -f "$HOST_CONFIG" ]] || die "host config not found: $HOST_CONFIG"

eval "$(
    uv run python - "$HOST_CONFIG" <<'PY'
import shlex
import sys
from worldsim.host_config import load_host_config

cfg = load_host_config(sys.argv[1])
for key, value in {
    "CFG_NAME": cfg.name,
    "CFG_REGION": cfg.region or "",
    "CFG_ADVERTISE_HOST": cfg.advertise_host,
    "CFG_SECURITY_GROUP_ID": cfg.security_group_id or "",
}.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

[[ "$CFG_NAME" == "r8a" ]] || die "expected r8a host config, got name=$CFG_NAME"
REGION="${REGION:-$CFG_REGION}"
[[ -n "$REGION" ]] || die "region missing; pass --region or set it in host config"

if [[ -z "$INSTANCE_ID" ]]; then
    INSTANCE_ID="$(
        aws ec2 describe-instances \
            --region "$REGION" \
            --filters \
                Name=tag:Name,Values=worldsim-r8a-benchmark-green \
                Name=instance-state-name,Values=pending,running,stopping,stopped \
            --query 'Reservations[].Instances[].InstanceId' \
            --output text
    )"
    [[ "$INSTANCE_ID" != *$'\t'* && "$INSTANCE_ID" != *" "* ]] || \
        die "multiple r8a instances matched; pass --instance-id explicitly: $INSTANCE_ID"
    [[ -n "$INSTANCE_ID" && "$INSTANCE_ID" != "None" ]] || \
        die "could not discover r8a instance by tag; pass --instance-id"
fi

instance_json="$(
    aws ec2 describe-network-interfaces \
        --region "$REGION" \
        --filters Name=attachment.instance-id,Values="$INSTANCE_ID" \
        --query 'NetworkInterfaces[0].{PublicIp:Association.PublicIp,AllocationId:Association.AllocationId,Groups:Groups[].GroupId}' \
        --output json
)"
stack_json="$(
    aws cloudformation describe-stacks \
        --region "$REGION" \
        --stack-name "$STACK_NAME" \
        --query 'Stacks[0].Outputs' \
        --output json 2>/dev/null || true
)"

uv run python - "$CFG_ADVERTISE_HOST" "$CFG_SECURITY_GROUP_ID" "$instance_json" "$stack_json" <<'PY'
import json
import sys

advertise_host, security_group_id, instance_raw, stack_raw = sys.argv[1:]
instance = json.loads(instance_raw)
outputs = {}
if stack_raw.strip():
    outputs = {item["OutputKey"]: item.get("OutputValue", "") for item in json.loads(stack_raw)}

failures = []
public_ip = instance.get("PublicIp") or ""
allocation_id = instance.get("AllocationId") or ""
groups = set(instance.get("Groups") or [])

if not allocation_id:
    failures.append("r8a public IP is ephemeral; no EIP allocation is associated")
if advertise_host != public_ip:
    failures.append(f"host config advertise_host={advertise_host} does not match EC2 public_ip={public_ip}")
if security_group_id and security_group_id not in groups:
    failures.append(f"host config security_group_id={security_group_id} is not attached to the instance ENI")
if not outputs:
    failures.append("CloudFormation stack is missing or unreadable")
elif outputs.get("AllocationId") != allocation_id:
    failures.append(
        f"CloudFormation allocation_id={outputs.get('AllocationId')} does not match ENI allocation_id={allocation_id}"
    )

print(f"public_ip={public_ip or '<none>'}")
print(f"allocation_id={allocation_id or '<none>'}")
print(f"host_config_advertise_host={advertise_host}")
print(f"cloudformation_stack={'present' if outputs else 'missing'}")

if failures:
    for failure in failures:
        print(f"ERROR: {failure}", file=sys.stderr)
    raise SystemExit(2)
print("r8a_control_plane=ok")
PY
