#!/usr/bin/env bash
# Deploy the CloudFormation control-plane stack for the canonical r8a host.
#
# This script intentionally manages only AWS control-plane identity: EIP
# allocation/association and operator SSH ingress. Benchmark containers,
# generated instances files, proxy maps, and Phase artifacts stay managed by
# the existing host setup scripts.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST_CONFIG="$REPO_ROOT/configs/benchmark_hosts/r8a.yaml"
TEMPLATE="$REPO_ROOT/infra/cloudformation/r8a-control-plane.yaml"
STACK_NAME="worldsim-r8a-control-plane"
INSTANCE_ID=""
SECURITY_GROUP_ID=""
REGION=""
CREATE_ELASTIC_IP="true"
EXISTING_ALLOCATION_ID=""
OPERATOR_CIDRS=()
WRITE_HOST_CONFIG=0
NO_EXECUTE_CHANGE_SET=0
ALLOW_REASSOCIATE_EXISTING_EIP=0

usage() {
    cat <<'USAGE'
deploy_r8a_control_plane.sh

Deploy the r8a CloudFormation control-plane stack.

Required:
  --operator-cidr <cidr|auto>       operator SSH CIDR, e.g. 128.84.124.235/32

Options:
  --host-config <path>              default: configs/benchmark_hosts/r8a.yaml
  --stack-name <name>               default: worldsim-r8a-control-plane
  --instance-id <id>                default: discover Name=worldsim-r8a-benchmark-green
  --security-group-id <sg-id>       default: host config security_group_id
  --region <region>                 default: host config region
  --existing-allocation-id <id>     associate an existing EIP instead of creating one
  --allow-reassociate-existing-eip  allow moving that EIP from another instance
  --extra-operator-cidr <cidr>      add up to four more SSH CIDRs
  --write-host-config               update advertise_host in the host config after deploy
  --no-execute-change-set           create and show the CloudFormation change set only
  -h, --help                        show this help

Notes:
  - The script refuses 0.0.0.0/0 for SSH.
  - If --operator-cidr auto is used, the script reads https://checkip.amazonaws.com.
  - After deploy, rerun r8a setup with scripts/scale_config.r8a-24x24.yml.
USAGE
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

while (($#)); do
    case "$1" in
        --host-config) HOST_CONFIG="$2"; shift 2 ;;
        --stack-name) STACK_NAME="$2"; shift 2 ;;
        --instance-id) INSTANCE_ID="$2"; shift 2 ;;
        --security-group-id) SECURITY_GROUP_ID="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --operator-cidr) OPERATOR_CIDRS+=("$2"); shift 2 ;;
        --extra-operator-cidr) OPERATOR_CIDRS+=("$2"); shift 2 ;;
        --existing-allocation-id)
            EXISTING_ALLOCATION_ID="$2"
            CREATE_ELASTIC_IP="false"
            shift 2
            ;;
        --allow-reassociate-existing-eip) ALLOW_REASSOCIATE_EXISTING_EIP=1; shift ;;
        --write-host-config) WRITE_HOST_CONFIG=1; shift ;;
        --no-execute-change-set) NO_EXECUTE_CHANGE_SET=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown arg: $1" ;;
    esac
done

require_cmd aws
require_cmd uv
[[ -f "$HOST_CONFIG" ]] || die "host config not found: $HOST_CONFIG"
[[ -f "$TEMPLATE" ]] || die "template not found: $TEMPLATE"

eval "$(
    uv run python - "$HOST_CONFIG" <<'PY'
import shlex
import sys
from worldsim.host_config import load_host_config

cfg = load_host_config(sys.argv[1])
for key, value in {
    "CFG_NAME": cfg.name,
    "CFG_REGION": cfg.region or "",
    "CFG_SECURITY_GROUP_ID": cfg.security_group_id or "",
}.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

[[ "$CFG_NAME" == "r8a" ]] || die "expected r8a host config, got name=$CFG_NAME"
REGION="${REGION:-$CFG_REGION}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-$CFG_SECURITY_GROUP_ID}"
[[ -n "$REGION" ]] || die "region missing; pass --region or set it in host config"
[[ -n "$SECURITY_GROUP_ID" ]] || die "security_group_id missing; pass --security-group-id or set it in host config"
[[ "${#OPERATOR_CIDRS[@]}" -ge 1 ]] || die "--operator-cidr is required"
[[ "${#OPERATOR_CIDRS[@]}" -le 5 ]] || die "CloudFormation template supports at most five operator CIDRs"
if [[ "$CREATE_ELASTIC_IP" == "false" && -z "$EXISTING_ALLOCATION_ID" ]]; then
    die "--existing-allocation-id must be non-empty when not creating an EIP"
fi

for i in "${!OPERATOR_CIDRS[@]}"; do
    if [[ "${OPERATOR_CIDRS[$i]}" == "auto" ]]; then
        require_cmd curl
        ip="$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')"
        [[ -n "$ip" ]] || die "failed to detect public IP"
        OPERATOR_CIDRS[$i]="$ip/32"
    fi
    canonical_cidr="$(
        uv run python - "${OPERATOR_CIDRS[$i]}" <<'PY'
import ipaddress
import sys

network = ipaddress.ip_network(sys.argv[1], strict=False)
if network.version != 4:
    raise SystemExit(1)
if network.prefixlen != 32:
    raise SystemExit(2)
print(str(network))
PY
    )" || {
        status=$?
        if [[ "$status" -eq 2 ]]; then
            die "operator SSH CIDR must be a single IPv4 /32, got: ${OPERATOR_CIDRS[$i]}"
        fi
        die "invalid operator CIDR: ${OPERATOR_CIDRS[$i]}"
    }
    OPERATOR_CIDRS[$i]="$canonical_cidr"
done

for i in "${!OPERATOR_CIDRS[@]}"; do
    for j in "${!OPERATOR_CIDRS[@]}"; do
        if [[ "$i" -lt "$j" && "${OPERATOR_CIDRS[$i]}" == "${OPERATOR_CIDRS[$j]}" ]]; then
            die "duplicate operator SSH CIDR: ${OPERATOR_CIDRS[$i]}"
        fi
    done
done

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

if [[ "$CREATE_ELASTIC_IP" == "false" ]]; then
    current_eip_instance="$(
        aws ec2 describe-addresses \
            --region "$REGION" \
            --allocation-ids "$EXISTING_ALLOCATION_ID" \
            --query 'Addresses[0].InstanceId' \
            --output text
    )"
    if [[ "$current_eip_instance" != "None" && "$current_eip_instance" != "$INSTANCE_ID" ]]; then
        if [[ "$ALLOW_REASSOCIATE_EXISTING_EIP" -ne 1 ]]; then
            die "EIP $EXISTING_ALLOCATION_ID is attached to $current_eip_instance; pass --allow-reassociate-existing-eip to move it"
        fi
        printf 'WARNING: moving EIP %s from %s to %s\n' \
            "$EXISTING_ALLOCATION_ID" "$current_eip_instance" "$INSTANCE_ID" >&2
    fi
fi

params=(
    "HostName=r8a"
    "InstanceId=$INSTANCE_ID"
    "SecurityGroupId=$SECURITY_GROUP_ID"
    "CreateElasticIp=$CREATE_ELASTIC_IP"
    "ExistingAllocationId=$EXISTING_ALLOCATION_ID"
    "OperatorSshCidr1=${OPERATOR_CIDRS[0]}"
)
for idx in 2 3 4 5; do
    array_index=$((idx - 1))
    value="${OPERATOR_CIDRS[$array_index]:-}"
    params+=("OperatorSshCidr${idx}=$value")
done

deploy_args=(
    cloudformation deploy
    --region "$REGION"
    --stack-name "$STACK_NAME"
    --template-file "$TEMPLATE"
    --parameter-overrides "${params[@]}"
    --tags Project=warp-taskgen Host=r8a ManagedBy=cloudformation
    --no-fail-on-empty-changeset
)
if [[ "$NO_EXECUTE_CHANGE_SET" -eq 1 ]]; then
    deploy_args+=(--no-execute-changeset)
fi

printf 'Deploying %s in %s for instance %s\n' "$STACK_NAME" "$REGION" "$INSTANCE_ID" >&2
aws "${deploy_args[@]}"

if [[ "$NO_EXECUTE_CHANGE_SET" -eq 1 ]]; then
    printf 'Change set created but not executed. Re-run without --no-execute-change-set to apply.\n' >&2
    exit 0
fi

outputs_json="$(
    aws cloudformation describe-stacks \
        --region "$REGION" \
        --stack-name "$STACK_NAME" \
        --query 'Stacks[0].Outputs' \
        --output json
)"
read -r OUT_ELASTIC_IP OUT_ALLOCATION_ID OUT_INSTANCE_ID <<< "$(
    uv run python - "$outputs_json" <<'PY'
import json
import sys

values = {item["OutputKey"]: item.get("OutputValue", "") for item in json.loads(sys.argv[1])}
print(
    values.get("ElasticIp", "") or "-",
    values.get("AllocationId", "") or "-",
    values.get("InstanceId", "") or "-",
)
PY
)"

actual_public_ip="$(
    aws ec2 describe-instances \
        --region "$REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text
)"

printf '\nr8a control plane deployed\n'
printf '  instance_id: %s\n' "$OUT_INSTANCE_ID"
printf '  public_ip:   %s\n' "$actual_public_ip"
printf '  allocation:  %s\n' "$OUT_ALLOCATION_ID"
printf '\nUpdate configs/benchmark_hosts/r8a.yaml advertise_host to:\n'
printf '  advertise_host: %s\n' "$actual_public_ip"

if [[ "$WRITE_HOST_CONFIG" -eq 1 ]]; then
    uv run python - "$HOST_CONFIG" "$actual_public_ip" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
new_host = sys.argv[2]
lines = path.read_text().splitlines()
updated = False
for index, line in enumerate(lines):
    if line.startswith("advertise_host:"):
        lines[index] = f"advertise_host: {new_host}"
        updated = True
        break
if not updated:
    raise SystemExit(f"advertise_host not found in {path}")
path.write_text("\n".join(lines) + "\n")
PY
    printf 'Updated %s\n' "$HOST_CONFIG"
fi

printf '\nNext required runtime step:\n'
printf '  scripts/setup_phase4_on_host.sh --host-config configs/benchmark_hosts/r8a.yaml --instances instances.scale.json --scale-config scripts/scale_config.r8a-24x24.yml\n'
