#!/usr/bin/env bash
# ssh_r5.sh — open a shell on r5 via AWS SSM.
#
# Why: SSH to r5 goes through sg-08792057943b27a65 pinned to the operator's
# home IP. Every IP change (travel, VPN, new ISP) breaks SSH until the SG
# rule is re-authorized. SSM uses IAM, not network ACLs — it keeps working
# from anywhere the operator can authenticate to AWS.
#
# Prereq (one-time, requires explicit approval per memory/feedback_sg_changes_explicit):
#   attach the worldsim-ec2-benchmark-backup instance profile to
#   i-03acfc08597207960 so it can register with SSM. Then:
#     aws ssm start-session --target i-03acfc08597207960 --region us-east-2
#
# Usage:
#   scripts/ssh_r5.sh                  # interactive shell
#   scripts/ssh_r5.sh 'cd /home/ubuntu/browser-sim && git log -1'   # one-shot
#
# Env overrides:
#   R5_INSTANCE_ID (default: i-03acfc08597207960)
#   R5_REGION      (default: us-east-2)

set -euo pipefail

INSTANCE_ID="${R5_INSTANCE_ID:-i-03acfc08597207960}"
REGION="${R5_REGION:-us-east-2}"

if ! command -v aws >/dev/null 2>&1; then
    echo "ERROR: aws CLI not found. brew install awscli" >&2
    exit 2
fi

if ! aws sts get-caller-identity --region "$REGION" >/dev/null 2>&1; then
    echo "ERROR: aws credentials not active. aws sso login / export AWS_PROFILE=..." >&2
    exit 2
fi

if [[ $# -eq 0 ]]; then
    exec aws ssm start-session \
        --target "$INSTANCE_ID" \
        --region "$REGION"
fi

# One-shot command mode: run the joined args as a single bash -lc invocation
# inside the target. Use run-command (not start-session) so output streams
# back cleanly instead of landing in a pseudo-terminal.
CMD="$*"
aws ssm send-command \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters "commands=[\"$CMD\"]" \
    --query 'Command.CommandId' \
    --output text >/tmp/ssh_r5.cmdid

CMD_ID="$(cat /tmp/ssh_r5.cmdid)"
echo "submitted command $CMD_ID; polling..." >&2

while true; do
    STATUS="$(aws ssm get-command-invocation \
        --region "$REGION" \
        --command-id "$CMD_ID" \
        --instance-id "$INSTANCE_ID" \
        --query 'Status' \
        --output text 2>/dev/null || echo Pending)"
    case "$STATUS" in
        Success|Failed|Cancelled|TimedOut)
            break ;;
    esac
    sleep 1
done

aws ssm get-command-invocation \
    --region "$REGION" \
    --command-id "$CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --query 'StandardOutputContent' \
    --output text

if [[ "$STATUS" != "Success" ]]; then
    echo "--- stderr ---" >&2
    aws ssm get-command-invocation \
        --region "$REGION" \
        --command-id "$CMD_ID" \
        --instance-id "$INSTANCE_ID" \
        --query 'StandardErrorContent' \
        --output text >&2
    exit 1
fi
