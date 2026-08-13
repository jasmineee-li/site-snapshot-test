# r8a Control Plane

r8a is the canonical WARP Taskgen benchmark host. Its AWS control-plane
identity should be managed declaratively; benchmark runtime state remains
generated on the host.

The canonical r8a scale topology is 24 GitLab replicas plus 24 Reddit/Postmill
replicas, generated from `scripts/scale_config.r8a-24x24.yml`. W48 runs use
that 48-instance pool with top-level `warp-taskgen phase 4
--phase-4-max-workers 48`; `--workers 48` belongs only to
`scripts/run_phase4_process_pool.py`.

## Boundary

CloudFormation owns:

- the canonical Elastic IP allocation;
- the Elastic IP association to the existing r8a EC2 instance;
- the operator SSH ingress rules that make remote job and lifecycle scripts
  usable from approved networks.

Existing scripts own:

- Docker compose generation;
- `instances.scale.json` / `instances.smoke.json`;
- proxy port maps;
- storage-state minting;
- phase artifacts and remote job execution.

This split keeps durable AWS resources drift-detectable without pretending that
per-run benchmark topology is static infrastructure.

Local generated files are scratch. It is normal for a laptop checkout to have
missing or stale `instances.scale.json`, `instances.smoke.json`,
`scripts/proxy_ports.conf`, and generated compose files. Regenerate them for the
selected host instead of treating the local copies as source of truth.

The tracked host config is a sanitized template. For a real operation, copy it
to the ignored local path and fill in the selected host's values:

```bash
cp configs/benchmark_hosts/r8a.yaml configs/benchmark_hosts/r8a.local.yaml
# edit r8a.local.yaml locally; never commit that file
```

## Deploy

Preview the stack:

```bash
scripts/deploy_r8a_control_plane.sh \
  --host-config configs/benchmark_hosts/r8a.local.yaml \
  --operator-cidr auto \
  --no-execute-change-set
```

Apply it:

```bash
scripts/deploy_r8a_control_plane.sh \
  --host-config configs/benchmark_hosts/r8a.local.yaml \
  --operator-cidr auto \
  --write-host-config
```

Use explicit CIDRs for shared operators:

```bash
scripts/deploy_r8a_control_plane.sh \
  --host-config configs/benchmark_hosts/r8a.local.yaml \
  --operator-cidr <approved-operator-cidr> \
  --extra-operator-cidr <additional-operator-cidr> \
  --write-host-config
```

To intentionally move an existing EIP allocation, make the cutover
explicit:

```bash
scripts/deploy_r8a_control_plane.sh \
  --host-config configs/benchmark_hosts/r8a.local.yaml \
  --operator-cidr auto \
  --existing-allocation-id <allocation-id> \
  --allow-reassociate-existing-eip \
  --write-host-config
```

Without `--allow-reassociate-existing-eip`, the deploy script refuses to move an
EIP that is still attached to another instance.

After the EIP changes, regenerate r8a runtime artifacts on the host:

```bash
scripts/setup_phase4_on_host.sh \
  --host-config configs/benchmark_hosts/r8a.local.yaml \
  --instances instances.scale.json \
  --scale-config scripts/scale_config.r8a-24x24.yml
```

`setup_phase4_on_host.sh` runs the r8a control-plane audit before regenerating
runtime topology, so a stale `advertise_host`, missing EIP association, wrong
security group, or missing SSH rule fails before new instance files or storage
state are minted.

For local inspection only, regenerate into a temp directory instead of writing
ignored root artifacts:

```bash
uv run python scripts/generate_compose_scale.py \
  --config scripts/scale_config.r8a-24x24.yml \
  --base-config instances.json \
  --host-config configs/benchmark_hosts/r8a.local.yaml \
  --out-dir "${TMPDIR:-.}/r8a-gen-audit" \
  --final-config-dir "$PWD"
```

## Audit

```bash
scripts/audit_r8a_control_plane.sh \
  --host-config configs/benchmark_hosts/r8a.local.yaml
```

The audit fails if r8a has an ephemeral public IP, if the checked-in
`advertise_host` no longer matches EC2, if the expected security group is not
attached, or if the CloudFormation stack allocation does not match the ENI.

## Lifecycle policy

The control plane also owns the three-layer defense against idle compute
billing and accidental termination. All four pieces work independently:
disabling any one of them does not break the others.

```
Layer 1: scripts/host_park.sh + scripts/host_resume.sh   (operator-driven)
Layer 2: EventBridge daily auto-stop (tag-gated)         (catches "forgot")
Layer 3: CloudWatch low-CPU stop alarm                   (backstop)
+ Termination protection                                 (always-on guard)
+ worldsim:sweep-in-progress tag                         (live-run gate)
```

### The sweep tag

`worldsim:sweep-in-progress=true` is the contract every layer respects.
Set the tag before any work that must not be interrupted; clear it once
the run and its archive are complete. `scripts/host_resume.sh` sets it
automatically; `warp_taskgen/phase_4/sweep_tag.py` (loaded by the Phase 4
runner) sets it best-effort at run start and clears it at run end.

### Operator workflow

```bash
# Before bed (or any long gap between sweeps):
scripts/host_park.sh --host-config configs/benchmark_hosts/r8a.local.yaml

# Before a new sweep:
scripts/host_resume.sh --host-config configs/benchmark_hosts/r8a.local.yaml

# After the sweep AND its archive complete:
aws ec2 delete-tags --region us-east-2 \
  --resources <instance-id> \
  --tags Key=worldsim:sweep-in-progress
```

If `host_park.sh` refuses because the tag is set but no Phase 4 process
is actually running, the runner crashed and left the tag stale. Clear
it manually (above) or pass `--force` (operator IAM identity is logged).

### Termination protection

`DisableApiTermination=true` is set on the instance via
`scripts/enable_r8a_termination_protection.sh`. CloudFormation cannot
manage this attribute on imported instances. See the "Terminating r8a"
edge case below for the deliberate replace-and-relaunch dance.

### Layer 2: EventBridge daily auto-stop

A daily `AWS::Scheduler::Schedule` (default `cron(0 3 * * ? *)` UTC)
invokes the `worldsim-r8a-auto-stop` Lambda. The Lambda:

1. Describes the instance and reads tags.
2. If the instance is already stopped or stopping: publishes a
   "skipped: already stopped" SNS message and returns.
3. If `worldsim:sweep-in-progress=true`: publishes a "skipped: sweep in
   progress" SNS message and returns.
4. Otherwise: calls `ec2:StopInstances` and publishes "auto-stopped".

Failures land in a 14-day DLQ (`worldsim-r8a-auto-stop-dlq`).

To temporarily disable for a multi-day sweep window, redeploy with
`AutoStopEnabled=false` (or disable the schedule in the AWS console).

Subscribe to notifications by passing `AutoStopNotificationEmail` at
stack deploy time, or out-of-band:

```bash
aws sns subscribe --region us-east-2 \
  --topic-arn $(aws cloudformation describe-stacks --region us-east-2 \
    --stack-name worldsim-r8a-control-plane \
    --query "Stacks[0].Outputs[?OutputKey=='AutoStopSnsTopicArn'].OutputValue" \
    --output text) \
  --protocol email --notification-endpoint you@example.com
```

### Layer 3: CloudWatch idle-stop alarm

`worldsim-r8a-idle-stop` fires after 24 consecutive 5-minute periods of
`CPUUtilization < 2%` (= 2 hours of fully-idle CPU). The native EC2
`ec2:stop` action is wired directly; no Lambda is involved.

Native action means the alarm does NOT check the sweep tag. The tight
threshold is what protects live runs: a real sweep moves CPU well above
2% during browser sessions, so two consecutive idle hours during a
sweep is implausible. If we observe false positives we switch to a
Lambda variant with tag checking; that has not been needed to date.

To temporarily disable: redeploy with `IdleStopAlarmEnabled=false`, or
in the AWS console disable the alarm action.

### Testing the layers

```bash
# Layer 1 dry-run:
scripts/host_park.sh --host-config configs/benchmark_hosts/r8a.local.yaml --dry-run

# Layer 2 manual invocation (with tag set => skip):
aws ec2 create-tags --region us-east-2 \
  --resources <instance-id> \
  --tags Key=worldsim:sweep-in-progress,Value=true
aws lambda invoke --region us-east-2 \
  --function-name worldsim-r8a-auto-stop "${TMPDIR:-.}/auto-stop-skipped.json"
# (expect SNS "skipped: sweep in progress")

# Layer 2 manual invocation (without tag => stops if running):
aws ec2 delete-tags --region us-east-2 \
  --resources <instance-id> \
  --tags Key=worldsim:sweep-in-progress
aws lambda invoke --region us-east-2 \
  --function-name worldsim-r8a-auto-stop "${TMPDIR:-.}/auto-stop-fired.json"
# (expect instance state -> stopping; SNS "auto-stopped")

# Layer 3 is verified by leaving the instance idle for 2h with Layer 2
# disabled and observing the alarm in the CloudWatch console.
```

## Edge Cases

- **Existing r8a has an ephemeral IP.** Associating an EIP releases the
  ephemeral address. Update the ignored `configs/benchmark_hosts/r8a.local.yaml`
  and regenerate
  host-local topology immediately after deploy.
- **Existing manual SSH rule duplicates a stack rule.** CloudFormation may reject
  duplicate security group ingress. Remove the manual duplicate or use a
  different operator CIDR before retrying.
- **Stopping r8a before EIP deploy.** The public IP may change on restart. Audit
  and redeploy before trusting public smoke or proxy URLs.
- **Reassociating an existing EIP.** This is a deliberate cutover. The deploy helper
  requires `--allow-reassociate-existing-eip` so an old host is not broken by a
  typo or stale allocation ID.
- **Terminating r8a.** The current root volume is delete-on-termination. Stop is
  safe; terminate is data destructive unless volume retention is changed first.
  Termination protection is the durable guard against accidental termination
  (the CLI call below fails until protection is explicitly cleared):
  ```bash
  scripts/enable_r8a_termination_protection.sh \
    --host-config configs/benchmark_hosts/r8a.local.yaml
  ```
  CloudFormation does not manage `DisableApiTermination` on imported instances,
  so this attribute is set out-of-band. To deliberately replace the instance
  (AMI rotate, family change), clear protection first, terminate, relaunch,
  then re-enable on the replacement:
  ```bash
  scripts/enable_r8a_termination_protection.sh \
    --host-config configs/benchmark_hosts/r8a.local.yaml \
    --disable
  aws ec2 terminate-instances --region us-east-2 --instance-ids <old>
  # ... launch replacement ...
  scripts/enable_r8a_termination_protection.sh \
    --host-config configs/benchmark_hosts/r8a.local.yaml   # on the new instance
  ```
- **Multiple r8a instances match the tag.** Pass `--instance-id` explicitly.
- **Operator network drift.** Re-run the stack with the newly approved `/32`
  value from the local config. Do not open SSH to `0.0.0.0/0` or broad network
  ranges; the deploy script canonicalizes operator IPs to `/32` and rejects
  broader CIDRs.
