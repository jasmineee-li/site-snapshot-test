# r8a Control Plane

r8a is the canonical WARP Taskgen benchmark host. Its AWS control-plane
identity should be managed declaratively; benchmark runtime state remains
generated on the host.

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

## Deploy

Preview the stack:

```bash
scripts/deploy_r8a_control_plane.sh \
  --operator-cidr auto \
  --no-execute-change-set
```

Apply it:

```bash
scripts/deploy_r8a_control_plane.sh \
  --operator-cidr auto \
  --write-host-config
```

Use explicit CIDRs for shared operators:

```bash
scripts/deploy_r8a_control_plane.sh \
  --operator-cidr 128.84.124.235/32 \
  --extra-operator-cidr 128.84.126.158/32 \
  --write-host-config
```

To intentionally move an existing EIP, such as the old r5 EIP, make the cutover
explicit:

```bash
scripts/deploy_r8a_control_plane.sh \
  --operator-cidr auto \
  --existing-allocation-id eipalloc-... \
  --allow-reassociate-existing-eip \
  --write-host-config
```

Without `--allow-reassociate-existing-eip`, the deploy script refuses to move an
EIP that is still attached to another instance.

After the EIP changes, regenerate r8a runtime artifacts on the host:

```bash
scripts/setup_phase4_on_host.sh \
  --host-config configs/benchmark_hosts/r8a.yaml \
  --instances instances.scale.json \
  --scale-config scripts/scale_config.r8a-24x24.yml
```

## Audit

```bash
scripts/audit_r8a_control_plane.sh
```

The audit fails if r8a has an ephemeral public IP, if the checked-in
`advertise_host` no longer matches EC2, if the expected security group is not
attached, or if the CloudFormation stack allocation does not match the ENI.

## Edge Cases

- **Existing r8a has an ephemeral IP.** Associating an EIP releases the
  ephemeral address. Update `configs/benchmark_hosts/r8a.yaml` and regenerate
  host-local topology immediately after deploy.
- **Existing manual SSH rule duplicates a stack rule.** CloudFormation may reject
  duplicate security group ingress. Remove the manual duplicate or use a
  different operator CIDR before retrying.
- **Stopping r8a before EIP deploy.** The public IP may change on restart. Audit
  and redeploy before trusting public smoke or proxy URLs.
- **Pulling over the r5 EIP.** This is a deliberate cutover. The deploy helper
  requires `--allow-reassociate-existing-eip` so an old host is not broken by a
  typo or stale allocation ID.
- **Terminating r8a.** The current root volume is delete-on-termination. Stop is
  safe; terminate is data destructive unless volume retention is changed first.
- **Multiple r8a instances match the tag.** Pass `--instance-id` explicitly.
- **Collaborator IP drift.** Re-run the stack with the new `/32` CIDR. Do not
  open SSH to `0.0.0.0/0`; the deploy script refuses that value.
