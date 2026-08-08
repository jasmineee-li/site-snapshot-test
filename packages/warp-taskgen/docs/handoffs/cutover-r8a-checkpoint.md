# r8a pre-sync checkpoint (2026-08-08)

Issue #35 requires a read-only checkpoint of the stopped r8a host before any
cutover source synchronization. This record intentionally contains only
sanitized operator evidence. It does not contain host-local configuration,
credentials, public addresses, run identifiers, or command-line secrets.

## Scope and decision

`safe_to_sync: false` (blocked). The remote checkout identity, source dirt,
remote job registry, and remote experiment-artifact state could not be read
through the approved operator path. A sync would therefore be unable to prove
that it preserves the remote checkout or any remote-only work. Do not sync
until an approved SSH path is restored and the three remote inventories below
are captured with the existing wrappers.

No remote checkout, job, topology, artifact, archive object, or credential was
modified, deleted, or overwritten during this inspection.

## Lifecycle evidence

- Host config: authoring-tree `configs/benchmark_hosts/r8a.yaml` (the explicit
  r8a config with the durable instance identity).
- Initial observation: `stopped`; root device `ebs`; instance type
  `r8a.24xlarge`; `worldsim:sweep-in-progress` was unset. This was obtained
  with `scripts/host_park.sh --dry-run`.
- To make a read-only inspection possible, the host was temporarily started
  with `scripts/host_resume.sh --no-tag`. EC2 system and instance status checks
  reached `ok`, and the existing r8a control-plane audit passed.
- The host was restored with the guarded `scripts/host_park.sh` wrapper. A
  final EC2 read confirmed `stopped`; the sweep tag remained unset.

## Remote checkout and source dirt

| Check | Result | Evidence/constraint |
| --- | --- | --- |
| Checkout SHA | unavailable | The existing `remote_job_list.sh`/SSH path timed out twice before any remote command ran. |
| Branch and tracked status | unavailable | Same SSH timeout; no local checkout was substituted for the remote one. |
| Source-like untracked paths | unavailable | Same SSH timeout; no remote filesystem scan was attempted through an alternate path. |

The absence of these values is itself a block: the checkpoint cannot certify
that a source sync would preserve remote-only tracked or untracked work.

## Remote jobs and experiment artifacts

| Inventory | Result | Evidence/constraint |
| --- | --- | --- |
| Registered remote jobs | unavailable | `scripts/remote_job_list.sh --json` timed out over the approved SSH path. No job was started, stopped, or bypassed. |
| Per-job status | unavailable | `scripts/remote_job_status.sh --latest` also timed out before resolving a job ID; do not infer that the registry is empty. |
| Host-local experiment artifacts | unavailable | The remote checkout and its logs could not be read; no artifact was copied or deleted. |
| Archive location | known, separate from source | The local gitignored archive index records the configured S3 prefix in `us-east-2`; a read-only bucket-head check succeeded. No run IDs or object contents are included here. |
| Local archive index snapshot | 21 entries | The index states it was last regenerated on 2026-05-11 22:22 UTC. This is historical index evidence, not proof of the current host contents. |

Runtime configuration, generated topology, vendor trees, storage state, and
credentials remain host-local by policy. None were read, synced, or published
by this checkpoint.

## Connectivity blocker and next safe action

The configured public SSH endpoint timed out on two invocations of the existing
remote-job wrapper. A bounded private-address SSH probe also timed out, and the
instance has no AWS Systems Manager managed-instance record. The control-plane
audit passed, so this checkpoint does not authorize changing security-group
ingress, adding a tunnel, overriding topology, or using an unapproved access
path.

After an approved operator access path is available, rerun the existing
`remote_job_list.sh` and `remote_job_status.sh` wrappers, capture the remote
checkout SHA/branch/tracked status/source-like untracked paths, and inventory
the remote artifact/archive state. Re-evaluate `safe_to_sync` only after all
three inventories are present and no registered job is active.
