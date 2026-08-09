# r8a pre-sync checkpoint (2026-08-08)

Issue #35 requires a sanitized, read-only checkpoint before synchronizing the
r8a source. The evidence below authorizes the reconciliation sync; host-local
configuration, credentials, current operator addresses, sensitive host
identities, and run identifiers are intentionally omitted.

## Scope and decision

`safe_to_sync: true`.

The checkpoint records the approved clean-sync stamp and aggregate runtime
inventories only. The tracked r8a configuration remains a public template; real
operations use an ignored `configs/benchmark_hosts/r8a.local.yaml` copy. No
workflow-switch changes are included in this reconciliation checkpoint.

The machine-readable evidence, including observation times, exact check
categories, sanitized aggregates, and SHA-256 references to private raw output,
is recorded in
[`cutover-r8a-checkpoint-evidence-2026-08-08.json`](../../../../docs/research/cutover-r8a-checkpoint-evidence-2026-08-08.json).
Its SHA-256 digest is
`f3afa91f403df42e533d0f1c08baa352e59e7ae2a5c46fa4733d5cb6b92c938d`.

## Sanitized evidence

- Clean sync stamp: `739b80b7`.
- The remote checkout has no `.git` directory by design; the source sync treats
  it as an execution workspace rather than a Git checkout.
- After applying the repository sync/ignore contract, the only remaining extra
  source-class path was one expected ignored local control file whose hash
  matched the authoring copy. No tracked source path was missing and no
  unexpected source-like path was reported.
- Remote job registry: 122 exited jobs and 0 active jobs at inspection time.
- The latest historical job was exited with return code 0; one expected Phase 4
  output was absent. No job was active, and the runtime logs/results remain
  preserved because they are excluded from rsync.
- Aggregate run-level canonical artifact-file counts: 48 Phase 1 benign-task
  files, 43 Phase 2 adversarial-task files, 31 Phase 3 contract files, and 12
  Phase 4 result files.
- Archive inventory: 21 archives with 21 Phase 0a benchmark manifests present.
- Lifecycle was restored to `stopped`; the sweep tag was absent/untagged after
  inspection.

These are aggregate, sanitized observations. They do not identify the current
operator network, instance, security-group, EIP, volume, collaborator, or run.

## Sync constraints

Only source and documentation paths listed in the reconciliation manifest are
eligible for import. Generated evidence (including the smoke archive document)
is permanently excluded. Preserve the package's public-path hygiene and keep
all credentials, endpoint values, and generated artifacts host-local.
