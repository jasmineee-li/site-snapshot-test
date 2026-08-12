# Artifact Policy

Use this before adding, deleting, restoring, or promoting generated run output.

## Runtime boundary

`logs/` is runtime output. Keep Phase 1/2/3/4 results outside tracked source
unless the spec declares a source fixture with a stable owner, regeneration
command, reviewable size/schema, and review contract. Task-bank metadata and
curated task-card plans may be source inputs; raw run JSON, screenshots, traces,
and remote-job output belong in `logs/` or an archive.

Before relying on an artifact, check the active host or run archive. Live state
is often newer than the checkout, but a remote host is operational state rather
than canonical source. Use `remote-runs.md` wrappers for active jobs.

Phase 2a planning checkpoints live under `phase_2/shards/`. Each shard JSON is
paired with a `.manifest.json` carrying only its label, input/output task IDs,
payload hash, Run ID, and Definition Digest. The manifest is resume evidence,
not a replacement for Phase 2 validation: paused resume rejects legacy,
missing, malformed, stale, or unbound manifests and reruns that shard. Partial
shards never promote `adversarial_plans.json`.

Phase 2b text-fill checkpoints live under
`phase_2/text_fill/checkpoints/`, one JSON envelope per plan/task. Each
envelope carries the Run ID, Definition Digest, plan-input content hash, text
settings, all payload ordinals, validation records, selected seed, and compact
diagnostics. Exact continuation rejects legacy, missing, malformed, stale,
tampered, failed, or unbound envelopes and reruns that unit. Checkpoints are
resume evidence only; `adversarial_tasks.json` is promoted once the selected
text-fill work is complete.

Phase 2c feasibility checkpoints live under
`phase_2/feasibility_checkpoints/`. Each task file is an atomic, Run-bound
envelope written only after the seed work unit has attempted cleanup. The
Phase 2c validator accepts only verified or infeasible evidence whose task
content, verifier/policy/catalog versions, and hash-only instance topology
match the current run; legacy, malformed, stale, tampered, or topology-drifted
files are rerun. These sidecars are evidence for crash recovery, not Phase 4
admission grants, and no source-data preflight checkpoint is created per task.

Derived Runs are initialized under the source root's sibling
`.warp-derived-runs/<request-key>/<run-id>/` collection. The request key is an
internal idempotency key, not a public Run ID. The atomic reservation lives at
`.warp-derived-runs/.reservations/<request-key>.json`; `derived_run.json`
records the same immutable source/child lineage and requested Definition
Digest, and `pipeline_state.json` is the child's authoritative resume
checkpoint. Do not merge these roots recursively into their source Run or treat
the collection lock as a run artifact. HF exports project `source_run_id` and
`definition_digest` when the persisted Run Definition is valid. Archive one
child by passing its request-key directory as the existing archive wrapper's
`--logs-dir` and its opaque child ID as `<run_id>`.
Materialization retries preserve normal phase/root artifacts only after the
reservation, `derived_run.json`, and child `pipeline_state.json` identities all
validate; incomplete roots remain fresh-root fail-closed.

Plain `resume` never materializes a child implicitly. Operators who intend to
execute a changed definition must invoke `derive-and-resume`; that operation
binds the child root and its local `last_run_state.json` pointer for dispatch.

## Promotion record

A promoted fixture or long-lived report records:

- producing commit SHA and source run;
- host or archive path;
- counts/status summary and content hashes;
- regeneration or restore command;
- owner, schema, and reason it is a fixture rather than runtime output.

Promotion never bypasses Phase 2c admission, Phase 4 encounter, or reward
evidence. A legacy generated file that remains tracked is historical debt, not a
template for new output.

## Recovery

Use git history to restore a historical artifact for a named investigation,
keep the restored file local, and remove it when the investigation ends. If the
investigation produces a fixture, document the owner and regeneration path in
the same change. For rigor runs, follow
`docs/handoffs/rigor-run-setup.md` and keep the archive manifest in git instead
of committing runtime blobs.

Completion means every artifact touched has a current owner, source/run
provenance, retention location, and promotion or recovery decision.

## Final cutover compatibility

The expand-in-place boundary keeps the canonical local root
`logs/<run_id>/` and the archive prefix
`s3://benchmark-archives/worldsim-runs/<run_id>/`. The existing archive
wrapper selects exactly one `--logs-dir` plus opaque `<run_id>`; it does not
rename paths, flatten feature directories, or recursively archive the
`.warp-derived-runs` collection. Archive transport preserves Phase 2a shard,
Phase 2b text-fill, and Phase 2c feasibility Run Artifacts without becoming a
checkpoint validator or migration system.

Legacy roots remain readable and identity-less: WARP does not invent a Run ID
or Definition Digest and does not treat unbound legacy output as a reusable
Checkpoint. The archive manifest's `run_id` is only the selected S3 namespace
for that archive; it is not authoritative lifecycle identity and must not make
Legacy artifacts reusable. New and Derived roots retain their persisted identity. See the
tracked [final cutover readiness note](../docs/handoffs/final-cutover-readiness-2026-08-11.md)
and `tests/test_archive_run_scripts.py` for the compatibility proof.
