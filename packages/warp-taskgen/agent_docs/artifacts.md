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
