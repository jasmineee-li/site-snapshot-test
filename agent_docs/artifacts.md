# Artifact Policy

Use this when deciding whether generated run output belongs in git.

## Rule

`logs/` is runtime output. Do not commit new files under `logs/` unless a spec
section explicitly declares the file to be a source fixture and gives it a stable
owner, regeneration path, and review contract.

For normal Phase 1, Phase 2, Phase 3, and Phase 4 runs, keep large artifacts
outside the tracked source tree. Preserve auditability with:

- the producing commit SHA
- the host or archive location
- counts and status summaries
- content hashes when promoting an artifact
- the command used to regenerate or restore the artifact

Git history remains the recovery path for artifacts that were previously
tracked. For new rigor runs, prefer an artifact archive such as the S3 path in
`docs/handoffs/rigor-run-setup.md`, then keep only the manifest or runbook in
git.

Some legacy generated artifacts may still be tracked while readiness debt is
being unwound. Treat those as deferred historical fixtures, not permission to
commit new runtime output under `logs/`.

## Where To Look

Do not assume git has the newest task artifacts. Phase commands write mutable
state under `logs/`, and r5 is often the freshest working copy during live
benchmark work.

Before changing, deleting, or relying on task artifacts, check the active host
or run archive first. Use repository remote-run wrappers for active jobs. If an
artifact is promoted into a paper, rigor run, or long-lived fixture, record the
producing code SHA, host or archive path, counts, hashes, and restore command in
git. Otherwise keep it out of tracked source.

## Promotion Criteria

A generated artifact may become tracked only when all of these are true:

- the spec or a handoff declares it a fixture rather than runtime output
- the artifact has a stable owner and regeneration command
- the artifact has a reviewable size and schema
- the producing code version and source run are recorded
- fixture promotion does not bypass Phase 2c admission or Phase 4 encounter
  science

Task-bank metadata and task-card plans may be tracked when they are curated
dataset inputs. Raw Phase 1/2/3/4 outputs should stay in `logs/` or an archive.

## Recovery

Restore a previously tracked artifact from git history only for a specific
historical investigation:

```bash
mkdir -p logs/phase_1 logs/phase_2
git show <commit>:logs/phase_2/adversarial_tasks.json \
  > logs/phase_2/adversarial_tasks.json
```

Keep restored files local unless the work explicitly converts them into
documented fixtures.

For rigor runs, follow `docs/handoffs/rigor-run-setup.md`: sync artifacts from
the run archive into `logs/`, run the preflight or live gate, and keep the
archive manifest in git rather than committing runtime JSON blobs.
