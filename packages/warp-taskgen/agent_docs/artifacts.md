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

## Where To Look For Current Artifacts

Do not assume git has the newest task artifacts. Phase 1, Phase 2, Phase 3, and
Phase 4 commands write mutable state under `logs/`, and r5 is often the freshest
working copy during live benchmark work.

Before changing, deleting, or relying on task artifacts, check the active host
or archive first. Use repository remote-run wrappers for active jobs.

Treat r5 as live operational state, not automatically canonical source. If a
remote artifact is promoted into a paper, rigor run, or long-lived fixture,
record the producing code SHA, host or archive path, counts, hashes, and restore
command in git. Otherwise keep it out of tracked source.

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

## Current Removed Artifacts

This branch removes generated artifacts that were tracked at baseline
`origin/feat/worldsim-v5` (`9b9f9a56`). The same current artifact shape was
observed on r5 at `/home/ubuntu/browser-sim` on 2026-04-28. The r5 checkout was
at detached `HEAD=07919d7e` with local source edits, and its main Phase 1/2 JSON
artifacts had filesystem mtimes around `2026-04-28T01:18:52Z`. r5 also had
newer Phase 2 exposure sidecars under `logs/phase_2/` and later remote-job logs,
so agents should check r5 before assuming the git baseline has the latest run
state.

These files are not the documented 113-task rigor floor from
`docs/handoffs/phase-2c-admission-floor.md`. The checked-in Phase 2 report at
the baseline says `phase_2_status=failed`, `instances=instances.smoke.json`,
`verified_count=92`, `infeasible_count=6`, and
`source_data_dropped_count=0`.

| Path | Classification | Count | SHA1 |
| --- | --- | ---: | --- |
| `logs/phase_1/benign_tasks.json` | Phase 1 generated output | 872 | `41599bd04da5ed977f5c7fd7efa6b03e1682f776` |
| `logs/phase_1/novel_tasks_gitlab.json` | Phase 1 generated cache/output | 30 | `6a3534bc8c068f521bd04f35ab6f71f6aafc6435` |
| `logs/phase_2/adversarial_tasks.json` | Phase 2c generated output, failed smoke run | 92 | `fe878076b86d546032c32b456dd6166278e26c66` |
| `logs/phase_2/adversarial_tasks.infeasible.json` | Phase 2c generated output, failed smoke run | 6 | `0ca26e9695859436836e83b2b9498b4b54f4619c` |
| `logs/phase_2/adversarial_tasks.dropped_source_data.json` | Phase 2c generated output, empty sidecar | 0 | `97d170e1550eee4afc0af065b78cda302a97674c` |
| `logs/phase_2/adversarial_tasks.map_quarantine.json` | Historical map quarantine reference | 76 | `6e800a9970ff6119528e48fec41ba96d81ccc82b` |
| `logs/phase_2/feasibility_report.json` | Phase 2c generated report, failed smoke run | 1 report | `fa63c65d6f7205494a4d6fa017e734044fbceeb6` |
| `logs/phase_2/new_task_resolver_dropouts.json` | Phase 2 generated dropout sidecar | 1 dropout | `dcc7044a4cb7bca15264c968e452d0dcb3fc90b9` |

The map quarantine file is intentionally not discarded as knowledge. It is
recoverable from git history and should be restored into `logs/` only for
explicit map redesign or migration work. If map is re-admitted, create a new
source fixture or archived artifact with a clear owner instead of re-tracking
runtime output under `logs/`.

## Recovery

Restore a previously tracked artifact from git history only for a specific
historical investigation:

```bash
mkdir -p logs/phase_1 logs/phase_2
git show 9b9f9a56:logs/phase_2/adversarial_tasks.map_quarantine.json \
  > logs/phase_2/adversarial_tasks.map_quarantine.json
```

Use the same pattern for the other removed files, or replace `9b9f9a56` with a
specific historical commit when investigating a different artifact snapshot.
Keep restored files local unless the work explicitly converts them into
documented fixtures.

For rigor runs, follow `docs/handoffs/rigor-run-setup.md`: sync artifacts from
the run archive into `logs/`, run the preflight or live gate, and keep the
archive manifest in git rather than committing runtime JSON blobs.
