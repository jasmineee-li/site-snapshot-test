# Phase 2c admission floor: 113 / 124 (91.1%)

> **ACTIVE OPERATIONAL PIN.** This records the accepted 113/124 Phase 2c admission floor verified on 2026-04-25. Replace it only when a newer admitted cohort is intentionally pinned.

Last verified: 2026-04-25T04:28:08Z on r5 (3.12.221.9), code at HEAD=48150838.

## Outcome

| | count | classification |
|---|---|---|
| verified | 113 | admitted to Phase 4 |
| dropped at preflight (`source_data_issue.not_found`) | 7 | quarantined by d8df9078 source-data preflight |
| infeasible (probe) | 4 | sidekiq render tail (this doc) |

Per-site:
- gitlab: 80 verified / 0 infeasible
- reddit: 33 verified / 4 infeasible / 7 quarantined

## The 4 `render_unverified` infeasibles — natural sidekiq tail

Four reddit tasks miss the 20s body-poll deadline plus the 3s retry breather. The seeded comment/post succeeds at the API layer, but the rendered HTML on the canonical view finishes past the gate's patience window. Shape varies between reverifies (different task IDs trip on different runs); count is stable around 4-5. Earlier runs saw the same tail on the GitLab side; the sidekiq-tail population is platform-load-dependent, not task-dependent.

Why this is genuine platform latency, not a code path:

- The 113 verified tasks rely on the same render gate; only ~4-5 consistently exceed the deadline.
- `feasibility.errors[0].render_evidence.urls_tried` shows multiple polls all returning a body without the seed signature.
- Increasing the body-poll deadline materially (e.g. to 60s) would slow every reverify by 3-5x for one-bucket recovery. Not a worthwhile trade given the floor is well above the rigor-run threshold.

## The 7 `source_data_issue.not_found` quarantines — stale Postmill anchors

Reddit anchors `/f/news/3`, `/f/news/5`, `/f/news/6`, `/f/headphones/4` are 200 only on `reddit_0` and 404 on `reddit_1`-`reddit_9` (9 / 10 replicas). The benchmark snapshot's subreddit IDs drifted from the current Postmill state. d8df9078 caches their 404 status during preflight and quarantines the affected tasks before the probe runs, surfacing them as `source_data_issue.not_found` rather than letting them appear as flaky `request_failed`. Reclaim only on Phase 2a regen against the current snapshot or restoration of the missing subreddits.

## Acceptance

This is the documented floor for shipping Phase 4. Future Phase 2c runs that produce ≥113 / 124 admission with the same residual shape pass. Counts below 113 with new failure shapes indicate regression and warrant investigation before proceeding.

## Rigor-run provenance pin

`logs/phase_2/adversarial_tasks.json` is mutable on r5 — Phase 2c reverifies,
integration tests, and ad-hoc scripts can overwrite it. For new rigor runs,
prefer an immutable run directory or archive plus `artifact_manifest.json`
instead of copying backup files inside canonical `logs/phase_2`.

Current setup writes provenance with:

```
scripts/setup_phase4_on_host.sh \
  --host-config configs/benchmark_hosts/r5.yaml \
  --instances instances.scale.json \
  --artifacts-source s3://benchmark-archives/worldsim-runs/<run_id>/
```

If you must preserve this historical floor in-place, use the backup commands
below and update the hashes. Do not hand-edit `feasibility.status`.

```
cp -p logs/phase_2/adversarial_tasks.json                   logs/phase_2/adversarial_tasks.rigor_run_pinned.bak.json
cp -p logs/phase_2/adversarial_tasks.infeasible.json        logs/phase_2/adversarial_tasks.infeasible.rigor_run_pinned.bak.json
cp -p logs/phase_2/adversarial_tasks.dropped_source_data.json logs/phase_2/adversarial_tasks.dropped_source_data.rigor_run_pinned.bak.json
cp -p logs/phase_2/feasibility_report.json                  logs/phase_2/feasibility_report.rigor_run_pinned.bak.json
```

Before launching Phase 4, restore from pinned copies:

```
cp -p logs/phase_2/adversarial_tasks.rigor_run_pinned.bak.json                   logs/phase_2/adversarial_tasks.json
cp -p logs/phase_2/adversarial_tasks.infeasible.rigor_run_pinned.bak.json        logs/phase_2/adversarial_tasks.infeasible.json
cp -p logs/phase_2/adversarial_tasks.dropped_source_data.rigor_run_pinned.bak.json logs/phase_2/adversarial_tasks.dropped_source_data.json
cp -p logs/phase_2/feasibility_report.rigor_run_pinned.bak.json                  logs/phase_2/feasibility_report.json
```

Verify with:

```
test "$(jq length logs/phase_2/adversarial_tasks.json)" = "113" || echo "WRONG COUNT"
jq -r '.verified_count, .infeasible_count, .source_data_dropped_count, .generated_at' logs/phase_2/feasibility_report.json
```

Pinned at 2026-04-25T04:28:08Z (current as of this doc):

```
sha1  8ecccb3946b62600ae78ced80fc3c0c5ec26227f  adversarial_tasks.rigor_run_pinned.bak.json
sha1  67af0ef164e1441538fb17ea2973b8b6bcf16865  adversarial_tasks.infeasible.rigor_run_pinned.bak.json
sha1  90b69de4e717b35312640be2946228a6e2c4aae4  feasibility_report.rigor_run_pinned.bak.json
```

If a future reverify intentionally produces a new admitted set, replace the pinned files and update both the SHAs above and the timestamp.

On r5, reverify with host-local topology. Do not rely on the CLI default
`instances.smoke.json` for Phase 2c:

```
uv run python -m worldsim.main phase 2c \
  --feasibility-instances instances.scale.json
```

## Not in scope

Phase 2a regeneration. Pursued only on explicit request — see `docs/handoffs/rigor-run-setup.md` for cost/risk framing.
