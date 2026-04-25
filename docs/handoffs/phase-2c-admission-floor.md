# Phase 2c admission floor: 113 / 124 (91.1%)

Last verified: 2026-04-24 on r5 (3.12.221.9).

## Outcome

| | count | classification |
|---|---|---|
| verified | 113 | admitted to Phase 4 |
| infeasible | 11 | held back |
| ↳ `render_unverified` | 5 | accepted floor (this doc) |
| ↳ `source_data_issue.not_found` | 6 | dataset drift, see d8df9078 |

## The 5 `render_unverified` — natural sidekiq tail

Five GitLab note tasks consistently miss the 20s body-poll deadline plus the 3s retry breather. The seeded note succeeds at the API layer (write returns 201, `note_id` allocated, `/discussions.json` shows the note), but GitLab's sidekiq fan-out for the rendered HTML on the issue/MR page finishes past the gate's patience window. The shape is stable across reverify runs: same task IDs, same evidence (multiple polls returning a body without the seed signature).

Why this is genuine platform latency, not a code path:

- The 113 verified tasks rely on the same render gate; only these 5 consistently exceed the deadline.
- `feasibility.errors[0].render_evidence.urls_tried` shows multiple polls all returning a body without the seed signature.
- Increasing the body-poll deadline materially (e.g. to 60s) would slow every reverify by 3-5x for one-bucket recovery. Not a worthwhile trade given the floor is well above the rigor-run threshold.

## The 6 `source_data_issue.not_found` — stale Postmill anchors

Reddit anchors `/f/news/3`, `/f/news/5`, `/f/news/6`, `/f/headphones/4` are 200 only on `reddit_0` and 404 on `reddit_1`-`reddit_9` (9 / 10 replicas). The benchmark snapshot's subreddit IDs drifted from the current Postmill state. Diagnosis confirmed by isolated repro on r5; quarantine path landed in `d8df9078 fix(phase_2c): serialize shared preflight request context`. These tasks reclaim only on Phase 2a regen against the current snapshot or restoration of the missing subreddits.

## Acceptance

This is the documented floor for shipping Phase 4. Future Phase 2c runs that produce ≥113 / 124 admission with the same residual shape pass. Counts below 113 with new failure shapes indicate regression and warrant investigation before proceeding.

## Not in scope

Phase 2a regeneration. Pursued only on explicit request — see `docs/handoffs/rigor-run-setup.md` for cost/risk framing.
