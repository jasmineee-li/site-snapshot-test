# PVPO beginFrame / Page.navigate deadlock

> **ARCHIVED, DO NOT USE FOR CURRENT PVPO SETUP.** This documents a broken
> architecture that used dedicated PVPO Chrome containers plus
> `--enable-begin-frame-control`. That flag suspended Chrome's normal compositor
> scheduler, while browser-use waited on `Page.navigate()` without issuing
> `HeadlessExperimental.beginFrame`. Navigation timed out before the agent could
> reason. Active PVPO now uses page-surface-stable capture on the runner-owned
> browser; dedicated PVPO browser containers and `pvpo_cdp_url` are not part of
> the current run path.

## What Broke

The old PVPO design launched separate Chrome containers with begin-frame control
enabled. Under that mode, Chrome does not advance frames on the normal schedule.
browser-use 0.12.6 does not call `HeadlessExperimental.beginFrame`, so initial
navigation could block waiting for a committed frame and then fail with
`Navigation failed` / `Page.navigate() timed out`.

The failure presented as many Phase 4 rows breaking at step 1 with
`task_broke_agent_exception` and no useful agent behavior. Some downstream
classification made those rows look like `injection_not_encountered`, but the
raw issue was browser navigation deadlock, not model resistance or placement
failure.

## Historical Resolution

The immediate fix was a per-session begin-frame pump, but that path was later
removed. Commit `35ef05f3` cut PVPO over to page-surface-stable capture on the
runner-owned browser and removed `worldsim/phase_4/pvpo_frame_pump.py`, the
dedicated browser container path, and related tests/scripts.

## Current Rule

Use the current technical spec, `agent_docs/domain-invariants.md`, and
`agent_docs/remote-runs.md` for PVPO behavior. Treat any old instruction to run
dedicated `pvpo-chrome-*` containers or depend on `pvpo_cdp_url` as historical
debugging context only.
