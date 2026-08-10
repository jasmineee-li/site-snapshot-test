# WARP agent router

This repository contains WARP, the browser-agent safety benchmark and the
evaluation-awareness study behind it. Read `README.md` when the repository
purpose or released artifacts matter; use the path-specific guides below for
implementation work.

## Route by path first

- For `packages/warp-taskgen/**`, read `packages/warp-taskgen/CLAUDE.md` before
  editing. It owns Taskgen architecture, safety invariants, and verification.
- For `eval_awareness_experiments/**`, `eval_awareness/**`, `probes/**`, or
  `models/**`, read `eval_awareness_experiments/CLAUDE.md` first. It owns the
  study workflow, data provenance, benchmark safety, and experiment logging.
- For issue or specification work, read `docs/agents/issue-tracker.md`.
- For triage decisions, read `docs/agents/triage-labels.md`.
- For terminology or a domain decision, read `docs/agents/domain.md` and then
  the owning path guide or specification.
- When launching or updating an experiment, read
  `eval_awareness_experiments/experiment_log.md` and add the new entry at the
  top of that log.

## Canonical delivery boundary

The repository-wide workflow is one focused change in one short-lived topic
worktree from `origin/main`, delivered as one PR to `main`, then removed after
merge. Path-specific guides own the commands, tests, and domain details; this
file owns the one-worktree/one-PR boundary. Keep implementation and review
scoped to the requested path. Do not create a second source snapshot or a
sync-back workflow.

## Stable names and safety

- Use **WARP** in new prose. Keep historical `worldsim` identifiers, schema
  names, environment variables, and infrastructure names when compatibility
  requires them; rename them only in explicitly scoped work.
- Run adversarial and state-changing browser tasks only on configured sandboxed
  benchmark infrastructure. Evaluation data collection may perform the
  documented read-only capture of public pages; do not use it for authenticated,
  private, destructive, or account-affecting actions.
- Preserve benchmark admission, exposure, visibility, scoring, readback, and
  safety checks. Treat vendored or upstream trees as read-only references unless
  the path guide explicitly assigns a patch workflow.
- Keep API keys and host credentials in the environment or approved local
  configuration. Never commit secrets or paste them into logs, fixtures, or
  experiment notes.

## Guidance hygiene

- Keep one source of truth for each rule. Do not copy package or experiment
  command catalogs into this router; discover commands from the owning guide,
  current environment, and each tool's `--help` output.
- Prefer existing specs, helpers, tests, and artifact formats over new
  abstractions. Record the evidence for completion: changed paths, validation,
  and any unresolved blocker.
- If a path-specific instruction conflicts with this router, stop and surface
  the conflict before changing behavior. Keep this file stable and small so
  nested guidance can carry branch-specific detail.

## Completion

Before handoff, verify the requested scope and relevant checks, then report
the changed paths, evidence, and any blocker. A clean diff is not a substitute
for the path guide's validation contract.
