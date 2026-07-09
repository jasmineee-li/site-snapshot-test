# CLAUDE.md

Guidance for Claude Code and other agentic tools working in this repository.
This is the public **WARP** repo: the code, benchmark, and analysis behind the
paper "WARP: Measuring and Mitigating Evaluation Awareness in Browser-Agent
Safety Benchmarks." Read `README.md` first for the project overview.

## Workflow

- Commit frequently and atomically. One commit per logical change, don't
  batch unrelated edits. Commit as soon as a unit of work is complete and
  verified, rather than accumulating diffs across a session.
- Always `git push` after every atomic commit on a non-main branch. Durable
  authorization: the user has pre-approved pushing here, so don't re-ask per
  commit. Protected: never push to `main` / `master` without explicit
  confirmation.
- Keep edits scoped, and prefer existing patterns and helpers over new
  abstractions.

## Experiment log

The experiment log lives in-repo (visible in PRs and clones) because the
project has external collaborators.

Path: `eval_awareness_experiments/experiment_log.md`, newest entry at the top,
one entry per experiment (`## YYYY-MM-DD — slug`, with Type / Groups / What /
Why / Parent / Key change / Files / Results / Notes). Add an entry when an
experiment is launched (a plan plus empty results is fine) and update it when
results land.

## What's here

The repo holds the two lines of work from the paper, plus shared infrastructure:

- **`packages/warp-taskgen/`** is the WARP benchmark-generation pipeline
  (Phases 0-4: recon, benign-task generation, adversarial-twin seeding,
  admission, and scored rollout with the eval-awareness iterator). This is the
  primary codebase. It has its own `CLAUDE.md` and `agent_docs/`; read those
  before working in it.
- **`eval_awareness/`, `eval_awareness_experiments/`, `probes/`, `models/`**
  hold the cross-benchmark evaluation-awareness study: measuring whether
  frontier models recognize IPI safety benchmarks (DoomArena, WASP, AgentDojo,
  InjecAgent) as evaluations, the causal eval-cue intervention, and
  residual-stream probes on open-weight CUA models. Start at
  `eval_awareness_experiments/EXPERIMENTS.md`.
- **`AgentLab/`** is vendored, read-only reference: the BrowserGym
  `GenericAgent` harness the eval runs go through. Do not treat it as
  first-party code and do not import from it into `warp-taskgen`; retype
  equivalent behavior if you need it.
- **`README.md`** front door, **`docs/`** handoff notes, **`data/`**,
  **`results/`**, **`notebooks/`**, **`read_logs/`** tracked artifacts and
  analysis, **`scripts/`** loose experiment tooling.

## Commands

The WARP pipeline is its own `uv` project:

```bash
cd packages/warp-taskgen
uv sync --extra dev
uv run warp-taskgen --help
```

For tests, lint, live gates, and pipeline internals, use
`packages/warp-taskgen/CLAUDE.md` and `packages/warp-taskgen/agent_docs/`
(`verification.md`, `domain-invariants.md`, `phase4-reporting-metrics.md`,
`trace-inspection.md`).

## Naming

This project was previously **WorldSim**; public docs use **WARP**. The
distribution, CLI, Modal app, HF dataset, and specs are all WARP-named. The
internal Python package, the `WORLDSIM_*` env vars, schema-version constants,
and some infra identifiers (`X-Worldsim-Token`, `worldsim-proxy.conf`)
intentionally stay `worldsim`: renaming them would break saved artifacts,
external runbooks, and the deployed proxy/compose stack. Use **WARP** in new
prose; leave `worldsim` code identifiers alone unless a change is explicitly
scoped to renaming them.

## Scope and safety

WARP is safety-evaluation infrastructure for browser-agent indirect prompt
injection. Work only on repo code, generated logs/traces, and configured
benchmark infrastructure. Do not target real services, credentials, users,
money, or production systems; all adversarial content is scoped to sandboxed
host-environment instances.
