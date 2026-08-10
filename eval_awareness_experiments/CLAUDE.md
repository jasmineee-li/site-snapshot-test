# Evaluation-awareness experiments

This tree measures whether browser-agent models recognize benchmark or
evaluation cues, how that recognition changes behavior, and how those signals
relate to safety outcomes. It contains website data, trajectories, benchmark
runners, judges, and analysis. Preserve provenance and distinguish experiment
design from observed results.

## Route before editing or running

- Experiment design, prompts, judges, analysis, or causal conditions: read
  `EXPERIMENTS.md` first.
- Dependency, environment, container, or benchmark setup: read `SETUP.md`, then
  `SAFETY_BENCHMARKS_HANDOFF.md` for safety-benchmark runners.
- Website extraction or trajectory import: read `data/IMPORTING.md` and, when
  its format description is needed, `data/README.md`.
- A new run, result interpretation, or handoff: read
  `experiment_log.md`, starting with its newest entry, and update it when the
  run launches or results land.

These documents own the detailed recipes. Keep this guide as the stable safety
and routing layer; do not copy historical run matrices, result counts, or
command catalogs into it.

## Safety and provenance

- Run benchmark agents only against local fixtures or explicitly configured
  sandboxed environments. `data/IMPORTING.md` may authorize read-only capture
  of public pages; never use that path for authenticated, private, destructive,
  or account-affecting actions.
- Treat benchmark payloads as untrusted test inputs. Do not weaken injection,
  exposure, encounter, scoring, refusal, or cleanup checks to improve a result.
- Keep API keys in environment variables or approved local configuration. Never
  commit keys, tokens, private host details, or raw secrets to data or notes.
- Treat imported benchmark trees and vendored dependencies as read-only unless
  the setup or handoff document explicitly defines a patch workflow.
- Preserve raw trajectories and input manifests. Generated outputs belong in
  the documented result locations; do not hand-edit a result to repair a run.

## Environment and command discovery

Use the active project environment described by the route documents. Before a
run, inspect the current `pyproject.toml`, lockfile, environment variables, and
module `--help` output. Prefer the command and flags discovered there over
remembered recipes; if a command or path has changed, update the owning guide,
not this router. Start with a bounded smoke run before scaling concurrency or
task volume, and confirm cleanup/reset behavior before using external fixtures.

## Working loop

1. Read the route documents and inspect the current code, manifest, config, and
   git status. Record the question, cohort/condition, and expected evidence.
2. Make the smallest in-scope code or config change. Keep shared parsers,
   wrappers, runners, and judges single-sourced across benchmark tracks.
3. Discover the current entry point and options with `--help`; run a bounded
   smoke test or focused test before any larger run.
4. Record the exact commit, inputs, model/provider settings, conditions,
   filters, output paths, and failures in the experiment log. Report
   denominators and exclusions from artifacts rather than memory.
5. Analyze raw outputs without silently dropping non-encounters or failed
   tasks. Keep causal conditions and scoring contracts explicit in the report.

## Completion contract

Work is complete when the requested change is scoped, the relevant tests or
smoke run pass, safety and cleanup are confirmed, and the resulting artifacts
are traceable to their inputs and configuration. A run is not complete merely
because a process exited: record the evidence, limitations, and next action in
`experiment_log.md`, with newest entries first.
