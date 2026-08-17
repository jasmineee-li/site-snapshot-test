# Evaluation-awareness setup

Use this guide for the shared Python environment and setup routing. It does not
own benchmark-specific service recipes or data-import procedures.

## Canonical environment

The repository root is one `uv` project. Do not create a parallel Conda
environment or install editable benchmark packages with an unscoped `pip`.

From the repository root:

```bash
uv sync --locked --group dev
uv run playwright install chromium
uv run python -m eval_awareness_experiments.run --help
```

Developer tooling is the PEP 735 `[dependency-groups]` `dev` group in the root
`pyproject.toml`; there is no `dev` extra, and `uv sync --extra dev` now fails.
`dev` is one of uv's default groups, so a bare `uv sync` installs it too, but
name `--group dev` explicitly: it survives a later `[tool.uv] default-groups`
entry that narrows the defaults. Add new developer tooling to that one group.
`[project.optional-dependencies]` is reserved for optional *runtime* extras —
today only `cua`, installed separately on model-serving hosts.

Use `uv run` for Python entrypoints so commands resolve against the locked root
environment. Inspect `pyproject.toml`, `uv.lock`, and the selected command's
`--help` before adding dependencies or copying a historical invocation.

`packages/warp-taskgen` is a separate uv project with its own `pyproject.toml`,
its own lockfile, and its own `dev` **extra**. Its documented
`uv sync --extra dev` is unaffected by anything above; run it from that
directory, not from the repository root.

Keep provider keys in the root `.env` or the approved process environment.
Never paste a key, token, authenticated browser state, or private host value
into a command, config committed to git, result, or handoff.

## Lint, format, and types

The root `[tool.ruff]` config lints this tree — `eval_awareness/`,
`eval_awareness_experiments/`, `models/`, `probes/`, and the root-level runner
scripts. Run it from the repository root before handing off a change:

```bash
uv run ruff check <changed-python-files>
uv run ruff check .
```

Ruff is lint-only here; `[tool.black]` still owns formatting. `packages/` is
excluded because `warp-taskgen` carries its own `[tool.ruff]` config and its
own acceptance gate. Do not silence a finding with a blanket `ignore` when a
targeted `per-file-ignores` entry or a `# noqa` with a reason will do.

`[tool.mypy]` type-checks the same tree plus `scripts/` and the two root runner
modules; `[tool.mypy] files` is the authoritative list. Run the remaining two
gates from the repository root as well:

```bash
uv run black --check .
uv run mypy
```

mypy takes no path argument here — it reads its own roots, so a bare
`uv run mypy` is the whole gate. Write a suppression as `# type: ignore[<code>]`
with a reason rather than a bare `# type: ignore`, so the gate reports it once
it stops being needed.

## Route by setup type

- Website extraction, manifests, or trajectory import: read
  `data/IMPORTING.md`. Its source-specific commands and platform requirements
  are authoritative for imports.
- DoomArena, WASP, AgentDojo, InjecAgent, EIA, or SafeArena setup: read
  `SAFETY_BENCHMARKS_HANDOFF.md` before installing or starting services. Use
  `setup_benchmarks.sh` only for the targets named by that handoff.
- Experiment selection, prompts, judges, and configs: read `EXPERIMENTS.md`.
- A new run or result: create the entry in `experiment_log.md` before scaling
  beyond a bounded smoke test.

Vendored and external benchmark trees are read-only unless their owning
handoff explicitly records a local patch workflow. Do not reinstall one
AgentLab checkout over another or mix packages from a historical environment
into the root `.venv`.

## Service and platform boundaries

Start only the services required by the selected source or benchmark and use
its documented health check. Some upstream Docker images require Linux amd64;
do not treat an emulation failure on macOS arm64 as an experiment result.

Benchmark agents run only against configured sandbox services. The documented
real-site importer may capture public pages read-only; it must not authenticate,
mutate state, or access private data. Stop local containers after capture or a
smoke run unless the active handoff explicitly assigns an operator to them.

## Completion

Setup is complete when the locked environment resolves, the selected entrypoint
prints help, required services pass their documented health checks, a bounded
smoke test writes the expected output, and cleanup succeeds. Record environment,
benchmark revisions, inputs, command, output path, and failures in
`experiment_log.md`; do not record cached sample counts or conclusions here.
