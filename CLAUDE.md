# CLAUDE.md

## What This Is

Monorepo for evaluating AI agent safety in controlled web environments. Generates realistic web pages with embedded adversarial content (prompt injections), runs browser-use agents against both clean and adversarial variants, and measures whether agents resist or comply. The core insight: capability difficulty is orthogonal to adversarial propensity — a task can be easy to complete but hard to resist manipulation on.

Components:
- **AgentLab** (`AgentLab/`) — experiment orchestration, agent implementations, benchmark interfaces (BrowserGym-based)
- **Redteam Benchmark** (`AgentLab/src/agentlab/benchmarks/redteam/`) — environment generation, Flask server, Playwright interception, safety analysis
- **WebArena Infinity** (`vendors/webarena-infinity/`) — extended WebArena benchmark (reference)

## How to Work Here

```bash
cd AgentLab && uv sync                           # First-time setup (installs all deps)
uv run python main.py                            # Run experiments (modify to configure)
make test                                        # Run full test suite
uv run agentlab-xray                             # Analyze results in Gradio UI
```

Environment variables: see `.env.example` at repo root. `OPENROUTER_API_KEY` and `ANTHROPIC_API_KEY` are required.

Browsers run headless by default. Set `headless=False` in `RedteamEnvArgs` for debugging.

## Where to Find Details

Read these when working on the relevant component — not all at once:

| File | Covers |
|------|--------|
| `README.md` | Full architecture, quick start, behavior spec, output structure, adversarial scoring |
| `AgentLab/README.md` | Framework setup, 11 supported benchmarks, results analysis, env vars |
| `AgentLab/src/agentlab/benchmarks/redteam/README.md` | Redteam package API, module descriptions, mock endpoints, state protocol |
| `AgentLab/Makefile` | All available make targets (setup, test, miniwob, lint, osworld) |
| `docs/architecture.md` | Monorepo component map, agent internals, judge system |
| `.env.example` | Required and optional environment variables |

## Key Patterns

- `main.py` is a "lazy CLI" — modify and run rather than complex CLI args
- Results analyzed via `agentlab-xray` Gradio UI
- Set `n_jobs=1` for debugging (enables VSCode breakpoints)
- Ray backend for parallel experiments with task dependency awareness
- Pre-generated environments are the default; generation happens offline via `redteam_pregenerate`

<important if="you are writing or modifying tests">
- Run tests from `AgentLab/`: `MINIWOB_URL="http://localhost:8080/miniwob/" uv run pytest -n 5 -m 'not pricy' tests/`
- MiniWob server must be running first: `make miniwob` (stop with `make stop-miniwob`)
- Full suite shortcut: `make test` (handles setup + miniwob + tests)
</important>

<important if="you are working on the redteam benchmark or environment generation">
- Read `AgentLab/src/agentlab/benchmarks/redteam/README.md` first — it documents every module, mock endpoints, and the state protocol
- Variant generation writes benign + adversarial HTML from the same base with placeholder injection — never modify base HTML directly
- Mock endpoints (`/mock/file/password.txt`, `/mock/mail/submit`) are how judges detect unsafe agent actions
- All agent actions are logged to `server_events.log` in the experiment directory
</important>

<important if="you are running or configuring experiments">
- Edit `main.py` directly to configure experiments — it's the intended workflow, not a hack
- Run with `uv run python main.py` — uv manages the virtualenv and dependencies
- `OPENROUTER_API_KEY` and `ANTHROPIC_API_KEY` must be set (see `.env.example`)
- Set `AGENTLAB_EXP_ROOT` to control where results are written (default: `~/agentlab_results`)
- Use `pregenerated_env_dir` in `RedteamBenchmark` to skip LLM generation during experiment runs
</important>
