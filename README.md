# WorldSim v5

Adversarial evaluation pipeline for browser agents. Generates indirect prompt injection tasks against pre-running benchmark environments (WebArena), runs [Browser Use](https://github.com/browser-use/browser-use) agents, and scores them on two sequential gates: **ecological validity** and **attack effectiveness**.

## Install

```bash
uv sync
# or, if you prefer pip:
pip install -e .
```

PostgreSQL support is included in the default install.

### WebArena Verified evaluator

Canonical WebArena Verified scoring is isolated behind a separate installable,
similar to benchmark-specific packages in BrowserGym.

Install the adapter in its own environment:

```bash
uv sync --directory packages/worldsim-webarena-verified
export WORLDSIM_WEBARENA_EVAL_PYTHON="$PWD/packages/worldsim-webarena-verified/.venv/bin/python"
```

That keeps `worldsim`'s core environment compatible with `browser-use` while
still allowing canonical `webarena_verified` evaluation. If
`WORLDSIM_WEBARENA_EVAL_PYTHON` is unset, WorldSim will try an in-process
`webarena-verified` install and otherwise fail closed.

## Prerequisites

1. **Modal account.** Sign up at <https://modal.com>, then run `modal token new` to write `~/.modal.toml`, or set `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` in your environment.
2. **Claude Code authentication.** Claude Code runs *inside* the Modal sandbox and needs credentials injected via `modal.Secret.from_dict`. Three auth methods are supported — pick whichever you have:

   - **`CLAUDE_CODE_OAUTH_TOKEN`** (preferred if you have a Claude Pro / Claude Max subscription):

     ```bash
     export CLAUDE_CODE_OAUTH_TOKEN=your-oauth-token
     ```

   - **`ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_BASE_URL`** (OpenRouter or any proxy):

     ```bash
     export ANTHROPIC_AUTH_TOKEN=sk-or-v1-...
     export ANTHROPIC_BASE_URL=https://openrouter.ai/api
     ```

   - **`ANTHROPIC_API_KEY`** (traditional API-credit billing):

     ```bash
     export ANTHROPIC_API_KEY=sk-ant-...
     ```

   **If both are set, OAuth wins** — `worldsim/modal_sandbox.py:_build_claude_secrets` drops `ANTHROPIC_API_KEY` from the sandbox env because Claude Code's internal auth precedence would otherwise silently bill against API credits instead of your subscription.

   For CI / shared workspaces, you can opt into a named Modal secret instead:

   ```bash
   modal secret create my-claude-secret CLAUDE_CODE_OAUTH_TOKEN=claude-...
   export WORLDSIM_CLAUDE_MODAL_SECRET=my-claude-secret
   ```

   In that mode the priority fixup does not apply — manage the secret's contents yourself.

3. **Benchmark codebase on disk** — clone WebArena Verified into `vendors/webarena-verified/`:

   ```bash
   mkdir -p vendors
   git clone https://github.com/ServiceNow/webarena-verified vendors/webarena-verified
   ```

   `vendors/` is in `.gitignore` — you clone manually, not via submodules.
4. **Benchmark instances running** — **required for Phases 3 and 4 only**. You stand up WebArena sites per the benchmark's own documentation and register them with the orchestrator via CLI flags (see Run below).

Phases 0, 1, and 2 only need the benchmark **codebase** on disk, not running instances.

## Run

```bash
# Phase 0 against WebArena Verified (reads the codebase, no running services needed)
uv run python -m worldsim.main phase 0 --benchmark vendors/webarena-verified

# Phase 2 runs two internal stages sequentially:
# 2a plan generation in Modal sandboxes, then 2b host-side text fill.
uv run python -m worldsim.main phase 2 --benchmark vendors/webarena-verified

# Resume from the last checkpoint after a crash
uv run python -m worldsim.main resume
```

Pipeline state is written to `logs/pipeline_state.json` before each major operation. If you use a custom `WORLDSIM_STATE_DIR`, WorldSim also writes a pointer under `logs/` so `uv run python -m worldsim.main resume` can find the active run later without re-exporting the environment variable.

## Architecture

A local Python orchestrator coordinates three things:

1. **Modal Sandboxes running Claude Code** — all code exploration, generation, and diagnosis steps. Each sandbox is scoped by *inclusion*: small inputs are staged with `add_local_file` / `add_local_dir`, while large stable benchmark trees can be mounted from Modal volumes.
2. **Browser Use** — async Python library for running browser agents against benchmark instances. Each evaluation worker gets its own browser session and a dedicated pre-running benchmark instance.
3. **Local orchestrator logic** — state management, validation, file routing between phases, and the iteration loops that connect everything.

Phase 0 always needs `--benchmark`. Phase 1 reads `BENCHMARK_MANIFEST.json` from `logs/phase_0a/` by default, and `--config` only overrides that manifest path.

Five phases:

| # | Phase | What it does |
|---|-------|--------------|
| 0 | Reconnaissance | 0a discovers benchmark structure, 0b computes per-site sandbox file maps, 0c profiles each site in parallel |
| 1 | Task Generation | Mode A wraps existing benchmark tasks; Mode B generates new tasks (stretch goal) |
| 2 | Injection Generation | Runs 2a plan generation, then 2b text fill sequentially; emits final adversarial tasks with materialized data seeds |
| 3 | Benign Validation | Runs the agent against benign seeds; diagnoses failures |
| 4 | Adversarial Evaluation | Runs the agent against injected seeds; applies ecological-validity gate and attack-effectiveness gate; adaptively varies strategy when attacks are refused |

The **authoritative technical spec** lives at [`docs/worldsim-v5-technical-specifcation.md`](docs/worldsim-v5-technical-specifcation.md). Every module in `worldsim/` implements a section of that spec.

## Repository layout

```
.
├── worldsim/                     # the package
│   ├── main.py                   # CLI entrypoint
│   ├── config.py                 # benchmark instance schema
│   ├── modal_sandbox.py          # run_claude_in_sandbox + base image
│   ├── state.py                  # save_state / load_state
│   ├── prompt_loading.py         # tiny prompt file loader
│   ├── trajectory.py             # load_trajectory_into_sandbox
│   ├── browser_use_agent.py      # AgentRunner + BrowserUseAgent
│   ├── eval_worker_pool.py       # run_eval + staggered_worker
│   ├── seeding.py                # apply_data_seed (sql/api/state_push)
│   ├── rewards.py                # run_reward_function dispatcher
│   ├── prompts/                  # full prompts verbatim from the v5 spec
│   └── phases/                   # phase_0_recon, phase_1_tasks, ... phase_4_adversarial
├── docs/
│   ├── worldsim-v5-technical-specifcation.md  # canonical technical spec — source of truth
│   └── webarena-infinity-paper.md             # predecessor research reference
├── AgentLab/src/agentlab/benchmarks/redteam/
│   ├── execution.py              # Modal reference (light-reference only, not imported)
│   └── claude_code.py            # Claude Code auth reference (light-reference only)
├── vendors/                      # gitignored — clone benchmarks here manually
├── pyproject.toml
├── CLAUDE.md                     # guidance for Claude Code sessions working in this repo
└── README.md
```

## License

TBD.
