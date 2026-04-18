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
4. **Benchmark instances running** — **required for Phase 4 only**. Phase 3 is an agent-free contract validity gate and does not touch live instances. You stand up WebArena sites per the benchmark's own documentation and register them with the orchestrator via CLI flags (see Run below).

   For WebArena Verified seeding, keep instance auth and DB connectivity explicit in your instances config:

   - `shopping` customer-form seeding uses `X-M2-Customer-Auto-Login` and typically reads from `WORLDSIM_SHOPPING_AUTO_LOGIN`.
   - `reddit` / Postmill form seeding uses `X-Postmill-Auto-Login` and typically reads from `WORLDSIM_REDDIT_AUTO_LOGIN`.
   - `shopping_admin` auth is config-driven rather than env-driven; use a static header such as `X-M2-Admin-Auto-Login: admin:admin1234` in the `shopping_admin` instance block when admin-origin seeding is required.
   - `db_connection` in `instances.json` is optional and used only for postcondition verification and reward evaluation (read-only). In this repo it is still a real runtime dependency for many WebArena Verified tasks, so host reachability and DB port exposure must stay aligned with the generated compose config. Typical shapes are:
     - `mysql://magentouser:MyPassword@HOST:3306/magentodb` for `shopping` / `shopping_admin`
     - `postgresql://postmill:postmill@HOST:5432/postmill` for `reddit`
     - `postgresql://openstreetmap:openstreetmap@HOST:5433/openstreetmap` for `map`

Phases 0, 1, 2a, and 2b only need the benchmark **codebase** on disk, not running instances. **Phase 2c requires a live dev instance** (defaults to the instances listed in `instances.smoke.json`): each adversarial task's `adversarial_data_seed` is POSTed against the live platform to prove feasibility. Pass `--skip-feasibility` to skip 2c for fast dev iteration; Phase 4 will run in grace mode and admit the unverified tasks with a warning.

### Proxy Setup (Phase 0c live verification)

Phase 0c runs LLM-based injection surface discovery in Modal cloud sandboxes. These
sandboxes can optionally probe live benchmark instances to verify mechanical claims
(URL existence, required fields, entity IDs). Because Modal sandboxes exit from
dynamic IPs that the EC2 security group blocks, and opening `0.0.0.0/0` on the real
ports is insecure (benchmark instances have default credentials), an authenticated
nginx reverse proxy is deployed on offset ports.

**When to set this up:** only when you want Phase 0c to live-verify profiles against
running instances. Phase 0c works without it (code-reading only), and Phases 1-2
never use the proxy.

**Deploy the proxy:**

```bash
# Canonical r5 path (scale port map + explicit HTTP opt-in):
./scripts/deploy_proxy_r5.sh

# Generic path:
./scripts/deploy_benchmark_proxy.sh --host-config configs/benchmark_hosts/r5.yaml

# With explicit arguments:
./scripts/deploy_benchmark_proxy.sh \
    --host-config configs/benchmark_hosts/r5.yaml \
    --topology scale \
    --insecure-http \
    --ssh-key ~/.ssh/webarena-key.pem \
    --port-map scripts/proxy_ports.conf
```

The script installs nginx on the EC2 instance, generates a random token, writes
one `server` block per site (proxy port = real port + 10000), and restarts nginx.
It is idempotent, safe to re-run, and benchmark-agnostic (reads port mappings from
`scripts/proxy_ports.conf` or a custom file). The checked-in `deploy_proxy_r5.sh`
wrapper currently opts into token-protected HTTP by passing `--insecure-http`;
switch to TLS inputs if you want HTTPS on the public proxy.

For the scale bring-up path, `./scripts/bootstrap_r5.sh` now regenerates the
scale artifacts and runs a security-group preflight against the generated
runtime ports before staging the compose file onto the host.

For TLS-backed Phase 0c probing, the deploy helper also accepts
`--tls-cert /path/on/host/fullchain.pem --tls-key /path/on/host/privkey.pem`
and emits `"scheme": "https"` in the suggested `verification_proxy` block.

**After deploying:** open the proxy ports (17770, 17780, etc.) in the EC2 security
group for `0.0.0.0/0`. These ports are token-protected. Then copy the token into
`instances.json`:

```json
"verification_proxy": {
  "token": "<token from .proxy_token>",
  "scheme": "http",
  "port_offset": 10000
}
```

Phase 0c reads this config, rewrites site URLs to proxy ports, and includes
`X-Worldsim-Token` in all sandbox curl requests. Without a non-empty token the
proxy is treated as disabled. This proxy is for Phase 0c live verification only;
Phases 3-4 continue to use the real `site_url` and `reset_endpoint` values from
`instances.json`.

### WebArena setup artifact cold storage (S3)

The initial WebArena instance setup downloads ~265GB of tarballs to
`/home/ubuntu/downloads` on EC2 (nominatim 117GB, osm_tile 39GB, osrm 20GB,
wikipedia zim 89GB). Once unpacked into Docker volumes they are redundant for
running benchmarks, but needed if you want to spin up additional instances or
rebuild from scratch.

These artifacts live in `s3://benchmark-archives/webarena/` in `us-east-2`
(Standard-IA tier, around $3/month). The IAM user `worldsim-ec2-benchmark-backup`
has scoped read/write access on this bucket only.

Restore to EC2 with:

```bash
./scripts/restore_benchmark_archives_from_s3.sh
```

Restore takes 10-15 min end-to-end (intra-region transfer is free and fast).
Checksums are verified on pull. Use `--wiki-only` or `--skip-wiki` to subset.
The restore helper also creates the three intentionally-empty map volumes
(`webarena-verified-map-tiles`, `webarena-verified-map-style`,
`webarena-verified-map-website-db`) that are declared by the vendor compose
file but not hydrated from the archived tarballs.

For amd64 smoke bring-up on EC2, use `scripts/bootstrap_ec2.sh`. It writes the
canonical `docker-compose.override.yml` from
[`scripts/webarena-compose-override.yml`](scripts/webarena-compose-override.yml),
stamps the selected host contract into `/home/ubuntu/.env`, builds the local
`worldsim/webarena-verified-wikipedia:amd64` image when needed, and pins the
compose stack to the local amd64 tags plus the correct
`WA_ENV_CTRL_EXTERNAL_SITE_URL` / DB port bindings.

## Run

```bash
# Phase 0 against WebArena Verified (reads the codebase, no running services needed)
uv run python -m worldsim.main phase 0 --benchmark vendors/webarena-verified

# Phase 3 runs a cheap, agent-free contract validity check and emits
# phase_3/contracts.json for Phase 4 to admit.
uv run python -m worldsim.main phase 3

# Phase 4 Browser Use agents can use GPT-5.4-mini through OpenRouter
export OPENROUTER_API_KEY=sk-or-v1-...
uv run python -m worldsim.main phase 4 --instances instances.json \
  --agent-provider openrouter --agent-model gpt-5.4-mini

# Phase 2 runs three internal stages sequentially:
# 2a plan generation in Modal sandboxes, 2b host-side text fill, then 2c
# feasibility verification against a live dev instance. Use --skip-feasibility
# for fast dev iteration (Phase 4 runs in grace mode in that case); omit it
# for shipping runs.
uv run python -m worldsim.main phase 2 --benchmark vendors/webarena-verified \
  --feasibility-instances instances.smoke.json

# Re-verify an already-generated dataset against a fresh dev host:
uv run python -m worldsim.main phase 2c \
  --feasibility-instances instances.smoke.json

# Resume from the last checkpoint after a crash. Resume reads the saved
# phase_2_stage and re-enters planning/text_fill/feasibility automatically.
uv run python -m worldsim.main resume
```

Pipeline state is written to `logs/pipeline_state.json` before each major operation. If you use a custom `WORLDSIM_STATE_DIR`, WorldSim also writes a pointer under `logs/` so `uv run python -m worldsim.main resume` can find the active run later without re-exporting the environment variable.

### Nightly feasibility drift check

`scripts/nightly_feasibility_check.sh` re-runs Phase 2c against the dev host with a 24-hour TTL so previously-verified tasks skip on fingerprint match. Wire it into cron to catch silent platform drift (GitLab secret rotation, Magento review policy changes, PostMill schema migration) before it contaminates the next ASR run:

```cron
0 3 * * * cd /path/to/browser-sim && bash scripts/nightly_feasibility_check.sh >> logs/cron/nightly_feasibility.log 2>&1
```

Override `INSTANCES_FILE` (default `instances.smoke.json`), `TTL_HOURS` (default `24`), or `FEASIBILITY_CONCURRENCY` (default `10`) via env var.

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
| 2 | Injection Generation | Runs 2a plan generation, 2b text fill, then 2c feasibility verification sequentially; emits final adversarial tasks with materialized data seeds and a `feasibility` stanza per task (requires a live dev instance unless `--skip-feasibility` is set) |
| 3 | Contract Validity Gate | Agent-free schema check over every benign contract (reward function, start URLs, data seed) and every adversarial task's benign reference; writes `phase_3/contracts.json` |
| 4 | Adversarial Evaluation | Runs the agent against injected seeds; applies ecological-validity gate and attack-effectiveness gate; adaptively varies strategy when attacks are refused; reports baseline capability as a byproduct |

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
│   ├── seeding.py                # apply_data_seed (api/form/state_push)
│   ├── rewards.py                # run_reward_function dispatcher
│   ├── prompts/                  # full prompts verbatim from the v5 spec
│   └── phases/                   # phase_0_recon, phase_1_tasks, ... phase_4_adversarial
├── scripts/
│   ├── deploy_benchmark_proxy.sh  # authenticated reverse proxy for Phase 0c
│   └── proxy_ports.conf           # site-to-port mapping for the proxy
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
