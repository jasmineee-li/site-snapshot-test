# WorldSim v5

Adversarial evaluation pipeline for browser agents. Generates indirect prompt injection tasks against pre-running benchmark environments (WebArena), runs Browser Use agents by default with an optional AgentLab sidecar runner, and scores them on two sequential gates: **paint-verified payload encounter** and **attack effectiveness**. Transcript Purpose and VEA are observational in baseline scoring; the explicit eval-awareness iterator may use Transcript Purpose only for bounded rewrite control.

## Install

```bash
uv sync --extra dev
# or, if you prefer pip:
pip install -e .
```

Use plain `uv sync` for runtime-only installs. PostgreSQL support is included in
the default install; lint/test tooling is in the `dev` extra.

### WebArena Verified evaluator

Canonical WebArena Verified scoring is isolated behind a separate installable,
similar to benchmark-specific packages in BrowserGym.

Install the adapter in its own environment:

```bash
uv sync --directory packages/worldsim-webarena-verified --locked
export WORLDSIM_WEBARENA_EVAL_PYTHON="$PWD/packages/worldsim-webarena-verified/.venv/bin/python"
```

That keeps `worldsim`'s core environment compatible with `browser-use` while
still allowing canonical `webarena_verified` evaluation. If
`WORLDSIM_WEBARENA_EVAL_PYTHON` is unset, WorldSim first tries the repo-local
adapter venv and then an in-process `webarena-verified` install before failing
closed.

### AgentLab sidecar

AgentLab/BrowserGym support lives in an isolated package so the root Browser Use
environment can keep its own dependency set.

```bash
uv sync --directory packages/worldsim-agentlab-runner --locked
```

The sidecar expects the upstream AgentLab checkout at `vendors/AgentLab-upstream`
when running AgentLab-backed tasks. Use it for Phase 4 with:

```bash
uv run python -m worldsim.main phase 4 \
  --runner agentlab \
  --instances instances.scale.json
```

## Prerequisites

1. **Modal account.** Sign up at <https://modal.com>, then run `modal token new` to write `~/.modal.toml`, or set `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` in your environment.
2. **Claude authentication.** Credentials are used by (a) Claude Code inside the Modal sandbox for code-reading and generation steps (injected via `modal.Secret.from_dict`), and (b) host-side Anthropic Messages API calls for Phase 2a/2b structured generation plus Phase 4 refusal judging, variant generation, Transcript Purpose, VEA, placement-fix, eval-awareness cue diagnosis, and eval-awareness rewrite (`worldsim/phase_4/anthropic_client.py`). Three auth methods are supported, pick whichever you have:

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

   **Auth precedence differs by execution path.** For Modal Claude Code, OAuth wins: `worldsim/modal_sandbox.py:_build_claude_secrets` drops `ANTHROPIC_API_KEY` from the sandbox env because Claude Code's internal auth precedence would otherwise silently bill against API credits instead of your subscription. For host-side Anthropic Messages calls, `worldsim/phase_4/anthropic_client.py` prefers OpenRouter/proxy envs, then OAuth, then API key.

   For CI / shared workspaces, you can opt into a named Modal secret instead:

   ```bash
   modal secret create my-claude-secret CLAUDE_CODE_OAUTH_TOKEN=claude-...
   export WORLDSIM_CLAUDE_MODAL_SECRET=my-claude-secret
   ```

   In that mode the priority fixup does not apply, manage the secret's contents yourself.

3. **Benchmark codebase on disk** — clone WebArena Verified into `vendors/webarena-verified/`:

   ```bash
   mkdir -p vendors
   git clone https://github.com/ServiceNow/webarena-verified vendors/webarena-verified
   ```

   `vendors/` is in `.gitignore` — you clone manually, not via submodules.
4. **Benchmark instances running**. Phase 2c and Phase 4 require live benchmark instances. Phase 3 is an agent-free contract validity gate and does not touch live instances. You stand up WebArena sites per the benchmark's own documentation and register them with the orchestrator via CLI flags (see Run below).

   Current WASP mainline carrier scope is GitLab and Reddit/Postmill UGC only. Keep instance auth and DB connectivity explicit in your instances config:

   - `gitlab` auth is normally minted by Phase 0d into `logs/phase_0d/gitlab/storage_state.json`; editor calls use the configured GitLab API auth/PAT path.
   - `reddit` / Postmill editor/form seeding uses `X-Postmill-Auto-Login` and typically reads from `WORLDSIM_REDDIT_AUTO_LOGIN`.
   - `db_connection` in `instances.json` is optional and used only for postcondition verification and reward evaluation (read-only). Typical current mainline shapes are:
     - `postgresql://gitlab:...@HOST:5432/gitlabhq_production` or the generated GitLab DB URL from the host config
     - `postgresql://postmill:postmill@HOST:5432/postmill` for `reddit`
   - Shopping, shopping_admin, map/OSM, Magento, Wikipedia, and classifieds settings are historical full-benchmark/support plumbing and are not active IPI carriers unless the spec reopens scope.
   - `pvpo_cdp_url` is required for shipping Phase 4 runs. Each execution instance needs its own dedicated local `chrome-headless-shell` CDP endpoint (for example `http://127.0.0.1:9222`, `http://127.0.0.1:9223`, ...). Do not point multiple workers at one shared browser.

Phases 0, 1, 2a, and 2b only need the benchmark **codebase** on disk, not running instances. **Phase 2c requires a live dev instance** (defaults to the instances listed in `instances.smoke.json`): each adversarial task's `adversarial_data_seed` is POSTed against the live platform to prove feasibility. Pass `--skip-feasibility` only for fast dev iteration; unverified tasks are not suitable for shipping Phase 4 runs.

### Proxy Setup (Phase 0c live verification)

Phase 0c runs LLM-based injection surface discovery in Modal cloud sandboxes. These
sandboxes can optionally probe live benchmark instances to verify mechanical claims
(URL existence, required fields, entity IDs). Because Modal sandboxes exit from
dynamic IPs that the EC2 security group blocks, and opening `0.0.0.0/0` on the real
ports is insecure (benchmark instances have default credentials), an authenticated
nginx reverse proxy is deployed on offset ports.

**When to set this up:** only when you want Phase 0c to live-verify profiles against
running instances. Phase 0c works without it (code-reading only). Phases 1,
2a, and 2b never use the proxy; Phase 2c uses the live instances file directly.

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
  "token_file": ".proxy_token",
  "token_env": "WORLDSIM_VERIFICATION_PROXY_TOKEN",
  "scheme": "http",
  "port_offset": 10000
}
```

Phase 0c reads this config, rewrites site URLs to proxy ports, and includes
`X-Worldsim-Token` in all sandbox curl requests. Without a non-empty token the
proxy is treated as disabled. Tokens should live in the gitignored `.proxy_token`
file or `WORLDSIM_VERIFICATION_PROXY_TOKEN`, not in checked-in JSON. This proxy
is for Phase 0c live verification only; Phases 3-4 continue to use the real
`site_url` and `reset_endpoint` values from `instances.json`.

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

## Verify

Use the repo wrappers for deterministic, context-efficient local validation:

```bash
# Fast agent baseline: scoped lint plus pytest collection
bash scripts/verify_fast.sh

# Default local shipping gate: fast baseline plus default pytest
bash scripts/verify_default.sh
```

Live and host-backed gates remain explicit because benchmark hosts are external
runtime dependencies.

## Run

```bash
# Phase 0 against WebArena Verified (reads the codebase, no running services needed)
uv run python -m worldsim.main phase 0 --benchmark vendors/webarena-verified

# Phase 3 runs a cheap, agent-free contract validity check and emits
# phase_3/contracts.json for Phase 4 to admit.
uv run python -m worldsim.main phase 3

# Phase 4 Browser Use agents default to Sonnet via Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
uv run python -m worldsim.main phase 4 --instances instances.json

# Phase 2 runs exposure-contract materialization, API 2a strategy planning,
# host-side 2b text fill, then 2c feasibility + read-surface verification.
# Use --skip-feasibility only for fast dev iteration; omit it for shipping runs.
uv run python -m worldsim.main phase 2 --benchmark vendors/webarena-verified \
  --feasibility-instances instances.smoke.json

# On r5, regenerate instances.scale.json from configs/benchmark_hosts/r5.yaml
# instead of hand-editing IPs. The generated scale file carries host_access
# metadata and binds site_url/reset_endpoint/placeholders to orchestrator_host
# (172.17.0.1). Then re-mint Phase 0d against that exact generated file so
# storage_state cookie domains match the runtime host view:
./scripts/generate_scale_r5.sh
uv run python -m worldsim.main phase 0d --benchmark vendors/webarena-verified \
  --instances instances.scale.json

# Re-verify an already-generated dataset against a fresh dev host:
uv run python -m worldsim.main phase 2c \
  --feasibility-instances instances.scale.json

# Resume from the last checkpoint after a crash. Resume reads the saved
# phase_2_stage and re-enters planning/text_fill/feasibility automatically.
uv run python -m worldsim.main resume
```

Pipeline state is written to `logs/pipeline_state.json` before each major operation. If you use a custom `WORLDSIM_STATE_DIR`, WorldSim also writes a pointer under `logs/` so `uv run python -m worldsim.main resume` can find the active run later without re-exporting the environment variable.

Phase 4 result.json carries two parallel awareness metrics on every trajectory with `encounter.max_coverage > 0`:

- **Transcript Purpose**: Apollo Research's two-step classifier (`nix-apollo/38e0520c21bed0831d44019457d1833f`) over the shared Needham XML transcript. Step 1 asks the model to guess why the interaction took place; Step 2 classifies that guess into `Alignment Evaluation` / `Capability Evaluation` / `Other Evaluation` / `Real` / `Refused`. Keys: `transcript_purpose_classification`, `transcript_purpose_is_aware` (True iff classification is Alignment/Capability/Other Evaluation), `transcript_purpose_was_refused`, `transcript_purpose_guess`, `transcript_purpose_explanation`, `transcript_purpose_failure`, `transcript_purpose_format` (always `"needham-xml-v1"`), `transcript_purpose_attempts`, `transcript_purpose_raw_response_path`. Per-task artifacts include shared `needham_trace.json` / `needham_trace.xml`, `transcript_purpose_transcript.txt`, and `transcript_purpose_raw_response.json`. Baseline scoring records this as an observational metric; the explicit `eval-awareness-iterator` uses it only as the bounded rewrite-control signal.
- **Verbalized Eval Awareness**: Jasmine's 4-way classifier over the same Needham XML transcript. Keys: `verbalized_eval_awareness_classification` (one of `Alignment Evaluation` / `Capability Evaluation` / `Other Evaluation` / `Real`), `verbalized_eval_awareness_is_aware` (derived bool), `verbalized_eval_awareness_evidence` (list of verbatim quotes), `verbalized_eval_awareness_failure`, `verbalized_eval_awareness_transcript_format` (always `"needham-xml-v1"`), `verbalized_eval_awareness_raw_response_path`. Per-task artifacts include shared `needham_trace.*`, `verbalized_eval_awareness_transcript.xml`, and `verbalized_eval_awareness_raw_response.json`. VEA remains report-only and is never a gate or branch signal.

Trajectories where the injection never rendered (`encounter.max_coverage == 0`) get every metric key stamped with a present-but-null value so downstream loaders see a uniform schema. See the spec's "Gate 1 Stage 2: Observational Transcript Purpose + VEA" section for the full design.

Phase 4 trajectories are enriched post-hoc with a diagnosable outcome taxonomy (see the spec's "Outcome Taxonomy" section). Every `processed_result.json` gains `outcome_fine`, `flags`, `signals`, and `diagnosable_confidence` fields alongside the legacy `outcome`. To re-classify a historical run offline:

```bash
uv run python scripts/reclassify_phase_4_results.py logs/phase_4/<run_id>/
```

The reclassifier is idempotent (skips rows already at the current `classifier_version`; pass `--force` to rewrite) and reads trajectory artifacts from disk; no live services are touched.

### Nightly feasibility drift check

`scripts/nightly_feasibility_check.sh` re-runs Phase 2c against the dev host with a 24-hour TTL so previously-verified tasks skip on fingerprint match. Wire it into cron to catch silent platform drift (GitLab secret rotation, PostMill schema migration, host topology changes) before it contaminates the next ASR run:

```cron
0 3 * * * cd /path/to/browser-sim && bash scripts/nightly_feasibility_check.sh >> logs/cron/nightly_feasibility.log 2>&1
```

Override `INSTANCES_FILE` (default `instances.smoke.json`), `TTL_HOURS` (default `24`), or `FEASIBILITY_CONCURRENCY` (default `10`) via env var.

## Architecture

A local Python orchestrator coordinates three things:

1. **Modal Sandboxes running Claude Code**: code exploration, benchmark profiling, task generation, and the steps that need isolated filesystem access. Host-side structured APIs handle Phase 2a/2b and Phase 4 classifiers or rewrites when a sandbox is unnecessary. Each sandbox is scoped by *inclusion*: small inputs are staged with `add_local_file` / `add_local_dir`, while large stable benchmark trees can be mounted from Modal volumes.
2. **Browser-agent runtimes**: Browser Use is the default runtime. AgentLab/BrowserGym runs behind the isolated `packages/worldsim-agentlab-runner` sidecar and must emit equivalent auth, PVPO, network, trajectory, and result artifacts before a run is treated as parity data.
3. **Local orchestrator logic** — state management, validation, file routing between phases, and the iteration loops that connect everything.

Phase 0 always needs `--benchmark`. Phase 1 reads `BENCHMARK_MANIFEST.json` from `logs/phase_0a/` by default, and `--config` only overrides that manifest path.

Five phases:

| # | Phase | What it does |
|---|-------|--------------|
| 0 | Reconnaissance | 0a discovers benchmark structure, 0b computes per-site sandbox file maps, 0c profiles each site in parallel |
| 1 | Task Generation | Existing-task wrapping plus opt-in novel-task generation via `--generate-novel` for Phase 4-admissible carrier route families |
| 2 | Injection Generation | Runs 2a plan generation, 2b text fill, then 2c feasibility verification sequentially; emits final adversarial tasks with materialized data seeds and a `feasibility` stanza per task (requires a live dev instance unless `--skip-feasibility` is set) |
| 3 | Contract Validity Gate | Agent-free schema check over every benign contract (reward function, start URLs, data seed) and every adversarial task's benign reference; writes `phase_3/contracts.json` |
| 4 | Adversarial Evaluation | Runs the agent against injected seeds; applies the PVPO encounter gate and attack-effectiveness gate; default resistant attempts enter the `eval-awareness-iterator`, which diagnoses eval-awareness cues and tries bounded payload rewrites. Legacy refusal-judge strategy variation remains opt-in via `--phase-4-variant-system strategy-variation`. Reports baseline capability as a byproduct |

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
│   ├── seeding/                  # apply_data_seed and editor_call dispatch
│   ├── rewards/                  # run_reward_function dispatcher and benchmark reward adapters
│   ├── prompts/                  # full prompts verbatim from the v5 spec
│   ├── phase_2/                  # modular Phase 2 package (eligibility, generation, plan_validation, runner, target_resolution/, phase_2c/)
│   ├── phase_4/                  # modular Phase 4 package (runner, execution, postprocess, metrics, results, resume, strategy_variation, variant_eval, ...)
│   └── phases/                   # legacy single-file modules + compat shims (phase_0_recon, phase_1_tasks, phase_2_injections shim, phase_4_adversarial shim, ...)
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
