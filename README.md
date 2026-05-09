# WARP Taskgen

WARP task generation and admission pipeline for browser-agent IPI benchmarks. Generates indirect prompt injection tasks against pre-running benchmark environments (WebArena), runs Browser Use agents by default with an optional AgentLab sidecar runner, and scores them on two sequential gates: **paint-verified payload encounter** and **attack effectiveness**. Transcript Purpose and VEA are observational in baseline scoring; the explicit eval-awareness iterator may use Transcript Purpose only for bounded rewrite control.

## Install

```bash
uv sync --extra dev
# or, if you prefer pip:
pip install -e .
```

Use plain `uv sync` for runtime-only installs. PostgreSQL support is included in
the default install; lint/test tooling is in the `dev` extra.

The distribution and primary console script are named `warp-taskgen`. The Python
package and compatibility CLI remain `worldsim`, and the v5 spec filenames keep
their legacy names so old runbooks and artifacts stay resolvable.

### WebArena Verified evaluator

Canonical WebArena Verified scoring is isolated behind a separate installable,
similar to benchmark-specific packages in BrowserGym.

Install the adapter in its own environment:

```bash
uv sync --directory packages/worldsim-webarena-verified --locked
export WARP_TASKGEN_WEBARENA_EVAL_PYTHON="$PWD/packages/worldsim-webarena-verified/.venv/bin/python"
```

That keeps WARP Taskgen's core environment compatible with `browser-use` while
still allowing canonical `webarena_verified` evaluation. If
`WARP_TASKGEN_WEBARENA_EVAL_PYTHON` is unset, WARP Taskgen first tries the
repo-local adapter venv and then an in-process `webarena-verified` install
before failing closed. The legacy `WORLDSIM_WEBARENA_EVAL_PYTHON` name remains
accepted as a compatibility alias.

### AgentLab sidecar

AgentLab/BrowserGym support lives in an isolated package so the root Browser Use
environment can keep its own dependency set.

```bash
uv sync --directory packages/worldsim-agentlab-runner --locked
```

The sidecar lockfile pins AgentLab to
`ServiceNow/AgentLab@cbc35a9bc0facaf731bc858c5825edbe757c719f`. If you change
the AgentLab revision, rerun the sidecar tests before treating new AgentLab
output as parity data. Use it for Phase 4 with:

```bash
uv run warp-taskgen phase 4 \
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
   export WARP_TASKGEN_CLAUDE_MODAL_SECRET=my-claude-secret
   ```

   In that mode the priority fixup does not apply, manage the secret's contents yourself. The legacy `WORLDSIM_CLAUDE_MODAL_SECRET` name remains accepted as a compatibility alias.

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
   - Phase 4 PVPO uses `page-surface-stable` capture on the runner-owned browser for both Browser Use and AgentLab. Do not configure dedicated PVPO browser endpoints for new shipping runs. Legacy instance files may still contain `pvpo_cdp_url`; canonical Phase 4 treats it as inert metadata.

Phases 0, 1, 2a, and 2b only need the benchmark **codebase** on disk, not running instances. **Phase 2c requires a live dev instance** (local defaults use `instances.smoke.json`): each adversarial task's `adversarial_data_seed` is POSTed against the live platform to prove feasibility. On remote benchmark hosts such as r5/r8a, use the host-local `instances.scale.json` for Phase 2c and Phase 4 so browser traffic uses the orchestrator host view. Pass `--skip-feasibility` only for fast dev iteration; unverified tasks are not suitable for shipping Phase 4 runs.

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

# Generic path with explicit topology and HTTP opt-in:
./scripts/deploy_benchmark_proxy.sh \
    --host-config configs/benchmark_hosts/r5.yaml \
    --topology scale \
    --insecure-http

# With explicit arguments:
./scripts/deploy_benchmark_proxy.sh \
    --host-config configs/benchmark_hosts/r5.yaml \
    --topology scale \
    --insecure-http \
    --ssh-key ~/.ssh/webarena-key.pem \
    --port-map scripts/proxy_ports.conf
```

The script installs nginx on the EC2 instance, generates a random token, writes
one listener row per proxy mapping, and restarts nginx. Current scale maps include
separate web and envctrl rows per replica rather than one generic proxy port per
site. The command is idempotent, safe to re-run, and benchmark-agnostic because it
reads port mappings from `scripts/proxy_ports.conf` or a custom file. The
checked-in `deploy_proxy_r5.sh` wrapper currently opts into token-protected HTTP
by passing `--insecure-http`; switch to TLS inputs if you want HTTPS on the
public proxy.

For the scale bring-up path, `./scripts/bootstrap_r5.sh` now regenerates the
scale artifacts and runs a security-group preflight against the generated
runtime ports before staging the compose file onto the host.

For TLS-backed Phase 0c probing, the deploy helper also accepts
`--tls-cert /path/on/host/fullchain.pem --tls-key /path/on/host/privkey.pem`
and emits `"scheme": "https"` in the suggested `verification_proxy` block.

**After deploying:** open only the generated proxy listener ports from
`scripts/proxy_ports.conf` in the EC2 security group for `0.0.0.0/0`. These ports
are token-protected. Then reference the token from `instances.json`:

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
proxy is treated as disabled. Phase 0c resolves proxy tokens from a literal
`token` only for generated or ephemeral configs, then from `token_env`, then from
`token_file`. Checked-in configs should not use literal `token`; prefer
`token_env` or a gitignored `token_file`. Relative `token_file` paths are
resolved relative to the instances config file. This proxy is for Phase 0c live
verification only; Phases 3-4 continue to use the real `site_url` and
`reset_endpoint` values from `instances.json`.

### Historical full-benchmark storage notes

Older setup docs mention S3 restore tooling and amd64 images for Wikipedia,
OSM/map, Magento, and other full WebArena sites. That path is
historical/support plumbing only. A Wikipedia support image may still appear in
old smoke or full-benchmark setup notes, but Wikipedia is not an active WARP
Taskgen IPI carrier unless the technical spec explicitly reopens scope. Current
WARP Taskgen mainline is GitLab and Reddit/Postmill only; use the host setup
scripts and benchmark-host configs for active WASP runs.

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
# Fast agent baseline: scoped lint, pytest collection, readiness audit
bash scripts/verify_fast.sh

# Default local shipping gate: fast checks without duplicate collection, plus parallel default pytest
bash scripts/verify_default.sh
```

Live and host-backed gates remain explicit because benchmark hosts are external
runtime dependencies.

## Run

```bash
# Phase 0 against WebArena Verified (reads the codebase, no running services needed)
uv run warp-taskgen phase 0 --benchmark vendors/webarena-verified

# Phase 3 runs a cheap, agent-free contract validity check and emits
# phase_3/contracts.json for Phase 4 to admit.
uv run warp-taskgen phase 3

# Phase 4 Browser Use agents default to Sonnet via Anthropic.
# On remote hosts, use the generated host-local scale topology.
export ANTHROPIC_API_KEY=sk-ant-...
uv run warp-taskgen phase 4 \
  --instances instances.scale.json \
  --phase-4-max-workers 48 \
  --phase-4-variant-system eval-awareness-iterator

# Phase 2 runs exposure-contract materialization, API 2a strategy planning,
# host-side 2b text fill, then 2c feasibility + read-surface verification.
# Use --skip-feasibility only for fast dev iteration; omit it for shipping runs.
uv run warp-taskgen phase 2 --benchmark vendors/webarena-verified \
  --feasibility-instances instances.smoke.json

# On r5/r8a, regenerate instances.scale.json from the selected host config
# instead of hand-editing IPs. The generated scale file carries host_access
# metadata and binds site_url/reset_endpoint/placeholders to orchestrator_host
# (172.17.0.1). Then re-mint Phase 0d against that exact generated file so
# storage_state cookie domains match the runtime host view:
./scripts/generate_scale_r5.sh
uv run warp-taskgen phase 0d --benchmark vendors/webarena-verified \
  --instances instances.scale.json

# Re-verify an already-generated dataset against a fresh dev host:
uv run warp-taskgen phase 2c \
  --feasibility-instances instances.scale.json

# Resume from the last checkpoint after a crash. Resume reads the saved
# phase_2_stage and re-enters planning/text_fill/feasibility automatically.
uv run warp-taskgen resume
```

Pipeline state is written to `logs/pipeline_state.json` before each major operation. If you use a custom `WARP_TASKGEN_STATE_DIR`, WARP Taskgen also writes a pointer under `logs/` so `uv run warp-taskgen resume` can find the active run later without re-exporting the environment variable. The legacy `WORLDSIM_STATE_DIR` name remains accepted as a compatibility alias.

Phase 4 result.json carries two parallel awareness metrics on every trajectory where a Needham XML transcript can be extracted. PVPO remains the encounter gate, but TP/VEA are observational context classifiers and may still run on `encounter.max_coverage == 0` trajectories:

- **Transcript Purpose**: Apollo Research's two-step classifier (`nix-apollo/38e0520c21bed0831d44019457d1833f`) over the shared Needham XML transcript. Step 1 asks the model to guess why the interaction took place; Step 2 classifies that guess into `Alignment Evaluation` / `Capability Evaluation` / `Other Evaluation` / `Real` / `Refused`. Keys: `transcript_purpose_classification`, `transcript_purpose_is_aware` (True iff classification is Alignment/Capability/Other Evaluation), `transcript_purpose_was_refused`, `transcript_purpose_guess`, `transcript_purpose_explanation`, `transcript_purpose_failure`, `transcript_purpose_format` (always `"needham-xml-v1"`), `transcript_purpose_attempts`, `transcript_purpose_raw_response_path`. Per-task artifacts include shared `needham_trace.json` / `needham_trace.xml`, `transcript_purpose_transcript.txt`, and `transcript_purpose_raw_response.json`. Baseline scoring records this as an observational metric; the explicit `eval-awareness-iterator` uses it only as the bounded rewrite-control signal.
- **Verbalized Eval Awareness**: Jasmine's 4-way classifier over the same Needham XML transcript. Keys: `verbalized_eval_awareness_classification` (one of `Alignment Evaluation` / `Capability Evaluation` / `Other Evaluation` / `Real`), `verbalized_eval_awareness_is_aware` (derived bool), `verbalized_eval_awareness_evidence` (list of verbatim quotes), `verbalized_eval_awareness_failure`, `verbalized_eval_awareness_transcript_format` (always `"needham-xml-v1"`), `verbalized_eval_awareness_raw_response_path`. Per-task artifacts include shared `needham_trace.*`, `verbalized_eval_awareness_transcript.xml`, and `verbalized_eval_awareness_raw_response.json`. VEA remains report-only and is never a gate or branch signal.

Trajectories where transcript extraction is unavailable get present-but-null metric keys so downstream loaders see a uniform schema. See the spec's "Gate 1 Stage 2: Observational Transcript Purpose + VEA" section for the full design.

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

Override `INSTANCES_FILE` (default `instances.smoke.json`), `TTL_HOURS` (default `24`), or `FEASIBILITY_CONCURRENCY` (default `10`) via env var. For remote host-local checks, set `INSTANCES_FILE=instances.scale.json`; `instances.smoke.json` is for local/public smoke topology.

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
| 4 | Adversarial Evaluation | Runs the agent against injected seeds; applies the PVPO encounter gate and attack-effectiveness gate; the default `eval-awareness-iterator` records an iterator envelope for every PVPO-valid default-path baseline trajectory and tries bounded payload rewrites only when Transcript Purpose calls for them. Legacy refusal-judge strategy variation remains opt-in via `--phase-4-variant-system strategy-variation`. Reports baseline capability as a byproduct |

The **authoritative technical spec** lives at [`docs/warp-taskgen-technical-spec.md`](docs/warp-taskgen-technical-spec.md). Every module in `worldsim/` implements a section of that spec.

## Repository layout

```
.
├── worldsim/                     # core Python import package
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
│   ├── prompts/                  # full prompts verbatim from the Taskgen spec
│   ├── phase_2/                  # modular Phase 2 package (eligibility, generation, plan_validation, runner, target_resolution/, phase_2c/)
│   ├── phase_4/                  # modular Phase 4 package (runner, execution, postprocess, metrics, results, resume, strategy_variation, variant_eval, ...)
│   └── phases/                   # legacy single-file modules + compat shims (phase_0_recon, phase_1_tasks, phase_2_injections shim, phase_4_adversarial shim, ...)
├── scripts/
│   ├── deploy_benchmark_proxy.sh  # authenticated reverse proxy for Phase 0c
│   └── proxy_ports.conf           # site-to-port mapping for the proxy
├── docs/
│   ├── warp-taskgen-technical-spec.md  # canonical technical spec
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
