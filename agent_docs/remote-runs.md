# Remote Runs and Host Ops

Use this before fresh-host setup, benchmark proxy work, r5 runs, Phase 4 rigor runs, or debugging live benchmark infrastructure.

## Operator Flow

Read `docs/handoffs/rigor-run-setup.md` for the full runbook. The short version:

1. Prepare or validate the host config, usually `configs/benchmark_hosts/r5.yaml`.
2. Sync code with `scripts/sync_to_r5.sh`.
3. Start long jobs with `scripts/remote_job_start.sh` and a job id.
4. Monitor with `scripts/remote_job_status.sh` or `scripts/remote_job_tail.sh`.
5. Stop only with `scripts/remote_job_stop.sh --job-id <id>`.

Avoid raw long-lived SSH pipes for runs. Never use broad `pkill -f` to stop jobs.
Do not sync a remote checkout while a registered job is still running. Common
Phase 0 -> 1 -> 2 and Phase 4 commands start fresh Python processes between
steps; syncing mid-chain can make one artifact use multiple code versions.
`sync_to_r5.sh` blocks active remote jobs by default. Use
`--allow-active-jobs` only for deliberate maintenance after recording why mixed
checkout provenance is acceptable.

`remote_job_start.sh` normalizes common project toolchain commands such as
`uv`, `uvx`, `pnpm`, `bun`, `npm`, `npx`, `node`, `modal`, and `claude` through
`bash -lc` before the detached runner starts them. This makes detached jobs use
the same login-shell PATH as an operator session and prevents false "stale"
jobs from tools installed by shell startup files. Metadata records both
`original_command` and normalized `command`, plus `command_execution.reason`.
Use an explicit `bash -lc '...'` command for multi-step runs. Set
`WORLDSIM_REMOTE_JOB_EXEC_MODE=direct` only when intentionally testing exact
argv/PATH behavior; launch failures are written as `exit.status=launch_failed`
and surfaced by `remote_job_status.sh`.

Use `--expected-output <path>` for the artifact that proves the job did real
work. For named Phase 1/2/3/4 runs, pass an explicit
`--state-dir logs/<run_name>` to `remote_job_start.sh`; that sets
`WORLDSIM_STATE_DIR` for the detached process and is how `worldsim.main` chooses
where `phase_*` artifacts are written. Do not pass `--output-dir` to
`worldsim.main phase`; the phase CLI is state-dir based. Use `--state-dir auto`
only for isolated experiments. Push required remote environment variables with
`scripts/remote_env_push.sh`; do not paste secrets into job commands or
hand-written shell history.

Modal-backed phases need two auth layers on remote hosts:

- Provider/app secrets in the repo `.env` or pushed environment, such as
  `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and auto-login values.
- The Modal client token config readable by the remote `ubuntu` user, normally
  `/home/ubuntu/.modal.toml` with mode `600`.

Do not assume syncing the repo or `.env` configures the Modal client. Phase 0c
and legacy sandbox-backed Phase 1 can pass project preflight yet still fail
Modal image prebuild with `Token missing` if the client config is absent. Verify
fresh hosts with a non-printing token check such as:

```bash
ssh ubuntu@<host> 'cd /home/ubuntu/browser-sim && uv run python -c "from modal.config import config; raise SystemExit(0 if config.get(\"token_id\") and config.get(\"token_secret\") else 1)"'
```

Treat launch attempts and evidence runs as separate artifacts. If a remote job
exits before it starts the measured phase, for example because auth, topology,
or prerequisite artifacts are missing, keep its remote job registry for
diagnostics but do not relaunch into the same state directory. Fix the
precondition, materialize a fresh state directory from the verified Phase 1-3
inputs, and start a new job whose `--expected-output` points at that fresh run.
This keeps failed setup attempts from becoming part of the evidentiary run
history and makes later summaries unambiguous.

Canonical r5 Phase 4 launch shape:

```bash
RUN=logs/<phase4_run_name>
SOURCE=logs/<verified_phase2_phase3_source>

scripts/remote_job_start.sh \
  --host-config configs/benchmark_hosts/r5.yaml \
  --remote-dir /home/ubuntu/browser-sim \
  --name <short-job-name> \
  --state-dir "$RUN" \
  --expected-output "$RUN/phase_4/results.json" -- \
  bash -lc '
    set -euo pipefail
    cd /home/ubuntu/browser-sim
    RUN="'"$RUN"'"
    SOURCE="'"$SOURCE"'"
    rm -rf "$RUN"
    mkdir -p "$RUN"
    for d in phase_0c phase_1 phase_2 phase_3; do
      cp -a "$SOURCE/$d" "$RUN/$d"
    done
    uv run python -m worldsim.main phase \
      --instances instances.scale.json \
      --sites reddit \
      --task-origin new_task \
      --max-tasks-per-site 16 \
      --phase-4-variant-system eval-awareness-iterator \
      --phase-4-eval-awareness-max-iterations 3 \
      --agent-provider openai \
      --agent-model gpt-5.2 \
      --agent-service-tier priority \
      --agent-llm-timeout 240 \
      --agent-step-timeout 300 \
      --agent-task-timeout 900 \
      --sandbox-model claude-sonnet-4-6 \
      4
  '
```

`eval-awareness-iterator` is the Phase 4 default. Pass it explicitly in remote
rigor commands so run intent is visible in job logs. For legacy `3+3+1`
comparability, use `--phase-4-variant-system strategy-variation` and set
`--phase-4-variant-budget adaptive-3-3-1` explicitly.

Current r5 topology:

- Runtime instances file: `instances.scale.json`
- Modal/public smoke instances file: `instances.smoke.json`
- Active sites: 21 GitLab replicas and 10 Reddit/Postmill replicas
- Smoke coverage: one GitLab replica and one Reddit/Postmill replica
- Runtime host view: `172.17.0.1`
- Public/proxy host view: `3.12.221.9`
- Proxy map has both web and envctrl listener rows, not one generic proxy port
  per site.

`remote_job_status.sh` is the first status surface. It reports process liveness,
heartbeat age, expected-output presence, and Phase 4 progress when available.
Tail logs only when status says output is stale or failed.
For Phase 4, treat `phase4_progress` staleness as stronger evidence than noisy
Browser Use stderr. Remote Phase 4 commands must pass `--agent-task-timeout`
explicitly; `--agent-llm-timeout` and `--agent-step-timeout` do not bound
session-start, CDP, or cleanup deadlocks.

High-concurrency Browser Use runs should cap expensive CDP state collection
separately from the outer worker count. `--phase-4-max-workers 48` means up to
48 live browser agents, but DOM snapshots and screenshots should not all hit
CDP simultaneously. The runner applies these PVPO-only backpressure defaults:

- `WORLDSIM_BROWSER_USE_DOM_STATE_CAP=16`
- `WORLDSIM_BROWSER_USE_SCREENSHOT_CAP=8`
- `WORLDSIM_BROWSER_USE_DEFAULT_ACTION_CAP=48`

For urgent W48 Browser Use runs, use `WORLDSIM_WORKER_STAGGER_DELAY_S=2.0`.
This reaches the full worker set in about 90 seconds, while avoiding the
sub-minute first-navigation stampede that can turn Browser Use's 8s page
readiness wait into `Navigation failed` / `DOMWatchdog` cascades. Reserve
lower values such as `0.5` for explicit stress tests, not paper-facing runs.
The Browser Use adapter also defaults high-concurrency event budgets to
`TIMEOUT_NavigateToUrlEvent=45.0`, `TIMEOUT_BrowserStateRequestEvent=60.0`,
and `TIMEOUT_BrowserConnectedEvent=60.0`; explicit environment values still
override these defaults.

Raise the outer worker cap only after `browser_runtime.json` shows low
`browser_use_*_watchdog_slow_calls`, low `browser_use_cdp_cancelled_requests_drained`,
and stable PVPO capture summaries. Do not treat a completed run as W48-clean if
duplicate CDP responses, DOM watchdog slow calls, or `pvpo_capture_degraded`
rows grow monotonically across tasks.

For paper-facing W48 Browser Use or AgentLab Phase 4 runs, prefer process
isolation over a single Phase 4 process with `--phase-4-max-workers 48` when
the single process shows runner event-bus/CDP contention. Use
`scripts/run_phase4_process_pool.py` from the remote job wrapper. The
supervisor launches normal
`worldsim.main phase 4` subprocesses, each with `--phase-4-max-workers 1`, a
single `--phase-4-task-id`, and a one-instance config. This preserves Phase 4
admission, seeding, PVPO, TP, VEA, eval-awareness iteration, rewards, and
readback semantics while isolating Browser Use or AgentLab sidecar event loops
and CDP clients by OS process. Page-surface-stable PVPO observes each worker's
normal browser session, so process-pool parallelism is no longer bounded by
dedicated PVPO CDP endpoint leases. If task distribution is site-skewed,
observed parallelism is still bounded by the number of tasks and benchmark
instances available for that site. For AgentLab parity runs, inspect each task's
`browser_runtime.json` for `browser_instance_scope="agent_run"` and
`agent_browser_connect_count=1`; BrowserGym auxiliary chat/UI launches are
reported separately under `auxiliary_browser_connect_count`.

Canonical process-pool shape:

```bash
scripts/remote_job_start.sh \
  --host-config configs/benchmark_hosts/r8a.yaml \
  --remote-dir /home/ubuntu/browser-sim \
  --name phase4-tier2-exact50-browseruse-gpt52-p48-processpool \
  --state-dir logs/<process_pool_run> \
  --expected-output logs/<process_pool_run>/phase_4/results.json -- \
  uv run python scripts/run_phase4_process_pool.py \
    --source-state-dir logs/<verified_phase1_phase3_source> \
    --instances instances.scale.json \
    --workers 48 \
    --runner browser_use \
    --agent-provider openai \
    --agent-model gpt-5.2 \
    --agent-service-tier priority \
    --agent-llm-timeout 240 \
    --agent-step-timeout 300 \
    --agent-task-timeout 2400 \
    --phase-4-variant-system eval-awareness-iterator \
    --phase-4-eval-awareness-max-iterations 3
```

While the process pool is running, start with `scripts/remote_job_status.sh`.
Its Phase 4 progress block includes active process-pool worker rows with worker
id, task id, current AgentLab/Phase 4 step, task trace dir, worker state dir,
and worker stderr path. Use `scripts/remote_job_tail.sh --worker-id <n>` or
`--task-id <id>` only after status points at a stale or failing worker. Task-id
tailing checks active `progress.json` first and completed
`phase_4/process_pool_summary.json` / `partial_manifest.json` second, so it can
tail completed process-pool workers after the summary is written. After
completion, inspect `phase_4/process_pool_summary.json` for per-worker
stdout/stderr paths, exit codes, timeouts, assigned instance indexes, and PVPO
endpoints. The final
`phase_4/results.json` is written only after every expected task has exactly
one valid worker result; missing, duplicate, or mismatched task IDs fail closed
instead of producing a partial canonical result. Failed process-pool runs may
write `phase_4/results.partial.json` and `phase_4/partial_manifest.json` for
operator inspection. Those files are never paper-eligible, the run still exits
nonzero, and only canonical `phase_4/results.json` counts as a complete Phase 4
artifact.

WorldSim uses two different network localities on r5. Treat the instances file
as an execution-locality contract, not just a dataset selector:

| Phase / caller | Where traffic originates | Correct r5 instances file | Why |
| --- | --- | --- | --- |
| Phase 0c profiling | Modal sandbox outside r5 | `instances.smoke.json` or equivalent public/proxy file | Modal cannot reach r5-only addresses such as `172.17.0.1`; it needs the authenticated public proxy. |
| Phase 0c host-side inventory enrichment | r5 orchestrator process | `--host-inventory-instances instances.scale.json` | Reddit DB enumeration and GitLab project inventory run on the host, not in Modal; they need the orchestrator-local topology. |
| Phase 2c render checks | r5 host / browser containers | `instances.scale.json` | AWS does not reliably hairpin public-IP traffic from the instance to itself; on-host browsers need `orchestrator_host`. |
| Phase 4 agent/PVPO | r5 host / browser containers | `instances.scale.json` | Same on-host browser topology as Phase 2c; storage-state cookies are host-bound. |

Do not collapse these into one rule. `instances.scale.json` is correct for
Phase 2c/4 and wrong for Modal Phase 0c. `instances.smoke.json` is correct for
Modal Phase 0c and wrong for on-host Phase 2c/4 unless regenerated for the
host's `orchestrator_host`.

Phase 0c has one mixed-locality edge case: its browser probes run in Modal, but
host-side inventory enrichment, such as Reddit forum DB enumeration, runs in
the detached orchestrator process. For fresh novel GitLab/Reddit generation on
r5, pass both localities explicitly:
`--instances instances.smoke.json --host-inventory-instances instances.scale.json`.
Do not rely on rewriting only the host portion of `instances.smoke.json`: scale
topology can use different per-replica DB/API ports, so a host-only rewrite can
silently point at the wrong service. Reddit forum enrichment must also intersect
the scale replica pool before advertising forum anchors; forums created by live
profiling or prior editor probes on a single replica are not stable benchmark
inventory. GitLab handle enrichment follows the same host-side API rule and
falls back to the original URL if the host-local candidate fails. If enrichment
falls back to static profile samples, treat any later source-data 404s as stale
inventory evidence, not as a carrier-render verdict.

On r5, also treat the benchmark source path as host-local. `sync_to_r5.sh`
intentionally excludes repo-local `vendors/`, so
`/home/ubuntu/browser-sim/vendors/webarena-verified` may be stale or incomplete.
Use `/home/ubuntu/vendors/webarena-verified` in remote Phase 0/1/2 commands
unless you have explicitly hydrated and verified the repo-local vendor tree.
`remote_job_start.sh` blocks the repo-relative WebArena Verified path by default
for this reason.

Generated topology artifacts are host-local too. `instances.scale.json`,
`instances.smoke.json`, their fragments, generated compose files, and proxy port
maps are gitignored and should be regenerated on the host from the selected
host config, not synced from a laptop checkout.

`scripts/remote_job_start.sh` enforces this split on
`remote_direct_restricted` hosts: it blocks Phase 2/2c/4 smoke inputs and
blocks Phase 0/0c scale inputs. Override with
`WORLDSIM_ALLOW_REMOTE_INSTANCE_TOPOLOGY_MISMATCH=1` only when intentionally
testing a different topology, and record why the run is not comparable.

## Resume and Compaction Checks

After context compaction or a long pause, treat any live artifact path as
topology-bound until checked. Before reusing a Phase 2/2c or Phase 4 state
directory on r5:

- Confirm the command uses `instances.scale.json` for on-host browser phases.
- Confirm the artifact's saved benchmark metadata was produced for the same
  instance pool. If Phase 2c refuses a metadata mismatch, rerun a small matching
  Phase 1 -> 2 job instead of forcing the old artifact through a different
  instance file.
- Treat `host_unreachable`, public-IP navigation from the host, and GitLab
  storage-state host mismatches as topology symptoms first, not as carrier,
  prompt, or strategy verdicts.

Remote metadata and logs live under:

```text
<remote-dir>/logs/remote_jobs/<job_id>/
```

Live hosts often contain the freshest Phase task artifacts under `logs/`.
Before relying on checked-in task JSON, audit the host or run archive and follow
`agent_docs/artifacts.md` for manifest, hash, and fixture-promotion rules.

## Fresh Host Gate

Run this on any fresh host before Phase 4:

```bash
scripts/setup_phase4_on_host.sh \
  --host-config configs/benchmark_hosts/r5.yaml \
  --instances instances.scale.json \
  --artifacts-source s3://benchmark-archives/worldsim-runs/<run_id>/
```

If `--artifacts-source` is omitted, the script expects matching
`phase_0c`, `phase_2`, and `phase_3` artifacts to already exist under the
selected state directory. Use `/home/ubuntu/vendors/webarena-verified` as the
benchmark source unless you have explicitly hydrated the repo-local
`vendors/webarena-verified` tree on the host.

Its preflight step runs `pytest -m preflight tests/preflight` and proves:

- Benchmark instance connectivity and page-surface-stable PVPO readiness.
- GitLab Phase 0d `storage_state` exists with non-empty cookies.
- The `worldsim-webarena-verified` evaluator venv resolves.

Do not skip this before rigor runs.

## Proxy Discipline

The source of truth for nginx proxy config is:

```text
scripts/deploy_benchmark_proxy.sh
```

Never hand-edit `/etc/nginx/conf.d/worldsim-proxy.conf` on the host. Verify parity with:

```bash
scripts/check_proxy_drift.sh \
  --host 3.12.221.9 \
  --topology scale \
  --insecure-http \
  --verify-runtime
```

If the drift checker grows full `--host-config` support, the runbook can switch
back to that form. Until then, pass the host explicitly so the command cannot
silently verify the wrong target.

The proxy uses token auth (`X-Worldsim-Token`) and offset ports from `scripts/proxy_ports.conf`. Phase 0c may use the proxy for live verification; Phases 3 and 4 use real `site_url` and `reset_endpoint` values from the instances file.

## PVPO Rigor Requirement

Phase 4 rigor runs use page-surface-stable PVPO. The runner captures the
visible viewport from its own browser with normal CDP `Page.captureScreenshot`
and accepts evidence only when pre/post DOM witness probes are stable. Dedicated
dedicated compositor-driving browser containers are removed from the active run
path; `pvpo_cdp_url` remains only for legacy remote-browser lifecycle
compatibility.

## Live Integration Command

For PR gates against a live stack:

```bash
scripts/run_integration_tests.sh --host-config configs/benchmark_hosts/r5.yaml --quiet
```

Use `--quiet` for agent sessions. It prints a one-line pass summary or full failure output.
With `--host-config` and no explicit `--instances`, the wrapper now generates a
temporary host-config-specific smoke instances file from `scripts/scale_config.yml`
and the selected host config. This is intentional: `sync_to_r5.sh` excludes
generated topology artifacts, and host setup may regenerate `instances.smoke.json`
with ports that differ from a stale laptop checkout. Pass `--instances` only
when you intentionally want a specific public/proxy or scale topology file. On
r5/r8 host-side Phase 2c/4 commands, continue using `instances.scale.json`.

**Topology-mismatch symptom and fix.** If the wrapper fails reddit-side tests
with `Connection refused` on a stale legacy smoke proxy port (for example
`19999`, derived from old local `reddit:9999`), Reddit is usually not down.
First rerun without an explicit `--instances` so the wrapper can generate the
host-specific smoke file. If you are deliberately testing scale topology, pass
the matching generated scale fixture:

```bash
bash scripts/run_integration_tests.sh \
    --host-config configs/benchmark_hosts/r5.yaml \
    --quiet

# Deliberate scale-topology check:
bash scripts/generate_scale_r5.sh --host-config configs/benchmark_hosts/r5.yaml
bash scripts/run_integration_tests.sh \
    --host-config configs/benchmark_hosts/r5.yaml \
    --instances instances.scale.json \
    --quiet
```

The default-installed proxy adapter in `tests/integration/conftest.py`
(`_DEFAULT_SITE_PORTS`) already covers both legacy `9999` and the scale band
`9900..9990`, so picking replica 0 (`reddit:9900`, `gitlab:8023`) flows through
the existing nginx listens automatically. The corrective action is the
fixture, not the proxy.
