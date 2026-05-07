# Rigor-run setup runbook

One-page reference for going from a fresh EC2 host to a Phase 4 rigor run.
Codifies the manual r5 setup from 2026-04-20 so the next host runs with
zero hand-patching.

## Sequence

1. **`scripts/bootstrap_r5.sh`** (or equivalent for the target host).
   Generates the scale compose, preflights the SG, brings benchmark
   containers up with env-ctrl responding.
2. **`scripts/setup_phase4_on_host.sh`** idempotent setup:

   ```
   scripts/setup_phase4_on_host.sh \
       --host-config configs/benchmark_hosts/r5.yaml \
       --instances instances.scale.json \
       --artifacts-source s3://benchmark-archives/worldsim-runs/<run_id>/
   ```

   1. uv + repo venv + evaluator venv (`packages/worldsim-webarena-verified`).
   2. Regenerate `instances.scale.json` / `instances.smoke.json` from
      `scripts/scale_config.yml` and the selected host config.
   3. Playwright Chromium + system libs when Playwright is installed.
   4. One `pvpo-chrome` Docker container per configured `pvpo_cdp_url`
      (content-hash stamped; rebuilds only when
      `worldsim/docker/chrome-headless-shell.Dockerfile` changes; containers
      are `--restart unless-stopped` because Phase 4 recycles the browser
      process after every task).
   5. Artifact sync (`logs/phase_0c`, `logs/phase_2/adversarial_tasks.json`,
      `logs/phase_3/contracts.json`). Prefer `aws s3 sync` from
      `s3://benchmark-archives/worldsim-runs/<run_id>/`, same-region AWS
      transfer, no egress, no SSH key dependency.
   6. Re-mint GitLab Phase 0d `storage_state.json` unconditionally so cookies
      bind to the current host topology.
   7. `pytest -m preflight tests/preflight`, the pass/fail gate.

   Before any Modal-backed Phase 0c or legacy sandbox-backed Phase 1 work,
   verify the host has both layers of auth:

   - repo/app environment such as `.env`, `OPENAI_API_KEY`,
     `ANTHROPIC_API_KEY`, and login helper values;
   - Modal client credentials for the remote user, normally
     `/home/ubuntu/.modal.toml` with mode `600`.

   Repo sync and `.env` sync do not install the Modal client config. A missing
   client config can pass project-level key checks and then fail at Modal image
   prebuild with `Token missing`.
3. **Launch or resume**:
   ```
   scripts/remote_job_start.sh \
       --host-config configs/benchmark_hosts/r5.yaml \
       --remote-dir /home/ubuntu/browser-sim \
       --name phase4-rigor \
       --state-dir logs/<run_name> \
       --expected-output logs/<run_name>/phase_4/results.json \
       -- \
       uv run python -m worldsim.main phase 4 \
           --instances instances.scale.json \
           --phase-4-variant-system eval-awareness-iterator \
           --agent-task-timeout 900
   ```
   To continue an interrupted pipeline from saved state, use the dedicated
   resume subcommand through the same remote job wrapper:
   ```
   scripts/remote_job_start.sh \
       --host-config configs/benchmark_hosts/r5.yaml \
       --remote-dir /home/ubuntu/browser-sim \
       --name phase4-resume \
       --state-dir logs/<run_name> \
       --expected-output logs/<run_name>/phase_4/results.json \
       -- \
       uv run python -m worldsim.main resume \
           --instances instances.scale.json \
           --agent-task-timeout 900
   ```

   For high-concurrency Browser Use evidence runs, use the process-pool runner
   when a single Phase 4 process shows Browser Use event-bus/CDP contention.
   The process pool is orchestration-only: each subprocess invokes normal
   `worldsim.main phase 4` with one task, one instance, and
   `--phase-4-max-workers 1`, then the supervisor merges a canonical
   `phase_4/results.json` after validating that every expected task has exactly
   one result.

   ```
   scripts/remote_job_start.sh \
       --host-config configs/benchmark_hosts/r8a.yaml \
       --remote-dir /home/ubuntu/browser-sim \
       --name phase4-rigor-p48-processpool \
       --state-dir logs/<run_name> \
       --expected-output logs/<run_name>/phase_4/results.json \
       -- \
       uv run python scripts/run_phase4_process_pool.py \
           --source-state-dir logs/<verified_phase1_phase3_source> \
           --instances instances.scale.json \
           --workers 48 \
           --runner browser_use \
           --phase-4-variant-system eval-awareness-iterator \
           --phase-4-eval-awareness-max-iterations 3 \
           --agent-provider openai \
           --agent-model gpt-5.2 \
           --agent-service-tier priority \
           --agent-task-timeout 2400
   ```

   Do not compare process-pool output to single-process W48 as an infrastructure
   stress result; compare it as the same Phase 4 measurement core with safer
   runner isolation. If `phase_4/process_pool_summary.json` reports missing
   worker results, duplicate task IDs, task ID mismatches, or worker timeouts,
   treat the run as incomplete until the failed task set is rerun or the root
   cause is classified.

## Remote job wrapper quickstart

Use the remote job scripts for any r5 command that may outlive an interactive
SSH session. They detach with `/dev/null` stdin, file-backed stdout/stderr, and
a registry under `<remote-dir>/logs/remote_jobs/<job_id>/`.

1. Sync the checkout without secrets or generated logs:
   ```
   scripts/sync_to_r5.sh \
       --host-config configs/benchmark_hosts/r5.yaml \
       --remote-dir /home/ubuntu/browser-sim
   ```
   `--ssh-key ~/.ssh/webarena-key.pem` is expanded locally before it reaches
   `rsync`; avoid nested quoted `$HOME` in hand-written `rsync -e` strings.

2. Start a named job. The human name is not the unique id; the script prints a
   timestamped `job_id` plus exact follow-up commands.
   ```
   scripts/remote_job_start.sh \
       --host-config configs/benchmark_hosts/r5.yaml \
       --remote-dir /home/ubuntu/browser-sim \
       --name phase1-route-diversity \
       --expected-output logs/phase_1/benign_tasks.json \
       -- \
       uv run python -m worldsim.main phase 1 --generate-novel
   ```

3. Inspect, tail, and stop by registry id:
   ```
   scripts/remote_job_status.sh --host-config configs/benchmark_hosts/r5.yaml --job-id <job_id>
   scripts/remote_job_tail.sh --host-config configs/benchmark_hosts/r5.yaml --job-id <job_id> --lines 120
   scripts/remote_job_tail.sh --host-config configs/benchmark_hosts/r5.yaml --job-id <job_id> --stderr
   scripts/remote_job_stop.sh --host-config configs/benchmark_hosts/r5.yaml --job-id <job_id>
   ```
   `--latest` and `--name <name>` are supported for status/tail/list convenience;
   stop intentionally requires `--job-id`.

Each job directory contains `metadata.json`, `command.argv.json`, `stdout.log`,
`stderr.log`, `heartbeat.json`, `pid`, `pgid`, and `exit.json` after completion
or an operator stop. Status verifies the remote process instead of trusting
metadata alone and warns when a live process has not written logs recently.

State directory note: `remote_job_start.sh` does **not** set
`WORLDSIM_STATE_DIR` by default because Phase 1/2/3/4 often read and write
canonical `logs/phase_*` artifacts. For named rigor runs, pass
`--state-dir logs/<run_name>` and register the expected Phase 4 result file.
Use `--state-dir auto` only for isolated experiments that should write under
`logs/remote_jobs/<job_id>/state`.

Failure recovery:

- If the local SSH session dies, use `remote_job_status.sh --latest` or
  `remote_job_list.sh` to recover the job id; logs remain on the host.
- If a job fails before the measured phase is underway, treat it as a setup
  attempt, not as a partial evidence run. Preserve the remote job logs for the
  root cause, fix the missing precondition, and relaunch into a fresh
  `--state-dir` copied from the verified Phase 1-3 inputs.
- If status says `running` but logs are quiet for more than 15 minutes, inspect
  `stderr.log` and expected outputs before deciding to stop.
- If stop refuses to signal, the recorded PID was reused or no longer matches
  the job wrapper. Inspect the registry files manually; do not fall back to
  `pkill -f`.
- These scripts only run repo commands on an already-prepared benchmark host.
  They do not start, stop, or repair benchmark containers, proxies, nginx, or
  EC2 lifecycle state.

## What the preflight covers

`tests/preflight/test_phase_4_preflight.py` asserts each of:

| check | failure remediation |
|---|---|
| every configured `pvpo_cdp_url` reachable and unique | rerun setup step 4 |
| each loopback `pvpo-chrome-<port>` has restart policy `unless-stopped` | rerun setup step 4 |
| `logs/phase_0d/gitlab/storage_state.json` has cookies | rerun `login_gitlab_r5.py` or setup step 6 |
| evaluator venv imports `webarena_verified` | rerun setup step 1 |

If preflight fails, the bash orchestrator exits non-zero and nothing is
launched.

## PVPO integration: why CDP connect (not local flags)

The PVPO launch flags (`--enable-begin-frame-control`,
`--run-all-compositor-stages-before-draw`, `--disable-checker-imaging`)
pause default frame rendering; frames only commit when
`HeadlessExperimental.beginFrame` is called explicitly. Applying them to
the local Chromium that Browser-Use launches hangs every `page.goto`
for 30s. The correct integration is the `chrome-headless-shell`
container (flags in its `CMD`, beginFrame calls at capture time only);
Browser-Use connects via `BrowserSession(cdp_url=...)`.

Phase 4 no longer supports a shared remote PVPO browser. Each instance
carries its own `pvpo_cdp_url`, and setup/preflight treat duplicate
endpoint assignment as a hard error.

Phase 4 also treats each dedicated PVPO browser as single-task-use. Browser
Use sessions close tabs and contexts, but the remote `chrome-headless-shell`
process survives the session. Under `--enable-begin-frame-control`, leaked
renderer state can keep consuming CPU after the task ends. For managed loopback
endpoints, the runner now restarts `pvpo-chrome-<port>` with `docker restart`
at task teardown and waits for `/json/version` to return on the same port. CDP
`Browser.close` is only a fallback for unmanaged endpoints; it does not reliably
kill the supervised parent process on r5. Use `WORLDSIM_PVPO_BROWSER_RECYCLE=0`
only for local smoke tests against a manually-started Chrome without restart
supervision.

## Historical Magento Base-URL Drift

Magento left active WASP scope on 2026-04-21. Keep this note only for reading
old shopping artifacts; it is not part of the active Phase 4 rigor setup.

Root cause: `scripts/generate_compose_scale.py:96` once baked
`WA_ENV_CTRL_EXTERNAL_SITE_URL` with the raw backend port. Every Phase 4
`reset_endpoint` POST triggered env-ctrl `_init()` which ran
`setup:store-config:set --base-url=<raw>` and reverted the repair on
every task.

Two-part fix:

1. `generate_compose_scale.py:96` now bakes
   `real_web + proxy_port_offset`, so `_init()` is idempotent with the
   proxy origin.
2. `sync_magento_base_urls.py` uses `config:set --lock-env` (writes
   `env.php`, top of precedence chain). Even if `_init()` regressed, the
   env.php value wins.

Loops every replica via `instances.json:replica_name` (not the hardcoded
non-indexed container names). Structured JSON summary per replica.

## Known-good agent model slugs

Model allowlist was removed — providers rotate catalogs faster than any
in-tree list stays accurate, and a rotting allowlist is worse than no
allowlist. `BrowserUseAgent.__init__` now logs the configured slug at
construction; if the run 404s mid-task, grep for that log line and
double-check the slug against the provider's current catalog.

## Env vars this runbook uses

- `WORLDSIM_WEBARENA_EVAL_PYTHON` — override evaluator venv Python (default is repo-relative `packages/worldsim-webarena-verified/.venv/bin/python`).
- `WORLDSIM_AUTO_MINT_STORAGE_STATE` — opt non-WebArena-Verified benchmarks in to runtime auto-heal. `true` is implicit for WebArena Verified.
- `GITLAB_HOST` / `GITLAB_STORAGE_STATE_PATH` — override defaults for `scripts/login_gitlab_r5.py`.
- `WORLDSIM_REPO_ROOT` — override sentinel-walk repo discovery.
- `WORLDSIM_BENCHMARK_ROOT`, override the WebArena Verified checkout used by
  `scripts/setup_phase4_on_host.sh` (default `/home/ubuntu/vendors/webarena-verified`).

## When something breaks mid-run

- **PVPO `max_coverage == 0` on every trajectory**: one or more per-instance
  containers are not reachable or the instances file points multiple workers
  at the same endpoint. Check `pytest -m preflight tests/preflight`, then
  inspect the corresponding `docker logs pvpo-chrome-<port> | tail -40` and
  `curl http://127.0.0.1:<port>/json/version`.
- **Host load climbs after each completed trajectory**: check
  `browser_runtime.json` for `pvpo_browser_recycle_status`. Anything other
  than `recycled` means the browser process was not restarted cleanly; rerun
  setup step 3 and inspect the `pvpo-chrome-<port>` restart policy plus
  `docker_restart_*` fields in the runtime artifact.
- **Evaluator subprocess error in rewards**: the evaluator venv isn't
  synced. `cd packages/worldsim-webarena-verified && uv sync --locked`.
- **Gitlab task fails with `AuthArtifactMissingError`**: the auto-heal
  didn't kick in. Check the `WORLDSIM_AUTO_MINT_STORAGE_STATE` env var
  and that the site has `form_login` configured in `instances.json`.
