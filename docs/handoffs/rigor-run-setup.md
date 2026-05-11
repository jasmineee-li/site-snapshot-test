# Rigor-run setup runbook

One-page reference for going from a fresh EC2 host to a Phase 4 rigor run.
Codifies the manual r5 setup from 2026-04-20 while making r8a the canonical
host-config-driven path for current runs.

## Sequence

1. **`scripts/bootstrap_r8a.sh`** for current r8a runs, or the selected-host
   equivalent. Audits the r8a control plane, generates the scale compose,
   preflights the SG, and brings benchmark containers up with env-ctrl
   responding.
2. **`scripts/setup_phase4_on_host.sh`** idempotent setup:

   ```
   scripts/setup_phase4_on_host.sh \
       --host-config configs/benchmark_hosts/r8a.yaml \
       --instances instances.scale.json \
       --artifacts-source s3://benchmark-archives/worldsim-runs/<run_id>/
   ```

   1. uv + repo venv + evaluator venv (`packages/warp-taskgen-webarena-verified`).
   2. Regenerate `instances.scale.json` / `instances.smoke.json` from the
      selected scale config and host config. Use
      `--scale-config scripts/scale_config.r8a-24x24.yml` for the r8a 24x24
      topology.
   3. Playwright Chromium + system libs when Playwright is installed.
   4. Page-surface-stable PVPO requires no dedicated browser endpoint setup.
      The runner-owned Browser Use or AgentLab browser is the capture surface.
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
       --host-config configs/benchmark_hosts/r8a.yaml \
       --remote-dir /home/ubuntu/browser-sim \
       --name phase4-rigor \
       --state-dir logs/<run_name> \
       --expected-output logs/<run_name>/phase_4/results.json \
       -- \
       uv run warp-taskgen phase 4 \
           --instances instances.scale.json \
           --phase-4-variant-system eval-awareness-iterator \
           --agent-task-timeout 900
   ```
   To continue an interrupted pipeline from saved state, use the dedicated
   resume subcommand through the same remote job wrapper:
   ```
   scripts/remote_job_start.sh \
       --host-config configs/benchmark_hosts/r8a.yaml \
       --remote-dir /home/ubuntu/browser-sim \
       --name phase4-resume \
       --state-dir logs/<run_name> \
       --expected-output logs/<run_name>/phase_4/results.json \
       -- \
       uv run warp-taskgen resume \
           --instances instances.scale.json \
           --agent-task-timeout 900
   ```

   For high-concurrency Browser Use evidence runs, top-level
   `warp-taskgen phase 4` commands use `--phase-4-max-workers`, not
   `--workers`. The legacy `python -m worldsim.main phase 4` path is accepted
   for compatibility, but runbooks should prefer `warp-taskgen`. Use the
   process-pool runner only when a single Phase 4 process shows Browser Use
   event-bus/CDP contention. The process pool is orchestration-only: each
   subprocess invokes normal Phase 4 with one task, one instance, and
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
           --source-state-dir logs/<source_with_phase_2_adversarial_tasks_and_phase_3_contracts> \
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

   In this command, `--workers 48` belongs to `scripts/run_phase4_process_pool.py`.
   Do not use it with top-level `warp-taskgen phase 4`; the remote job guard
   rejects that spelling for top-level Phase 4.

   Do not compare process-pool output to single-process W48 as an infrastructure
   stress result; compare it as the same Phase 4 measurement core with safer
   runner isolation. If `phase_4/process_pool_summary.json` reports missing
   worker results, duplicate task IDs, task ID mismatches, or worker timeouts,
   treat the run as incomplete until the failed task set is rerun or the root
   cause is classified. After `process_pool_summary.json` or
   `partial_manifest.json` exists, `scripts/remote_job_tail.sh --task-id <id>`
   resolves the completed worker's stdout/stderr without needing the numeric
   worker id.

   If the failed rows are rerun successfully, build a separate repaired run with
   `scripts/repair_process_pool_partial.py` instead of editing partial artifacts.
   The repair output must include `phase_4/process_pool_repair_manifest.json`
   and `paper_eligible="operator_review_required"` so the row replacements are
   reviewed before any paper-facing promotion. Iterator-checkpoint timeout
   salvage remains inspection evidence only; it does not make a failed
   process-pool merge canonical.

## Archive a completed run to S3

After a rigor run completes and the results are accepted as evidence, archive
the run directory to `s3://benchmark-archives/worldsim-runs/<run_id>/` so the
trajectories survive even when the host disk is recycled or the AMI is
rotated. The script lives at `scripts/archive_run_to_s3.sh` and follows the
AWS-recommended two-pass discipline (sync, verify with dryrun, optional delete).

```
scripts/archive_run_to_s3.sh <run_id>
```

This will:

1. Validate the run directory exists and write `ARCHIVE_MANIFEST.json` capturing
   the git sha, branch, hostname, file count, and byte count at archive time.
2. `aws s3 sync` the run directory to `s3://benchmark-archives/worldsim-runs/<run_id>/`
   with `STANDARD_IA` storage class and `--only-show-errors` operator logging.
3. Verify with `aws s3 sync --dryrun` — must report zero `upload:` lines.
4. Tag the manifest object so future audits can filter by `project=warp-taskgen`,
   `run_id=<id>`, `archived_at=<ISO>`, `git_sha=<sha>`, `purpose=paper-evidence`.

After the local copy is no longer needed (and disk pressure warrants it), pass
`--delete-local` for the same invocation to remove the local directory in the
same pass. Verification still runs first; `--delete-local` will refuse to delete
if verification didn't run.

```
scripts/archive_run_to_s3.sh <run_id> --delete-local
```

Resume after interruption: just rerun the same command. `aws s3 sync` is
natively idempotent — it skips files already at the destination with matching
size and mtime. The two-pass dryrun verification will catch any partial state.

Refresh the local archive index after archiving:

```
scripts/regen_archive_index.sh
```

That regenerates `docs/runs/archive-index.md` (gitignored) from S3 metadata
plus the local HuggingFace dataset references. Shows gaps where HF references
a run_id that has no S3 archive.

Hardlink note: `du -sb` counts hardlinks once; S3 expands them into separate
objects. The S3 byte total will exceed the local `du -sb` total — this is
expected and the verification script accounts for it. Don't panic at the size
diff; the dryrun check is the authoritative integrity gate.

Sequencing note for paper handoff:

1. Phase 4 completes and produces `results.json` plus the HF dataset export.
2. The HF export commits to `data/hf/<dataset_id>/`.
3. `scripts/archive_run_to_s3.sh <run_id>` writes the trajectories to S3.
4. `scripts/regen_archive_index.sh` updates the local index.
5. Only then is the run "fully landed". A run with HF export but no S3 archive
   is incomplete because reviewers / future you cannot retrieve the underlying
   trajectories that back the HF rows.

## Remote job wrapper quickstart

Use the remote job scripts for any r5 command that may outlive an interactive
SSH session. They detach with `/dev/null` stdin, file-backed stdout/stderr, and
a registry under `<remote-dir>/logs/remote_jobs/<job_id>/`.

1. Sync the checkout without secrets or generated logs:
   ```
   scripts/sync_to_host.sh \
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
       uv run warp-taskgen phase 1 --generate-novel
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

State directory note: `remote_job_start.sh` does **not** set a state dir by
default because Phase 1/2/3/4 often read and write canonical `logs/phase_*`
artifacts. For named rigor runs, pass `--state-dir logs/<run_name>`; the
wrapper currently injects `WORLDSIM_STATE_DIR` for compatibility, and the CLI
also accepts `WARP_TASKGEN_STATE_DIR` as the canonical state-dir env alias.
Register the expected Phase 4 result file. Use `--state-dir auto` only for
isolated experiments that should write under `logs/remote_jobs/<job_id>/state`.

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
| `logs/phase_0d/gitlab/storage_state.json` has cookies | rerun `login_gitlab_r5.py` or setup step 6 |
| evaluator venv imports `webarena_verified` | rerun setup step 1 |

If preflight fails, the bash orchestrator exits non-zero and nothing is
launched. Current Phase 4 does not require dedicated PVPO CDP endpoints, so
preflight does not inspect or supervise legacy PVPO browser containers.

## PVPO Integration

Phase 4 uses page-surface-stable PVPO. The capture callback runs inside the
same Browser Use or AgentLab browser session that is executing the task:

1. evaluate the DOM witness probe;
2. capture the visible viewport with normal CDP `Page.captureScreenshot`;
3. evaluate the DOM witness probe again;
4. accept the screenshot only when URL, witness, background, and geometry are
   stable across the pre/post probes.

Do not provision dedicated PVPO browser endpoints or browser-recycle
infrastructure for new rigor runs. Some generated legacy-compatible instance
files still contain `pvpo_cdp_url`; canonical page-surface-stable PVPO ignores
that metadata.

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

- `WARP_TASKGEN_WEBARENA_EVAL_PYTHON` — override evaluator venv Python
  (default is repo-relative
  `packages/warp-taskgen-webarena-verified/.venv/bin/python`).
- `WORLDSIM_AUTO_MINT_STORAGE_STATE` — opt non-WebArena-Verified benchmarks in to runtime auto-heal. `true` is implicit for WebArena Verified.
- `GITLAB_HOST` / `GITLAB_STORAGE_STATE_PATH` — override defaults for `scripts/login_gitlab_r5.py`.
- `WORLDSIM_REPO_ROOT` — override sentinel-walk repo discovery.
- `WORLDSIM_BENCHMARK_ROOT`, override the WebArena Verified checkout used by
  `scripts/setup_phase4_on_host.sh` (default `/home/ubuntu/vendors/webarena-verified`).

## When something breaks mid-run

- **PVPO `max_coverage == 0` on every trajectory**: the runner-owned browser is
  not reaching the seeded surface, the instances file points workers at the
  wrong host view, or reset/envctrl is drifting between tasks. Inspect per-task
  `pvpo/capture_summary.json`, `browser_runtime.json`, reset endpoint
  reachability, and the site/envctrl container logs for the affected instance.
- **Host load climbs after each completed trajectory**: inspect
  `browser_runtime.json` for runner-owned browser lifecycle fields, slow CDP
  calls, timeout counts, and degraded PVPO capture summaries.
- **Evaluator subprocess error in rewards**: the evaluator venv isn't
  synced. `cd packages/warp-taskgen-webarena-verified && uv sync --locked`.
- **Gitlab task fails with `AuthArtifactMissingError`**: the auto-heal
  didn't kick in. Check the `WORLDSIM_AUTO_MINT_STORAGE_STATE` env var
  and that the site has `form_login` configured in `instances.json`.
