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

WorldSim uses two different network localities on r5. Treat the instances file
as an execution-locality contract, not just a dataset selector:

| Phase / caller | Where traffic originates | Correct r5 instances file | Why |
| --- | --- | --- | --- |
| Phase 0c profiling | Modal sandbox outside r5 | `instances.smoke.json` or equivalent public/proxy file | Modal cannot reach r5-only addresses such as `172.17.0.1`; it needs the authenticated public proxy. |
| Phase 2c render checks | r5 host / browser containers | `instances.scale.json` | AWS does not reliably hairpin public-IP traffic from the instance to itself; on-host browsers need `orchestrator_host`. |
| Phase 4 agent/PVPO | r5 host / browser containers | `instances.scale.json` | Same on-host browser topology as Phase 2c; storage-state cookies are host-bound. |

Do not collapse these into one rule. `instances.scale.json` is correct for
Phase 2c/4 and wrong for Modal Phase 0c. `instances.smoke.json` is correct for
Modal Phase 0c and wrong for on-host Phase 2c/4 unless regenerated for the
host's `orchestrator_host`.

On r5, also treat the benchmark source path as host-local. `sync_to_r5.sh`
intentionally excludes repo-local `vendors/`, so
`/home/ubuntu/browser-sim/vendors/webarena-verified` may be stale or incomplete.
Use `/home/ubuntu/vendors/webarena-verified` in remote Phase 0/1/2 commands
unless you have explicitly hydrated and verified the repo-local vendor tree.
`remote_job_start.sh` blocks the repo-relative WebArena Verified path by default
for this reason.

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

## Fresh Host Gate

Run this on any fresh host before Phase 4:

```bash
scripts/setup_phase4_on_host.sh
```

Its preflight step runs `pytest -m preflight tests/preflight` and proves:

- PVPO CDP endpoints are reachable and uniquely assigned.
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
scripts/check_proxy_drift.sh --verify-runtime
```

The proxy uses token auth (`X-Worldsim-Token`) and offset ports from `scripts/proxy_ports.conf`. Phase 0c may use the proxy for live verification; Phases 3 and 4 use real `site_url` and `reset_endpoint` values from the instances file.

## PVPO Rigor Requirement

Phase 4 rigor runs need `chrome-headless-shell` Docker containers from `worldsim/docker/chrome-headless-shell.Dockerfile`. Native macOS Chrome does not support `HeadlessExperimental.beginFrame`; without the container, PVPO falls back to zero coverage and trajectories route to placement-fix.

Each execution instance needs its own `pvpo_cdp_url`. Do not point multiple workers at one shared browser.

## Live Integration Command

For PR gates against a live stack:

```bash
scripts/run_integration_tests.sh --host-config configs/benchmark_hosts/r5.yaml --quiet
```

Use `--quiet` for agent sessions. It prints a one-line pass summary or full failure output.
