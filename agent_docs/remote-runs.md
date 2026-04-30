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

For r5 jobs that run browsers on the host, including Phase 2c render checks and
Phase 4 agent/PVPO runs, use the generated `instances.scale.json`. Do not use
`instances.smoke.json` on-host unless you have regenerated it for the host's
`orchestrator_host`. The smoke file may point at the public advertised IP; AWS
does not reliably hairpin browser traffic from the instance back to its own
public IP, which causes false `host_unreachable` render failures and host-bound
storage-state mismatches. `instances.scale.json` is generated from the host
config and uses `orchestrator_host` such as `172.17.0.1`.
`scripts/remote_job_start.sh` enforces this for known on-host browser phases
and blocks Phase 2/2c defaults or explicit smoke-instance inputs before the
remote job starts. Override with
`WORLDSIM_ALLOW_REMOTE_INSTANCE_TOPOLOGY_MISMATCH=1` only when intentionally
testing an external-browser topology.

Phase 0c is the topology exception. Its profiling work runs in Modal sandboxes,
not in the on-host browser process, so it cannot reach r5-only addresses such
as `172.17.0.1`. Use an externally reachable/proxied instance file such as
`instances.smoke.json` for Phase 0/0c live probing, then switch back to
`instances.scale.json` for Phase 2c and Phase 4. The remote launcher blocks
`phase 0 --instances instances.scale.json` on `remote_direct_restricted` hosts
for this reason.

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
