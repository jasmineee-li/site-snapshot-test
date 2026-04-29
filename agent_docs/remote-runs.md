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

For r5 jobs that run browsers on the host, including Phase 2c render checks and
Phase 4 agent/PVPO runs, use the generated `instances.scale.json`. Do not use
`instances.smoke.json` on-host unless you have regenerated it for the host's
`orchestrator_host`. The smoke file may point at the public advertised IP; AWS
does not reliably hairpin browser traffic from the instance back to its own
public IP, which causes false `host_unreachable` render failures and host-bound
storage-state mismatches. `instances.scale.json` is generated from the host
config and uses `orchestrator_host` such as `172.17.0.1`.

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
