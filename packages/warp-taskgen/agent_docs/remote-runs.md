# Remote Runs and Host Operations

Use this before fresh-host setup, r8a work, remote jobs, proxy changes, Phase 4
rigor runs, or live integration debugging. It is an operator route, not a
second source for script help; use each command's `--help` for flags.

## Canonical flow

1. Validate the ignored operator config
   `configs/benchmark_hosts/r8a.local.yaml` and audit the selected host.
2. Transfer the accepted checkout with `scripts/sync_to_host.sh`; this is an
   operational deployment, not a second writable Taskgen source.
3. Start a named job with `scripts/remote_job_start.sh`, an explicit
   `--state-dir`, and an `--expected-output` artifact.
4. Watch `scripts/remote_job_status.sh` first; use
   `scripts/remote_job_tail.sh` only after status identifies stale or failed
   output.
5. Stop by job id with `scripts/remote_job_stop.sh` and preserve the registry.

Syncing while a registered job runs mixes checkout versions across phases.
Use `--allow-active-jobs` only for deliberate maintenance with the provenance
impact recorded. Keep secrets in the pushed environment, never in commands or
shell history.

## Launch contract

The canonical selected-host shape is:

```bash
scripts/remote_job_start.sh \
  --host-config configs/benchmark_hosts/r8a.local.yaml \
  --remote-dir /srv/warp-taskgen \
  --name <short-job-name> \
  --state-dir logs/<run> \
  --expected-output logs/<run>/phase_4/results.json -- \
  bash -lc 'cd /srv/warp-taskgen && uv run warp-taskgen phase 4 \
    --instances instances.scale.json \
    --phase-4-variant-system eval-awareness-iterator \
    --phase-4-eval-awareness-max-iterations 3 \
    --agent-task-timeout 900'
```

Pass an explicit whole-trajectory `--agent-task-timeout` on remote Phase 4
jobs. LLM and step timeouts do not bound session start, CDP, or cleanup
deadlocks. The default variant system is `eval-awareness-iterator`; legacy
comparability uses `strategy-variation` with the named `adaptive-3-3-1`
budget.

For high-concurrency Browser Use, keep expensive DOM/screenshot CDP calls under
their separate caps and inspect `browser_runtime.json` before increasing the
outer worker cap. Top-level Phase 4 uses `--phase-4-max-workers`; the
process-pool wrapper alone owns `--workers`.

Process-pool runs preserve the normal admission, seeding, PVPO, TP, VEA,
iterator, reward, and readback contracts. A complete pool writes exactly one
valid result per expected task to `phase_4/results.json`; missing, duplicate,
or mismatched task ids fail closed. `results.partial.json` and
`partial_manifest.json` are inspection artifacts, not paper-eligible results.
Repair into a separate run with
`scripts/repair_process_pool_partial.py --help`; do not edit partial output.

## Locality and topology

Treat instance files as execution-locality contracts:

| Caller | Origin | Instance view |
| --- | --- | --- |
| Phase 0c browser probes | Modal sandbox | `instances.smoke.json` or an equivalent public/proxy view |
| Phase 0c host inventory | selected host orchestrator | `--host-inventory-instances instances.scale.json` |
| Phase 2c render checks | selected-host browser | `instances.scale.json` |
| Phase 4 agent/PVPO | selected-host browser | `instances.scale.json` |

The mixed Phase 0c path needs both files when fresh GitLab/Reddit inventory is
generated: `--instances instances.smoke.json
--host-inventory-instances instances.scale.json`. Do not rewrite only the host
portion of a smoke file; replica DB/API ports and inventory must come from the
selected scale topology.

Generated instances, fragments, compose files, and proxy maps are host-local
and gitignored. Regenerate them from the selected host config. Browser-facing
ports must avoid Chromium's restricted-port list; repair topology generation,
not task seeds or placement-fix variants. A topology mismatch is an
infrastructure symptom, not a carrier, prompt, or strategy verdict.

## Fresh host, proxy, and lifecycle

On a fresh host, run:

```bash
scripts/setup_phase4_on_host.sh \
  --host-config configs/benchmark_hosts/r8a.local.yaml \
  --instances instances.scale.json \
  --artifacts-source s3://benchmark-archives/worldsim-runs/<run_id>/
```

The setup sequence regenerates topology, syncs Phase 0c/2/3 artifacts, mints
host-bound GitLab storage state, and runs `pytest -m preflight tests/preflight`.
That preflight proves non-empty storage state and evaluator-venv resolution; it
does not prove live benchmark task connectivity. Current PVPO is
page-surface-stable on the runner-owned browser and needs no dedicated browser
container or PVPO endpoint. Historical `pvpo_cdp_url` metadata is ignored; no
legacy browser route is active.

The nginx source of truth is `scripts/deploy_benchmark_proxy.sh`. Check parity
without hand-editing the generated host file:

```bash
scripts/check_proxy_drift.sh \
  --host-config configs/benchmark_hosts/r8a.local.yaml \
  --topology scale --insecure-http --verify-runtime
```

Token auth is `X-Worldsim-Token`; proxy ports come from
`scripts/proxy_ports.conf`. Phase 0c may use the proxy, while on-host Phases 3
and 4 use the real `site_url` and `reset_endpoint` values in the selected
instances file.

When no rigor sweep is running, park the host with
`scripts/host_park.sh --host-config configs/benchmark_hosts/r8a.local.yaml` and
resume with `scripts/host_resume.sh --host-config configs/benchmark_hosts/r8a.local.yaml`.
Read the lifecycle policy in
`docs/infra/r8a-control-plane.md` before disabling an auto-stop layer.

## Resume and evidence

After compaction or a pause, verify the instance pool before reusing state:

- on-host Phase 2c/4 uses regenerated `instances.scale.json`;
- saved benchmark metadata matches that pool;
- host-unreachable, public-IP navigation, and host-bound cookie errors are
  investigated as topology symptoms first.

For Phase 2a planning, Phase 2b text fill, or a Phase 4 run, request a cooperative pause without
terminating the remote job. Target the normal run root or the process-pool
output root:

```bash
WARP_TASKGEN_STATE_DIR=<run-root> uv run warp-taskgen pause
WARP_TASKGEN_STATE_DIR=<run-root> uv run warp-taskgen status
```

Wait for `status=paused` before stopping infrastructure. `status=pausing`
means an admitted atomic unit or process-pool child is still draining. A paused
process pool prints and persists its full `scripts/run_phase4_process_pool.py
--resume ...` command; use that exact wrapper command. Generic `warp-taskgen
resume` deliberately refuses a pool root. Process-pool termination without a
cooperative pause remains the existing inspect/repair workflow. Phase 2a pause
drains admitted planning shards and resumes through their Run-bound manifests;
Phase 2b text fill drains admitted task units and resumes through exact
Run-bound checkpoints; Phase 2c feasibility and Phases 0, 1, and 3 reject
pause. Do not simulate pause by killing their workers.

A result-affecting resume override on an identified Run materializes an
isolated child and prints its exact resume command. Preserve both environment
assignments so the child state and discovery mirror stay inside that child:

```bash
WARP_TASKGEN_STATE_DIR=<child-root> \
WARP_TASKGEN_RESUME_POINTER=<child-root>/last_run_state.json \
uv run warp-taskgen resume
```

The child restarts conservatively from Phase 0a and does not inherit parent
Phase 2/4 artifacts. To archive it with the existing wrapper, use the opaque
child ID with `--logs-dir <child-root-parent>`; do not archive the whole
`.warp-derived-runs` collection recursively.

Remote metadata is under `<remote-dir>/logs/remote_jobs/<job_id>/`. Audit the
active host or archive before trusting checked-in task JSON, then apply
`agent_docs/artifacts.md` promotion rules. A setup/launch failure gets a fresh
state directory after the precondition is fixed; do not relaunch into the same
evidentiary run.

For the live PR gate, use the current host config and quiet wrapper:

```bash
scripts/run_integration_tests.sh \
  --host-config configs/benchmark_hosts/r8a.local.yaml --quiet
```

With no explicit `--instances`, the wrapper generates a host-config-specific
smoke file. Pass `--instances instances.scale.json` only for an intentional
scale-topology check.

Completion means the job has one explicit topology, state directory,
expected-output contract, and timeout; status evidence identifies the final
artifact; and any live gate or topology mismatch is recorded as infrastructure
evidence rather than model behavior.
