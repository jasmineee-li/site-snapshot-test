# Fresh-host bootstrap

## The sequence

From a bare r5 instance to Phase-4-ready requires a specific order. `scripts/setup_phase4_on_host.sh` codifies everything the operator had to do by hand on the 2026-04-20 r5 setup, so the short version is "run the script". But when it fails, you need to know which step.

Order matters:

1. **Bootstrap.** `scripts/bootstrap_r5.sh` (or `bootstrap_ec2.sh` on older instances) installs system deps, pulls the benchmark Docker images, and starts env-ctrl on the instance. This is a prerequisite; `setup_phase4_on_host.sh` assumes it has already run and all benchmark containers are up with env-ctrl responding.
2. **uv + venvs.** The orchestrator uses `uv`; the WebArena-Verified evaluator ships as its own package with its own venv (`worldsim-webarena-verified`). Both must resolve before anything downstream works.
3. **Playwright system deps.** Browser-Use runs Playwright against chromium; Playwright needs libnss3, libxss1, etc. on Linux.
4. **pvpo-chrome container.** The `chrome-headless-shell` Docker container (see `rigor-containers.md`) must be built and running with the right flags before Phase 4 can compute non-zero PVPO coverage.
5. **Artifact sync.** Phase 0c / Phase 2 / Phase 3 artifacts (task lists, AGENT_CONTEXT, adversarial task JSON) have to land on the host. `--artifacts-source s3://benchmark-archives/worldsim-runs/<run-id>/` pulls them; if omitted, the script looks locally and fails loudly if anything's missing.
6. **Storage_state mint.** GitLab authenticated sessions ship as `storage_state.json` per site, minted in Phase 0d. If the host's `orchestrator_host` differs from the one the state was minted against, the cookies will not validate. The 2026-04-03 fix (unconditional Phase 0d re-mint bound to `orchestrator_host`) handles this, but preflight will catch it if the re-mint is skipped.
7. **Preflight gate.** `pytest -m preflight tests/preflight`. This must pass before the script exits green. What it proves, from `tests/preflight/test_phase_4_preflight.py`:
   - Every configured PVPO CDP endpoint is reachable and uniquely assigned (no two instances sharing `127.0.0.1:9222`).
   - GitLab `storage_state.json` exists and has non-empty cookies.
   - The `worldsim-webarena-verified` evaluator venv resolves.

The script accepts `--skip-pvpo-container` / `--skip-gitlab-mint` flags for partial re-runs during debugging. Don't use them on a fresh green-field setup.

## CLI reference

```bash
scripts/setup_phase4_on_host.sh \
  --host-config configs/benchmark_hosts/r5.yaml \
  --instances instances.scale.json \
  --artifacts-source s3://benchmark-archives/worldsim-runs/<id>/
```

Flags:

- `--host-config` (required) — the host YAML that carries `HOST_IP`, `orchestrator_host`, etc.
- `--instances` (default `instances.scale.json`) — the `BenchmarkConfig` the orchestrator reads.
- `--artifacts-source` — S3 URI for a prior-run's artifact bundle. If omitted, local artifacts are used.
- `--skip-pvpo-container` — skip pvpo-chrome container step (for debugging).
- `--skip-magento-sync` — vestigial flag from pre-WASP-alignment scope; Magento is out of scope as of 2026-04-21, so this is a no-op on modern branches.
- `--skip-gitlab-mint` — skip Phase 0d storage_state re-mint (debug only).

## S3 archive fallback

Benchmark archives live at `s3://benchmark-archives/webarena/` in `us-east-2` — ~265 GB of WebArena setup tars. If bootstrap fails and the benchmark containers won't come up from scratch, `scripts/restore_benchmark_archives_from_s3.sh` restores the tarballs in ~10-15 minutes. This is the "instance got wiped" recovery path, not the normal setup path.

## When preflight fails

Read the pytest output. Each assertion names which invariant broke:

- **PVPO CDP unreachable.** The pvpo-chrome container didn't start or is bound to the wrong port. Check `docker ps` and the container logs.
- **PVPO CDP collision.** Two instances configured with the same endpoint. Fix `instances.scale.json` so each entry has a unique `pvpo_cdp_url`.
- **Storage_state empty or missing.** Phase 0d minter didn't run or crashed. Re-run with `--skip-gitlab-mint=0` (the default).
- **Evaluator venv unresolvable.** `uv sync` in the evaluator subtree failed or the venv was never created. Run the sync manually and inspect the error.

Do not "fix" a preflight failure by editing `feasibility.status` downstream or by skipping the preflight flag. Fix the underlying issue.
