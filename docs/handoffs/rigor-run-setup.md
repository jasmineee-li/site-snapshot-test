# Rigor-run setup runbook

One-page reference for going from a fresh EC2 host to a Phase 4 rigor run.
Codifies the manual r5 setup from 2026-04-20 so the next host runs with
zero hand-patching.

## Sequence

1. **`scripts/bootstrap_r5.sh`** (or equivalent for the target host).
   Generates the scale compose, preflights the SG, brings benchmark
   containers up with env-ctrl responding.
2. **`scripts/setup_phase4_on_host.sh`** — idempotent, 7 steps:
   1. uv + repo venv + evaluator venv (`packages/worldsim-webarena-verified`).
   2. Playwright Chromium + system libs.
   3. one `pvpo-chrome` Docker container per configured `pvpo_cdp_url`
      (content-hash stamped; rebuilds only when
      `worldsim/docker/chrome-headless-shell.Dockerfile` changes; containers
      are `--restart unless-stopped` because Phase 4 recycles the browser
      process after every task).
   4. Artifact sync (`logs/phase_0c`, `logs/phase_2/adversarial_tasks.json`,
      `logs/phase_3/contracts.json`). Prefer `aws s3 sync` from
      `s3://benchmark-archives/worldsim-runs/<run_id>/` — same-region intra
      -AWS transfer, no egress, no SSH key dependency.
   5. Mint gitlab Phase 0d `storage_state.json` (skipped when present).
   6. `sync_magento_base_urls.py` across every shopping replica using
      `config:set --lock-env` (writes `app/etc/env.php`, top of Magento's
      precedence chain).
   7. `pytest -m preflight tests/preflight` — the pass/fail gate.
3. **Launch**:
   ```
   uv run python -m worldsim.main phase 4 \
       --instances instances.scale.json --resume
   ```

## What the preflight covers

`tests/preflight/test_phase_4_preflight.py` asserts each of:

| check | failure remediation |
|---|---|
| every configured `pvpo_cdp_url` reachable and unique | rerun setup step 3 |
| each loopback `pvpo-chrome-<port>` has restart policy `unless-stopped` | rerun setup step 3 |
| Magento base_url matches proxy origin | rerun `sync_magento_base_urls.py` |
| `logs/phase_0d/gitlab/storage_state.json` has cookies | rerun `login_gitlab_r5.py` (setup step 5) |
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
renderer state can keep consuming CPU after the task ends. The runner now sends
CDP `Browser.close` at task teardown and waits for `/json/version` to disappear
and return on the same port. Docker's restart policy supplies the fresh process.
Use `WORLDSIM_PVPO_BROWSER_RECYCLE=0` only for local smoke tests against a
manually-started Chrome without restart supervision.

## Magento base_url drift: root cause + defense in depth

Root cause: `scripts/generate_compose_scale.py:96` baked
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

## When something breaks mid-run

- **PVPO `max_coverage == 0` on every trajectory**: one or more per-instance
  containers are not reachable or the instances file points multiple workers
  at the same endpoint. Check `pytest -m preflight tests/preflight`, then
  inspect the corresponding `docker logs pvpo-chrome-<port> | tail -40` and
  `curl http://127.0.0.1:<port>/json/version`.
- **Host load climbs after each completed trajectory**: check
  `browser_runtime.json` for `pvpo_browser_recycle_status`. Anything other
  than `recycled` means the browser process was not observed restarting; rerun
  setup step 3 and inspect the `pvpo-chrome-<port>` restart policy.
- **Magento 502 / base_url shows `3.12.221.9:7770`**: the compose env var
  regression returned. Run `docker exec webarena-verified-shopping_0 env
  | grep WA_ENV_CTRL_EXTERNAL_SITE_URL` — must show the proxy port.
- **Evaluator subprocess error in rewards**: the evaluator venv isn't
  synced. `cd packages/worldsim-webarena-verified && uv sync --locked`.
- **Gitlab task fails with `AuthArtifactMissingError`**: the auto-heal
  didn't kick in. Check the `WORLDSIM_AUTO_MINT_STORAGE_STATE` env var
  and that the site has `form_login` configured in `instances.json`.
