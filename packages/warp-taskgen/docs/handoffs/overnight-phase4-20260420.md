# Phase 4 Overnight Run — Status Summary

**Run ID:** `20260420_122257`
**Agent model:** `gemini-3-flash-preview` (via OpenRouter)
**Launched:** 2026-04-20 12:22:57 UTC
**Killed:** 2026-04-20 13:42 UTC (~80 min elapsed, at 100/174 tasks)
**Outcome:** Run aborted after OpenRouter credit exhaustion

## TL;DR

The run did not produce rigor-result data. Two orthogonal blockers corrupted the signal:

1. **PVPO zero-coverage fallback** — the chrome-headless-shell Docker container was not running on r5, so all trajectories report `max_coverage = 0.000` and route to placement-fix. This is the CLAUDE.md-documented degraded mode: "correct behavior but not a rigor result."
2. **OpenRouter 402 Insufficient Credits at T+47 min** — first 402 at 13:10 UTC. After that, agents failed-fast on every task without real execution. Of 96 tasks with a `result.json`, only ~5 had non-agent-exception outcomes; the remainder are credit-exhaustion noise, not meaningful trajectories.

## Results distribution (as saved)

- `total result.json`: 96 / 174 admitted
- `outcome_fine`:
  - `task_broke_agent_exception`: 91 (credits + PVPO cascade)
  - `task_broke_injection_unreached`: 2
  - `task_broke_wrong_answer`: 1
  - `task_broke_self_abandoned`: 1
  - `?`: 1
- `final_status`:
  - `injection_not_encountered`: 95 (all PVPO-zero)
  - `?`: 1
- `encounter.max_coverage`: nonzero = 0/95 (PVPO fallback)
- `steps`: mean 5.2, max 14

**Non-agent-exception outcomes: 5.** Only those 5 represent real task execution; the ~91 agent_exception tasks are mostly credit-failure fail-fast.

## Cost

Run cost tracked to `logs/phase_4/cost_report_20260420_082253.json`. Pre-run baseline was $0.38 across calibration attempts (4 iterations). Full-run incremental is likely under ~$15 since 402s don't bill and most agents cut out early on PVPO-zero.

## Caveats / notes on the overnight flow

Full infrastructure was built up on r5 tonight. This is NOT trivial to reproduce — document the steps before rerunning.

1. **SG port 22 opened** to `128.84.126.158/32` (rule `sgr-03428b4d595a893c7`). **Revoked post-run** (per safety plan).
2. **Repo rsynced** to `/home/ubuntu/browser-sim/` on r5, including `.env` (user-authorized; chmod 600 applied).
3. **uv installed** on r5 via `curl | sh` (user-authorized one-time).
4. **Playwright chromium system deps** installed via `sudo playwright install-deps chromium`.
5. **26-replica scale stack deployed** (shopping 4, shopping_admin 4, gitlab 8, reddit 4, map 4, wikipedia 6). Project name `webarena-verified-envs` preserved to share volume namespace with the prior smoke stack. Map replicas 1-3 crashloop on shared-RO-volume chown (non-blocking — 0 map tasks).
6. **Magento base_url hand-set to `http://127.0.0.1:PORT/`** for all 8 Magento replicas. Shopping_0 and shopping_3 silently revert after `cache:flush` — re-applied right before run launch. Pattern suggests an init hook or config-cache re-read that isn't documented.
7. **Magento health check patched** (`worldsim/phase_4/magento_health.py` on r5) to decode Luma's `\uXXXX` JS escapes in rendered BASE_URL. Was a real regex bug — should be upstreamed.
8. **gitlab storage_state minted** via `scripts/login_gitlab_r5.py` (patched HOST to 127.0.0.1:8023). Copied to `/home/ubuntu/.build/webarena-verified/logs/phase_0d/gitlab/` so the benchmark-root path check finds it. `--skip-host-bound-storage-state-auth` required since one storage_state covers 4 replicas.
9. **WORLDSIM_WEBARENA_EVAL_PYTHON** path in `.env` fixed from laptop path to r5 path (`/home/ubuntu/browser-sim/packages/warp-taskgen-webarena-verified/.venv/bin/python`). Separate venv created for the eval package.
10. **instances.trimmed.json** built from the generator output with `verification_proxy` removed (no proxy deployed) and `map` entry removed (no map tasks, no storage_state).
11. **Model slug:** `openai/gpt-5.4-mini` via OpenRouter returns 404 ("no endpoints handle these parameters") — browser-use's tool_choice/response_format params aren't supported for that slug today. Swapped to `gemini-3-flash-preview` which does work.

## To resume this run

1. **Add OpenRouter credits** (the immediate blocker).
2. **Build + run chrome-headless-shell Docker container** (per `worldsim/docker/chrome-headless-shell.Dockerfile`) — or decide that `--enable-begin-frame-control` isn't needed on Linux and patch PVPO to run without it. Required for real encounter detection (non-zero `max_coverage`).
3. **Investigate shopping_0 / shopping_3 base_url drift** — why does `core_config_data` revert for those two replicas specifically? Scale_config.yml's `WA_ENV_CTRL_SKIP_RECONFIGURE: "true"` should prevent init-time overrides.
4. **Re-open SG port 22** to your laptop IP (rule was revoked tonight, need a new authorize).
5. **Confirm `openai/gpt-5.4-mini` slug on OpenRouter** — either the slug is different or it requires custom param filtering. For now, `gemini-3-flash-preview` is a known-good default.
6. **Kick off with** `uv run python -m worldsim.main phase 4 --instances /home/ubuntu/scale_out/instances.trimmed.json --benchmark /home/ubuntu/.build/webarena-verified --skip-host-bound-storage-state-auth --agent-model gemini-3-flash-preview` (or update model once OpenRouter slug is confirmed).

## Files changed on laptop this session

- `instances.smoke.local.json` (new, unused — built before pivot to r5 run)
- `logs/phase_4/20260420_122257/` (new — partial trajectory data)
- `logs/phase_4/full_run_20260420_082253.log` (orchestrator log)
- `logs/phase_4/cost_report_20260420_082253.json`

## Files changed on r5 (not synced back)

- `worldsim/phase_4/magento_health.py` patch (Luma escape decode)
- `scripts/login_gitlab_r5.py` HOST const (3.12.221.9 → 127.0.0.1)
- `.env` WORLDSIM_WEBARENA_EVAL_PYTHON path

## Recommendation

Treat this as a dry-run that surfaced the real blocker list. Do not interpret the 96 result.json files as representative of model/attack behavior — the outcome distribution is dominated by credit exhaustion and PVPO fallback, not agent behavior under adversarial pressure. Rerun after items 1-3 above.
