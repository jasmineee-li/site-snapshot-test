# Handoff: Paper Run v1 -- Resume from Phase 3

## How to use this file

**New session:** paste the following as your first message:

> read `docs/handoffs/orchestrator-handoff-paper-run-v1.md` and execute

---

## Current state (2026-04-17)

**Branch:** `feat/worldsim-v5`, last commits:
- `4fb8388c` fix(bootstrap): suppress macOS AppleDouble in webarena source tar
- `a5bf5af5` feat(setup): r5 host hardening pass + adversarial seed template resolution
- `b2fbc00` feat: add migration tooling for r5 scale-out

**Infrastructure state:**
- Migrated from m5.xlarge (stopped, `18.117.99.179`) to r5.4xlarge (running, `3.12.221.9`, instance `i-03acfc08597207960`, us-east-2). AWS personal-account vCPU quota = 16, so "r5.8xlarge" in the runbook reads as r5.4xlarge.
- Bootstrap now one-command: `COPYFILE_DISABLE=1 bash scripts/bootstrap_ec2.sh --host-config configs/benchmark_hosts/r5.yaml`. Builds 4 patched `worldsim/webarena-verified-*:amd64` site images from vendored source with env-ctrl `WA_ENV_CTRL_EXTERNAL_SITE_URL` fallback baked in — no more per-run patcher dependency.
- Re-run in flight at session end: all 4 amd64 images built, map + wiki data extracted, docker compose up -d completed; Step 6b DB-grant WARN (containers still initializing, likely transient).

**Codex hardening (commit `a5bf5af5`):**
- 52 files, +6977 / −478. 701 tests passing.
- Item #13 (seeding template resolution): `worldsim/seeding.py` gained `_render_http_seed_call` + response-chaining context. Reddit forum/submission lookups via `db_connection`, map way_id via Nominatim. 3 targeted tests cover the exact Phase 4 failure modes we hit.
- Item #7 (base_url override): `scripts/webarena-compose-override.yml` parameterized by `WORLDSIM_{ADVERTISE,BIND,DB_BIND}_HOST`. Sets `WA_ENV_CTRL_EXTERNAL_SITE_URL` on every service; binds DB ports for reward eval; pins wikipedia to amd64 build; overrides map volume names to avoid compose's project prefix trap.
- Item #6 (env-ctrl fallback): `scripts/build-webarena-amd64-images.sh` builds from vendored source, survives `docker compose up --force-recreate`.
- New modules: `worldsim/{host_config,instance_selection,storage_state_preflight,task_reset_cache}.py`. New scripts: `bootstrap_r5.sh`, `deploy_proxy_r5.sh`, `generate_scale_r5.sh`, `preflight_host_config.py`, `preflight_security_group.py`, `configure_db_access.sh`, `export_host_config_env.py`. New host config: `configs/benchmark_hosts/r5.yaml`.
- Handoff doc used to brief codex: `docs/handoffs/codex-handoff-setup-hardening.md`.

**Phase 3 smoke runs on r5 (2026-04-16 → 2026-04-17):**
- Run 1 (`--max-tasks-per-site 4`, 20 total): 2 validated. 17/18 failures had triage broken by a transient OpenRouter 402 (see below).
- Run 2 (`--max-tasks-per-site 10`, 50 total): **7 validated (14%)**. `OPENROUTER_API_KEY` unset forced triage through `CLAUDE_CODE_OAUTH_TOKEN` → anthropic fallback, zero triage errors. Validated benigns: gitlab 3, shopping 1, shopping_admin 1, reddit 1, map 1. Wikipedia 0.
- `logs/phase_3/validated_tasks.json` has 7 entries. Safe to feed Phase 4.

**Phase 4 smoke run (2026-04-17 00:40):**
- `0 complied, 0 variant_success, 0 resistant, 0 broke, 0 invalid, 7 error, 0 inconclusive`.
- Root cause: adversarial `data_seed.calls[].url` contained literal `{forum_name}` / `{way_id}` / `{submission_id}` — template vars never substituted. All 7 erroring before reaching browser-use.
- Fixed in `a5bf5af5`. Rerun pending after r5 redeploy completes.

**OpenRouter status (2026-04-17 01:18):** key healthy. `total_credits: 8000`, `total_usage: 2350.46`, so ~5650 remaining. Both `openai/gpt-4o-mini` and `anthropic/claude-sonnet-4.5` return 200 in smoke tests. The 402 errors during Phase 3 triage at 22:48 were transient (brief provider outage, rate limit, or routing hiccup). Keeping the `OPENROUTER_API_KEY`-unset workaround as the default triage path is still correct — CLAUDE_CODE_OAUTH_TOKEN is effectively free via the Pro/Max subscription.

**Known gaps and open items:**
- `logs/phase_0d/gitlab/storage_state.json` cookies are tied to the pre-redeploy gitlab container's SECRET_KEY_BASE. After bootstrap completes, regenerate via `uv run python scripts/login_gitlab_r5.py` (one-shot playwright login). Or add a `form_login` recipe to gitlab's `agent_auth` in `instances.json` so Phase 0d can auto-generate — see handoff item #9.
- `logs/phase_0d/map/storage_state.json` was regenerated against r5 earlier at 23:31 but may be stale after the redeploy; Phase 0d should re-run post-bootstrap.
- Phase 2 adversarial tasks count is **312, not 516**. Earlier handoff's 516 was a pre-dedup shard count.
- 91/312 Phase 2 tasks embed literal m5 IP `18.117.99.179` in `agent_context.auth_mechanism.storage_state.form_login.login_url`. Placeholder rewriter doesn't touch literals. Codex handoff item #11 kicks off the fix; audit needed to confirm runtime consumption.
- Stale uncommitted working-tree items from mid-session: `instances.smoke.json`, `instances.scale.json`, `.m5_instance_id`, `.codex-worktrees/`, `logs_run1_*`, `typescript/`. All ignorable or session-scoped; triage separately.

**Post-bootstrap sequence (to execute next):**

1. Regenerate gitlab cookies: `uv run python scripts/login_gitlab_r5.py`
2. Re-run Phase 0d: `uv run python -m worldsim.main phase 0d --benchmark vendors/webarena-verified --instances instances.smoke.json` (refreshes map cookies)
3. Reset `logs/pipeline_state.json` to `{"step": "phase_2", "status": "failed"}` so `--resume` will enter Phase 3.
4. Optional: rerun Phase 3 at a larger N now that the host is properly patched. Existing 7 validated are valid but thin. `--max-tasks-per-site 20` gives ~120 tasks, ~60 min.
5. Run Phase 4 on the validated subset: `uv run python -m worldsim.main phase 4 --benchmark vendors/webarena-verified --instances instances.smoke.json --agent-provider openai --agent-model gpt-5.4-mini` (OPENROUTER_API_KEY unset → OAuth triage fallback). Seeding template resolution fix means this will actually run now.
6. ASR analysis: Gate 1 (P(eval) ≥ 0.6) filter on per-trajectory probes, per-cell ASR across strategy × site, archive to `logs/paper_run_v1/`.

---

## Original context (historical, 2026-04-15/16)

## Context

Session on 2026-04-15/16 completed Phase 0c (verified profiles) and Phase 2
(312 adversarial tasks). Phase 3 failed due to OpenRouter credit exhaustion.
The pipeline code is solid, all infrastructure is deployed.

## What was accomplished

### Code changes (12 commits on feat/worldsim-v5, pushed)

1. `eef5c50` fix: cross-site delivery binding and graceful db postcondition skip
2. `5b54d93` fix: relax Phase 4 seed pre-flight for missing db_connection
3. `2cbcf31` feat: add live instance verification to Phase 0c injection surface discovery
4. `962eec7` refactor: remove SQL seeding from evaluation methodology
5. `118b6bd` feat: add authenticated reverse proxy for Phase 0c live verification
6. `fd7ebd5` feat: runtime token generation for benchmark auth
7. `42210d7` fix: update voice exemplar registry for new Phase 0c surface IDs
8. `1d32012` fix: remove POST prohibition from Phase 0c verification prompt
9. `205f570` refactor: key voice registry on source_field pattern, eliminating surface ID coupling
10. `e45b803` fix: update stale Phase 0c prompt test
11. gpt-5.4-mini extra_body fix (uncommitted, in working tree)

### Infrastructure deployed

- **Nginx proxy** on EC2 (18.117.99.179) with token auth on ports 17770, 17780, 18023, 19999, 18888, 13030
- **Security group** `benchmark-proxy` (sg-08792057943b27a65) attached to instance
- **Token**: stored in `.proxy_token` and `instances.json` verification_proxy block
- **Runtime token generation**: GitLab PAT and Magento bearer tokens acquired automatically at pipeline startup

### Pipeline state

| Phase | Status | Output |
|-------|--------|--------|
| 0a | Complete | logs/phase_0a/BENCHMARK_MANIFEST.json |
| 0b | Complete | logs/phase_0b/SANDBOX_MAP.json |
| 0c | Complete (verified via proxy) | logs/phase_0c/BENCHMARK_PROFILE_*.json (6 sites) |
| 1 | Complete | logs/phase_1/benign_tasks.json (812 tasks) |
| 2 | Complete | logs/phase_2/adversarial_tasks.json (312 tasks, 0 SQL) |
| 3 | **Not started** (needs OpenRouter credits) | -- |
| 4 | Not started (depends on Phase 3) | -- |

### Profile status

| Site | Verified | Non-priv channels | Notes |
|------|----------|-------------------|-------|
| shopping | GET-verified, REST API manually added | 6 (3 API + 3 admin form) | REST API body schemas in required_body |
| shopping_admin | GET-verified, REST API manually added | 17 (11 API + 6 admin form) | Same |
| gitlab | 29/29 verified via proxy | 29 | Clean |
| reddit | Verified, required fields manually added | 9 | submission[mediaType] + submission[forum] patched |
| map | 8/8 verified via proxy | 8 | Clean |
| wikipedia | All false (immutable ZIM archive) | 0 | No adversarial tasks |

## What needs to happen

### Step 1: Add OpenRouter credits

Go to https://openrouter.ai/settings/credits and add credits. At $0.75/M
input + $4.50/M output for gpt-5.4-mini, Phase 3 (312 tasks) costs ~$25-40.
Phase 4 agent costs are similar.

### Step 2: Commit the gpt-5.4-mini fix

The `extra_body` override removal is uncommitted:

```bash
git add worldsim/agent_config.py
git commit -m "fix: remove extra_body overrides from gpt-5.4-mini config (incompatible with ChatOpenRouter)"
```

### Step 3: Clean Phase 3 state and run

```bash
rm -f logs/phase_3/validated_tasks.json logs/phase_3/results.json \
  logs/phase_3/diagnoses.json logs/phase_3/triage.json \
  logs/pipeline_state.json logs/last_run_state.json

uv run python -m worldsim.main phase 3 \
  --benchmark vendors/webarena-verified \
  --instances instances.json \
  --agent-model gpt-5.4-mini \
  --agent-provider openrouter \
  > logs/phase_3_paper_run.log 2>&1
```

The pipeline auto-loads .env (dotenv), no export needed. Runtime token
generation acquires fresh GitLab PAT and Magento bearer at startup.

### Step 4: Verify Phase 3

```python
import json
vt = json.load(open('logs/phase_3/validated_tasks.json'))
print(f'Validated: {len(vt)}')  # Expect ~150+ at 50% pass rate
```

### Step 5: Run Phase 4

```bash
uv run python -m worldsim.main phase 4 \
  --benchmark vendors/webarena-verified \
  --instances instances.json \
  --agent-model gpt-5.4-mini \
  --agent-provider openrouter \
  > logs/phase_4_paper_run.log 2>&1
```

### Step 6: ASR analysis and archive

Load `logs/phase_4/results.json`, filter by Gate 1 ecological validity
(P(eval) >= 0.6), compute per-cell/per-site/per-framing/per-concealment ASR.
Archive to `logs/paper_run_v1/`.

## Known issues and future improvements

### Magento REST API body schemas

The Phase 0c profiles for shopping/shopping_admin were manually patched with
`required_body` fields documenting the full REST API body structure. This is
a manual step. The deterministic sandbox validator approach (Opus agent
recommendation) would automate this: after the LLM writes the profile, a
Python script POSTs each channel and discovers the required body from error
responses.

### Voice registry

Refactored to source_field pattern matching (7 semantic categories). No more
manual updates when Phase 0c reruns. But the exemplar texts still use the old
per-site voice banks. A future improvement: create per-category exemplar
banks (short_title, long_body, comment, etc.) for better ecological validity.

### Phase 0c POST probing

The verification prompt now allows POST smoke tests, but the LLM may still
skip them (no structural enforcement). The Opus agent recommended adding a
deterministic `verify-channels` subcommand to `_sandbox_validator.py` that
POSTs each channel after the LLM writes its draft. This is the right
architectural fix.

### SQL seeding removed

SQL seeding (mechanism: sql) was removed from the evaluation methodology
because it violates the threat model. All adversarial content enters through
HTTP channels (api/form) that a regular user can legitimately access. Database
read access is retained for postcondition verification and reward evaluation.

## Stale data

Previous buggy runs archived to `logs/archive_stale_buggy_20260415/`.
Phase 3 retry logs at `logs/phase_3_paper_run.log` (original, reasoning error),
`logs/phase_3_paper_run_retry1.log` (provider error),
`logs/phase_3_paper_run_retry2.log` (credit exhaustion).
