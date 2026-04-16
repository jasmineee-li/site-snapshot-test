# Handoff: Paper Run v1 -- Resume from Phase 3

## How to use this file

**New session:** paste the following as your first message:

> read `docs/handoffs/orchestrator-handoff-paper-run-v1.md` and execute

---

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
