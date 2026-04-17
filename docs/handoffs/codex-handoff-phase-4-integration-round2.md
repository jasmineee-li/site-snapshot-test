# Codex handoff — Phase 4 integration round 2

**Branch**: `feat/worldsim-v5`
**Base commit**: `af5f2de9` (after item #14 landed + 4 followup fixes tonight)
**Scope**: fix the remaining resolver-vs-live-instance integration issues so a demo-grade Phase 4 run (6 adversarial tasks unlocked by 7 validated benigns, minus the map-quarantined one) completes with non-zero `variant_success + resistant + complied + broke`.
**Out of scope**: Phase 3, agent model choice, paper-grade scaling. Those come after this handoff lands.

---

## Context — what's already fixed, why we're still not green

Tonight, after item #14 (pure-create seed contract) landed in `917211e4`, 5 followup patches shipped:

| SHA | Fix |
|---|---|
| `db365ca9` | Migration stripped duplicate `{{PAYLOAD_TEXT}}` from redundant top-level `body` (65 tasks affected) |
| `0c289c51` | `validate_seed_template_contract` updated to count placeholder in `body + target.create + target.update` per api_call (item #14's authoritative location) |
| `11d9eb37` | Reddit `_auth_username` expanded to read `agent_auth.authentication.credentials.username` (the path real instances.smoke.json uses) |
| `af5f2de9` | `_apply_http_seed_call` catches `psycopg2.OperationalError` under `WORLDSIM_SKIP_DB_POSTCONDITION=1` — soft-skip when site's DB isn't TCP-exposed |

After those fixes, Phase 4 reached the resolver-apply stage for all 6 tasks. Final summary at `05:22:13`:

```
Phase 4 failed — 6 tasks: 0 complied, 0 variant_success, 0 resistant, 0 broke,
                          0 invalid, 2 seed_preflight_mismatch, 4 error, 0 inconclusive
```

The plumbing works. What's left is 3 distinct runtime-integration defects in the live-instance assumptions, summarized below.

All of these would have been caught by a real `@pytest.mark.integration` suite run against the live r5 stack. The handoff deferred that suite to "local run after each resolver lands" and it never ran. **That testing gap is the root cause of tonight's whack-a-mole.** Addressing it properly is part of this round-2 handoff (see §4).

---

## 1. Gitlab `_ensure_project` POST 400 — resolver assumes idempotent-or-fresh (P0)

**Observed** (`logs/phase_4_demo_v7.log`, tasks `adv-005`, `adv-177-error-recovery-plaintext`, `adv_gitlab_011`, 3 of 6 failures):

```
worker 0 failed task adv-005: 400 Client Error: Bad Request
  for url: http://3.12.221.9:8023/api/v4/projects
  File ".../worldsim/seed_resolvers/gitlab.py", line 153, in _ensure_project
    project = _gitlab_request_json(...)
```

**Hypothesis**: `_ensure_project` does a POST to create `webagent-task-<task_id>` without first checking whether a project with that path already exists for `byteblaze`. On a re-run (or when any prior attempt left projects behind), the POST collides → 400.

Alternative: gitlab rejects the `webagent-task-<task_id>` slug because task_id contains characters gitlab doesn't accept in project paths (e.g., `-`, uppercase, underscores in a way gitlab disallows). Run the actual POST body against a live gitlab manually to see the real error payload.

**Fix path**:
1. Check the live instance for an existing project matching the target's `name_template` / `path_template` before POSTing. `GET /api/v4/users/:user_id/projects?search=<template>` returns matches; reuse if present.
2. If not present, POST. On 400, parse the response body's `message` field — gitlab returns structured validation errors like `"has already been taken"` or `"can contain only letters"`. Map those to `ResolverError(kind, detail)` with a specific kind so preflight diagnoses it cleanly.
3. Sanitize the path template: `re.sub(r"[^a-zA-Z0-9-]", "-", task_id)`, collapse runs, strip leading/trailing `-`. Gitlab path slugs are strict.

**Verify**: add an integration test (`tests/integration/test_seed_resolver_gitlab_live.py`, marked `@pytest.mark.integration`) that:
- Runs against a live gitlab from env var `LIVE_GITLAB_URL` + `LIVE_GITLAB_TOKEN`.
- Creates a project twice in a row with the same name_template, asserts second call reuses first (no 400).
- Creates a project with a task_id that has special chars, asserts the sanitized slug is accepted.

---

## 2. Shopping MariaDB grant doesn't include our host (P0)

**Observed** (`logs/phase_4_demo_v7.log`, task `ADV-007`, 1 of 6):

```
worker 0 failed task ADV-007: 
  pymysql.err.OperationalError: (1130, 
  "Host 'nat-128-84-124-0-13.cit.cornell.edu' is not allowed to 
  connect to this MariaDB server")
```

Note: the 5 resolvers and seeding code DO reach MariaDB — this isn't the 127.0.0.1-bind issue we softened with `WORLDSIM_SKIP_DB_POSTCONDITION`. This is gitlab's postgres problem inverted: shopping's MariaDB IS externally exposed but `configure_db_access.sh` on the host grants access only to a hardcoded allow-list that doesn't include dynamic NAT'd home IPs.

**Hypothesis**: `scripts/configure_db_access.sh` runs `GRANT ... TO 'magentouser'@'<specific_host>'` with a fixed list (probably `127.0.0.1`, `localhost`, maybe the r5's own IP). Our orchestrator connects from the local laptop through NAT, which presents as a Cornell CIT host that's not in the grant.

**Fix path**:
1. Read `scripts/configure_db_access.sh` — confirm the grant list.
2. Add a wildcard `@'%'` grant for each of the DB users (`magentouser`, `gitlab`, `postmill`, `renderer`). Risk: opens DB access wider than ideal, but these are test-instance DBs inside a private VPC security group; acceptable.
3. Alternatively: detect the orchestrator's public IP at Phase 4 startup and grant from that specific host. Brittle when the IP shifts.
4. Alternatively alt: route DB connections through the SSH tunnel we already use for scp. Adds complexity; skip.

Do option 2 (wildcard `@'%'`). Document the security trade-off in the script's header comment.

**Verify**: `nc -z 3.12.221.9 3306` from the orchestrator's location, then `python3 -c "import pymysql; pymysql.connect(host='3.12.221.9', port=3306, user='magentouser', password='MyPassword', database='magentodb').cursor()"` — returns a live connection.

---

## 3. 2 tasks fail preflight (reddit) — likely still an auth-path or base-state mismatch (P1)

**Observed**: the Phase 4 summary shows `2 seed_preflight_mismatch`. The specific tasks aren't in the error log because preflight mismatches are recorded as result status, not as ERROR lines. Inspect `logs/phase_4/<latest_run>/results.json` (or the equivalent Phase 4 output structure) to find the specific tasks + mismatch `detail`.

**Hypothesis candidates**:
- Reddit's submission/comment resolver has a base-state probe that still fails for a path unrelated to `user_bio` (the path we fixed in `11d9eb37`).
- A reddit task targets a resource_type the resolver doesn't handle (e.g., `forum_moderator_edit` or similar) and raises `ResolverError("unsupported_resource_type")` during dry-run.
- A gitlab task has a base-state probe for a resource it can't resolve (e.g., `group` or `repo_file` which we noted in the earlier review as having limited resolver coverage).

**Fix path**:
1. Read `logs/phase_4/<latest>/results.json` to find the 2 preflight-mismatch task IDs and their `mismatch.detail` strings. Post those to the handoff PR before picking a fix.
2. If they're `unsupported_resource_type`: either implement the missing resource type in the resolver, OR move the offending tasks to the quarantine file alongside map until a proper resolver lands. Prefer the former for tasks covering types already in scope; prefer the latter for types genuinely out of scope (e.g., admin-only endpoints).
3. If they're base-state-probe failures: fix the probe path (same style as `11d9eb37`).

**Verify**: rerun Phase 4 with the same 6 tasks; preflight mismatch count drops to 0 or the mismatched tasks are moved to quarantine and drop out of the "N/236 adversarial tasks have validated benign counterparts" count cleanly.

---

## 4. Testing strategy — the real structural fix (P0 for durability)

The first three items above are tactical. This one is architectural. **The absence of integration tests is why tonight was whack-a-mole.**

Item #14's §14.11 called for `@pytest.mark.integration` tests per site, gated behind `LIVE_INSTANCE_URL` env var, runnable via `scripts/run_integration_tests.sh`. Codex shipped the handoff item without creating either. Unit tests mock every external dependency, so each of the 5 fixes tonight (template duplication, validator, reddit auth, DB postcondition, and the 3 remaining) passed 798 green unit tests the moment it was written.

**Fix path**:

1. Create `scripts/run_integration_tests.sh` that reads a `configs/benchmark_hosts/r5.yaml` (or user-specified) host config, sets `LIVE_INSTANCE_URL_<site>` env vars, and invokes `uv run pytest -m integration tests/`.

2. Create per-site integration test files:
   - `tests/integration/test_seed_resolver_gitlab_live.py`
   - `tests/integration/test_seed_resolver_shopping_live.py`
   - `tests/integration/test_seed_resolver_shopping_admin_live.py`
   - `tests/integration/test_seed_resolver_reddit_live.py`
   - (map stays quarantined per §15)

   Each exercises: auth token acquisition, resolver create() against live service, get-or-create idempotency (run twice, second call should reuse), resolver update() against real singleton, preflight dry-run, postcondition verification with real DB.

3. Wire into a CI job (GitHub Actions or similar) that spins up the webarena stack via docker compose, runs integration tests, tears down. Not blocking for tomorrow's demo; blocking for the next time anyone touches resolver or seeding code.

4. Document in `CLAUDE.md` or a new `docs/testing.md`: "Any PR that modifies `worldsim/seed_resolvers/**`, `worldsim/seeding.py`, or `worldsim/phases/phase_4_adversarial.py` must run `scripts/run_integration_tests.sh` locally and paste output in the PR description."

**Verify**: integration tests catch the 5 issues we fixed tonight when run against the `af5f2de9` HEAD. Rolling back any of the 5 fixes should cause the corresponding integration test to fail.

---

## Acceptance criteria for round 2

After codex lands this round:

1. `uv run pytest tests/ -q` — 801+ pass, 2 skipped. No regressions.
2. `scripts/run_integration_tests.sh` against live r5 — all pass (or explicit SKIP with reason for any that require fixture state we don't have).
3. Demo Phase 4 run:
   ```
   set -a && source .env && set +a
   unset OPENROUTER_API_KEY
   export WORLDSIM_SKIP_DB_POSTCONDITION=1   # if gitlab's postgres still 127.0.0.1-bound
   uv run python -m worldsim.main phase 4 \
     --benchmark vendors/webarena-verified \
     --instances instances.smoke.json \
     --agent-provider openai --agent-model gpt-5.4-mini
   ```
   Expected: `Phase 4 complete — 6 tasks: N complied, M variant_success, K resistant, ... 0 error, ≤2 seed_preflight_mismatch`.
4. Commit the DB grant widening + any gitlab path-slug sanitizer + reddit preflight fix.
5. Keep `WORLDSIM_SKIP_DB_POSTCONDITION` as a flag (don't rip it out) — useful for future DBs that aren't TCP-exposed.

---

## Commit chain tonight (for your reference)

- `917211e4` feat(seed): item #14 pure-create seed contract + resolver package + dataset migration
- `db365ca9` fix(seed-migration): remove duplicate {{PAYLOAD_TEXT}} placeholder
- `0c289c51` fix(phase2): update seed_template validator for item-#14 target contract
- `11d9eb37` fix(resolver-reddit): expand _auth_username to read agent_auth credentials
- `af5f2de9` fix(seeding): soft-skip DB postcondition verification when DB unreachable

All pushed to `origin/feat/worldsim-v5`.

---

## Why this is not a codex-quality problem

Codex shipped the item #14 spec faithfully and followed the handoff's test-coverage instructions. The handoff itself deferred integration tests. The defects we hit were in the spec's grey areas (credential path conventions, DB reachability, gitlab idempotency semantics, host grants) that unit tests with mocks can't reach. The fix is to add the integration tests that should have existed before item #14 landed, so the NEXT contract change doesn't go through the same discovery cycle.
