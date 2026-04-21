# WorldSim v5 -- Current Progress

> **Historical session log — pre-PVPO + pre-WASP-scope state.** This file is a chronological snapshot of work through 2026-04-17.
>
> **Two subsequent cutovers invalidated parts of this document:**
>
> 1. **PVPO cutover (2026-04-19)** deleted the sandbox `probe_ecological_validity` + `_run_ecological_validity_fix_loop`, the `probe-ecological-validity.md` and `fix-ecological-validity.md` prompts, the C1a directive-canary signal, and the `TROJAN-ACK-<hex>` directive token. See [`docs/handoffs/codex-handoff-paint-verified-oracle.md`](./handoffs/codex-handoff-paint-verified-oracle.md).
>
> 2. **WASP-aligned scoping (2026-04-21)** restricted the pipeline to GitLab + Reddit only and deleted shopping/shopping_admin/wikipedia/map support. Any reference below to those sites or to scripts named `setup-wikipedia-robust.sh`, `build-wikipedia-amd64.sh`, `setup-map-robust.sh`, `restore_benchmark_archives_from_s3.sh`, `fix_magento_base_url.sh`, `sync_magento_base_urls.py`, `patch_form_to_api.py`, `worldsim/editors/shopping*.py`, or `worldsim/phase_4/magento_health.py` is **historical** — those modules were deleted. The current dataset is 84 tasks (78 GitLab + 6 Reddit). See [`docs/handoffs/wasp-aligned-scoping-decision.md`](./handoffs/wasp-aligned-scoping-decision.md).

Last updated: 2026-04-17

## 2026-04-17 update

- **Phase 3 restructured to agent-free contract validity gate.** Following the WebArena-Infinity three-layer test model, Phase 3 no longer runs the target agent; it schema-validates contracts and emits `phase_3/contracts.json`. Phase 4 admits every `validity_status: "valid"` entry, and baseline capability is reported as Phase 4's `capability_benign_under_attack` on ecologically valid trajectories. The earlier triage / diagnosis / fix-loop machinery and `validated_tasks.json` / `diagnoses.json` / `triage.json` are gone; progress notes below describing them are historical.

## 2026-04-14 evening session update (newest first)

## 2026-04-14 evening session update (newest first)

For the full handoff, see `docs/handoffs/orchestrator-handoff-phase-2-v2.md`. Summary:

### What shipped this session

- **GitLab bearer auth moved to runtime PAT brokerage.** The preferred path is now `token_generator: "gitlab_pat"` with live validation and refresh, including deduped parallel acquisition across instances. The file-backed `logs/phase_0d/gitlab/personal_access_token.txt` helper remains as a legacy break-glass path only and now revalidates before reuse.
- **Phase 2 orchestrator patches landed (uncommitted):**
  - Per-shard persistence in `_generate_injections_for_site` — each validated shard writes to `logs/phase_2/shards/<label>.json` as soon as it completes. Crash-safe.
  - Fail-open publish in the main phase_2 entrypoint — partial `adversarial_tasks.json` is always written when ≥1 shard succeeds, tagged `status="partial_complete"` with `partial=true` + `generation_failures=[...]`. Only fully-failed runs (zero shards) return exit 1.
  - `asyncio` import added to `modal_sandbox.py` (the prior patch used `asyncio.gather` without the import, killing all 43 shards in the first retry attempt).
- **Modal sandbox watchdog infrastructure added externally** to `worldsim/modal_sandbox.py` (uncommitted). Adds `SandboxWatchdogState`, silence-based abort thresholds, rate-limit grace windows, and a retriable timeout error class. Still under review; retain and inspect the diff before committing.
- **Three Phase 2 rerun attempts this session, all failed with different root causes:**
  1. OAuth, 43 shards simultaneous → Modal SDK stream-wedge (hit the prior session's bug, already patched).
  2. OAuth, patched → all 43 shards completed but 12 failed to produce output due to OAuth simultaneous-burst rate-limit; orchestrator's fail-closed logic then discarded the 31 successful shards in memory. Cost ~$78.88 sunk. ← This is what motivated the fail-open + per-shard persistence patches above.
  3. OpenRouter (fallback) → 21 of 43 shards refused immediately with "I'm not going to help, even given the security research framing." ~50% refusal rate on proxy-auth path. 10 shards produced output before being killed.

### Major decision: Phase 2 v2 (plan/text decomposition) architecture

Phase 2 single-shot design is auth-coupled to OAuth because the prompt asks the model to generate adversarial content text. Proxy-auth paths (OpenRouter) evaluate the prompt under a stricter policy regime and refuse at ~50%. OAuth refuses at ~0% but is rate-limited.

**The refactor splits Phase 2 into two internal sub-stages within the single `worldsim phase 2` command, mirroring Jasmine's `behavior-gen/pipeline.py` decomposition:**

- **Phase 2a — Plan.** Pure schema work: `target_surface_id`, `framing`, `concealment`, `delivery_mechanism`, `delivery_channel`, `attack_objective`, `required_tokens`, `length_budget`, `reward_function`. NO payload text. Runs under Claude Code in Modal, OAuth-or-OpenRouter compatible (no refusal because no harm-adjacent content is emitted).
- **Phase 2b — Text fill.** Parallel structured-output API calls (not Claude Code). Per-plan: one async call that composes UGC in the plan's declared voice with required tokens. Self-healing 3-variant retry loop (`standard`, `creative_writing`, `testing_compliance`) + OAuth fallback. Empirically: OpenRouter Sonnet 4.5 ACCEPTS 3 of 4 tested framings for UGC composition.

Comprehensive spec with acceptance criteria, implementation order, and voice-exemplar design is now in `docs/TODO-adversarial-rigor-mvp.md:639+`. Codex is implementing this next. The new handoff (`2026-04-14-phase-2-v2-handoff.md`) includes a ready-made subagent verification prompt the orchestrator runs before executing anything.

### Empirical smoke test result (OpenRouter refusal asymmetry)

| Framing sent to OpenRouter Sonnet 4.5 | Result |
|---|---|
| "You are a security researcher generating adversarial indirect prompt injections..." | REFUSED |
| "Compose a single customer review in the voice of a platform-trust admin with these exact tokens..." | ACCEPTED |
| UGC composition + markdown-fenced concealment requirement | ACCEPTED |
| UGC composition + "safety eval composer" system prompt + angry-consumer voice | ACCEPTED |

Mechanism unproven; behavior reproducible. Justifies the Phase 2b decomposition without needing to close the mechanism question.

### Immediate next action

**Codex implements Phase 2 v2 per `docs/TODO-adversarial-rigor-mvp.md:639+`.** After Codex commits, the next orchestrator follows `docs/handoffs/orchestrator-handoff-phase-2-v2.md`, starting with the subagent verification check.

### Open sequence (after Codex completes)

1. Subagent verification of v2 implementation (6-section checklist in the handoff)
2. Clean up contaminated state (nuke `logs/phase_2/shards/` — OpenRouter-generated, research-integrity concern)
3. Phase 2 run (2a then 2b sequentially) → audit plans and payloads (≥60 plans, ≥20 cells, 5 sites; ≥95% first-attempt text-fill success, ≤5% OAuth fallback)
5. Phase 3 gitlab smoke (critical patch-validation checkpoint)
6. Phase 3 full on `gpt-5.4-mini` via OpenRouter (budget TBD after smoke, 3–8 hr)
7. Phase 4 full on `gpt-5.4-mini` via OpenRouter (budget TBD after smoke)
8. Per-cell ASR analysis; archive to `logs/paper_run_v1/`; produce `docs/paper_run_v1_summary.md`

### Cost ledger (sunk, updated)

| Phase / attempt | Cost | Notes |
|---|---|---|
| Phase 0a | $1.95 | |
| Phase 0c (on Opus) | $86.34 | should have been Sonnet; $60 waste |
| Phase 3 smokes (old schema) | $44.66 | |
| Phase 2 attempt #1 (OAuth, stream-wedged) | $112.25 | 5 shards rescued, later superseded |
| Phase 2 attempt #2 (OAuth, patched, rate-limit burst) | $78.88 | 31 successful shards discarded ← fixed by fail-open patch |
| Phase 2 attempt #3 (OpenRouter, refusals) | ~$2 | 21 immediate refusals |
| **Total sunk** | **~$326** | |
| **Remaining (Phase 2 v2 + Phase 3 + Phase 4)** | **~$80–150** | OAuth-subscription absorbs the majority; real spend is OpenRouter share |

### Root causes of this session's failures (for future runs)

1. **OAuth simultaneous-burst rate-limit cliff:** launching 43 shards in ~80ms triggers rate-limit rejection for ~10–12 shards that die with `Fatal error in message reader: Command failed with exit code 1`. Mitigated by `DEFAULT_SANDBOX_CONCURRENCY` semaphore (added externally, currently =250 which is effectively uncapped; recommend 4–15 for OAuth runs).
2. **Fail-closed orchestrator discarded in-memory successful shards:** the prior code's `site_failures → return 1` in `phase_2_injections.py:227-241` threw away 31 shards' worth of output when 12 failed. Fixed by fail-open publish + per-shard persistence to disk.
3. **OpenRouter proxy-auth refuses adversarial prompts:** ~50% immediate-refusal rate on the Phase 2 prompt as written. Fixed structurally by the v2 decomposition (plan stays refusal-safe, text is UGC-framed).
4. **GitLab PAT bootstrap broken on modern GitLab:** CSRF token moved from hidden input to meta tag; response moved from HTML scrape to JSON body. Fixed in commit `b1d1e1a`.

---

## Branch

`feat/worldsim-v5`, 73 commits ahead of `main`. Latest committed changes:

```
40557d1 feat: tier Phase 0c profiling and emit agent context artifacts
47985c3 feat: carry benchmark agent context into task generation and runtime prompts
ede6432 fix: restore Claude Code semantics in Modal sandbox runs
98f4576 docs: refresh the v5 spec and progress notes for the tiered rerun
```

All four changes above are now committed.

## Current Live Run State

The first run has been archived to:

`logs_run1_no_tiered_site_discovery_and_missing_benchmark_specific_task_template/`

Archive operation:

```bash
mv logs logs_run1_no_tiered_site_discovery_and_missing_benchmark_specific_task_template
mkdir -p logs
cp -r \
  logs_run1_no_tiered_site_discovery_and_missing_benchmark_specific_task_template/phase_0a \
  logs_run1_no_tiered_site_discovery_and_missing_benchmark_specific_task_template/phase_0b \
  logs/
```

After the archive, `logs/` was initially seeded with `phase_0a/` and `phase_0b/` from the archived run. However, those outputs are also invalid (see below) and must be deleted before the rerun.

Why a full rerun from Phase 0a is required:

- **`ede6432` fixed a critical sandbox runner bug.** The archived run's Modal sandboxes were not passing the `claude_code` preset through the Agent SDK. This means Phase 0a (benchmark discovery) and Phase 0c (per-site profiling) ran under generic SDK semantics, not Claude Code semantics. The resulting manifest and profiles may be unreliable.
- The archived run also predates the tiered Phase 0c implementation and the benchmark-specific `agent_context` fix.
- No archived sandbox output can be trusted as canonical for the current codebase.

Canonical rerun boundary: start again at **Phase 0a** (full pipeline from scratch).

## What's Proven (Archived Run 1)

All items below refer to the archived run at
`logs_run1_no_tiered_site_discovery_and_missing_benchmark_specific_task_template/`.
They ran against real Modal sandboxes and produced artifacts on disk.

- **Phase 0a -- Benchmark Discovery (invalidated).** Single Modal sandbox analyzed the full WebArena Verified codebase. Produced `BENCHMARK_MANIFEST.json` (6 sites, 813 tasks). However, this sandbox ran without the `claude_code` preset (`ede6432`), so the output is not trustworthy under the current contract. Must rerun.
- **Phase 0b -- Sandbox Filesystem Mapping (invalidated).** Pure local Python, no LLM. Output is deterministic given the manifest, but since the manifest must be regenerated, 0b must rerun too.
- **Legacy Phase 0c -- Per-Site Profiling (invalidated).** The archived run used both the older single-stage flow and the broken sandbox runner. Superseded by tiered Phase 0c and the Claude Code semantics fix.
- **Phase 1 -- Task Wrapping (Mode A).** Loaded 813 tasks from the WebArena Verified dataset and wrapped them into benign task bundles for downstream phases. Output: `benign_tasks.json`. No LLM required.
- **Phase 2 -- Injection Generation.** 43 Modal sandbox runs across 6 sites with per-site task sharding. Shopping (182 tasks) and gitlab (160 tasks) were the largest. Produced 671 adversarial tasks total. In-sandbox validation caught schema errors before sandbox exit. Total $82.99 spend across 624 turns.
- **Phase 3 artifact production.** The archived run produced `validated_tasks.json`, `results.json`, and `diagnoses.json` under `phase_3/`, proving the Phase 3 path executed far enough to materialize downstream artifacts.
- **Modal sandbox primitive.** `run_claude_in_sandbox` creates a sandbox, stages files via `add_local_file` / `add_local_dir`, runs Claude Code through `claude-agent-sdk` via `_sdk_runner.py`, and returns file outputs plus a `_summary` key with cost, token usage, session ID, and per-model breakdowns.
- **Three Claude Code auth paths.** OAuth (`CLAUDE_CODE_OAUTH_TOKEN`), OpenRouter (`ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_BASE_URL`), and direct API key (`ANTHROPIC_API_KEY`). Priority: OAuth > OpenRouter > API key. Named Modal secret support via `WORLDSIM_CLAUDE_MODAL_SECRET`.
- **Volume-based benchmark upload.** `upload_to_volume` does one-time content-addressed upload, skips if already populated, and eliminates per-call benchmark hashing overhead.
- **Cost tracking with resume persistence.** `CostTracker` records per-sandbox cost data, persists to `cost_report.json`, and reloads on `--resume` without double-counting.

## What's Implemented (Current Code Complete, Unit-Tested)

Everything below reflects the current working tree.

- **Phase 0c -- Tiered per-site profiling.** Profiling is now split into Tier 1 and Tier 2. Tier 1 runs three sandboxes in parallel per site: verification capabilities, data model, and agent context. Tier 2 then runs injection-surface / task-coverage analysis using validated Tier 1 outputs as inputs.
- **Artifact-specific Phase 0c validation.** `_sandbox_validator.py` now validates `VERIFICATION_CAPABILITIES.json`, `DATA_MODEL.json`, `AGENT_CONTEXT.json`, and `INJECTION_SURFACE.json` independently. Host-side retries append concrete validation errors back into the retry prompt before re-running a tier.
- **`agent_context` as a first-class artifact.** Phase 0c now emits `AGENT_CONTEXT_{site}.json` containing response-format requirements, auth details, discovered vendor prompt templates, and benchmark/site context.
- **Shared benchmark-specific prompt builder.** `worldsim/agent_prompt.py` builds runtime prompts from `agent_context`, appending auth guidance, discovered credentials, response-schema requirements, and per-task format requirements sourced from `instantiation_dict`.
- **Phase 1 Mode A propagation.** Wrapped benchmark tasks now preserve `instantiation_dict` and embed `agent_context` when Phase 0c output is available.
- **Phase 1 Mode B propagation and cache hardening.** Novel-task generation now receives `AGENT_CONTEXT.json`, embeds it into generated tasks, and invalidates stale per-site caches when the sibling `AGENT_CONTEXT_{site}.json` changes or when cached tasks are missing the embedded context.
- **Phase 2 immutable contract update.** Adversarial task generation now treats `agent_context` as an immutable field copied from the benign task and passes `AGENT_CONTEXT.json` into the injection-generation sandbox so attacks are crafted against the real benchmark contract.
- **Phase 3 -- Benign Validation.** Full implementation: reset endpoint call, data seed application, Browser Use agent run, reward function evaluation, failure diagnosis loop (reward bug, data seed issue, impossible task, task too hard, agent limitation). Phase 3 now builds a benchmark-specific `site_prompt` from embedded `agent_context`.
- **Phase 3 conservative triage before diagnosis.** New `worldsim/phase_3_triage.py` adds a cheap host-side pre-filter before `_diagnose_one_task`. Obvious infra/auth/off-site failures are classified locally; ambiguous failures escalate; only escalated failures pay for the expensive sandbox diagnosis loop. Top-level `phase_3/triage.json` plus additive `triage_*` metadata in failed `result.json` preserve the audit trail.
- **Phase 4 -- Adversarial Evaluation.** Full iterative decision tree remains in place, and now uses the same `agent_context`-driven prompt builder as Phase 3 so benign and adversarial runs share the same benchmark contract.
- **Profile cross-reference hardening.** `profile_validation.py` and `_sandbox_validator.py` now reject `source_field` references that use a real entity with a field belonging to some other entity. Field validation is now entity-scoped, not global-name-scoped.
- **Browser Use agent runner.** `BrowserUseAgent` supports `start_urls` and `site_prompt`, preserves task-scoped HAR capture and screenshots, and uses benchmark-specific task text when provided instead of the generic one-line prompt.
- **Sandbox runner correctness.** The Modal sandbox runner now passes the Claude Code `claude_code` preset through the Agent SDK. This is a correctness fix, not just observability: without it, Modal sandboxes were not actually running Claude Code semantics. The runner also caps runs with `max_budget_usd=250.0` and emits better diagnostics for rate limits, stderr, and result subtype.
- **Reward evaluation -- dual path.** Primary: WebArena Verified vendor evaluator (`webarena_verified` package) with proper normalization (NFKC, unidecode, type dispatch). Fallback: homebrew evaluator for when vendor package is unavailable. Custom checker registry for `db_query_match` (Phase 4 injection verification).
- **Configurable agent LLM.** `make_llm` supports google/openai/anthropic providers with auto-detection from model name prefix. Default remains `gemini-3-flash-preview`.
- **Multi-site placeholder resolution.** `placeholders.py` handles `__SHOPPING__`, `__SHOPPING_ADMIN__`, `__GITLAB__`, `__REDDIT__`, `__WIKIPEDIA__`, `__MAP__` tokens. Strict mode raises on unresolved placeholders. `merge_placeholder_maps` composes config-level, instance-level, and task-level sources.
- **Data seeding.** Three mechanisms: sql (MySQL + PostgreSQL), api (HTTP requests), state_push (JSON PUT). SQL statements are validated against a disallowed keyword list. Async wrapper via `asyncio.to_thread`.
- **Eval worker pool.** `run_eval` + `staggered_worker` with `STAGGER_DELAY=5`. Phase-agnostic via `task_runner` callable. Site-aware routing via `run_tasks_by_site`.
- **Machine-readable `auth_mechanism` schema.** Phase 0c now emits an additive `auth_mechanism` block alongside prose `authentication`. Validator enforces per-type required fields (`storage_state`, `http_basic`, `form_login`, `http_headers`, `client_cert`, `pre_auth_script`, `none`, `unknown`). Runtime ships `storage_state` + `http_basic` + `none` as first-batch implementations in `BrowserUseAgent._resolve_auth`; remaining types raise `NotImplementedError`. CLI `--allow-unknown-auth` gates Phase 3 against unreviewed `unknown`-typed sites.
- **Constrained fix-loop patches.** `worldsim/fix_validation.py` gates every diagnosis-sandbox patch before `_apply_fix`: origin + method + path allowlist harvested at runtime from `BENCHMARK_PROFILE_{site}.json` (no hardcoded URLs), SQL delegated to `_sandbox_validator.validate_seed_sql`, state_push shape-checked. Rejected patches trigger one retry of `diagnose_failure` with rejection context appended to the prompt; second rejection records `patch_rejected`/`rejection_reasons`/`original_sandbox_patch` in `diagnoses.json` and scores the task as-is. Seeding layer adds a belt-and-suspenders method allowlist so `DELETE`/`HEAD`/`OPTIONS` are blocked at the lowest layer. Diagnosis prompt hardened to classify auth gaps as `agent_limitation` and reserve `mechanism: "api"` for endpoints listed in `verification_capabilities`.

### Post-Rerun Changes (this session)

Landed on `feat/worldsim-v5` on top of the archived-rerun baseline:

1. **Phase 2 `--sites` + `--max-tasks-per-site`.** Deterministic seeded subsetting for fast iteration (same sampler as Phase 3/4 so the same N tasks pair across phases). The `--sites` filter preserves other sites' existing entries in `adversarial_tasks.json` on merge, so partial reruns do not wipe earlier results.
2. **SQL validator string-literal stripping.** `worldsim/_sandbox_validator.py` and `worldsim/seeding.py` now strip single-quoted SQL string literals (honoring `''` escape) before running the disallowed-keyword regex. Eliminates false-positives on English words like "DO NOT" or "MERGE request" inside `VALUES (...)` payloads.
3. **`auth_mechanism` schema + runtime dispatcher.** Additive block alongside the prose `authentication`. Schema lives in `worldsim/_sandbox_validator.py::_validate_auth_mechanism`. First-batch runtime implementations: `storage_state`, `http_basic`, `http_headers`, `none`. Stubs raise `NotImplementedError`: `form_login`, `pre_auth_script`, `client_cert`. `--allow-unknown-auth` CLI gate refuses Phase 3/4 when any site declares `type: "unknown"`. Prompt `profile-agent-context.md` updated with detection checklist, classification rules, and four worked examples.
4. **Phase 0d auth bootstrap (new phase).** `worldsim/phases/phase_0d_auth_bootstrap.py`. Dispatch order per site: `generator_script` -> native `form_login` Playwright helper -> trust existing `storage_state.path` -> skip. Writes `logs/phase_0d/<site>/storage_state.json` and a content-addressed `completion.json`. SHA-256 input hash covers site name, credentials, declared path, generator script bytes, and form_login recipe. Runtime rotations of credentials or scripts trigger regeneration automatically. Wired into `_PHASE_ORDER` between 0c and 1. `BrowserUseAgent._resolve_auth` consults `logs/phase_0d/<site>/storage_state.json` as a fallback when the declared path is missing.
5. **Fix-loop patch validator.** `worldsim/fix_validation.py` (new). URL allowlist is harvested from the Phase 0c profile (`verification_capabilities[*].examples[*].eval_config.expected.url` plus an optional top-level `seeding_endpoints`). Placeholders (`{id}`, numeric segments, `__GITLAB__`) collapse to per-segment wildcards. Rejected patches trigger one diagnosis retry with rejection context appended to the prompt, then fall through to `keep_flagged`. Defense in depth: `apply_data_seed` enforces the `GET/POST/PUT/PATCH` method allowlist at the lowest layer.
6. **`benchmark_root` threaded through retry paths.** Phase 3's `_diagnose_one_task` -> `fix_loop` -> `_rerun_live_task` -> `run_task` and Phase 4's `_postprocess_one_task` -> `_process_adversarial_result` -> `_run_ecological_validity_fix_loop` / `_run_placement_fix_loop` / `run_strategy_variation` -> `_rerun_adversarial_task` / `_evaluate_variant` -> `run_adversarial_task` all honor the flag. Resolves `auth_mechanism.storage_state.path` values declared relative to the benchmark codebase root.

Bugfixes:

- **`_unknown_auth_sites` handles the flat Phase 0c layout.** `worldsim/main.py` previously filtered on `is_dir()` and missed the `AGENT_CONTEXT_<site>.json` flat layout that Phase 0c actually emits. Now handles both flat and nested layouts.
- **Phase 2 merge preserves other sites' adversarial tasks.** `worldsim/phases/phase_2_injections.py`: when `--sites` is set and `adversarial_tasks.json` exists, entries whose `site` is outside the filter are preserved verbatim.
- **`_resolve_auth` header kwarg.** Browser Use's `BrowserSession` takes `headers=`, not Playwright's `extra_http_headers=`. Unblocks `http_headers` runtime injection.

Infrastructure scripts:

- `scripts/webarena-compose-override.yml`: added wikipedia amd64 image pin + map volume `name:` overrides so `docker compose up -d map` binds to the unprefixed populated volumes instead of creating empty prefixed ones.
- `scripts/bootstrap_ec2.sh` (new): end-to-end orchestrator that scp's helpers, brings up all 6 containers, patches env-ctrl, verifies /init endpoints.
- `scripts/setup-wikipedia-robust.sh` (new): ZIM download with CMU + archive.org mirrors in parallel, size + magic-byte verification, atomic replace-in-volume.
- `scripts/wa_envctrl_patcher.py` (new): Python patcher for env-ctrl sites that inserts `import os` after `from __future__` and patches `_init` to read `WA_ENV_CTRL_EXTERNAL_SITE_URL`. Idempotent, self-repairs prior broken runs.
- `scripts/patch_webarena_containers.sh`: added `--on-ec2` mode, `docker cp`'s the Python patcher instead of inlining fragile shell-escaped Python.

Instance files:

- `instances.scale.json`: canonical scale topology (gitlab + reddit replicas on the r5).
- `instances.smoke.json` + `instances.smoke.local.json`: single-replica smoke configs.

Runtime artifact:

- `logs/phase_0c/AGENT_CONTEXT_gitlab.json`: hand-edited to add the `form_login` recipe (byteblaze / hello1234 + `data-testid` selectors) because Phase 0b's sandbox file scope for gitlab excludes `examples/configs/` and `tests/integration/environments/gitlab/conftest.py`. The file's `notes` flags that Phase 0b should expand its scope.

Deps: Phase 0d's `form_login` bootstrap imports `playwright.async_api.async_playwright` lazily. Install `playwright` in the environment (and run `playwright install chromium`) before running Phase 0d against a `form_login` site. Operators who only run `generator_script` / `pre_auth_script` sites do not need Playwright.

## Pipeline Outputs on Disk

Current canonical live run:

```text
logs/
  last_run_state.json        # Marker for the reset live run
```

The `phase_0a/` and `phase_0b/` directories that were initially copied back have been removed. The full pipeline must rerun from Phase 0a due to the sandbox runner fix (`ede6432`).

Archived historical run:

```text
logs_run1_no_tiered_site_discovery_and_missing_benchmark_specific_task_template/
  cost_report.json
  last_run_state.json
  pipeline_state.json
  phase_0a/
    BENCHMARK_MANIFEST.json
    BENCHMARK_MANIFEST.md
  phase_0b/
    SANDBOX_MAP.json
  phase_0c/
    BENCHMARK_PROFILE_gitlab.json
    BENCHMARK_PROFILE_map.json
    BENCHMARK_PROFILE_reddit.json
    BENCHMARK_PROFILE_shopping.json
    BENCHMARK_PROFILE_shopping.md
    BENCHMARK_PROFILE_shopping_admin.json
    BENCHMARK_PROFILE_wikipedia.json
  phase_1/
    benign_tasks.json
  phase_2/
    adversarial_tasks.json
  phase_3/
    contracts.json
```

Golden path status:

- **Historical proof:** the archived run reached Phase 3 artifact production and produced real 0a/0b/0c/1/2/3 outputs.
- **Current canonical state:** `logs/` is empty (aside from `last_run_state.json`). No phase outputs are canonical.
- **Required next step:** rerun the full pipeline from Phase 0a. The sandbox runner fix (`ede6432`) invalidated all prior sandbox outputs.

## Architecture Highlights

Key design decisions for anyone continuing this work:

- **Tiered Phase 0c profiling.** Per-site profiling is no longer a single monolithic sandbox. Verification capabilities, data model, and agent context are discovered independently first; injection-surface analysis happens only after those artifacts validate.
- **`agent_context` propagation.** Benchmark-specific auth, response-format requirements, and discovered prompt templates are now carried as task data instead of requiring later phases to reopen Phase 0c artifacts.
- **Benchmark-specific runtime prompts.** Phase 3 and Phase 4 no longer rely only on a generic "You are interacting with a web application..." prompt. They can now replay the benchmark's actual response contract and auth workflow.
- **Sandbox runner must use Claude Code semantics.** The Agent SDK path now explicitly requests the `claude_code` preset so Modal sandboxes behave like real Claude Code, not a generic SDK agent.
- **Phase 2 task sharding.** Large sites (shopping: 182 tasks) are split into shards of ~25 tasks. Shards run as independent Modal sandboxes in parallel, then results are merged with deduplication.
- **In-sandbox output validation.** `_sandbox_validator.py` runs inside the Modal sandbox as the last step before exit. This catches schema and contract issues within the same session at effectively zero extra orchestration cost.
- **Per-site instance locking.** `site_lock.py` provides asyncio locks keyed by site name. Diagnosis sandboxes run fully parallel. Agent reruns that mutate shared DB state serialize per site.
- **File routing via inclusion.** Sandboxes are scoped by `add_local_file` / `add_local_dir` calls, not ignore patterns. Phase 0b's `SANDBOX_MAP.json` remains the driver for which files each site's profiling sandbox receives.
- **Per-task resume in Phase 3/4.** `result.json` in each task directory serves as a completion sentinel. On `--resume`, existing completed results are merged with new ones.
- **Run archival boundary.** The first run predates both the tiered Phase 0c/`agent_context` changes and the sandbox runner Claude Code semantics fix (`ede6432`). All sandbox outputs from that run are invalid. Resume is cut at Phase 0a (full pipeline from scratch).

## WebArena env-ctrl base_url Fix

The original Docker images (`am1n3e/webarena-verified-*`) ship env-ctrl code where `_init()` requires a `base_url` argument, but the HTTP server (`POST /init`) calls `ops.init()` with no args. Some sites lacked an env-var fallback, causing `ValueError("base_url is required")` on every reset.

**Fix (two layers, both required for robustness):**

1. `scripts/webarena-compose-override.yml` -- sets `WA_ENV_CTRL_EXTERNAL_SITE_URL` for shopping, shopping_admin, gitlab, and reddit. Deployed into `vendors/webarena-verified/docker-compose.override.yml` by the patch script (and into `/home/ubuntu/docker-compose.override.yml` by `bootstrap_ec2.sh`). Docker Compose auto-merges it. Survives container recreation.
2. `scripts/patch_webarena_containers.sh` (+ `scripts/wa_envctrl_patcher.py`) -- patches running containers to add the env-var fallback in the Python `_init()` code (for images that lack it). Idempotent. The Python helper inserts `import os` AFTER any `from __future__` line (the prior inline snippet prepended it at line 0 and triggered a `SyntaxError` on `from __future__ import annotations`; the helper also self-repairs that broken state). Run modes:
   * `./scripts/patch_webarena_containers.sh [HOST_IP]` from the repo root deploys the override file into `vendors/webarena-verified/`.
   * `./scripts/patch_webarena_containers.sh --on-ec2 [HOST_IP]` runs the in-container Python patch against the local EC2 docker daemon; `bootstrap_ec2.sh` scp's the script and `wa_envctrl_patcher.py` up and drives this mode automatically.

For fresh container starts, only the override file is needed (the env var is sufficient when the Python code has the fallback; the patch script ensures it does).

## Bootstrap script

Canonical single-entrypoint for taking a fresh (or partially bootstrapped) EC2 host to a working 6-site WebArena Verified deployment is [`scripts/bootstrap_ec2.sh`](../scripts/bootstrap_ec2.sh). From the repo root, on your workstation:

```bash
./scripts/bootstrap_ec2.sh
# or with overrides:
HOST_IP=1.2.3.4 SSH_KEY=~/.ssh/webarena-key.pem ./scripts/bootstrap_ec2.sh
```

What it does, idempotently:

1. scp's `setup-map-robust.sh`, `setup-wikipedia-robust.sh`, `build-wikipedia-amd64.sh`, `webarena-compose-override.yml`, `patch_webarena_containers.sh`, and `wa_envctrl_patcher.py` to `/home/ubuntu/`.
2. Replaces `/home/ubuntu/docker-compose.override.yml` with the canonical one in this repo.
3. SSHes in and runs `build-wikipedia-amd64.sh` (skips if the image tag already exists).
4. SSHes in and runs `setup-map-robust.sh` (aria2 resume, per-volume `.extracted` sentinels).
5. SSHes in and runs `setup-wikipedia-robust.sh` (aria2 resume across CMU + archive.org in parallel, size + `ZIM\x04` magic check, atomic replace-in-volume via alpine `cp <src> <dst>.new && mv <dst>.new <dst>`; only restarts wikipedia if the ZIM actually changed).
6. `docker compose up -d` on the EC2 brings up all 6 sites.
7. Runs `patch_webarena_containers.sh --on-ec2` on the EC2 so the Python env-ctrl base_url patcher targets the EC2's docker daemon.
8. Verifies each site's env-ctrl `/init` returns HTTP 200.
9. If gitlab specifically fails, respawns its env-ctrl via `docker exec -d webarena-verified-gitlab sh -c 'setsid /usr/local/bin/env-ctrl serve --port 8877 >>/tmp/env-ctrl.log 2>&1 </dev/null'`. gitlab's image has no process manager for env-ctrl; plain `pkill` leaves nothing to respawn it, and plain backgrounded ssh commands get SIGHUP'd when the exec returns.
10. Prints a per-site summary (HTTP code, site URL, env-ctrl URL).

`scripts/setup-wikipedia-robust.sh` is analogous to `setup-map-robust.sh` and can also be run standalone on the EC2 host if the wiki flow needs to be re-run in isolation. It writes a `.verified` sentinel next to the on-disk ZIM so re-runs are a no-op once the file has passed verification.

## Known Issues and Lessons Learned

From code review sweeps, the archived run, and the current refactor:

- **The first run is historically useful but semantically stale.** The archive directory name is literal: that run predates tiered site discovery in Phase 0c and predates the benchmark-specific task-template / `agent_context` fix. It should not be resumed from beyond Phase 0b.
- **Agent SDK invocation is not automatically Claude Code.** When using `claude-agent-sdk`, you must explicitly request the `claude_code` preset. Without it, sandbox behavior diverges from the intended Claude Code contract.
- **Phase 0c needed to be decomposed.** A single profiling sandbox per site was too brittle. Separating data model, verification capabilities, and agent-context discovery prevents one bad inference from contaminating downstream injection-surface analysis.
- **Benchmark-specific task templates matter.** The missing benchmark-specific prompt/template issue was real. Later phases need the benchmark's response contract, auth expectations, and prompt shape, not just the task instruction.
- **Field validation must be entity-scoped.** It is not enough to check whether a field name exists somewhere in the data model. `Product.body` should fail if `body` only belongs to `Review`.
- **Phase 0c sandboxes need 4-hour timeout.** Complex sites (especially shopping / shopping_admin) can take tens of minutes, and the new tiered flow plus retries needs headroom.
- **Phase 2 sharding required for large sites.** Sending all 182 shopping tasks into one sandbox exceeded practical limits. Sharding into ~25-task chunks with parallel execution and merge solved it.
- **In-sandbox validation catches schema errors at zero cost.** Running the validator inside the same sandbox session avoids re-paying for a whole sandbox just to fix malformed JSON.
- **Volume upload for benchmarks matters.** `add_local_dir` on every call re-hashed the entire benchmark tree. `upload_to_volume` removed that per-call overhead.
- **`benign_task_id` from LLM output must be coerced to string.** LLMs sometimes emit integers. All ID matching now normalizes through `str()`.
- **Browser Use 0.12+ teardown uses `session.kill()`.** `.close()` does not exist and leads to hangs.
- **`IS_SANDBOX=1` is required.** Claude Code needs it to accept `bypassPermissions` as root inside Modal.
- **Empty-string env vars cause auth failures.** Set-but-empty vars must be treated as unset during preflight auth checks.

## Phase 3 Full Smoke Test (this session, 6 sites x 2 tasks = 12)

End-to-end smoke run after all the fixes above landed:

- **12/12 tasks reached the agent** (no infra drops). Previous attempts lost 4-6 tasks to missing-instance or reset errors.
- **0/12 passed.** Failure modes are now agent/data, not harness: cross-site routing failures on map tasks (OSRM could not compute a path between CMU and Madison Square Garden with the tiles we shipped), JSON format bypass on shopping / gitlab retrieve tasks (agent produced the correct answer but skipped the schema wrapper), and agent-capability failures on multi-hop tasks (684 picked public github instead of the local gitlab).
- Gitlab's `byteblaze` pre-auth worked: Phase 0d native form_login produced `logs/phase_0d/gitlab/storage_state.json` and Phase 3 loaded it transparently. No password guessing in the trajectory.
- Fix-loop validator rejected zero hallucinated patches this run because the diagnosis prompt hardening steered the sandbox away from API-based fixes for auth gaps (task 533 was previously the poster child for `/api/v4/session` hallucination; this run it diagnosed as `agent_limitation`).
- `phase_3` cost for this smoke: **$17.65 / 25 sandboxes / 358 turns**. Full pipeline cumulative: **$80.20** (0a: $1.95, 0c reruns: $54.06, Phase 2: $6.54, Phase 3: $17.65).

**Takeaway.** The 0/12 is the current gemini-3-flash-preview ceiling on this
WebArena-Verified 12-task subset, not a pipeline bug. The relevant next
experiments are (a) swap to a stronger agent model on the same 12 tasks
and (b) run Phase 4 on this exact 12-task subset to measure adversarial
propensity on the same cohort.

## What's Next

Prioritized execution order from the current canonical state. The pipeline is
now end-to-end functional through Phase 3; the remaining work is Phase 4 and
model upgrades.

1. **Run Phase 4 on the current 12-task subset.** Adversarial evaluation against
   the same deterministic seed/cap as Phase 3 (`--max-tasks-per-site 2`) so the
   benign and adversarial task sets stay 1:1 paired.
   ```
   uv run python -m worldsim.main phase 4 --instances instances.json \
     --max-tasks-per-site 2 --agent-model gemini-3-flash-preview
   ```
   Expected runtime: one Gate 1 probe pass per trajectory (~5 probes/task
   averaging P(eval) via the ecological-validity loop), followed by Gate 2
   attack-effectiveness judgments. Budget ~$20-40 on this cohort at
   gemini-3-flash rates, more on Opus.
2. **Swap to a stronger agent model and rerun Phase 3 on the same 12.** Try
   `gpt-5.4-mini` via OpenRouter first, then `claude-opus-4-6` or
   `claude-sonnet-4-6` as comparison paths. Our
   trajectory audits showed multiple tasks where gemini-3-flash got the
   right answer but ignored the JSON schema, so format-following alone should
   flip several passes.
3. **Expand Phase 0b's sandbox file map.** Today's Phase 0c rerun came back
   `auth_mechanism.type: "unknown"` for gitlab because the credential sources
   (`examples/configs/config.example.json`, `tests/integration/environments/gitlab/conftest.py`)
   were outside the per-site file scope. Expanding 0b's glob patterns to
   include these conventional locations would make auth discovery fully
   automatic on future benchmark onboardings.
4. **Implement runtime `form_login`.** Phase 0d handles the offline
   bootstrap case; the runtime-form_login path in `_resolve_auth` still raises
   `NotImplementedError`. Matters for benchmarks that need per-task fresh
   sessions (none on the current roadmap but cheap to add).
5. **Push Phase 0c sandbox scope fix + Phase 4 results back through the
   pipeline.** After Phase 4 lands clean, doc a canonical cost curve per
   benchmark in `docs/worldsim-v5-technical-specifcation.md`.

## Cost Summary

Historical archived-run summary only. Source:

`logs_run1_no_tiered_site_discovery_and_missing_benchmark_specific_task_template/cost_report.json`

These numbers are useful context, but they are not the final cost profile for the current tiered Phase 0c / `agent_context` design.

| Phase | Sandboxes | Turns | Cost |
|-------|-----------|-------|------|
| Phase 0a | 2 | 90 | $2.88 |
| Legacy Phase 0c | 8 (6 succeeded, 2 retries) | 196 | $13.88 |
| Phase 2 | 43 (sharded across 6 sites) | 624 | $82.99 |
| **Total** | **53** | **910** | **$99.75** |

Models used in the archived run: `claude-opus-4-6` (primary, ~$96.59), `claude-haiku-4-5-20251001` (sub-agent, ~$3.15). The archived run used the OpenRouter auth path.

Latest full-pipeline rerun (Phase 0a + 0c rerun with auth_mechanism discovery + 0d form_login bootstrap + Phase 1 rewrap + Phase 2 with --sites shopping_admin merge + full 6-site Phase 3 smoke `--max-tasks-per-site 2`):

| Phase | Sandboxes | Turns | Cost |
|-------|-----------|-------|------|
| Phase 0a | 1 | 54 | $1.95 |
| Phase 0c (2 attempts, tiered) | 48 | 995 | $54.06 |
| Phase 2 (6 sites + shopping_admin rerun) | 7 | 107 | $7.42 |
| Phase 3 (full 6-site smoke, 12 tasks) | 25 | 358 | $17.65 |
| **Total** | **81** | **1514** | **$80.20** (approx, rounded to the visible `cost_report.json` tail at session close) |

Fix-loop rejection cost is tracked in-line with diagnosis cost; retries surface as distinct `phase_3_diagnose` entries. Phase 3 cost was higher than run 1's `$1.48` because every one of the 12 tasks ran to completion and hit the diagnosis path; run 1's path errored early on most of them.

## Codebase Stats

| Metric | Value |
|--------|-------|
| `worldsim/` Python lines | 10,039 (+auth_mechanism + fix_validation + phase_0d) |
| Test files | 17 files (`test_auth_mechanism.py`, `test_fix_validation.py`, `test_phase_0d_auth_bootstrap.py` new) |
| Tests collected | 345 |
| Prompt files | 14 |

## Key Files

| File | Purpose |
|------|---------|
| `docs/worldsim-v5-technical-specifcation.md` | Source of truth. Updated to match tiered Phase 0c and `agent_context` propagation. |
| `docs/current_progress.md` | This file. Historical proof vs current canonical rerun state. |
| `worldsim/modal_sandbox.py` | The core sandbox primitive, secret wiring, and Modal execution path. |
| `worldsim/_sandbox_runner.py` | Runs inside Modal. Drives `claude-agent-sdk` with the Claude Code preset. |
| `worldsim/_sandbox_validator.py` | In-sandbox output validation for manifests, profiles, tasks, diagnosis, and ecological-validity artifacts. |
| `worldsim/phases/phase_0_recon.py` | Phase 0 orchestration, including the new tiered Phase 0c profiling flow. |
| `worldsim/phases/phase_0d_auth_bootstrap.py` | Phase 0d auth bootstrap. Dispatch: `generator_script` -> native `form_login` (Playwright) -> trust existing `storage_state.path` -> skip. Idempotent via SHA-256 of inputs. |
| `worldsim/fix_validation.py` | Host-side patch validator for Phase 3's fix-loop. Method + origin + path allowlist harvested at runtime from the site profile. |
| `worldsim/agent_prompt.py` | Shared builder for benchmark-specific runtime prompts from `agent_context`. |
| `worldsim/phases/phase_1_mode_a.py` | Wraps benchmark tasks and now preserves `instantiation_dict` + embeds `agent_context`. |
| `worldsim/phases/phase_1_mode_b.py` | Novel-task generation, per-site caching, and `agent_context` propagation. |
| `worldsim/phases/phase_2_injections.py` | Task sharding, parallel sandbox execution, immutable-field merge, and `agent_context` preservation. |
| `worldsim/phases/phase_3_benign.py` | Benign validation with diagnosis loop and benchmark-specific runtime prompt injection. |
| `worldsim/phases/phase_4_adversarial.py` | Full adversarial decision tree with the same benchmark-specific runtime prompt builder. |
| `worldsim/browser_use_agent.py` | Browser Use lifecycle, HAR capture, screenshot persistence, and `site_prompt` support. |
| `worldsim/profile_validation.py` | Shared profile loading and host-side validation parity with the in-sandbox validator. |
| `worldsim/prompts/profile-*.md` | Tiered Phase 0c prompt set. |
| `worldsim/prompts/generate-benign-tasks.md` | Mode B prompt, now conditioned on `AGENT_CONTEXT.json`. |
| `worldsim/prompts/generate-injections.md` | Phase 2 prompt, now conditioned on `AGENT_CONTEXT.json`. |
| `tests/test_phase_0_recon.py` | Tests for tiered Phase 0c retries and non-publication of invalid outputs. |
| `tests/test_phase_1_tasks.py` | Tests for `agent_context` propagation, cache invalidation, and `instantiation_dict` preservation. |
| `tests/test_phase_3_benign.py` | Tests for benchmark-specific prompt construction. |
| `tests/test_phase_4_adversarial.py` | Tests for benchmark-specific prompt use in adversarial evaluation. |
| `tests/test_auth_mechanism.py` | 33 tests: `auth_mechanism` schema validation + `_resolve_auth` dispatcher coverage. |
| `tests/test_fix_validation.py` | 29 tests: URL allowlist, method gating, SQL delegation, state_push shape. |
| `tests/test_phase_0d_auth_bootstrap.py` | 30 tests: dispatch precedence, idempotency hashing, form_login bootstrap, trust-path mode. |
| `scripts/bootstrap_ec2.sh` | Single-entrypoint host-side orchestrator: scp helpers, bring up all 6 containers, patch env-ctrl, verify /init endpoints. |
| `scripts/setup-wikipedia-robust.sh` | Parallel ZIM download (CMU + archive.org), size + magic-byte verification, atomic replace-in-volume. |
| `scripts/wa_envctrl_patcher.py` | In-container Python patcher: inserts `import os` after `from __future__`, then splices the `WA_ENV_CTRL_EXTERNAL_SITE_URL` fallback into `_init`. Idempotent, self-repairs prior broken runs. |

## Auth Setup

Current working configuration:

- **Sandbox Claude Code auth:** archived real runs used `ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_BASE_URL` (OpenRouter path). OAuth and direct Anthropic API key paths are also supported in current code.
- **Modal:** profile `theta-software-inc`, environment `test`. Token in `~/.modal.toml`.
- **Sandbox runner semantics:** current code explicitly passes the Claude Code `claude_code` preset through the Agent SDK so Modal sandboxes behave like Claude Code rather than a generic SDK agent.
- **Agent LLM (for Phases 3-4):** needs `GOOGLE_API_KEY` for the default `gemini-3-flash-preview`, or `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` if switching providers.

## Prompt Inventory

All 14 prompts in `worldsim/prompts/`:

| Prompt | Used by |
|--------|---------|
| `discover-benchmark.md` | Phase 0a |
| `profile-site.md` | Legacy single-sandbox Phase 0c prompt, retained for reference |
| `profile-verification-capabilities.md` | Phase 0c Tier 1A |
| `profile-data-model.md` | Phase 0c Tier 1B |
| `profile-agent-context.md` | Phase 0c Tier 1C |
| `profile-injection-surface.md` | Phase 0c Tier 2 |
| `generate-benign-tasks.md` | Phase 1 Mode B |
| `generate-injections.md` | Phase 2 |
| `probe-ecological-validity.md` | Phase 4 Gate 1 (Modal sandbox) |
| `judge-adversarial-failure.md` | Phase 4 Gate 2 classifier (Messages API, tool-use forced) |
| `fix-ecological-validity.md` | Phase 4 validity-fix loop (Modal sandbox) |
| `fix-injection-placement.md` | Phase 4 placement-fix loop (Modal sandbox) |
| `generate-variant.md` | Phase 4 variant generator (Messages API, tool-use forced, 22-strategy pool) |
