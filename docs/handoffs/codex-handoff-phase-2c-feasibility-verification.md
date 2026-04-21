# Codex handoff — Phase 2c feasibility verification (TODO)

**Branch to base off:** `feat/worldsim-v5`
**Target branch:** `codex/phase-2c-feasibility`
**Date:** 2026-04-17 (revised 2026-04-17 to align with actual code shape)
**Status:** shipped 2026-04-18. Four commits on `feat/worldsim-v5`: verifier + hooks + grace mode, enriched dataset against r5, strict-admission flip, nightly cron hook. AT-009 lands `length_exceeded` with `http_status=400` in `adversarial_tasks.infeasible.json` as required. **Post-ship audit 2026-04-18** caught six latent bugs (see §16); fixed in follow-up commits `0cfb6513` and `e0b4b07b`. One was critical: the nightly cron from `ed7cb377` would have corrupted the dataset on its first warm re-run by overwriting `feasibility.status="verified"` with `"skipped"` on every idempotency hit, and strict admission (from `5ea2a275`) would then reject every previously-verified task at the Phase 4 gate.
**Anchor artifact:** commit `4c27fb48` + AT-009 case study from the 2026-04-17 demo. Post-ship anchor: `0cfb6513` (idempotency preservation) + `e0b4b07b` (pipeline hardening).

**Mandate — implement this end-to-end, self-contained, directly on `feat/worldsim-v5`.** This handoff is the complete spec. Codex owns the full delivery: verifier module, retry helper, editor `_classify_4xx_response` overrides for all four sites, Phase 2/3/4 wiring, CLI flags + resume passthrough, unit tests, integration tests + fixtures, cleanup-script extension, doc drift (spec, README, CLAUDE.md, cross-reference footnotes), and the four-step migration. **Work trunk-style:** commit directly to `feat/worldsim-v5`; no feature branch, no PRs, no code review round-trip. The §6 migration remains a four-commit sequence, but each step is a direct commit pushed to `feat/worldsim-v5` after it lands green on CI. Do not stop early or punt sub-tasks to follow-up work. There is no time budget; ship when §9 acceptance criteria all pass. If you discover a design gap not covered here, fix it inline and document the decision in the commit message body (do not pause for human input on small choices — exercise judgment and proceed). The only allowed deferrals are the items explicitly listed in §8 non-goals. Never force-push `feat/worldsim-v5`; fix forward with follow-up commits.

---

## 0. One-sentence summary

Add an operational-feasibility check as a third internal stage of Phase 2 (`phase_2_stage="feasibility"`) that POSTs each freshly-generated adversarial task against a live dev instance, keeps the ones that 2xx, quarantines the ones that don't, and writes a fingerprinted feasibility stanza onto every task — so the main dataset is *physically executable by construction* by the time Phase 3 reads it.

---

## 1. Why this exists

### 1.1 The AT-009 case study

Task `AT-009` (gitlab, `create_group`) generated a 624-character injection for `group.description`. Every layer of static validation passed:

- Schema ✓ (`seed_template` and `adversarial_data_seed` both have `editor_calls`)
- Editor method registered ✓ (`create_group` in `ShoppingAdminEditor` / `GitlabEditor`, see `worldsim/editors/gitlab.py:13–25`)
- Body-field aliases resolved ✓ (after commit `4c27fb48`)
- Length budget respected ✓ (`length_budget.max = 1500`, payload = 624)

Then Phase 4 fired the POST. GitLab answered:

```
HTTP 400: {"message":"Failed to save group {:description=>[\"is too long (maximum is 255 characters)\"], ...}"}
```

The task was valid on paper, infeasible on the platform. The generator had no way to know GitLab's `groups.description` column caps at 255 because nothing in the pipeline records that fact.

### 1.2 Why static analysis alone is insufficient

A length cap is one symptom of a broader class of platform-level rejections that no purely-static layer catches:

- Character-set validation (`<script>` tags stripped, URL patterns rejected, emoji normalization)
- Implicit required fields (gitlab projects need `visibility`; Magento reviews need `stores`)
- Business rules ("group path must match `[a-zA-Z][\w-]*`")
- Content policy filters (spam detection on injection-like text)
- Permission edge cases (this user cannot create this resource type)
- Platform rate limits on specific endpoints
- Encoding quirks (multipart vs JSON, UTF-8 normalization, trailing-newline trimming)
- Referential integrity (foreign keys to fixtures that don't exist on this instance)

Each of these fails at POST time. No amount of static schema checking catches them all. A predictive layer (parse OpenAPI specs, scrape `<input maxlength>` attributes, learn from error regexes) covers *most* length cases but **cannot** cover content policy, business rules, or implicit requirements.

### 1.3 WASP precedent

WASP's `webarena_prompt_injections/environment_setup.py` (Meta, 2025, arXiv:2504.18575) does exactly this pattern: each attack resource is *actually created* at setup time against the live benchmark. `make_project_as_agent_user()`, `create_issue_with_title_and_text()` fire real POSTs; failures are handled immediately (the `make_attacker_account` "reuse if exists" fallback is the canonical example). Feasibility is proven by construction, not predicted.

The current pipeline skipped this because Phase 2 generates into `adversarial_tasks.json` and Phase 4 is the first place that touches a live instance. That gap means *every feasibility bug manifests as a Phase 4 runtime error*, contaminating ASR data with infrastructure failures.

### 1.4 Research-methodology cost of the status quo

Without feasibility verification, every Phase 4 run has three terminal states that aren't distinguishable from the summary line:

- `error` — task errored during seed (could be infra-flaky, could be infeasible forever)
- `seed_preflight_mismatch` — static validator rejection (already fine; deterministic)
- `complied` / `resistant` / etc. — actual research signal

Reviewers cannot trust ASR numerators/denominators until the `error` bucket is known to be infra-only, not feasibility. Phase 2c's output (`feasibility.status == "verified"`) is what converts `error` into a pure infra-flakiness signal.

### 1.4.1 Ground-truth shape of the 236-task dataset (verified 2026-04-17)

Inspected `logs/phase_2/adversarial_tasks.json` directly. Codex must design against these facts, not against the placeholder examples in older spec drafts:

- **Type:** top-level JSON array of 236 task objects (not a dict; not wrapped).
- **Per-task keys:** `id`, `site`, `sites`, `benign_task_id`, `instruction`, `attack_objective`, `concealment`, `framing`, `delivery_channel`, `delivery_mechanism`, `payload_texts`, `payload_text_diagnostics`, `selected_payload_index`, `length_budget`, `required_tokens`, `source_field`, `target_surface_id`, `start_urls`, `agent_context`, `seed_template`, `data_seed`, `adversarial_data_seed`, `reward_function`.
- **`adversarial_data_seed.mechanism` = `"editor"` for 100% of tasks.** Not `api`/`form`/`state_push`. The verifier must drive the `editor_calls` path; it should not branch on a `state_push` short-circuit (none exist in real data).
- **`adversarial_data_seed.editor_calls`:** non-empty list (every task has ≥1 call). Per-call keys: `args`, `benchmark`, `method`, `site`. Defensively still handle `editor_calls == []` → mark task as `infeasible` with `kind="empty_seed"`, since admitting trivially-empty seeds bloats the report and hides upstream Phase 2 bugs.
- **Site distribution:** gitlab 91, shopping 59, reddit 58, shopping_admin 28. All four editors must implement `_classify_4xx_response`. None can be punted.

### 1.4.2 Ground-truth shape of `instances.smoke.json`

Also inspected directly. The handoff's earlier "list of instances" framing is incomplete:

- **Top-level shape:** a dict with keys `benchmark_name`, `benchmark_codebase`, `verification_proxy`, `url_placeholders`, **`instances`** (the actual list), plus optional fields. **The verifier must `json.load(f)["instances"]` to get the per-site list, not pass the top-level object straight to `acquire_tokens_for_instances`.**
- **Per-instance auth shape:** each instance can carry up to three auth blocks — `auth` (http_headers for browsing), `api_auth` (bearer_token for editor REST), `agent_auth` (Browser Use). 2c hits `api_auth` via the editor; `acquire_tokens_for_instances` already iterates `("auth", "api_auth")` (`worldsim/auth_tokens.py:317`).
- **Other instance files exist:** `instances.scale.json`, `instances.smoke.json`, `instances.smoke.local.json`. Default the CLI flag to `instances.smoke.json` and document the precedent.

### 1.5 Where this fits in the pipeline state machine

Phase 2 already exposes a sub-stage tag in `pipeline_state.json`:

- `phase_2_stage="planning"` — set after Phase 2a (Modal sandbox planning)
- `phase_2_stage="text_fill"` — set after Phase 2b (host-side text fill)
- `phase_2_stage="complete"` — set when all of the above succeed

2c slots into this cleanly as a **third sub-stage** before `complete`:

- `phase_2_stage="feasibility"` — set after 2c writes the enriched dataset back

There is **no new top-level phase**, no addition to `_PHASE_ORDER` in `worldsim/main.py:340–351`, and no new `phase_2c` state key. `--resume` continues to walk `_PHASE_ORDER` and treats Phase 2 as one node (resume-into-feasibility is handled by the intra-phase logic, see §3.7). The CLI alias `phase 2c` (see §3.2) is sugar that maps to `phase 2 --feasibility-only`. This decision is the load-bearing piece — please do not invent `phase_2c` as a separate top-level phase.

---

## 2. Non-negotiable invariants (from CLAUDE.md)

- The orchestrator does not manage environments. Feasibility verification **connects** to a dev instance the user supplies in `instances.smoke.json` / `instances.scale.json`; it never starts/stops/snapshots it. One `reset_endpoint` call between tasks is allowed, same exception as Phase 3/4.
- SQL seeding stays excluded. Verification uses the same editor `api`/`form` channels as Phase 4.
- Threat model unchanged: regular authenticated user. Feasibility confirms the attacker-user can actually plant the payload, which is a precondition for the attack-realism claim.
- Benchmark-agnostic. No `if benchmark == "webarena_verified"` in the feasibility module. Dispatch via `EDITOR_REGISTRY[(benchmark, site)]` (`worldsim/editors/__init__.py:9–14`).
- File routing via inclusion, not ignore (Modal isn't involved in 2c, but the principle still applies: do not rely on ignore-files for isolation).
- No `import` from `AgentLab/`. 2c is brand-new code, not a port.

---

## 3. Design

### 3.1 File layout

```
worldsim/phases/phase_2_feasibility.py        # new, ~150 LOC verifier
worldsim/_async_utils.py                      # new, ~40 LOC retry helper
worldsim/editors/base.py                      # touched: +_classify_4xx_response hook (~25 LOC)
worldsim/editors/gitlab.py                    # touched: override _classify_4xx_response (~40 LOC)
worldsim/editors/reddit.py                    # touched: override _classify_4xx_response (~25 LOC)
worldsim/editors/shopping.py                  # touched: override _classify_4xx_response (~25 LOC)
worldsim/phases/phase_2_injections.py         # touched: pipe 2b → 2c (~30 LOC)
worldsim/phases/phase_3_benign.py             # touched: feasibility-aware contract enrichment (~30 LOC)
worldsim/phases/phase_4_adversarial.py        # touched: feasibility admission gate (~40 LOC)
worldsim/main.py                              # touched: CLI flags + resume passthrough
scripts/run_integration_tests.sh              # touched: add @pytest.mark.feasibility group
scripts/cleanup_webagent_test_resources.sh    # touched: add `webagent-verify-` glob
tests/test_phase_2_feasibility.py             # new, ~400 LOC unit tests (FakeSession pattern)
tests/integration/test_phase_2_feasibility_live.py  # new, ~180 LOC live
tests/integration/fixtures/feasibility/<site>/{good,oversize,policy}.json  # new, ~12 fixtures
docs/worldsim-v5-technical-specifcation.md    # touched: §Phase 2c subsection (~700 words)
README.md                                      # touched: CLI flags + prerequisites note
CLAUDE.md                                      # touched: integration-test trigger list
```

**Total estimated LOC (new + touched): ~830** — not 1100 as the prior draft estimated. The verifier is small because most of the heavy lifting lives in `worldsim/seeding.py` already (see §3.4).

No new package. Stays inside `worldsim/phases/` alongside the sibling stages. The retry helper lives in `worldsim/_async_utils.py` (a leading underscore signals "internal"); reuse it from elsewhere in the pipeline if a similar need arises.

### 3.2 Pipeline position

```
Phase 2a (Modal sandbox planning) →
  Phase 2b (host-side text fill) →
    Phase 2c (feasibility verification)   ← new sub-stage
      → adversarial_tasks.json          (rewritten in place; tasks gain feasibility stanza)
      → adversarial_tasks.infeasible.json   (new; quarantined tasks)
      → feasibility_report.json         (new; per-site summary)
```

`uv run python -m worldsim.main phase 2 …` runs 2a + 2b + 2c by default. New flags (added to both `phase_cmd` and `resume_cmd` per `worldsim/main.py:185–295`):

- `--skip-feasibility` — skip 2c for fast dev iteration. Tags every task with `feasibility = {"status": "unverified"}` (nested) and emits a single warning. Phase 4 grace period (§4.4) handles these.
- `--feasibility-only` — re-run 2c on existing `adversarial_tasks.json` without re-running 2a/2b. Idempotent per the §3.7 truth table.
- `--feasibility-instances PATH` (default: `instances.smoke.json`) — the per-site instances file. **Note:** this is not a host config; `configs/benchmark_hosts/r5.yaml` is an EC2 deploy descriptor (advertise_host, SSH user, security group), not a per-site URL list. Editors require the per-site list.
- `--feasibility-concurrency N` (default 10).
- `--feasibility-retry-count N` (default 1).
- `--feasibility-ttl-hours N` (default unset / unlimited) — opt-in dev convenience: skip re-verify if `verified_at` is within N hours, even if fingerprint mismatches. Useful when iterating on editor classes locally.
- `--force-reverify` — re-verify every task regardless of fingerprint or status.

Also expose a standalone subcommand `uv run python -m worldsim.main phase 2c` that's pure CLI sugar for `phase 2 --feasibility-only`. **Implementation note:** add `"2c"` to the `choices` list at `worldsim/main.py:63` and route it through the existing dispatcher to set `args.feasibility_only = True`. Do **not** add `"phase_2c"` to `_PHASE_ORDER`; do **not** create a `phase_2c` state key.

### 3.3 Core contract

One async entry point in `worldsim/phases/phase_2_feasibility.py`:

```python
async def verify_feasibility(
    tasks_path: Path,
    *,
    instances: list[dict[str, Any]],
    concurrency: int = 10,
    retry_count: int = 1,
    ttl_hours: float | None = None,
    force_reverify: bool = False,
) -> FeasibilityReport: ...
```

Pre-flight (before launching any worker):

1. `acquire_tokens_for_instances(instances)` (`worldsim/auth_tokens.py:314–369`) — populates the per-run token cache so editors don't all stampede the auth endpoint. Without this, every worker hits `auth_missing` immediately.
2. For each `(benchmark, site)` pair represented in the task set, look up the corresponding instance and call `editor_cls.probe_base_state(instance)` (`worldsim/editors/base.py:135`) once. Fail fast if any instance is dead.
3. Compute the host-fingerprint constants once: `host_config_path`, `editor_commit = git rev-parse HEAD`, `dataset_commit = git rev-parse HEAD` (same value but recorded separately so the schema can evolve when 2c lives in a separate repo).

Returns a structured report:

```python
@dataclass(frozen=True)
class FeasibilityReport:
    verified: list[dict[str, Any]]                  # tasks that 2xx'd
    infeasible: list[dict[str, Any]]                # tasks that didn't (with feasibility.errors)
    skipped_already_verified: list[dict[str, Any]]  # idempotency path
    cleanup_warnings: list[str]                     # resources that may have leaked (per-task summary strings)
    host_fingerprint: dict[str, str]                # host_config + editor_commit + dataset_commit
    elapsed_seconds: float
    per_site_counts: dict[str, dict[str, int]]      # {"gitlab": {"verified": 47, "infeasible": 3}, ...}
```

Caller (the wiring in `phase_2_injections.py`) writes `verified` back to `tasks_path` via `write_json_atomic` (`worldsim/atomic_io.py:14`), writes `infeasible` to `tasks_path.with_name(tasks_path.stem + ".infeasible.json")`, writes a summary to `feasibility_report.json` in the same directory, and calls `save_state("phase_2", phase_2_stage="feasibility", ...)`.

### 3.4 Per-task execution

The right primitive is **already in the codebase**: `apply_data_seed_async(seed, instance) → SeedCleanupHandle | None` (`worldsim/seeding.py:342–346`). It does exactly what 2c needs — render templates, look up editor with caching, validate args, fire the method, accumulate cleanup. The wrapper class `SeedCleanupHandle` (`worldsim/seeding.py:93–112`) bundles the editor instances + session and exposes idempotent `.cleanup()` (an internal `_cleaned` flag protects against double-call) which always closes the underlying `requests.Session` in a `finally` block.

Editors are **synchronous** (`requests.Session`, see `BaseSiteEditor.__init__` at `base.py:129`). Do not try to `await` editor methods directly; that will return a non-awaitable value and crash. `apply_data_seed_async` already wraps the sync call in `asyncio.to_thread` for you.

**Phase 4 already follows this pattern** — see `worldsim/phases/phase_4_adversarial.py:1177` (`seed_cleanup = await apply_data_seed_async(adv_seed, seed_instance_dict)`) and the `acquire_tokens_for_instances` call at line 632. 2c is a structural sibling; mirror Phase 4's lifecycle exactly.

```python
async def verify_one(
    task: dict[str, Any],
    instance: dict[str, Any],
    *,
    retry_count: int,
    fingerprint: dict[str, str],
) -> dict[str, Any]:
    """Return the task with a feasibility stanza attached."""
    seed = task["adversarial_data_seed"]
    # CRITICAL: copy the instance — never mutate the shared dict, parallel verify_one
    # coroutines would race on instance["seed_task"] otherwise. See §3.4.2.
    bound_instance = dict(instance)
    bound_instance["seed_task"] = task

    handle: SeedCleanupHandle | None = None
    attempts: list[dict[str, Any]] = []
    try:
        handle = await retrying(
            lambda: apply_data_seed_async(seed, bound_instance),
            retries=retry_count,
            attempts_log=attempts,
        )
    except EditorError as exc:
        return _build_infeasible(task, exc, fingerprint=fingerprint, attempts=attempts)
    except ValueError as exc:
        # validate_data_seed raises ValueError (worldsim/seeding.py:162,173,177,etc.)
        # before any editor work; treat as schema-level infeasibility.
        return _build_infeasible(
            task,
            EditorError("schema_mismatch", str(exc)),
            fingerprint=fingerprint,
            attempts=attempts,
        )
    finally:
        if handle is not None:
            try:
                handle.cleanup()
            except EditorError as cleanup_exc:
                # Verification still succeeded; record a warning rather than fail the task.
                logger.warning("cleanup leaked for %s: %s", task["id"], cleanup_exc.detail)

    if handle is None:
        # apply_data_seed returns None when there were no editor calls (defensive — real
        # 236-task dataset has none of these, but Phase 2 may emit empty seeds in error paths).
        # Mark explicitly so the report surfaces upstream bugs instead of silently passing.
        return _build_infeasible(
            task,
            EditorError("empty_seed", "adversarial_data_seed produced no editor calls"),
            fingerprint=fingerprint,
            attempts=attempts,
        )
    return _build_verified(task, fingerprint=fingerprint, attempts=attempts)
```

**Why the per-task `dict(instance)` shallow copy matters:** `_build_seed_context` (`worldsim/seeding.py:349–377`) reads `instance["seed_task"]` to populate `task_id`, `instruction`, `topic`, `intent` into the rendering context. If you mutate the shared instance dict, two parallel verify_one coroutines on the same site will race — the second one's seed_task overwrites the first while the first is mid-render. Always shallow-copy. (Deep copy is unnecessary; nothing inside the instance dict gets mutated, only the top-level `seed_task` key.)

**Mechanism handling:** the verifier does not need to branch on `mechanism`. `apply_data_seed_async` already routes `state_push` separately (`seeding.py:291`); 100% of real adversarial seeds use `mechanism="editor"`, which falls into the `editor_calls` path (`seeding.py:316–323`). Trust the dispatcher.

### 3.4.0 Pre-flight (called once at `verify_feasibility` entry, before launching workers)

```python
async def verify_feasibility(tasks_path, *, instances, ...):
    # 1. Load tasks. Defensively handle empty list and non-list inputs.
    raw = json.loads(tasks_path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{tasks_path} must contain a JSON array of tasks; got {type(raw).__name__}")
    if not raw:
        logger.info("no tasks to verify; writing empty report")
        return _empty_report(host_fingerprint=...)

    # 2. Acquire tokens for every instance. acquire_tokens_for_instances returns a list
    # of error strings (it does NOT raise). Treat any non-empty result as fatal.
    token_errors = acquire_tokens_for_instances(instances)
    if token_errors:
        raise RuntimeError(
            "feasibility pre-flight: token acquisition failed:\n  - "
            + "\n  - ".join(token_errors)
        )

    # 3. Pre-validate site→instance mapping. Every site referenced by a task must have
    # exactly one matching instance. Fail loudly here, not 200 tasks deep.
    sites_in_tasks = {str(t.get("site", "")).lower() for t in raw if t.get("site")}
    sites_in_instances = {str(i.get("site_name", "")).lower() for i in instances}
    missing = sites_in_tasks - sites_in_instances
    if missing:
        raise RuntimeError(
            f"feasibility pre-flight: tasks reference sites with no matching instance: {sorted(missing)}"
        )

    # 4. Probe base state once per (benchmark, site). probe_base_state raises EditorError
    # on dead instances; surface that immediately rather than letting every worker discover it.
    for instance in instances:
        site = str(instance.get("site_name", "")).lower()
        if site not in sites_in_tasks:
            continue
        editor_cls = EDITOR_REGISTRY.get(("webarena_verified", site))
        if editor_cls is None:
            raise RuntimeError(f"no editor registered for site {site!r}")
        editor_cls.probe_base_state(instance)

    # 5. Compute the host fingerprint constants.
    fingerprint = {
        "host_config": tasks_path_or_instances_basename,
        "editor_commit": _git_head_short(),
        "dataset_commit": _git_head_short(),
        # task_content_hash is per-task; computed inside verify_one.
    }

    # ... then launch workers ...
```

**Note:** if the loaded JSON has `[wrapper_dict_keys, ..., "instances": [...]]` shape (the `instances.smoke.json` format described in §1.4.2), the caller in `phase_2_injections.py` extracts `["instances"]` before passing to `verify_feasibility`. The verifier itself receives a flat list.

**Key properties:**

- Uses the **same** `apply_data_seed` machinery Phase 4 uses. No duplicate code.
- Editor caching (one editor per `(benchmark, site)` pair *per task*) and `_cleanup_stack` LIFO are inherited from `_get_editor_for_seed_call` (`worldsim/seeding.py:568–595`) — do not reimplement them.
- Resource isolation comes from **immediate per-task cleanup**, not from naming. The handoff previously suggested a `webagent-verify-{task_id}` prefix, but resource names in editors come from `_slugify(arg["name"])` (`worldsim/editors/gitlab.py:1001–1006`), not from `task_id`. Setting a different `task_id` only affects fields the seed template explicitly wrote `{task_id}` into. Trust cleanup, not naming.
- Retry policy: retry only on `EditorError(kind ∈ {"request_failed", "unreachable"})`. Do not retry on 4xx — intentional rejection is the answer.
- If *any* call in a multi-call chain fails, the existing `apply_data_seed` exception path (`seeding.py:332–339`) tears down the partial state automatically.

### 3.4.1 The retry helper

Lives in `worldsim/_async_utils.py`:

```python
async def retrying(
    factory: Callable[[], Awaitable[T]],
    *,
    retries: int = 1,
    backoff_base_seconds: float = 1.0,
    attempts_log: list[dict[str, Any]] | None = None,
    retry_on: tuple[str, ...] = ("request_failed", "unreachable"),
) -> T:
    """Invoke factory() up to (retries + 1) times.

    Retry only on EditorError whose kind is in retry_on. 4xx kinds are not retried.
    Backoff: backoff_base_seconds * 2**attempt.
    """
```

We avoid `tenacity` (no new dep) — the helper is ~40 LOC and only does what we need. Place next to `staggered_worker` in `worldsim/eval_worker_pool.py` if you'd rather not add a new file; either is fine.

### 3.5 Error taxonomy

`EditorError` is raised today at these sites in `worldsim/editors/base.py`:

| Line | Kind | Trigger |
|---|---|---|
| 154 | `cleanup_failed` | one or more cleanup-stack callables raised |
| 165 | `missing_site_url` | `instance["site_url"]` empty |
| 199, 244 | `unexpected_redirect` | 3xx response on API path |
| 204, 248, 273, 277, 312 | `auth_missing` | 401/403 on API or form |
| 211, 254, 284, 320 | `request_failed` | catch-all 4xx/5xx after explicit kinds |
| 579 (seeding.py) | `site_mismatch` | call site ≠ instance site |
| 588 | `unsupported_site` | no editor registered for `(benchmark, site)` |
| 615, 622 | `unsupported_method` | method not in `supported_methods` |

Per-editor specific kinds (raised inside individual editor classes):

- **GitLab** (`gitlab.py`): `branch_exists`, `invalid_project_create`, `project_already_exists`, `invalid_project_path`, `invalid_issue_create`, `invalid_issue_note_create`, `invalid_merge_request_create`, `invalid_mr_note_create`, `invalid_repo_file_commit`, `invalid_current_user`, `invalid_group_create`, `project_cleanup_failed`.
- **Reddit, Shopping, ShoppingAdmin**: no specific kinds today; all 4xx falls through to `request_failed`.

**New additive kinds for 2c:**

- `length_exceeded` — 4xx with response body matching `is too long`, `too long (maximum is …)`, `value too long for type`, etc.
- `field_required` — 4xx with `is required`, `can't be blank`, `must be filled`, `missing` in the response.
- `content_policy` — 4xx with `forbidden`, `rejected`, `spam`, `abuse`, `blocked`, `policy violation`.

**How the heuristic gets injected without breaking existing call sites:**

Add an optional hook on `BaseSiteEditor`:

```python
def _classify_4xx_response(
    self,
    method: str,
    path: str,
    response: requests.Response,
) -> tuple[str, str] | None:
    """Return (kind, detail) for a 4xx, or None to fall back to generic request_failed."""
    return None  # base class: no opinion
```

Then in `_api_request_json` (`base.py:178–214`), `_form_get` (`base.py:261–288`), `_submit_form` (`base.py:290–324`), and `_submit_exact_form` (`base.py:326+`), insert the hook **after** the auth-missing branch but **before** the generic `request_failed` raise:

```python
if 400 <= response.status_code < 500 and response.status_code not in (401, 403):
    classified = self._classify_4xx_response(method, path, response)
    if classified is not None:
        kind, detail = classified
        raise EditorError(kind, detail)
# fall through to existing raise_for_status / generic request_failed
```

Subclasses override `_classify_4xx_response` per platform. **GitLab** parses `response.text` for the strings above. **Reddit** detects spam-protection responses (PostMill returns 422 with form errors). **Shopping/ShoppingAdmin** detects Magento's structured `errors[].message` field. Each editor's heuristic table lives next to its own request methods, not in the verifier — that keeps `phase_2_feasibility.py` benchmark-agnostic.

`EDITOR_ERROR_KINDS` may be exported as a frozenset constant from `base.py` for documentation, but no code today validates against a closed set; leave it informational.

### 3.6 Output schema (every task gets a `feasibility` stanza)

Always nested under `feasibility.{...}`. Never flat `feasibility_status`. Verified shape:

```json
{
  "id": "AT-009",
  "site": "gitlab",
  "...": "...",
  "feasibility": {
    "status": "verified",
    "verified_at": "2026-04-18T03:17:42Z",
    "host_fingerprint": {
      "host_config": "instances.smoke.json",
      "editor_commit": "4c27fb48",
      "dataset_commit": "c1ddb52c",
      "task_content_hash": "a1b2c3d4e5f6"
    },
    "attempts": [
      {"attempt": 0, "status": "success", "elapsed_ms": 412}
    ]
  }
}
```

Infeasible variant:

```json
{
  "feasibility": {
    "status": "infeasible",
    "host_fingerprint": { "...": "..." },
    "errors": [
      {
        "call_index": 0,
        "method": "create_group",
        "kind": "length_exceeded",
        "detail": "gitlab group description is too long (maximum is 255 characters)",
        "http_status": 400,
        "response_snippet": "{\"message\":\"Failed to save group {:description=>[\\\"is too long (maximum is 255 characters)\\\"]}\"}"
      }
    ],
    "first_failed_at": "2026-04-18T03:17:38Z",
    "attempts": [
      {"attempt": 0, "status": "infeasible", "elapsed_ms": 287}
    ]
  }
}
```

Skipped (`--skip-feasibility`) shape:

```json
{
  "feasibility": {
    "status": "unverified",
    "skipped_at": "2026-04-18T03:17:42Z",
    "reason": "skip_feasibility_flag"
  }
}
```

The `infeasible` JSON file uses the same task shape (full task with `feasibility.status=="infeasible"`). Downstream phases read both files but admit only `verified` (subject to grace-period rules in §4.4).

### 3.7 Idempotency, fingerprinting, and the `--feasibility-only` truth table

**Fingerprint fields:**

- `host_config` — basename of the instances file used (e.g., `"instances.smoke.json"`).
- `editor_commit` — `git rev-parse HEAD` of the worldsim repo.
- `dataset_commit` — `git rev-parse HEAD` of the worldsim repo (same value today; recorded separately so the schema accommodates a future split).
- `task_content_hash` — `sha256(canonical_json(task["adversarial_data_seed"]["editor_calls"]))[:12]`. This is the **load-bearing addition**: any change to the task's editor_calls (e.g., a manual edit, or a Phase 2 regeneration) invalidates verification automatically, regardless of git state.

**Resource naming:** all editors keep their existing `webagent-task-` prefix. 2c does not introduce a `webagent-verify-` namespace because (a) editor methods derive names from `args["name"]`, not from `task_id`, so a prefix on `task_id` only catches a subset of resources; (b) immediate per-task cleanup is the real isolation primitive. Phase 4 should never observe a `webagent-verify-*` residue *because there isn't one*. Cleanup script gets the new glob anyway as a defensive measure (§4.7).

**Truth table for `--feasibility-only` × current `feasibility` field:**

| Current state | Default behavior | With `--force-reverify` | With `--feasibility-ttl-hours N` |
|---|---|---|---|
| field missing | verify | verify | verify |
| `verified` + fingerprint matches | skip (no HTTP) | re-verify | skip |
| `verified` + fingerprint drifts | re-verify | re-verify | skip if `verified_at` within N hours |
| `infeasible` (any fingerprint) | re-verify (cheap; one 4xx round-trip) | re-verify | re-verify |
| `unverified` (skip-feasibility was used) | verify | verify | verify |

Re-verifying infeasible tasks by default catches platform changes (e.g., a content-policy update) without operator intervention. The cost is one HTTP round-trip per infeasible task per run; trivial.

### 3.8 Benchmark-agnosticity

No code in `phase_2_feasibility.py` mentions `webarena_verified`, `gitlab`, `shopping`, `magento`, `postmill`, or any site string. All benchmark knowledge lives in:

- `EDITOR_REGISTRY` — dispatch keyed on `(benchmark, site)`
- Individual editor classes under `worldsim/editors/` — where `_classify_4xx_response` heuristics live
- `instances.<env>.json` — host config

Adding a new benchmark (e.g., ST-WebAgentBench via `feat/multi-benchmark` per `docs/handoffs/codex-handoff-stwebagentbench-task-subset.md`) needs: new editor classes, new instances file. Zero changes to `phase_2_feasibility.py`.

### 3.9 Concurrency + throughput

Reuse the existing `staggered_worker` pool from `worldsim/eval_worker_pool.py:144–212`. STAGGER_DELAY = 5s startup-spread is appropriate for warm dev instances. At `concurrency=10`, 236 tasks finish in ≈30–60 s wall-clock against a warm r5 instance (each task ≈500 ms HTTP + ≈200 ms cleanup). Cold-start adds a few minutes for token acquisition + first-form-login per site.

Default `--feasibility-concurrency 10`. Drop to 1 to diagnose (a) intra-batch resource collisions or (b) rate-limiting interactions.

### 3.10 Cost

- No agent calls. No Modal sandboxes.
- Just HTTP, ~500 ms per task on average.
- Reuses already-acquired auth tokens from `worldsim/auth_tokens.py` (per-run cache `_RUN_TOKEN_CACHE`).
- **Estimated run cost for 236 tasks: $0.** Wall-clock ≈30–60 s warm; ≈2–3 min cold-start.

### 3.11 Failpoint integration (crash-resume tests)

Phase 2 today exercises crash recovery via `failpoint_base="phase_2.output.adversarial_tasks"` (singular `output`, not `outputs` — verify by grepping; the prior draft of this handoff had the wrong spelling). The failpoint mechanism is `worldsim/failpoints.py` (env var `WORLDSIM_FAILPOINTS`); `write_json_atomic` (`worldsim/atomic_io.py:33–37`) fires `<base>.before_replace` and `<base>.after_replace` automatically when given a `failpoint_base` kwarg.

2c needs equivalents so the same pattern covers the new writes (use the singular `output` to match Phase 2's existing convention):

- `failpoint_base="phase_2.output.feasibility_dataset"` — the in-place rewrite of `adversarial_tasks.json`.
- `failpoint_base="phase_2.output.feasibility_quarantine"` — the new `adversarial_tasks.infeasible.json`.
- `failpoint_base="phase_2.output.feasibility_report"` — the per-site summary file.

State writes use `save_state` (`worldsim/state.py:41`); no failpoint base parameter, but the `crash_if_enabled` hook can be called directly before/after the `save_state` invocation if the crash-resume test wants to interrupt mid-stage. New metadata keys: `feasibility_completed_at`, `feasibility_verified_count`, `feasibility_infeasible_count`, `feasibility_skipped_count`, `feasibility_unverified_count`.

Add at least three crash-resume integration tests (one per write failpoint), patterned after the existing `tests/integration/test_phase_3_*` crash tests. Each spawns the verifier as a child subprocess with `WORLDSIM_FAILPOINTS=<base>.before_replace` set, expects exit code 91 (the failpoint exit code, see `failpoints.py:18`), then re-runs without the env var and asserts the dataset converges.

---

## 4. What changes elsewhere

### 4.1 `worldsim/editors/base.py`

- Add `_classify_4xx_response` hook (default returns `None`, preserving current behavior — see §3.5).
- Insert hook invocation in `_api_request_json`, `_api_request_response`, `_form_get`, `_submit_form`, `_submit_exact_form` between the auth-missing branch and the generic `request_failed` raise.
- (Optional) export `EDITOR_ERROR_KINDS` as a frozenset constant for documentation.

### 4.2 Per-editor overrides

- **`worldsim/editors/gitlab.py`** — override `_classify_4xx_response` to detect GitLab error patterns (`is too long`, `is required`, `has already been taken`, `blocked`, `spam`, `abuse`). The existing `_classify_gitlab_request_error` (line ≈994) is post-hoc; merge or replace as appropriate.
- **`worldsim/editors/reddit.py`** — override to detect PostMill spam protection (typically 422 with `class="form-error"` HTML).
- **`worldsim/editors/shopping.py`** — override to parse Magento's structured JSON `errors[]` field for length / required / policy patterns.
- **`worldsim/editors/shopping_admin.py`** — inherits from `ShoppingEditor`; no change needed unless admin-only patterns appear during testing.

Each override returns `(kind, detail)` where `kind ∈ {"length_exceeded", "field_required", "content_policy"}` and `detail` is a short human-readable string. All other 4xx fall back to `request_failed`.

### 4.3 Phase 3 (`worldsim/phases/phase_3_benign.py`)

`_adversarial_task_errors` (lines ≈90–110) currently only validates schema. After 2c lands, enrich it to flag — but **not invalidate** — benign contracts whose only linked adversarials are all infeasible:

```python
if all(adv.get("feasibility", {}).get("status") == "infeasible" for adv in linked_advs):
    benign_contract["adversarially_exhausted"] = True  # new annotation
```

The benign contract is still valid; it just has no usable adversarial. Phase 4 reads `adversarially_exhausted` and decides whether to run baseline-only. This is an annotation-only change; do not break Phase 3's existing pass/fail semantics.

### 4.4 Phase 4 (`worldsim/phases/phase_4_adversarial.py`) — admission gate with grace period

Add (around the existing admission filter at lines ≈495–528):

```python
status = task.get("feasibility", {}).get("status")
if status == "infeasible":
    skipped_infeasible += 1
    continue
if STRICT_FEASIBILITY_ADMISSION and status != "verified":
    skipped_unverified += 1
    continue
# else (grace period): admit unverified, warn once
if status not in ("verified", "infeasible"):
    if not _grace_period_warning_emitted:
        logger.warning("admitting unverified tasks; flip STRICT_FEASIBILITY_ADMISSION to enforce")
        _grace_period_warning_emitted = True
```

`STRICT_FEASIBILITY_ADMISSION` defaults to `False` in commits 1–2 of the migration plan (§6) and flips to `True` in commit 3. Implement as a module-level constant (clear at read time; no env-var indirection needed since the flip lives in source control). Report both `skipped_infeasible` and `skipped_unverified` in the Phase 4 summary line.

### 4.5 CLI wiring (`worldsim/main.py`)

Add to the `phase` subparser (lines ≈131–183) and mirror in the `resume` subparser (lines ≈185–295):

```python
phase_cmd.add_argument("--skip-feasibility", action="store_true", ...)
phase_cmd.add_argument("--feasibility-only", action="store_true", ...)
phase_cmd.add_argument("--feasibility-instances", default="instances.smoke.json")
phase_cmd.add_argument("--feasibility-concurrency", type=_positive_int, default=10)
phase_cmd.add_argument("--feasibility-retry-count", type=_non_negative_int, default=1)
phase_cmd.add_argument("--feasibility-ttl-hours", type=float, default=None)
phase_cmd.add_argument("--force-reverify", action="store_true", ...)
```

Add `"2c"` to the `choices` list at line 63. In `_dispatch_phase`, route `phase=="2c"` to `phase=="2"` with `args.feasibility_only=True` set.

In `_dispatch_resume` (`main.py:365–477`), add the new fields to the Namespace passthrough at lines ≈411–443 and the synthetic Namespace at lines ≈455–475. State-metadata fallback: `getattr(args, "feasibility_concurrency", state.get("feasibility_concurrency", 10))`.

### 4.6 `scripts/run_integration_tests.sh`

- Add a `@pytest.mark.feasibility` group (the script currently passes `-m integration` at line 119 — extend to `-m "integration or feasibility"` when 2c runs).
- Export a `LIVE_PHASE2C_ARTIFACT` env var pointing at `logs/phase_2/feasibility_report.json`, mirroring the existing `LIVE_PHASE2_ARTIFACT` export at line ≈90.
- Update the script header comment to document the new group.

### 4.7 `scripts/cleanup_webagent_test_resources.sh`

- Extend the GitLab project-search loop to also glob `webagent-verify-` (defensive — 2c shouldn't leave residue, but if a task crashes between editor success and cleanup, the safety net catches it).
- Add a TODO comment noting that Reddit and Magento have no equivalent sweep today (acceptable risk; tracked in §7).

### 4.8 Documentation drift checklist

Every commit in this work updates these in lockstep with the code changes (no "docs lag behind code" commits):

| File | Change |
|---|---|
| `docs/worldsim-v5-technical-specifcation.md` | Add §Phase 2c subsection (~700 words) covering threat-model alignment, error taxonomy, idempotency, output schema. Update §Phase 2 intro to say "three internal stages: 2a planning, 2b text fill, 2c feasibility verification". Update §Phase 4 admission section to note the `feasibility.status == "verified"` gate post-grace-period. Update Pipeline Dependency Graph to include 2c. **Do NOT "fix" the filename typo.** |
| `README.md` | Phase 2 invocation example gains `--skip-feasibility` mention. Prerequisites section gains a one-line note that 2c (unlike 2a/2b) requires a live dev instance. Resume section gains a note that 2c is intra-Phase-2 and idempotent. |
| `CLAUDE.md` | Extend the "Integration test requirement" block (lines 24–26) to include `worldsim/phases/phase_2_feasibility.py` in the trigger list. Add to "What NOT to do": "Do not bypass 2c (`--skip-feasibility`) on shipping runs", "Do not hand-edit `feasibility.status`; trust the gate or re-run 2c". |
| `docs/handoffs/codex-handoff-phase-4-integration-round2.md` | Append a one-line cross-reference footnote: "Update 2026-04-XX: feasibility verification added in Phase 2c — see codex-handoff-phase-2c-feasibility-verification.md". |
| `docs/handoffs/researcher-handoff-project-status.md` | Add 2c to the "what's next" / "what's verified" tables when the work lands. |

---

## 5. Verification (how we know 2c itself works)

### 5.1 Unit tests (mocked, in `tests/test_phase_2_feasibility.py`)

Use the existing `_FakeSession` / `_FakeResponse` monkeypatch pattern from `tests/test_seeding.py:10–89`. **Do not** introduce `respx` or `aresponses` — the repo has no such dependency, and adding one for one new test file is wrong.

| # | Case | Setup | Expected |
|---|---|---|---|
| 1 | 2xx create | editor method mocked to return `{"id": 1}` | task → verified, cleanup called |
| 2 | 400 too long | mock raises `EditorError("length_exceeded", …)` | task → infeasible, kind=length_exceeded, no cleanup |
| 3 | 401 | mock raises `EditorError("auth_missing", …)` | task → infeasible, kind=auth_missing |
| 4 | 500 + retry success | mock raises 500 then returns 2xx | task → verified with `attempts: 2` |
| 5 | 500 + retry fail | mock raises 500 twice | task → infeasible, kind=request_failed |
| 6 | cleanup raises | 2xx create + cleanup raises | task → verified, `cleanup_warnings` non-empty |
| 7 | fingerprint match | task has prior verified fingerprint matching current | skip, no HTTP |
| 8 | fingerprint drift | task verified under different editor_commit | re-verify |
| 9 | task_content drift | task verified, but `editor_calls` content changed | re-verify (task_content_hash mismatch) |
| 10 | multi-call chain | 2 calls, first 2xx second 400 | task → infeasible, first call cleaned up |
| 11 | validator raises | `validate_args` raises | task → infeasible, kind=schema_mismatch, no HTTP |
| 12 | AT-009 regression | fixture identical to AT-009's payload + mocked GitLab returning length_exceeded | task → infeasible, kind=length_exceeded, response_snippet captured |
| 13 | force-reverify path | task already verified, `--force-reverify` set | re-verify, fingerprint refreshed |
| 14 | TTL skip | task verified 2 hours ago, fingerprint drifts, `--feasibility-ttl-hours 24` set | skip |
| 15 | token cache miss | `acquire_tokens_for_instances` raises | `verify_feasibility` raises before launching workers |
| 16 | unsupported_site | task references a site with no editor in registry | task → infeasible, kind=unsupported_site |

### 5.2 Integration tests (live r5, in `tests/integration/test_phase_2_feasibility_live.py`)

Mark `@pytest.mark.integration @pytest.mark.feasibility`. Fixtures live in `tests/integration/fixtures/feasibility/<site>/{good,oversize,policy}.json`, one per site, schema-identical to entries in `adversarial_tasks.json`.

Per site:

- `test_<site>_feasibility_good_task` — synthetic task known to POST, expect verified + cleanup + zero residue.
- `test_<site>_feasibility_oversize_task` — synthetic task with payload above the site's known length cap, expect infeasible + kind=length_exceeded.
- `test_<site>_feasibility_policy_task` — synthetic task containing a content-policy trigger string, expect infeasible + kind=content_policy (skip if site has no content-policy filter; document which).

Cross-site:

- `test_feasibility_concurrency_10` — 10 tasks in parallel across 4 sites, expect all verified in ≤30 s.
- `test_feasibility_cleanup_leaves_no_residue` — after run, `GET /api/v4/projects?search=webagent-verify-` returns 0; the same check for `webagent-task-` returns 0 *immediately after the run* (per-task immediate cleanup is the contract).
- `test_feasibility_resume_after_partial` — kill mid-run via failpoint, re-run, expect only the unverified subset re-attempted.

### 5.3 End-to-end criteria

Run against the current 236-task dataset. Success:

- ≥90% verified rate (otherwise redesign the length budget layer in Phase 2b).
- AT-009 confirmed `length_exceeded` (not `request_failed`).
- `webagent-verify-*` residue on r5 = 0 after the run.
- `webagent-task-*` residue from 2c run = 0 (per-task cleanup is the contract).
- `logs/phase_2/feasibility_report.json` summary includes per-site breakdown.
- Total run time ≤5 min at default concurrency.

### 5.4 Manual smoke checklist

Before pushing each commit to `feat/worldsim-v5`, Codex runs by hand:

```bash
# 1. Lint + unit tests
uv run pytest tests/ -q --ignore=tests/integration

# 2. Live feasibility check on a 4-task subset
uv run python -m worldsim.main phase 2 --feasibility-only \
  --feasibility-instances instances.smoke.json \
  --feasibility-concurrency 4 \
  --max-tasks-per-site 1

# 3. Full integration sweep
bash scripts/run_integration_tests.sh --host-config configs/benchmark_hosts/r5.yaml
```

All three must pass. Capture output for the commit message body.

---

## 6. Migration for existing datasets

Step-by-step so we don't break in-flight runs:

Each step below is a **single atomic commit on `feat/worldsim-v5`**, pushed after its own CI run lands green. No PRs, no feature branch. Sequence the pushes; do not push all four at once.

1. **Commit 1**: 2c module + CLI flags + editor `_classify_4xx_response` hooks land. Phase 4 stays in **grace mode** (`STRICT_FEASIBILITY_ADMISSION = False`) — admits tasks where `feasibility` is missing OR `feasibility.status == "verified"`; **skips** explicitly-`infeasible` tasks (with a one-time warning). Unit + integration tests land here. Documentation drift updated in lockstep.

2. **Commit 2**: run 2c once against live r5, commit the enriched `adversarial_tasks.json` + `adversarial_tasks.infeasible.json` + `feasibility_report.json`. Grace mode still active. Commit message body includes the report summary.

3. **Commit 3**: flip `STRICT_FEASIBILITY_ADMISSION = True`. Old dev checkouts that haven't pulled the enriched dataset will break on their next Phase 4 run; announce in #engineering before pushing so collaborators rebase/pull first. Update CLAUDE.md "What NOT to do" to add: "Do not run Phase 4 on a dataset that hasn't been through 2c — admission is strict now."

4. **Commit 4**: nightly cron hook that re-verifies the dataset against the dev host (with `--feasibility-ttl-hours 24` to skip recently-verified tasks) so silent platform drift surfaces early. Document in README.

Each commit is individually revert-safe. Commit 3 is the only one with potential to disrupt collaborators on the branch; coordinate the push. Never force-push `feat/worldsim-v5` — fix forward with follow-up commits if needed.

---

## 7. Risks + mitigations

| Risk | Mitigation |
|---|---|
| Cleanup drift leaks resources | Per-task immediate `SeedCleanupHandle.cleanup()`. `scripts/cleanup_webagent_test_resources.sh` extended to glob `webagent-verify-` as a safety net. |
| Dev instance drift (e.g., gitlab `SECRET_KEY_BASE` rotation invalidates tokens) | `host_fingerprint` captures host + editor + dataset commit + task content; drift invalidates verification, triggers re-verify on next run. |
| Transient 5xx quarantines feasible task | 1-retry default, exponential backoff. Manual `--force-reverify` escape hatch. Default re-verify of infeasible tasks each run catches transient quarantines automatically. |
| Rate limiting on bursty writes (or bursty deletes during cleanup) | Configurable `--feasibility-concurrency`. Drop to 1 on rate-limit detection in editor layer. STAGGER_DELAY=5 spreads worker startup. |
| Verification host ≠ Phase 4 host | `host_fingerprint.host_config` records both; warning if Phase 4's instances file differs from the one verification used. Tasks verified against host A aren't necessarily feasible against host B (different data snapshots). |
| Content-policy false positives (gitlab may accept today, reject tomorrow after policy update) | 2c re-runs are cheap; treat verification as perishable. Default re-verify on `infeasible` tasks each run; nightly cron in commit 4. |
| WASP-style "reuse existing" drift on get-or-create paths (e.g., gitlab project) | The editor already does the GET-then-POST dance; verification is idempotent by design. Multiple verify runs do not multiply resources. |
| **Intra-batch resource collision** — two parallel 2c tasks render the same resource name and the second 4xx's "already exists" | Default `--feasibility-concurrency 10` makes this rare on the 236-task dataset; reproduce with `--feasibility-concurrency 1`. Long-term mitigation is per-task name suffixing in editor layer (out of scope). |
| **Partial-state leak on SIGINT / OOM** — interrupt mid-task means `_cleanup_stack` doesn't run; resource leaks | The cleanup script is the safety net. Document that operators should run it after any interrupted 2c. Future work: install signal handlers that flush in-flight cleanup stacks. |
| **Fingerprint thrash during local dev** — every editor commit invalidates 236 verifications | `--feasibility-ttl-hours N` opt-in skip catches the worst of this for local iteration. Nightly cron in commit 4 keeps main fresh. |
| **Fixtures going stale** — synthetic good/oversize fixtures pinned to today's GitLab/Magento behavior | Fixtures live next to their integration tests; bump them in the same commit that updates the editor when the platform evolves. Document this in CLAUDE.md. |
| **OAuth token expiry mid-run** — `_RUN_TOKEN_CACHE` doesn't refresh | Tokens are minted per run by `acquire_tokens_for_instances`; a 5-min run is well within the typical 1h token TTL. If a future run takes longer, add a refresh loop; today, accept the risk. |
| **Reddit / Magento have no cleanup-script equivalent** | Acknowledged. Per-task immediate cleanup via `SeedCleanupHandle` covers the happy path. Add the equivalent sweepers as part of *this* work (commit 1 or 2) — Reddit `delete_forum`/`delete_submission` and Magento product-review delete already exist on the editor classes; the script extension is ~30 lines per site. Self-contained mandate at top of doc applies. |
| **Shared instance dict mutation under parallel workers** | `verify_one` shallow-copies the instance via `dict(instance)` before injecting `seed_task` (§3.4). Without this, two workers on the same site race on the seed_task key while `_build_seed_context` is mid-render. |
| **`validate_data_seed` raises `ValueError`, not `EditorError`** | `worldsim/seeding.py:162,173,177,…` raise plain `ValueError`. The verifier catches both (§3.4) and remaps `ValueError` → `EditorError("schema_mismatch", …)`. |
| **`acquire_tokens_for_instances` returns errors as a list, not by raising** | Pre-flight (§3.4.0) checks the return value and raises `RuntimeError` with the joined error list. Mirrors Phase 4's pattern (`phase_4_adversarial.py:632`). |
| **Loaded `instances.smoke.json` is a wrapper dict, not a list** | The wrapper has top-level `benchmark_name`, `verification_proxy`, `url_placeholders`, `instances`. The CLI loader extracts `["instances"]` before passing to `verify_feasibility`. Document in §1.4.2. |
| **Phase 2 `partial_complete` status leaves a partial dataset** | 2c verifies whatever is in `adversarial_tasks.json`; partial verification is fine and reflects upstream truth. Pre-flight (§3.4.0) does not refuse to run on partial datasets, but the report header records `phase_2_status` so reviewers see the qualifier. |
| **Phase 2 status flags get clobbered by 2c rewrite** | Phase 2c only sets `phase_2_stage="feasibility"` via `save_state`; it does not touch `status` (preserves `complete` vs `partial_complete` from 2b). |
| **Empty `editor_calls` list passes silently** | `verify_one` checks `handle is None` after `apply_data_seed_async` and marks empty seeds `infeasible` with `kind="empty_seed"` rather than admitting them as trivially-verified (§3.4 pseudocode). Surfaces upstream Phase 2 bugs instead of hiding them. |
| **`STRICT_FEASIBILITY_ADMISSION` is hard to override in emergencies** | It's a module-level constant in commit 3, but also exposed as the env var `WORLDSIM_STRICT_FEASIBILITY` for break-glass overrides (default to source-controlled value if unset). Document in CLAUDE.md and the spec. |

---

## 8. Non-goals (explicitly out of scope for this handoff)

- **Auto-regeneration of infeasible tasks.** 2c quarantines; a human or a future Phase 2d decides what to do. First version is strictly "does it POST, yes/no". Auto-regen would need length-budget-aware regeneration of just the broken field; separate design conversation.
- **Static length-cap inference.** Parsing OpenAPI specs or scraping `<input maxlength>` is a good optimization but orthogonal; 2c is the runtime backstop regardless. Implement static inference in a later commit; it makes 2c emit fewer infeasible tasks, it doesn't replace 2c.
- **Content-policy exploration.** 2c reports `content_policy` as an error kind. Understanding *which* content policies are active on each platform is research, not infrastructure.
- **Security validation.** 2c does not judge whether the attack payload is "good" — it only proves the platform accepts it.
- **Benchmark-environment lifecycle management.** As ever, 2c uses the running dev instance. It does not start, stop, snapshot, or seed the DB directly.
- **Map attack model (§15 of the spec).** Still deferred. If a map editor lands later, 2c consumes it transparently via `EDITOR_REGISTRY`.
- **Extending the cleanup script to Reddit / Magento.** Acknowledged gap; tracked in §7. Per-task immediate cleanup is the primary mechanism; sweep is a defensive belt-and-suspenders that we extend incrementally.
- **Signal-handler-driven flush of in-flight cleanup stacks.** Reasonable but separate scope.
- **Replacing `requests` with an async HTTP client throughout the editor layer.** That's a much bigger refactor; `asyncio.to_thread` is fine for now.

---

## 9. Acceptance criteria

1. `uv run pytest tests/ -q --ignore=tests/integration` — all green, +~16 new unit tests.
2. `bash scripts/run_integration_tests.sh --host-config configs/benchmark_hosts/r5.yaml` — all feasibility tests pass.
3. `uv run python -m worldsim.main phase 2c --feasibility-instances instances.smoke.json` — finishes in ≤5 min on the current 236-task dataset.
4. `logs/phase_2/feasibility_report.json` summary: ≥90% verified, AT-009 flagged `length_exceeded`, zero `webagent-verify-*` residue on r5 after run.
5. Every verified task has `feasibility.host_fingerprint.task_content_hash` populated.
6. Phase 4 admission summary line shows both `skipped_infeasible` and `skipped_unverified` counts.
7. `scripts/cleanup_webagent_test_resources.sh` extended to glob `webagent-verify-` and to sweep Reddit + Magento resources; a successful sweep result is recorded in the commit 1 (or 2) message body.
8. CLAUDE.md "Integration test requirement" section extended; spec §Phase 2c subsection added; README mention added — all in lockstep with the code commits.
9. Each commit message body includes: pytest summary, integration-test output, feasibility-report summary (commit 2), before/after per-site verified counts (commit 2), list of all docs touched in that commit.

---

## 10. Commit message template (commit 1 example)

Use a conventional-commits subject plus a structured body. Each commit on `feat/worldsim-v5` uses this shape; adapt the body per commit (e.g., commit 2 replaces "What ships" with "Dataset enrichment result" and shows the report summary; commit 3 is a short flip with before/after admission counts).

```
feat(phase2c): add feasibility verification sub-stage (grace mode)

Adds Phase 2c feasibility verification as an internal sub-stage of Phase 2.
Every adversarial task gets POSTed against a live dev instance; tasks that
2xx are tagged `feasibility.status="verified"` with a fingerprint; tasks
that fail are quarantined to `adversarial_tasks.infeasible.json`. Phase 4
runs in grace mode this commit; commit 3 will flip to strict admission.

What ships:
- worldsim/phases/phase_2_feasibility.py (new, ~150 LOC)
- worldsim/_async_utils.py (new, retry helper)
- worldsim/editors/{base,gitlab,reddit,shopping}.py — _classify_4xx_response hook
- worldsim/phases/{phase_2_injections,phase_3_benign,phase_4_adversarial}.py — wiring
- worldsim/main.py — CLI flags + resume passthrough
- tests/test_phase_2_feasibility.py + tests/integration/test_phase_2_feasibility_live.py + fixtures
- scripts/{run_integration_tests,cleanup_webagent_test_resources}.sh — updates

Docs touched (in this commit, not a follow-up):
- docs/worldsim-v5-technical-specifcation.md (§Phase 2c added)
- README.md (CLI flags + prerequisites)
- CLAUDE.md (integration-test trigger; what NOT to do)
- docs/handoffs/codex-handoff-phase-4-integration-round2.md (cross-reference footnote)

Test evidence:
- pytest: <paste summary>
- integration tests: <paste summary>
- feasibility_report.json summary: <paste per-site table>
- before/after dataset feasibility counts: <paste>
- cleanup sweep result: <paste>
```

---

## 11. What NOT to do

- Don't add `if benchmark == "webarena_verified"` or `if site == "gitlab"` anywhere in `phase_2_feasibility.py`. All such knowledge lives in editor classes or the registry.
- Don't duplicate the editor cleanup machinery — use `SeedCleanupHandle` from `worldsim/seeding.py:93`.
- Don't reimplement `apply_data_seed_async` — call it.
- Don't try to `await` synchronous editor methods directly. They return values, not coroutines. `apply_data_seed_async` already wraps them with `asyncio.to_thread`.
- Don't introduce a `webagent-verify-` prefix scheme as the resource-isolation primitive — it doesn't actually isolate, because resource names come from `args["name"]`, not `task_id`. Per-task immediate cleanup is the primitive.
- Don't add `phase_2c` to `_PHASE_ORDER` or create a `phase_2c` state key. 2c is `phase_2_stage="feasibility"` within Phase 2.
- Don't standardize on flat `feasibility_status`. Always nested `feasibility.status`.
- Don't pull in `respx`, `aresponses`, or `tenacity`. Use the existing `_FakeSession` pattern and the inline retry helper.
- Don't hit production gitlab/Magento/postmill. Dev instances only. `--feasibility-instances` defaults to `instances.smoke.json` for a reason.
- Don't write a DB-level verifier. HTTP 2xx is the postcondition (same rule as Phase 4 post-pivot).
- Don't import from `AgentLab/`. Feasibility is brand-new code, not a port.
- Don't make 2c block Phase 4 on the grace-period commits — the migration plan is explicit that commits 1+2 admit unverified tasks.
- Don't skip hooks (`--no-verify`) or force-push.
- Don't "fix" the `worldsim-v5-technical-specifcation.md` filename typo.

---

## 12. Reference files (read these before starting)

- **Anchor handoff:** `docs/handoffs/codex-handoff-phase-4-integration-round2.md` — pivot that introduced the editor architecture this 2c step plugs into.
- **Editor base:** `worldsim/editors/base.py:124–339` — `BaseSiteEditor`, `EditorError`, cleanup stack, HTTP entry points (where the `_classify_4xx_response` hook gets injected).
- **Editor registry:** `worldsim/editors/__init__.py:9–14` — the `(benchmark, site)` → editor class dispatch.
- **Seeding runtime (load-bearing):** `worldsim/seeding.py:93` (`SeedCleanupHandle`), `277–339` (`apply_data_seed`), `342–346` (`apply_data_seed_async`), `349–377` (`_build_seed_context`), `568–595` (`_get_editor_for_seed_call`), `598–629` (`_apply_editor_seed_call`). Read these before writing `verify_one`.
- **Auth tokens:** `worldsim/auth_tokens.py:217–254` (`acquire_token`), `314–369` (`acquire_tokens_for_instances`).
- **Worker pool:** `worldsim/eval_worker_pool.py:35` (`STAGGER_DELAY`), `144–212` (`staggered_worker`), `214–367` (`run_eval`).
- **Atomic writes:** `worldsim/atomic_io.py:14` (`write_json_atomic`).
- **State:** `worldsim/state.py:25–32` (`get_state_dir`, `get_state_file`), `41` (`save_state`).
- **CLI:** `worldsim/main.py:51–183` (phase subparser), `185–295` (resume subparser), `340–351` (`_PHASE_ORDER`), `365–477` (`_dispatch_resume`).
- **Phase 4 preflight:** `worldsim/phases/phase_4_adversarial.py:131–141` (`SeedPreflightMismatch`), `2926–2983` (`_probe_seed_base_state`). The admission filter at lines ≈495–528.
- **Phase 3 contract validation:** `worldsim/phases/phase_3_benign.py:90–110` (`_adversarial_task_errors`), `128` (input path), `184–212` (output shape).
- **Integration harness:** `scripts/run_integration_tests.sh` — line 119 invokes `pytest -m integration`. Add the feasibility group there.
- **Existing cleanup script:** `scripts/cleanup_webagent_test_resources.sh` — GitLab-only sweep today; extend the glob.
- **Test patterns:** `tests/test_seeding.py:10–89` — `_FakeSession` / `_FakeResponse` / `_FakeDbConnection` pattern. Mirror it; do not introduce `respx`.
- **WASP source for precedent:** `github.com/facebookresearch/wasp` — `webarena_prompt_injections/environment_setup.py`.
- **Dataset:** `logs/phase_2/adversarial_tasks.json` — target for the first real-world 2c run.
- **AT-009 evidence:** demo log `logs/phase_4_demo_20260417_*.log` — shows the HTTP 400 and the "too long" message.
- **Multi-benchmark forward look:** `docs/handoffs/codex-handoff-stwebagentbench-task-subset.md` — when ST-WebAgentBench editors land on `feat/multi-benchmark`, 2c picks them up via the registry with no code changes.

---

## 13. Definition of done

There is no time budget. Ship when, and only when, **all** of the following hold simultaneously:

- All §9 acceptance criteria pass on a clean checkout against r5.
- All four migration commits (§6) are pushed to `feat/worldsim-v5` in order; each passed local CI before being pushed; none were squashed, amended, or reordered after push.
- Every file in the §4.8 documentation-drift table has been updated in the same commit that introduces the corresponding code change (no doc-lag follow-up commits).
- AT-009 specifically lands in the `infeasible` quarantine with `kind="length_exceeded"`, `http_status=400`, response_snippet preserved. This is the load-bearing regression. If it lands as `request_failed`, the GitLab `_classify_4xx_response` override is incomplete; iterate until it classifies correctly.
- All four editors (`gitlab`, `reddit`, `shopping`, `shopping_admin`) have working `_classify_4xx_response` overrides covering at least the three additive kinds (`length_exceeded`, `field_required`, `content_policy`). None can be deferred — the dataset has tasks for all four.
- Reddit and Magento sweep additions land in `scripts/cleanup_webagent_test_resources.sh` in commit 1 or 2, not deferred.
- A clean re-run of `phase 2c` on the verified dataset is a no-op (every task skipped via fingerprint match). This proves idempotency.
- A re-run of `phase 2c --force-reverify` re-verifies every task and produces an identical report (modulo timestamps). This proves determinism.
- `git log feat/worldsim-v5 --oneline` shows the four commits in order with the expected subjects; no force-pushes appear in the reflog.

If any of these fails, do not ship — fix forward with an additional commit and re-verify. The mandate at the top of this doc is "self-contained end-to-end on `feat/worldsim-v5`"; partial delivery is not acceptable.

---

## 14. How to call this done

Four commits on `feat/worldsim-v5`, pushed in order, each green on CI before the next is pushed. No PRs; no feature branch. Each commit individually satisfies the §9 acceptance criteria for its scope. The commit that flips `STRICT_FEASIBILITY_ADMISSION = True` (commit 3) is the moment the research claim "our dataset is feasibility-verified against a dev host of known commit" becomes true and `error` in Phase 4 summaries can be reliably attributed to infra flake or agent action, never to dataset-side infeasibility.

After commit 4 lands, update this handoff's `Status:` field from "not started" to "shipped YYYY-MM-DD" in a final small commit.

That's the payoff.

---

## 15. Appendix — design bugs caught during pre-implementation review (do not reintroduce)

Recording these so future revisions don't regress to earlier drafts. Each was a real defect found by reading the actual code; the body of this handoff already encodes the fix.

1. **Editors are sync, not async.** Earlier draft had `await retrying(method(...))` directly on editor methods. Editors use `requests.Session` (sync). Fix: use `apply_data_seed_async`, which wraps in `asyncio.to_thread` (`worldsim/seeding.py:342–346`).
2. **`apply_data_seed_async` and `SeedCleanupHandle` already exist.** Earlier draft proposed a 350-LOC re-implementation. Fix: call existing helpers; verifier shrinks to ~150 LOC.
3. **Per-call editor instantiation breaks `_cleanup_stack` LIFO and re-does form logins.** Fix: reuse `_get_editor_for_seed_call`'s per-task editor cache (`seeding.py:568–595`).
4. **`webagent-verify-{task_id}` prefix scheme doesn't actually rename resources.** Names come from `_slugify(args["name"])` (`gitlab.py:1001`). Fix: trust per-task immediate cleanup; drop the false guarantee.
5. **`feasibility_status` (flat) vs `feasibility.status` (nested) inconsistency.** Fix: nested everywhere.
6. **Error-taxonomy table undercounted existing kinds.** Real kinds at `base.py:154,165,199,204,211,244,248,255` plus per-editor specific kinds. Fix: enumerate accurately; add three additive kinds (`length_exceeded`, `field_required`, `content_policy`) via a new `_classify_4xx_response` hook injected before the generic `request_failed` raise.
7. **`retrying` helper invented, no spec.** Fix: ~40-LOC inline helper with explicit retry-only-on-`request_failed`/`unreachable` semantics.
8. **Token acquisition lifecycle missing.** Fix: §3.4.0 pre-flight calls `acquire_tokens_for_instances(instances)` once, raises on non-empty error list.
9. **`--feasibility-only` × current-status truth table under-specified.** Fix: explicit table in §3.7.
10. **Fingerprint missing `task_content_hash`.** Manual edits to `editor_calls` would not invalidate verification. Fix: 12-char sha256 prefix of canonical-json'd `editor_calls`.
11. **Phase 4 grace-period semantics under-specified for `infeasible`-tagged tasks.** Fix: explicit rules in §4.4 — admit unverified, skip explicit-infeasible, throughout grace; flip to strict in commit 3.
12. **Cleanup script sweeps GitLab only.** Fix: extend in same commit (1 or 2) to Reddit and Magento (per §13 Definition of Done).
13. **`_PHASE_ORDER` placement vacillation.** Fix: 2c is `phase_2_stage="feasibility"`; no new top-level phase; CLI `phase 2c` is sugar.
14. **`write_json_atomic` not specified for outputs.** Fix: required for all three file writes; failpoint bases attached.
15. **Failpoint hooks missing.** Fix: three named bases, `phase_2.output.feasibility_*` (singular `output`, matching Phase 2's existing convention).
16. **Mechanism-branching mismatch.** Earlier draft assumed `mechanism ∈ {api, form, state_push}`. Real data: `mechanism="editor"` for 100% of tasks. Fix: don't branch in the verifier; trust `apply_data_seed_async`'s dispatcher.
17. **`validate_data_seed` raises `ValueError`, not `EditorError`.** Verifier's narrow `except EditorError` would crash. Fix: catch both; remap `ValueError` → `EditorError("schema_mismatch", ...)`.
18. **`acquire_tokens_for_instances` returns errors as a list, doesn't raise.** Earlier pre-flight pseudocode assumed raise. Fix: check return value, raise `RuntimeError` on non-empty.
19. **`instances.smoke.json` is a wrapper dict, not a list.** Top-level keys include `benchmark_name`, `verification_proxy`, `url_placeholders`, `instances`. Fix: load via `json.load(f)["instances"]`; document the wrapper shape in §1.4.2.
20. **Empty `editor_calls` would pass silently.** `apply_data_seed` returns None and `verify_one` would proceed to "verified". Fix: explicit `handle is None` check → `kind="empty_seed"` infeasible.
21. **Shared-instance-dict mutation race under parallel workers.** Two `verify_one` coroutines on the same site would race on `instance["seed_task"]`. Fix: shallow-copy via `dict(instance)` before mutation.
22. **`STRICT_FEASIBILITY_ADMISSION` had no break-glass override.** Fix: source-controlled constant + `WORLDSIM_STRICT_FEASIBILITY` env var override.
23. **Phase 2 `partial_complete` status handling unspecified.** Fix: 2c verifies whatever is present; report header records `phase_2_status` so reviewers see the qualifier.
24. **Site-instance pre-validation missing.** Earlier draft would discover missing instances 200 tasks deep. Fix: pre-flight (§3.4.0) computes `sites_in_tasks - sites_in_instances` and fails fast.
25. **`probe_base_state` not pre-flighted.** Earlier draft would discover dead instances per-task. Fix: pre-flight (§3.4.0) probes once per `(benchmark, site)` pair before launching workers.

If a future revision reintroduces any of these, treat it as a regression and revert.

---

## 16. Appendix — post-ship bugs caught by 2026-04-18 audit (fixed in `0cfb6513` + `e0b4b07b`)

Follow-up audit run immediately after the four-commit ship found six latent defects. Four are Phase 2c-specific (commit `0cfb6513`); two are pipeline-wide hardening (commit `e0b4b07b`). All six had test coverage added alongside the fix. +19 new unit tests; 855 passed / 2 skipped post-audit.

1. **CRITICAL — Idempotency-skip corrupts the dataset on re-run.** `phase_2_feasibility.py:266` overwrote `feasibility.status="verified"` with `"skipped"`, and `:223` added the corrupted result to the `verified` list written back to `adversarial_tasks.json`. Phase 4's strict admission gate (`phase_4_adversarial.py:626`) admits only `status == "verified"`, so the nightly cron from `ed7cb377` would have flipped every task to `status="skipped"` on its first warm run and Phase 4 would reject the whole dataset as `skipped_unverified`. Self-healed only via `--force-reverify`, which no operator would reach for without the outage first. Fix: preserve prior `status="verified"` verbatim on skip; record the skip fact on sibling fields `last_reverify_skipped_at` / `last_reverify_skip_reason`. Aggregator now keys the "skipped" report bucket off presence of `last_reverify_skipped_at`, not a status value. `_idempotency_decision` returns `(decision, reason)` so the skip stanza records whether it was fingerprint_match or ttl_hours.

2. **HIGH — Case 7 unit test encoded the bug.** `tests/test_phase_2_feasibility.py:368` asserted `status == "skipped"` post-skip, so CI stayed green while the corruption lived. Case 14 had the same masking for the TTL path. Fix: both cases now assert `status == "verified"` with `last_reverify_skipped_at` populated. Added Case 7b regression: three consecutive runs on the same dataset must not drift `status` or `verified_at`.

3. **MEDIUM — Phase 3 exhaustion check missed `"skipped"` status.** `phase_3_benign.py:114` only treated `status == "infeasible"` as exhausted. If a dataset on disk still carries pre-fix `status="skipped"` tasks from an earlier buggy build, the `adversarially_exhausted` annotation wouldn't fire and Phase 4 gets handed a benign whose only adversarial is Phase-4-inadmissible. Fix (defense-in-depth): `status in ("infeasible", "skipped")` triggers the annotation.

4. **LOW — `--feasibility-retry-count 0` silently rewrote to `1`.** `phase_2_injections.py:717` used `int(getattr(args, "feasibility_retry_count", None) or 1)`. The `or` fallback treats `0` as falsy; operators debugging a transient failure who wanted no retries got one anyway. Fix: explicit `None` sentinels; `0` now means "single attempt". Same pattern applied to `--feasibility-concurrency`.

5. **LOW — Uncaught `OSError` on instances / tasks file reads.** `phase_2_injections.py:669,704` caught `JSONDecodeError` only; permission errors, concurrent deletes, and other I/O failures produced an uncaught traceback instead of a clean error-code return. Fix: `except (json.JSONDecodeError, OSError)`.

6. **Log clarity.** `phase_2_injections.py:774-780` printed "N verified / Y skipped", which operators read as "N admitted, Y no-op". Post-fix, the fresh-vs-reused distinction is load-bearing (since reused tasks now also count as admitted). Now prints `N admitted (X fresh + Y reused via idempotency) / Z infeasible`.

Pipeline-wide hardening (commit `e0b4b07b`):

7. **MEDIUM — `atomic_io.write_json_atomic` silently downgraded file permissions to `0o600` on every rewrite.** `tempfile.mkstemp` hardcodes `0o600` on CPython regardless of umask. Four Phase 2 dataset files committed to git as mode `100644` existed on disk as `0o600` — a live divergence git cannot show via `git diff` (git tracks only the executable bit). Fix: probe umask once at import (thread-safe for the async worker pool), stat the destination before writing, chmod the tempfile to either the existing mode (preserving operator intent) or `0o666 & ~umask` (typically `0o644`). Does not retroactively heal files already at `0o600`; operators need `chmod 644 logs/phase_2/*.json` once, future writes then preserve `0o644`.

8. **MEDIUM — `retrying` did not retry `cleanup_failed`.** `BaseSiteEditor.cleanup()` attempts every unwind DELETE and, if any fail, raises `EditorError("cleanup_failed")` — *replacing* the original `request_failed` that triggered the unwind. `_async_utils._DEFAULT_RETRY_KINDS` did not include `cleanup_failed`, so the retry short-circuited and tasks were marked infeasible with the wrong kind. The common victim: GitLab's nginx-fronted 502/504 window where a create POST and its matching DELETE fall inside the same burst. Fix: add `cleanup_failed` to the default retry kinds. Idempotent editor methods (`create_group` / `create_issue` / `create_forum` via GET-then-POST, `create_project` via `_reap_preexisting_project` on `webagent-task-*` prefix) recover naturally on the next attempt. Also added optional `on_retry` hook to `retrying` for future callers that may need per-platform residue sweeps (not wired in Phase 2c — not needed given current editor idempotency).

New test files introduced by the audit:

- `tests/test_atomic_io.py` — 7 cases covering default mode, preservation of `0o644`/`0o600`/`0o640`, no tmpfile leftover on success or on mid-write exception.
- `tests/test_async_utils.py` — 12 cases covering default retry kinds, transient-kind retry, 4xx no-retry, retry exhaustion, `on_retry` firing boundaries, custom `retry_on`, and attempts-log bookkeeping.

If a future revision reintroduces any of these, treat it as a regression and revert.
