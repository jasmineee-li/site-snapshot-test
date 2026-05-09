# Snapshot Architecture Cutover — Comprehensive Handoff

> **SUPERSEDED 2026-04-21.** This cutover was NOT pursued. See [`wasp-aligned-scoping-decision.md`](./wasp-aligned-scoping-decision.md) for the adopted alternative (drop Magento, restrict to content-heavy environments per WASP precedent).
>
> Three Opus critique agents exposed two foundational flaws in this plan: (1) Showstopper #1 (Phase 3 contamination) was based on a false premise — `worldsim/phases/phase_3_benign.py` does zero `reset_endpoint`/`apply_data_seed` calls. (2) Showstopper #2 (reset_endpoint conflict) collapsed when investigation showed WebArena env-ctrl `init` doesn't restore container baseline. Plus the Magento driver Step 5 (`indexer:reindex review_summary`) is a hard CLI error — no such indexer exists in stock Magento 2.4. The user's research into WASP §3.1 confirmed Magento was excluded for **methodological** reasons (transactional, not content-heavy), making the entire cutover unnecessary. **Preserved here as decision provenance only.**

**Status:** SUPERSEDED. Original status was PROPOSAL UNDER REVIEW.
**Origin:** 2026-04-21 conversation following the killed Phase 4 run on r5 (~$9, 48 trajectories) caused by Magento review-pending bug. Three layers of defense shipped on `feat/worldsim-v5` (commits `1fc8a2aa`, `da2c8618`, `f364ee74`) work but feel architecturally brittle.
**Authors:** synthesis from 6 subagent investigations (3 research + 3 adversarial critique, Opus + Sonnet mix).
**Audience:** future operator picking up this work. Read top-to-bottom.

---

## Why this exists

WorldSim v5's per-task editor seeding model fights every platform's user-facing API at runtime (Magento moderation, Postmill spam queues, GitLab CSRF). The current 3-layer fix works but is unprecedented — no published IPI benchmark seeds a moderation-walled platform via runtime API calls. The field's actual best practice is to **pre-position adversarial content in the environment snapshot**, not to seed at runtime.

Two precedents:
- **VWA-Adv** (Wu et al., ICLR 2025) — adversarial content placed at dataset-construction time, restored via WebArena reset-token mechanism. Closest to "best architecture."
- **ST-WebAgentBench** (Levy et al., ICLR 2026) — `mysql -u admin -p < demo_data.sql` loaded once. Simple but tightly coupled to one platform.

The proposed cutover takes VWA-Adv's snapshot model + WebArena's `reset_endpoint` contract + ST-WebAgentBench's "use whatever DB mechanism is most reliable" + WASP's threat-model writeup. Encapsulated behind a `SnapshotDriver` Protocol with per-platform implementations.

**This document is the comprehensive plan.** Three Opus subagents attacked it adversarially; their critiques reshaped the design. Read the SHOWSTOPPERS section before doing anything else.

---

## SHOWSTOPPERS — must resolve before Stage 1

All three Opus critiques converged on these. Each is binary: until resolved, the cutover is dead-on-arrival.

### SHOWSTOPPER 1 — Phase 3 contamination

`worldsim/phases/phase_3_benign.py` measures benign agent capability. It uses the same `reset_endpoint` as Phase 4. **If `reset_endpoint` now restores to a snapshot containing 174 adversarial reviews, Phase 3's benign runs will see those reviews.** This contaminates the benign baseline — agents-vs-pre-injected-environment is not a benign baseline, it's a different experiment. Per CLAUDE.md Principle #4, baseline capability is a Phase 4 byproduct ("benign-under-attack"); the snapshot model invalidates that guarantee.

**Three options, none free:**

- **A. Two snapshots per host** — `snap_benign` (vanilla) and `snap_adversarial` (with payloads). `reset_endpoint` takes a snapshot-name parameter. **Requires env-ctrl modification** (out of scope per Principle #1: "The orchestrator does not manage environments"). Negotiate with env-ctrl maintainer or fork.
- **B. Two benchmark hosts** — r5 for Phase 4 (adversarial-loaded), r6 for Phase 3 (benign baseline). Doubles infra cost. Doubles drift surface.
- **C. Keep editor path alive for Phase 3 only** — Stage 3 deletion is partial. Phase 4 uses snapshot, Phase 3 uses editors. Less clean architecturally but lower blast radius.

**Decision required from operator before Stage 1.** Option C is the pragmatic recommendation: it accepts the asymmetry to ship faster and avoids env-ctrl modifications.

### SHOWSTOPPER 2 — `reset_endpoint` semantics conflict

WebArena's env-ctrl `reset_endpoint` (`http://3.12.221.9:7771/init`) restores the **container's baseline image**. Adversarial data we INSERT lives ON TOP of that baseline. **After the first `reset_endpoint` call, the snapshot is gone.**

**Three options, all bad:**

- **A. Bake adversarial data into a new container baseline image.** Requires `docker commit` + image push + env-ctrl reconfiguration. ~20 min per platform per Phase 2b regen. Operator-side workflow.
- **B. Re-apply SQL after every reset.** Programmatically calls SQL load after each `reset_endpoint` POST. **Functionally equivalent to per-task seeding, just batched.** Negates much of the snapshot benefit but works without env-ctrl changes.
- **C. Skip reset_endpoint between tasks.** Cross-task contamination from agent mutations. Unacceptable for rigor runs.

**Decision required from operator before Stage 1.** Option A is architecturally cleanest but slow per Phase 2b iteration. Option B is the pragmatic recommendation if Phase 2b regen frequency is high. Option C is wrong.

### SHOWSTOPPER 3 — Threat-model writeup

CLAUDE.md Principle #1 (binding):
> "Data seeding is per-task only, via `apply_data_seed` (api/form channels only). SQL seeding is excluded from the evaluation methodology because it violates the threat model (a regular authenticated user cannot write to the database directly)."

The 2026-04-15 commit `962eec78` killed SQL seeding for this reason. Reversing requires an explicit, defensible amendment that survives paper review (per critique #1: "expect an R2 rejection without it").

**The defensible framing** (synthesizing critiques 1 + 2):
> "Adversarial content is part of the *environment construction*, not part of the *attack*. The threat model is the agent encountering attacker-authored text on a legitimate web page; the mechanism by which that text arrived in the page is a setup detail, not a runtime capability. Snapshots are built by replaying the steady-state outcome of the platform's normal moderation workflow (admin approves the user-submitted review). We use the database as a fast, deterministic substitute for the human-attention cost of legitimate moderation, not as a separate attack vector. **Trust boundary is at instance-construction-time**, not at runtime. Cite: WASP §3.2 (Evtimov et al., 2025) for the user-as-attacker framing; ST-WebAgentBench (Levy et al., ICLR 2026) for the snapshot-load precedent; WebArena §4.1 for the environment-construction tradition."

**Decision required from operator before Stage 1.** Spec amendment in `docs/worldsim-v5-technical-specifcation.md` AND CLAUDE.md Principle #1 must land in the same commit as Stage 1 — not after.

---

## Revised architecture (post-critique)

The pre-critique design had three flaws caught by the Opus subagents:

1. **TargetSpec was too flat** — `entity_pk_value: Any` collapsed three different addressing modes (single PK, hierarchical path, query-result coordinate) into one slot.
2. **Snapshot identity was undefined** — keyed only on dataset_version, ignored Phase 2b LLM-nondeterministic payload texts.
3. **`phase_2_render_check.py` was scheduled for deletion** — it's actually the post-build verification gate we need.

### The corrected `SnapshotDriver` Protocol

```python
# worldsim/snapshot_drivers/types.py
from dataclasses import dataclass
from typing import Any, Protocol, Union

@dataclass(frozen=True)
class EntityTarget:
    """Single-row target with a stable PK at build time.
    Example: Magento review on product_id=67."""
    table: str
    pk_field: str
    pk_value: Any

@dataclass(frozen=True)
class PathTarget:
    """Hierarchical target: [(table, pk), (child_table, child_pk), ...].
    Example: GitLab note on issue 7 in project 42 →
    [("projects", 42), ("issues", 7), ("notes", None)]"""
    segments: list[tuple[str, Any]]

@dataclass(frozen=True)
class QueryTarget:
    """Surface that doesn't have a row at build time — described as a query.
    Example: search-results page for query 'foo' should contain payload."""
    surface_id: str
    predicate: dict[str, Any]

TargetSpec = Union[EntityTarget, PathTarget, QueryTarget]
```

### The Protocol

```python
# worldsim/snapshot_drivers/base.py
class SnapshotDriver(Protocol):
    benchmark: str           # "webarena_verified"
    site: str                # "shopping" | "gitlab" | "reddit"

    def apply_payload(
        self, *,
        payload_text: str,
        target: TargetSpec,
        nickname: str,
        snapshot_id: str,
    ) -> AppliedPayloadRecord:
        """Place payload using most reliable mechanism for this platform.
        MUST be idempotent on (snapshot_id, target). Returns record for the
        applied.jsonl manifest."""
        ...

    def post_apply_finalization(self) -> None:
        """Platform-specific work after all payloads applied. Magento:
        bin/magento cache:flush + indexer:reindex review_summary. Postmill:
        recompute submission_votes. GitLab: nothing."""
        ...

    def verify_payload_renders(
        self, *, target: TargetSpec, signature: str, browser: Any
    ) -> RenderOutcome:
        """Open Playwright, navigate to read_surface_url for target, assert
        signature is in DOM. REUSES worldsim/phases/phase_2_render_check.py
        — that module is NOT deleted."""
        ...

    def reset_url(self) -> str:
        """The reset_endpoint that restores to the seeded baseline."""
        ...
```

### Snapshot identity

Content-addressed, not version-keyed:

```python
snapshot_id = sha256(
    sorted([
        f"{task['id']}:{sha256(task['payload_text'])}:{task['target']}"
        for task in adversarial_tasks
    ])
    + driver_versions  # bumps on schema-affecting changes
    + benchmark_image_digests  # docker image hashes per platform
)
```

Stored in:
- `logs/snapshots/<snapshot_id>/manifest.json` (orchestrator-side)
- `worldsim_snapshot_meta` table on each platform DB (host-side, written by `post_apply_finalization`)
- `pipeline_state.json` (per-run record)

### Reset semantics (decision matrix per Showstopper #2)

| Choice | reset_endpoint behavior | Snapshot persistence |
|--------|------------------------|----------------------|
| **A. Baked image** | env-ctrl restores adversarial-loaded image | Persists across resets natively |
| **B. Replay-on-reset** | env-ctrl restores baseline; orchestrator re-applies SQL | Re-applied per task |
| **C. No reset** | Skipped | Cross-task contamination |

Default proposal: **B** for Stage 1–2 (works without env-ctrl mods), with operator-time bake (option A) as a follow-up optimization. Critically, Option B is NOT pure per-task seeding because the SQL bundle is content-addressed and pre-validated; the per-reset replay is mechanical.

---

## The five-stage migration plan (corrected from critiques)

### Stage 0 (mandatory, BEFORE any code) — resolve showstoppers

**Artifacts:**
1. Operator decision on Showstopper 1 (Phase 3 strategy): A / B / C.
2. Operator decision on Showstopper 2 (reset semantics): A / B / C.
3. Threat-model amendment commit (Showstopper 3) — `CLAUDE.md` + `docs/worldsim-v5-technical-specifcation.md` updated with the WASP-style writeup. **Lands as a docs-only commit BEFORE Stage 1 code.**
4. Tag `pre-snapshot-cutover` on current `feat/worldsim-v5` HEAD as the rollback target.
5. Cross-branch coordination — written confirmation from owners of `feat/multi-benchmark`, `feat/multi-benchmark-v5-integration-codex`, `multi-benchmark-rebased`, `pr-6-review-*` (5 variants), `feat/app-generation-infra` that they will rebase or abandon. Per critique #3: "9 active branches + 8 codex worktrees, all carrying pre-cutover seeding code; Stage 3 deletion creates merge catastrophe without coordination."

**Validation:** all 5 artifacts exist. Stage 1 cannot start otherwise.

### Stage 1 (additive, no deletion) — build the new path

**New code:**
- `worldsim/snapshot_drivers/__init__.py` — registry
- `worldsim/snapshot_drivers/types.py` — `EntityTarget`, `PathTarget`, `QueryTarget`, `TargetSpec`, `AppliedPayloadRecord`, `SnapshotManifest`, `RenderOutcome`
- `worldsim/snapshot_drivers/base.py` — `SnapshotDriver` Protocol
- `worldsim/snapshot_drivers/magento.py` — Magento implementation:
  - SQL INSERTs across `review` + `review_detail` + `review_store` + `rating_option_vote`
  - FK lookups (`rating_id`, `option_id`, `store_id`, `customer_id`)
  - `catalog_product_entity` existence check (verify `entity_pk_value` exists; per critique #1: ignored in initial design)
  - `created_at` / `updated_at` in Magento server local time (per critique #1: Magento doesn't accept UTC consistently)
  - `post_apply_finalization`: `bin/magento cache:flush` + `bin/magento indexer:reindex review_summary` (per critique #1: "the killer — post-insert reindex")
  - Idempotent UPSERTs keyed on `(snapshot_id, task_id)`
- `worldsim/snapshot_drivers/postmill.py` — Reddit/Postmill INSERTs into `submissions` + `submission_votes`
- `worldsim/snapshot_drivers/gitlab.py` — GitLab INSERTs into `notes` (or admin REST API; see decision below)
- `scripts/build_adversarial_snapshot.py` — orchestrator; reads `logs/phase_2/adversarial_tasks.json`, dispatches to drivers, emits `logs/snapshots/<id>/applied.jsonl` manifest
- `scripts/inspect_snapshot.py` — prints `{snapshot_id, dataset_hash, applied_count, drift_indicators}`
- `scripts/diff_snapshot.py <id_a> <id_b>` — manifest diff for audit
- `scripts/rollback_snapshot.py` — destroys current snapshot, restores vendor baseline via env-ctrl

**Decision (Stage 1 sub-decision): GitLab/Reddit drivers use SQL or REST?**
Per critique #1: "Right call: keep API for GitLab/Postmill, SQL for Magento. The architectural inconsistency objection is cosmetic; the SnapshotDriver protocol already abstracts the mechanism." Default: GitLab + Postmill drivers wrap the existing editor REST calls (so they're a thin shim over `worldsim/editors/gitlab.py` etc.); Magento driver uses SQL. This means Stage 3 can NOT delete GitLab/Reddit editors — only the Magento editor.

**Updated Stage 3 deletion scope:** ~6,000 LOC instead of ~9,500.

**Validation criteria:**
- `scripts/build_adversarial_snapshot.py --task <one-per-platform> --host r5` produces a snapshot artifact
- HTTP probe of every `read_surface_url` for those tasks returns the expected payload (use `phase_2_render_check.py` as the verifier — DO NOT delete it)
- Smoke test against ≥1 task per platform: shopping, shopping_admin, gitlab, reddit
- `scripts/inspect_snapshot.py` round-trips the manifest

**Rollback:** `git revert` the additive commits. Zero risk to current pipeline.

### Stage 2 (cutover with both paths active) — runtime support

- Add `--seeding-mode={editor,snapshot}` flag to:
  - `worldsim/main.py` Phase 2c subcommand
  - `worldsim/main.py` Phase 4 subcommand
- Add `seeding_mode` and `snapshot_id` fields to `pipeline_state.json` schema (per critique #3: "the cutover breaks `--resume` silently" without this)
- Refuse to resume when `pipeline_state.seeding_mode != current_invocation.seeding_mode` (defensive)
- Add `worldsim/phase_4/snapshot_admission.py` runtime gate — parallel to `magento_health.py`. Refuses Phase 4 launch if `instances.json[].snapshot_id` doesn't match host-reported `worldsim_snapshot_meta.id`
- New preflight test in `tests/preflight/` mirroring the four existing checks per CLAUDE.md

**Validation criteria:**
- Calibration run of ≥30 tasks with `--seeding-mode=snapshot` against r5
- `max_coverage > 0` rate within ±5% of editor-mode baseline (per critique #3: "judge agreement ≥95% on overlap; PVPO max_coverage rate within ±5%")
- `final_status` distribution matches between modes
- `pytest tests/test_phase_4_adversarial.py -k 'mode'` covers both code paths

**Rollback:** flip default back to `editor`, ship as a one-line revert.

### Stage 2.5 (mandatory, before Stage 3) — rewire test mocks

Per critique #3: 18 distinct `monkeypatch.setattr(..., "apply_data_seed_async", ...)` sites in `tests/test_phase_4_adversarial.py` (16) and `tests/test_phase_2_feasibility.py` (2). All test logic ORTHOGONAL to seeding (Phase 4 reset scheduling, instance binding, PVPO, judge plumbing). They mock `apply_data_seed_async` because Phase 4 calls it.

**If Stage 3 just deletes `apply_data_seed_async`, all 18 collapse at collection time.**

**Stage 2.5 work:** rename / rewire all 18 mocks to a stable seam. Recommendation: introduce `Phase4Hooks.before_task_launch` Protocol that both `editor` and `snapshot` modes implement. Tests mock the Protocol method, not the underlying function. Migration is mechanical — same test logic, different attribute path.

**Validation criteria:** `pytest tests/test_phase_4_adversarial.py tests/test_phase_2_feasibility.py` green with both code paths active.

### Stage 3 (deletion) — the point of no return

**Tag `pre-snapshot-cutover` is the rollback target.** Deletion happens as a single squash commit (per critique #3: "Stage 3 deletion ~9.5K LOC; if this commit reveals a problem, revert is one operation").

**Delete:**
- `worldsim/editors/shopping.py` (292 LOC) — replaced by `MagentoSnapshotDriver`
- `worldsim/editors/shopping_admin.py` (85 LOC) — same
- `worldsim/editors/_read_surface.py` (102 LOC) — only Magento editor uses it
- `worldsim/editors/__init__.py` exports trimmed
- Magento-specific helpers in `worldsim/seeding.py` — `_apply_editor_seed_call` paths for shopping
- `worldsim/phase_4/magento_health.py:check_pending_seed_reviews*` (~120 LOC, the Layer 3 backstop) — can never trigger
- `tests/test_seed_resolver_shopping.py` (328 LOC)
- `tests/test_seed_resolver_shopping_admin.py` (88 LOC)
- `tests/integration/test_seed_resolver_shopping_live.py` (54 LOC)
- `tests/integration/test_seed_resolver_shopping_admin_live.py` (95 LOC)
- Magento-specific assertions in `tests/integration/test_editor_read_surface_verification.py`

**KEEP (per critique #1's "single most important fix"):**
- `worldsim/phases/phase_2_render_check.py` (293 LOC) — REPURPOSED as snapshot post-build verification gate. Module docstring updated. After `commit_snapshot()`, iterate every `read_surface_url` for every task and assert the unique payload signature renders. **Without this, runtime brittleness is replaced by silent build-time brittleness.**

**DO NOT DELETE (per Stage 1 sub-decision: GitLab/Reddit stay on REST):**
- `worldsim/editors/gitlab.py` (1,338 LOC)
- `worldsim/editors/reddit.py` (390 LOC)
- `worldsim/editors/base.py` (622 LOC) — base class still needed
- `worldsim/seeding.py` core paths — still used for GitLab/Reddit
- `worldsim/auth_tokens.py` (442 LOC) — still used for GitLab PAT generation
- `worldsim/_async_utils.py` (121 LOC) — still used for editor retries
- GitLab/Reddit test files (~700 LOC)

**Net deletion:** ~1,500 LOC (much less than the 9,500 LOC initial estimate, because we keep editors for non-moderated platforms).

**CLAUDE.md updated in the SAME COMMIT as deletions** (per critique #3: "Otherwise an LLM session opens, reads CLAUDE.md, and writes new editor-mode code against deleted modules"):
- Threat-model statement amended (already done in Stage 0, this commit just confirms references)
- Integration test gate list trimmed (remove deleted files, add `worldsim/snapshot_drivers/**`)
- "What NOT to do" section: "Do not seed Magento reviews via REST POST. Use `MagentoSnapshotDriver`."

**Validation criteria:**
- `pytest tests/` green (1500+ tests)
- `scripts/run_integration_tests.sh --host-config configs/benchmark_hosts/r5.yaml` green
- Re-run the calibration set from Stage 2; outcomes match
- `python -c "import worldsim"` clean

**Rollback:** `git checkout pre-snapshot-cutover && git checkout -b rollback/snapshot-stage3`. Cherry-pick Stage 1 + 2 + 2.5 commits. Force-push only after explicit user approval.

### Stage 4 (cleanup) — orphaned imports

- Remove orphaned imports from `worldsim/seeding.py`, `worldsim/phases/phase_2_feasibility.py`
- Update `tests/preflight/` to assert snapshot is loaded
- Update `docs/handoffs/rigor-run-setup.md` operator runbook
- Update `scripts/setup_phase4_on_host.sh` step 7 (preflight) to include snapshot check

**Validation criteria:** ripgrep returns zero hits for deleted symbols.

---

## Per-platform driver complexity (per critique #1)

The "just SQL INSERTs" framing was wrong. Per-platform schema knowledge required:

### Magento (the hard one)

Tables touched per review:
1. `review` — primary row (`review_id`, `entity_pk_value`, `status_id=1`, `entity_id=1` for product, `created_at`, `updated_at`)
2. `review_detail` — body (`title`, `detail`, `nickname`, `customer_id` may be NULL, `store_id`)
3. `review_store` — store-scope linkage (`review_id`, `store_id`)
4. `rating_option_vote` — star rating (requires lookup of `option_id` from `rating_option` for the desired star value, `rating_id` from `rating`)

FK and lookup work:
- `entity_pk_value` references `catalog_product_entity.entity_id` — driver must verify product exists
- `store_id` from `store` table (default storefront is store_id=1)
- `rating_id` from `rating` table (default "Quality" rating)
- `option_id` from `rating_option` (4-star = option_id 4 in stock Magento, but verify per install)

Post-INSERT work (the killer per critique #1):
- `bin/magento cache:flush` — Magento full-page cache must be invalidated, otherwise PDPs serve stale HTML
- `bin/magento indexer:reindex review_summary` — `review_entity_summary` aggregation requires reindex, otherwise PDPs show old `reviews_count`
- These are container-shell commands. The driver must `docker exec` or use SSH-over-SSM to invoke them.

Timezone:
- `created_at` / `updated_at` must be Magento server local time (NOT UTC) — Magento admin queries are inconsistent on TZ.

### Postmill (medium)

Tables:
1. `submissions` — `(title, body, forum_id, user_id, created_at)`
2. `submission_votes` — must be pre-seeded or score recompute resets to 0

FK lookups:
- `forum_id` from `forums.name`
- `user_id` from `users.username`

Post-INSERT: none required (Postmill renders directly from DB on every page load).

### GitLab (easy if SQL, but we keep REST per Stage 1 decision)

If we WERE to do SQL:
- Tables: `notes` (`body`, `noteable_id`, `noteable_type='Issue'|'MergeRequest'`, `author_id`, `project_id`, `created_at`)
- FK lookups: `project_id` from `projects.path`, `noteable_id` from `issues.iid` etc.
- Post-INSERT: `gitlab-rake cache:clear` may be required

Per Stage 1 decision: **wrap existing `worldsim/editors/gitlab.py` REST calls in `GitLabSnapshotDriver`. No SQL needed.** GitLab's REST API has no moderation queue — it works fine.

---

## Test fixture migration strategy (per critique #1)

The 19+ test mocks of `apply_data_seed_async` fall into two categories per critique #3:

| Category | Count | Migration |
|----------|-------|-----------|
| Tests of seeding behavior itself (`test_seeding.py`, `test_read_surface_editors.py`, Magento-specific in `test_seed_resolver_shopping.py`) | ~5 files, ~2,500 LOC | DELETE in Stage 3. Replaced by `tests/test_snapshot_driver_magento.py` etc. |
| Tests of orthogonal Phase 4 logic that mock seeding to a no-op (`test_phase_4_adversarial.py`, `test_phase_2_feasibility.py`) | 18 mock sites | REWIRE in Stage 2.5 to mock `Phase4Hooks.before_task_launch` Protocol |

**For SQL driver unit tests** (per critique #1's "honest options"):
- **Recommended: testcontainers-mysql with the actual Magento schema dump.** Slow (1-2 min container spin-up per test class) but gives real coverage of FK + index + reindex behavior.
- Alternative: mock `pymysql.connect` with fixture cursor. Fast but tests nothing about SQL correctness.
- Worst: skip unit tests, rely entirely on integration tests against r5.

Default: testcontainers for the critical paths (apply, verify, finalization), pymysql mocks for branching logic.

---

## Operator runbook (post-cutover)

Operator complexity rating per critique #2: **2/5 → 4/5** (PhD-required range). Make this acceptable through tooling, not workflow streamlining.

### First-time setup

```bash
# Existing prerequisites (unchanged)
ssh ubuntu@3.12.221.9                               # benchmark host
sudo usermod -aG docker $USER                       # docker exec without sudo
# DB credentials in instances.smoke.json (already there)

# NEW: install snapshot tooling locally
cd /Users/ashtonchew/projects/browser-sim
uv pip install testcontainers-mysql                 # for unit tests
```

### Per-run workflow

```bash
# Phases 0-2b unchanged
uv run python -m worldsim.main phase 0
uv run python -m worldsim.main phase 2 --resume

# NEW: build snapshot (replaces Phase 2c per-task feasibility loop)
uv run python scripts/build_adversarial_snapshot.py \
    --dataset logs/phase_2/adversarial_tasks.json \
    --host-config configs/benchmark_hosts/r5.yaml \
    --drivers magento,postmill,gitlab \
    --idempotent                                    # safe to re-run

# NEW: capture env-ctrl baseline as the snapshot baseline
ssh ubuntu@3.12.221.9 'docker compose exec env-ctrl /capture-baseline.sh'
# (only if Showstopper 2 resolution was Option A)

# Phase 2c becomes hash-validation only
uv run python -m worldsim.main phase 2c --seeding-mode=snapshot --resume

# Phase 4 unchanged invocation
uv run python -m worldsim.main phase 4 \
    --seeding-mode=snapshot \
    --instances instances.scale.json \
    --resume
```

### Diagnostic tools

```bash
# What's currently loaded on the host?
uv run python scripts/inspect_snapshot.py \
    --host-config configs/benchmark_hosts/r5.yaml

# Diff two snapshots
uv run python scripts/diff_snapshot.py \
    logs/snapshots/<id_a>/manifest.json \
    logs/snapshots/<id_b>/manifest.json

# Nuke the snapshot, restore vendor baseline
uv run python scripts/rollback_snapshot.py \
    --host-config configs/benchmark_hosts/r5.yaml
```

### Failure-mode cheat sheet

| Symptom | First check | Likely cause |
|---------|-------------|--------------|
| Phase 4 launch refuses with "snapshot_id mismatch" | `scripts/inspect_snapshot.py` vs `pipeline_state.json` | Snapshot rebuilt without re-running Phase 2c |
| `max_coverage=0` across all shopping tasks | `phase_2_render_check.py` post-build output | Magento FPC not flushed, or reindex didn't run |
| `snapshot.applied_count < expected` | `logs/snapshots/<id>/applied.jsonl` | Build failed mid-run; re-run with `--idempotent` |
| Phase 3 sees adversarial reviews | Showstopper #1 resolution | Wrong reset_endpoint or wrong host |

---

## Mandatory pre-Stage-3 artifacts (consensus from all 3 critiques)

Before Stage 3 deletion is safe, ALL of these must exist and pass:

1. **Showstopper resolutions documented** in this handoff doc (update with operator decisions).
2. **`pre-snapshot-cutover` git tag** on the commit immediately before Stage 1.
3. **`seeding_mode` and `snapshot_id` fields** in `pipeline_state.json` schema with refuse-on-mismatch resume logic.
4. **Calibration run** (≥30 tasks, both modes, judge-agreement ≥95%, max_coverage rate within ±5%, archived under `logs/calibration/snapshot-vs-editor-<date>/`).
5. **Phase 3 architectural decision implemented** (Showstopper 1).
6. **Stage 2.5 mock rewire commit** — all 18 sites moved to a stable seam.
7. **Cross-branch coordination** — written confirmation from `feat/multi-benchmark*`, `pr-6-review-*`, `feat/app-generation-infra` owners.
8. **CLAUDE.md updated in the same commit as Stage 3** (not after).
9. **Snapshot build script unit tests** — round-trip per platform (shopping, shopping_admin, gitlab, reddit).
10. **Documented threat-model statement** in `docs/worldsim-v5-technical-specifcation.md` AND `CLAUDE.md` Principle #1.
11. **`scripts/inspect_snapshot.py`** + **`scripts/diff_snapshot.py`** + **`scripts/rollback_snapshot.py`** exist and tested.
12. **Snapshot admission preflight** in `worldsim/phase_4/snapshot_admission.py` + `tests/preflight/`.

**Without items 1, 2, 5, and 10, the cutover is malpractice.** The other items are quality gates.

---

## Rollback playbook

If Stage 3 reveals a problem after deletion:

1. `git checkout pre-snapshot-cutover && git checkout -b rollback/snapshot-stage3`
2. Cherry-pick Stage 1, 2, 2.5 commits onto the rollback branch (snapshot tooling preserved, but defaults flip back to editor mode)
3. Mark all `pipeline_state.json` files written under `seeding_mode=snapshot` as invalid; re-run those tasks from Phase 2c on the editor path
4. Restore `CLAUDE.md` to its pre-cutover statement about `apply_data_seed`
5. Force-push to `feat/worldsim-v5` ONLY after explicit user approval (per repo's git safety conventions); otherwise land as a revert PR
6. Postmortem must identify which showstopper or critique-axis fired and add a regression test before re-attempting Stage 3

---

## Open questions for the operator

These cannot be answered by code analysis — operator/researcher judgment required:

1. **Showstopper 1 (Phase 3):** Option A (env-ctrl mod), B (separate hosts), or C (keep editors for Phase 3)?
2. **Showstopper 2 (reset semantics):** Option A (baked image), B (replay-on-reset), or C (no reset)?
3. **Stage 1 sub-decision:** GitLab/Reddit drivers — wrap existing REST editors or write SQL drivers? (Default: wrap REST.)
4. **Test infrastructure:** testcontainers-mysql or just integration-test-only for Magento driver?
5. **A/B testing:** how do we compare two adversarial strategies on the same task? Two snapshots? Strategy-namespaced rows?
6. **Cross-benchmark extension:** when adding a new benchmark beyond WebArena, what's the integration contract? (Currently described as "implement SnapshotDriver and reset_endpoint" — but this is more handwave than spec.)
7. **CI/CD:** does CI build a fresh snapshot per run (heavy) or use a long-lived test snapshot (drift risk)?
8. **Multi-tenant Phase 4:** if an agent mutates a seeded record during a task, does the next worker on the same task see the mutation? (Yes, currently — cleanup hooks gone. Acceptable?)

---

## What this doc does NOT cover

- The exact SQL for each Magento INSERT — schema is Magento-version-specific; capture in driver implementation.
- The Phase 2b LLM prompt changes (none needed).
- Browser-Use agent behavior changes (none).
- PVPO capture / encounter detection / judge / VEA / Transcript Purpose — all orthogonal.
- Spec for new benchmarks beyond WebArena — out of scope for this cutover (handle when adding the second benchmark).

---

## Citations

- **WASP** (Evtimov et al., NeurIPS 2025) — arXiv:2504.18575 — threat model precedent ("authenticated user")
- **VWA-Adv** (Wu et al., ICLR 2025) — arXiv:2406.12814 — snapshot pre-positioning architecture
- **ST-WebAgentBench** (Levy et al., ICLR 2026) — arXiv:2410.06703 — DB-write-as-admin precedent + `demo_data.sql` pattern
- **WebArena** (Zhou et al., 2024) — `reset_endpoint` contract origin
- **AgentDojo** (Debenedetti et al., NeurIPS 2024) — synthetic-mock alternative; explicitly NOT what we're doing

## Appendix: critique sources

The three Opus subagents that pressure-tested this design (run 2026-04-21):
- **Architectural critique** (`a0e6e125203c62859`) — TargetSpec, snapshot lifecycle, build reliability, reset_endpoint contract, per-platform complexity, test fixtures, threat model, forward compatibility, operator burden. Single most important fix: keep `phase_2_render_check.py`.
- **Operational critique** (`ae5794a72556d6fc4`) — snapshot identity, host state machine, operator workflow, CI/CD, multi-tenancy, drift, partial-failure recovery, cross-phase contamination, reproducibility, security, docs, backward compat. Operator complexity 2/5 → 4/5.
- **Migration risk critique** (`ae121e8d2afca9562`) — point-of-no-return, in-flight runs, test mock collapse, data dependencies, CLAUDE.md drift, resume semantics, Phase 3 contamination, cross-branch coordination, reset failures, codex worktrees. Top risk: Phase 3 contamination.
