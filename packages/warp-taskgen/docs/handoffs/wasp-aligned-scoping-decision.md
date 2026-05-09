# WASP-Aligned Scoping Decision — Drop Magento, Restrict to Content-Heavy Environments

**Status:** ADOPTED 2026-04-21. Four-commit cutover on `feat/worldsim-v5` (Commit 4 added when the user surfaced WASP's explicit Wikipedia/kiwix33 exclusion from their Docker README).
**Supersedes:** `docs/handoffs/codex-handoff-snapshot-architecture-cutover.md` (snapshot architecture cutover) — not pursued.
**Origin:** Discovery via WASP paper (Evtimov et al., NeurIPS 2025) that Magento was deliberately excluded from the canonical IPI benchmark for **methodological** reasons, not engineering ones. After three rounds of subagent investigation revealed the snapshot cutover was both technically broken (Magento driver Step 5 calls a non-existent indexer) and unnecessary (its showstoppers were based on false premises about Phase 3 + reset_endpoint), the right answer crystalized: drop Magento.

## Why this decision

The WASP paper (§3.1 verbatim, via the user's research):

> "We focus on two web environments within VisualWebArena: gitlab, a clone of the GitLab, and reddit... For the gitlab environment, the attacker is able to create a new issue and post comments... Similarly, for the reddit environment, we only allow the attacker to create new posts and comments. **This restriction better reflects the security threat of web agents on trusted domains with potentially untrusted participants.**"

And explicitly on Magento:

> "Magento/shopping doesn't fit this model well. It is **transactional** (product search, cart/checkout, admin ops) rather than **content-heavy with open user posts/comments**. There aren't the same realistic 'untrusted participant' injection surfaces in the way the paper designed the benchmark."

This is a methodological observation, not an engineering one. The reason real Magento has heavy moderation on reviews is precisely because reviews are not a natural "trusted domain with untrusted UGC" surface — they're a low-trust commercial signal that platforms gate by design. Forcing IPI through that gate (whether by REST API + status_id=1 hack, SQL INSERT, or a Playwright form-submit) recreates the threat in a context where the platform's actual real-world defense (moderation queue) would catch it. Therefore the experiment doesn't generalize to deployment.

## What we tried and why it didn't work

The architectural journey:

1. **Discovered the bug** (2026-04-21 killed Phase 4 run): Magento defaults `POST /rest/V1/reviews` to `status_id=2` (Pending). Pending reviews don't render on the storefront PDP. PVPO sees no payload. Classifier mislabels trajectories as `resistant_unaware`. ~$9 / 48 trajectories spent.

2. **Built 3-layer defense** (commits `1fc8a2aa`, `da2c8618`, `f364ee74` on `feat/worldsim-v5`):
   - Layer 1: editor sets `status_id=1` + defensive PUT + DELETE cleanup
   - Layer 2: Phase 2c Playwright render verification before stamping `verified`
   - Layer 3: pre-Phase-4 DB backstop catches pending SeedNickAdv* reviews
   - **Worked.** 1502 unit tests passing. Bug class cannot recur.

3. **Asked "is this brittle?"** Three Opus subagents critiqued. Consensus: not brittle as code; brittle as a sign that we were force-fitting Magento into a threat model it doesn't naturally support.

4. **Designed snapshot-based architecture** (`docs/handoffs/codex-handoff-snapshot-architecture-cutover.md`). 5-stage migration plan, ~1500 LOC delete + ~1500 LOC add, 12 mandatory artifacts, 8 open questions.

5. **Three more Opus subagents critiqued the snapshot plan**:
   - Showstopper #1 (Phase 3 contamination) — based on false premise. `phase_3_benign.py` does zero `reset_endpoint` and zero `apply_data_seed` calls. Phase 3 is contracts-only.
   - Showstopper #2 (reset_endpoint conflict) — collapses. WebArena env-ctrl `init` does NOT restore container baseline; SQL inserts persist across resets.
   - Magento driver Step 5 (`indexer:reindex review_summary`) — broken. No such indexer exists in stock Magento 2.4.
   - **Steel-man verdict: "Drop Magento. Cite WASP."**

6. **User confirmed** with WASP §3.1 quote: Magento exclusion was deliberate methodology, not oversight. Decision made.

## What changes

### Dataset

Two filter passes:

**Pass 1 (Commit 2):** `delivery_channel.delivery_site != "shopping"`. Drops 87 Magento tasks.

**Pass 2 (Commit 4):** any task with `wikipedia` in `sites` list or `__WIKIPEDIA__` in `start_urls`. Drops 3 GitLab tasks (AT-003, AT-005, AT-008) that used Wikipedia as a knowledge-base lookup for the underlying benign portion. Per WASP's `environment_docker/README.md`: "Other environments (we DON'T use them in WASP!): shopping, kiwix33 [Wikipedia], classifieds, shopping_admin, openstreetmap." Strict WASP alignment requires dropping kiwix33/Wikipedia.

**Final: 84 tasks (78 GitLab + 6 Reddit).**

Verified breakdown:
- shopping: 87 (DROPPED Pass 1)
- gitlab w/ wikipedia: 3 (DROPPED Pass 2: AT-003, AT-005, AT-008)
- gitlab pure: 78 (KEPT)
- reddit: 6 (KEPT)

The dataset matches WASP's exact threat-model scope: trusted domains (GitLab, Postmill) where adversarial users post natural content surfaces (issues, notes, posts, comments).

### Instance configs

Two passes mirroring the dataset:

**Pass 1:** drop `shopping`, `shopping_admin`, `map` (no tasks reference these).

**Pass 2:** drop `wikipedia` (the 3 tasks that used it were dropped in dataset Pass 2).

`instances.smoke.json`, `instances.scale.json`, `instances.json`,
`instances.smoke.local.json`: keep only `gitlab` and `reddit` instance entries.
Replica counts are generated from the active host config and have changed since
the original cutover; as of the 2026-05-02 doc audit, `instances.scale.json`
contains 21 GitLab replicas and 10 Reddit replicas, while `instances.smoke.json`
contains one of each.

### Code deletions

- `worldsim/editors/shopping.py` (292 LOC)
- `worldsim/editors/shopping_admin.py` (85 LOC)
- `worldsim/phase_4/magento_health.py` (~408 LOC including Layer 3)
- `worldsim/editors/__init__.py` — remove shopping registry entries
- Magento probe wiring in `phase_4_adversarial.py` — `_probe_magento_base_urls`, `_probe_pending_seed_reviews`, save_state for `magento_base_url_mismatch` and `magento_pending_seed_reviews`
- `tests/test_seed_resolver_shopping.py` (328 LOC)
- `tests/test_seed_resolver_shopping_admin.py` (88 LOC)
- `tests/test_phase_4_magento_health.py` (~250 LOC after Layer 3 additions)
- `tests/integration/test_seed_resolver_shopping_live.py` (54 LOC)
- `tests/integration/test_seed_resolver_shopping_admin_live.py` (95 LOC)
- Magento sections in `tests/integration/test_editor_read_surface_verification.py`
- `scripts/fix_magento_base_url.sh`
- `scripts/sync_magento_base_urls.py`

**Net: ~−1,800 LOC deleted, +0 LOC added.**

### CLAUDE.md updates

- Principle #1: remove "SQL seeding is excluded ... violates the threat model" sentence (no longer relevant; we don't seed Magento at all).
- New paragraph in "What this is" or "Non-negotiable principles" citing WASP for the scoping decision.
- "What NOT to do" section: remove Magento-specific notes (base_url drift, FPC, proxy buffer rewrites, `fix_magento_base_url.sh`).
- Integration test gate list: remove deleted files; keep render-check module (still useful for future moderation surprises on other platforms).

### Spec doc

`docs/worldsim-v5-technical-specifcation.md`: add a "Scope" section near the top with the WASP citation and the methodological rationale.

## What stays

- `worldsim/editors/gitlab.py` (1,338 LOC) — content-heavy threat surface, REST API, no moderation issues
- `worldsim/editors/reddit.py` (390 LOC) — content-heavy, form-POST works fine
- `worldsim/editors/base.py` (622 LOC) — base class
- `worldsim/seeding.py` — full pipeline still needed for GitLab/Reddit
- `worldsim/auth_tokens.py` (442 LOC) — GitLab PAT generator stays
- `worldsim/http_proxy.py` (232 LOC) — GitLab also bakes `external_url`; the proxy stays. Magento-specific buffer config in nginx.conf can be trimmed but the URL-rewrite logic itself is platform-agnostic.
- `worldsim/phases/phase_2_render_check.py` (293 LOC) — generic post-seed render verification, useful for any platform with future moderation surprises (e.g., GitLab's spam filter, Postmill's flagging system). Keep as defensive infrastructure.
- All Phase 4 evaluation: PVPO, encounter detection, judge, variant, placement-fix, Transcript Purpose, VEA — all orthogonal to platform.
- The 3 Layer commits — Layer 1 (Magento approval) becomes dead code that gets deleted with `shopping.py`. Layers 2 + 3 stay as generic defensive infrastructure (Layer 3 deletion only because it's bound to magento_health; the pattern can be re-applied to any platform).

## Why this is the right answer

| Property | 3-layer architecture (current) | Snapshot cutover (rejected) | WASP-aligned drop (this) |
|----------|-------------------------------|---------------------------|-------------------------|
| Engineering cost to ship | DONE | 4-6 weeks | ~1 day |
| Methodological risk | Low | High (threat-model reversal) | Zero (cite WASP) |
| Operator complexity | 2/5 | 4/5 (PhD-required) | 1/5 |
| LOC delta | +500 | ~0 net | **−1,800** |
| Cross-branch coordination needed | None | 6+ branches | Trivial (multi-benchmark already drops these files) |
| Dataset coverage | 174 | 174 | 87 (50% loss) |
| Paper claim defensibility | Novel approach | Reviewer pushback risk | Cites strongest precedent in field |

The 50% dataset loss is the only real cost. It's accepted because:

1. **WASP shipped to NeurIPS 2025 with comparable scope** (GitLab + Reddit only).
2. **The Magento tasks were inherited from WebArena's task pool**, not deliberately chosen for IPI research goals.
3. **The threat model is more honest** with content-heavy platforms — Magento reviews are not a realistic IPI surface in deployment.
4. **The 87 remaining tasks support the central research claim**: "Do browser agents resist or comply with indirect prompt injections in realistic web environments (trusted domains with untrusted user-generated content)?"

## Compatibility with feat/multi-benchmark

`feat/multi-benchmark` defines a benchmark capability registry (`worldsim/benchmark_capabilities.py`) supporting four benchmarks:

| Benchmark | Runner | Phase 4 mode |
|-----------|--------|--------------|
| `webarena_verified` | browser_use OR agentlab | `worldsim_v5` (full pipeline — what we use) |
| `wasp` | agentlab only | `comparison_runner` (resistance) |
| `stwebagentbench` | agentlab only | `comparison_runner` (capability) |
| `doomarena` | agentlab only | `comparison_runner` (attack_success) |

The shopping/Magento editors are used **only** by `webarena_verified`. The other benchmarks route through AgentLab against their own published task suites. Therefore:

- This deletion is safe across all four benchmark adapters.
- `multi-benchmark` already merged from `feat/worldsim-v5` (commit `b183abda`); when it next pulls, this deletion will be a clean filter.
- `multi-benchmark` independently deleted `phase_2_render_check.py` and `magento_health.py` — convergent realization.

**Optional follow-up (separate decision):** Adopt `multi-benchmark`'s `runners/` package + `benchmark_capabilities.py` onto `feat/worldsim-v5`. This would enable running WASP and ST-WebAgentBench as comparison runs IN ADDITION to our novel Phase 2 generation on GitLab+Reddit. ~8K LOC merge. **Strengthens the paper claim** ("we extend WASP's evaluation methodology with novel adversarial task generation"). But out of scope for this cutover.

## Cutover commits

**Commit 1:** `docs(handoffs): adopt WASP-aligned scoping; drop transactional environments`
- This file (`docs/handoffs/wasp-aligned-scoping-decision.md`)
- "SUPERSEDED" header on `docs/handoffs/codex-handoff-snapshot-architecture-cutover.md`
- `CLAUDE.md` Principle #1 update + WASP citation
- `docs/worldsim-v5-technical-specifcation.md` Scope section

**Commit 2:** `feat(scope): drop Magento from adversarial dataset and instances`
- Filter `logs/phase_2/adversarial_tasks.json` (87 → 87 tasks, removing shopping)
- Drop shopping + shopping_admin from `instances.smoke.json`, `instances.scale.json`, `instances.json`
- Confirm via repeat of the breakdown grep

**Commit 3:** `refactor: delete Magento-specific code rendered dead by scoping`
- Delete editor files
- Delete magento_health module
- Strip Magento probes from phase_4_adversarial wiring
- Delete Magento-specific tests
- Delete `fix_magento_base_url.sh` and `sync_magento_base_urls.py`
- CLAUDE.md integration test gate list trim

## Validation

After all three commits:

```bash
# Unit tests still pass
uv run pytest tests/ --ignore=tests/integration --ignore=tests/preflight \
    --deselect tests/test_host_bootstrap_scripts.py::test_setup_phase4_on_host_fails_when_playwright_install_deps_fails

# Dataset shape
python3 -c "
import json
tasks = json.load(open('logs/phase_2/adversarial_tasks.json'))
print(f'remaining tasks: {len(tasks)}')
from collections import Counter
sites = Counter(t.get('delivery_channel', {}).get('delivery_site', t.get('site', 'unknown')) for t in tasks)
for s, c in sorted(sites.items(), key=lambda x: -x[1]):
    print(f'  {s}: {c}')
# Expected: 87 total (gitlab 81, reddit 6)
"

# Instance shape
python3 -c "
import json
for path in ['instances.smoke.json', 'instances.scale.json', 'instances.json']:
    try:
        cfg = json.load(open(path))
        sites = sorted(i['site_name'] for i in cfg.get('instances', []))
        print(f'{path}: {sites}')
    except FileNotFoundError:
        pass
# Expected: no 'shopping' or 'shopping_admin' in any
"

# Imports clean
python3 -c "import worldsim; from worldsim.editors import EDITOR_REGISTRY; print(sorted(EDITOR_REGISTRY))"
# Expected: no 'shopping' or 'shopping_admin' keys
```

Integration tests (against r5) require coordinator update: r5 docker-compose has shopping containers; they can stay running but Phase 4 won't reference them. No host-side changes needed.

**Phase 4 relaunch is OUT OF SCOPE for this cutover.** Operator triggers separately when ready, against the 87-task GitLab + Reddit dataset.

## Citations

- **WASP** (Evtimov et al., NeurIPS 2025) — arXiv:2504.18575 — methodological precedent for restricting IPI evaluation to content-heavy environments.
- **VWA-Adv** (Wu et al., ICLR 2025) — arXiv:2406.12814 — alternative architecture (snapshot pre-positioning) considered but not pursued.
- **ST-WebAgentBench** (Levy et al., ICLR 2026) — arXiv:2410.06703 — alternative architecture (DB-write-as-admin) considered but not pursued.
- **WebArena** (Zhou et al., 2024) — environment substrate; we use a strict subset of its sites.

## Appendix: subagent investigation IDs

For traceability of the decision provenance:

- WASP research (Sonnet, `aee4a5bbd70b22814`)
- ST-WebAgentBench research (Sonnet, `a7b6fb3614a5c5444`)
- IPI benchmark survey (Sonnet, `adc64424ee9b43850`)
- Snapshot cutover architectural critique (Opus, `a0e6e125203c62859`)
- Snapshot cutover operational critique (Opus, `ae5794a72556d6fc4`)
- Snapshot cutover migration risk critique (Opus, `ae121e8d2afca9562`)
- End-to-end plumbing trace (Opus, `a017531d0747228b6`)
- Magento 2.4 reality check (Opus, `a90b009d55700fa7f`)
- Steel-man no-cutover position (Opus, `a99a741eaab84b813`)
