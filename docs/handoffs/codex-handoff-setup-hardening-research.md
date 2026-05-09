# Research Report: Seed Target Abstraction for Browser-Agent Adversarial Eval

> **HISTORICAL RESEARCH NOTE.** Preserved for seed-target design rationale; do
> not treat resolver recommendations as current implementation work without
> checking current code and the technical spec.

Author: research subagent (opus), 2026-04-17
Context: background research for item #14 in `docs/handoffs/codex-handoff-setup-hardening.md`
Trigger: after #13 (seed template resolution) landed in commit `a5bf5af`, Phase 4 on r5 still 7/7 errored with concrete-ID 404/400/401s. Proposed fix was a constraint-based target resolver layer (~2500-3000 LOC, 2-3 days). This report investigates whether that scope is right.

**TL;DR:** The seed-target abstraction you're building is sound but you're overshooting — WASP (the closest published analog, Meta's 2025 VisualWebArena-based IPI benchmark) solves the exact same problem by **creating fresh resources per run** via a thin editor wrapper rather than by building a per-site resolver for every lookup pattern; ship the resolver module you already started (`worldsim/seed_resolvers/`) but keep it at ~1 small resolver per site with a bias toward creating rather than discovering resources, and treat "universal JSON lookup spec" as a premature abstraction.

---

## Section 1 — How comparable benchmarks solve this

**WASP (Facebook Research, 2025, arXiv 2504.18575)** is the closest published analog to WorldSim v5 — a prompt-injection benchmark built on top of VisualWebArena, explicitly restricting attacker surfaces to "elements attackers could plausibly manipulate" (GitLab comments, Reddit posts), which is the same threat model you describe. Reading `webarena_prompt_injections/prompt_injector.py` and `environment_setup.py` in their repo, WASP **does not hardcode resource IDs**. Instead:

- Task configs reference resources by **owner/name** pairs (e.g., `byteblaze/dotfiles`) and **template variables** like `{gitlab_domain}/{project_owner}/{project_name}/-/deploy_keys`, never numeric IDs like `/projects/5/`.
- `environment_setup.py` contains editor helpers (`GitlabEditor`, `RedditEditor`) that **create fresh resources per run** via authenticated GitLab/Reddit API calls: `create_issue_with_title_and_text(...)`, `create_post_with_title_and_text(...)`, `make_project_as_agent_user()`, `make_attacker_account()` (with "fallback handling if the username already exists").
- `_inject_gitlab_prompts()` captures the URL returned by the create call and stuffs it into the task object at runtime — there is no offline resolver that maps `{resource_type: mr_note, constraints: …}` → concrete URL; they just create the MR from scratch and keep the URL.

So WASP's answer to your 404 problem is: **don't look up resources, create them**. Cleanup is paired in `environment_cleanup.py`. This works because attacker-surface resources (comments, issues, posts, throwaway MRs) are by definition things an authenticated user can create, which is consistent with your threat model anyway (the `apply_data_seed` `api`/`form` allowlist is the same constraint).

**WebArena / WebArena-Verified (Zhou et al. 2023; ServiceNow 2024):** Task configs in `config_files/test.raw.json` do embed numeric IDs directly in `start_url` and `eval.program_html[].url` — and this is precisely why the original WebArena has well-known reproducibility issues (see [Issue #98](https://github.com/web-arena-x/webarena/issues/98) — `require_login=true`, `require_reset=false` on all 812 tasks means later tasks read state left by earlier ones). WebArena-Verified's answer wasn't to add a resolver — it was to **pre-capture network traces (HAR)** and evaluate offline against fixed traces, sidestepping the live-env ID-drift problem entirely. This is a valid alternative design, but incompatible with v5's live Phase-4 injection model.

**ChenWu98/agent-attack (ICLR 2025, Wu et al., arXiv 2406.12814):** Modified VisualWebArena to inject adversarial image perturbations. Reproducibility is achieved by **pre-generating** adversarial examples and committing them, plus a `scripts/process_data.py` step that replaces URL placeholders with actual URLs at setup. They pin snapshot state rather than resolving dynamically.

**InjecAgent (Zhan et al. ACL 2024, arXiv 2403.02691):** Tool-integrated agents, not browser — 1,054 cases over 17 user tools and 62 attacker tools. Simulates tool outputs rather than planting in a live env, so the resource-ID problem doesn't arise.

**ST-WebAgentBench (Shlomov et al., ICML 2025, arXiv 2410.06703):** 222 tasks across GitLab/ShoppingAdmin/CRM on the WebArena infrastructure. Inherits WebArena's same ID pinning, doesn't address drift.

**Mind2Web-Live (Zheng et al. 2024):** 542 tasks over real public websites with "key-node" state checks. Because targets are the live public web, they have to tolerate drift; the answer is loose semantic evaluators rather than structural ID lookups. Not applicable to v5's sandboxed threat model.

**Greshake et al. 2023 ("Not What You've Signed Up For", AISec '23, arXiv 2302.12173):** The foundational IPI paper. Demonstrations, not a benchmark harness — no reusable lookup contract.

**Conclusion for Section 1:** The only published benchmark that solves the exact problem you're solving (WASP) does it by **creating fresh resources per run, not by resolving stable IDs**. Your proposed `{constraints: {owner: "current_user", mr_state: "opened", select: "newest"}}` is strictly more general than what anyone in the published literature has built.

## Section 2 — Best-practice patterns from adjacent domains

Three relevant analogs:

1. **Load-test correlation (k6, JMeter).** "Correlation" is the load-testing term for exactly your problem: a recorded session captures dynamic values (session IDs, CSRF tokens, row IDs) that won't be valid on replay. Both tools solve it the same way: **extract from prior responses**. k6's BrowserRecorder auto-detects and substitutes dynamic values; JMeter has a Correlation Recorder Plugin. The extraction grammar is path-based (`$.data.id`, regex over HTML), not constraint-based. Your `_extract_response_seed_context` + `{forum_id}`-style placeholder substitution (lines 436–464 of `worldsim/seeding.py`) is exactly this pattern — and it's the lightest-weight thing that works.

2. **Pact provider states.** Provider tests declare a **state name** like `"an MR exists owned by current_user in state=opened"`, and the provider registers a handler that establishes that state before each interaction. Pact also supports **generators** that inject runtime-determined IDs from provider-state setup back into the contract. This is philosophically identical to your proposal, but Pact's production experience suggests: (a) keep state names coarse-grained (there are only a handful per service), (b) do the setup via whatever ORM/API the provider already has, (c) return the IDs the consumer needs through a well-defined `state_change_url` callback. No one writes a general "constraint resolver".

3. **factory_boy / FactoryBot.** The dominant integration-test fixture pattern is not "look up the right row" — it's **create-then-use**, with a `get_or_create` escape hatch (`django_get_or_create`). This is the same pattern WASP uses (`make_attacker_account` falls back if the user exists). Nobody writes property-based predicate resolvers for integration tests at any scale I've seen — the maintenance tax is too high.

**Adjacent-domain takeaway:** Across load-testing, contract-testing, and fixture-factory ecosystems, the dominant patterns are (a) extract-from-response (correlation) and (b) create-then-use (factories). Constraint-based resolvers exist academically (property-based testing à la Hypothesis) but are not used for stateful integration tests against real services, because the cost of specifying the constraints well exceeds the cost of just creating the resource.

## Section 3 — Honest assessment

**You are not doing catch-up.** No published browser-agent adversarial benchmark ships a reusable target-abstraction layer; WASP solves the problem by being sloppier than you're proposing (per-site editor helpers, duplicate code across GitLab/Reddit, not a unified interface). You would be building something that does not exist.

**But the proposed scope is over-engineered for your scale.** Three specific concerns:

1. **~2500–3000 LOC, 5 sites, per-site resolvers** is 500–600 LOC per site. WASP's entire `webarena_prompt_injections/environment_setup.py` is substantially smaller. The extra LOC is going into the *constraint language* (`select: "newest"`, `mr_state: "opened"`, `owner: "current_user"`), which is the part nobody else builds. Every constraint DSL feature is one you'll maintain and debug forever.

2. **The create-then-use alternative is strictly simpler and closer to the published norm.** Instead of `{constraints: {mr_state: "opened", select: "newest"}}`, write `{create: {resource: "mr", title: "...", body: "..."}}` and let the resolver POST it and capture the new URL. You already have auth wiring. This is ~50 LOC per site, not ~500.

3. **Most users will only add 1–2 benchmarks.** The per-site resolver cost is paid by every benchmark, not amortized across users. This reinforces "keep each resolver small and boring."

**Verdict:** Build the abstraction, keep the interface, but pick the simplest resolver shape — `create` first, `find_or_create` second, `select` constraints only when truly unavoidable (e.g., the benchmark forbids creating new resources, which your threat model doesn't). The research contribution here isn't novel — it's **systematizing what WASP hand-rolled**. That's valuable and publishable, but the value comes from the systematization, not from constraint expressiveness.

## Section 4 — Design recommendation

**Recommended (primary) design.** Keep the `target` dict in the seed JSON — the shape you've already committed in `validate_data_seed` is good. Narrow the resolver contract to three operations in priority order:

1. `create` — POST a fresh resource via the existing HTTP seed path; capture the returned URL/ID into `seed_context`. This should be the default and should cover ~80% of cases.
2. `find_or_create` — GET a list endpoint, pick the first match by a small whitelist of fields (`owner`, `state`, `title_contains`), create if none.
3. `select` — read-only lookup with the constraint language. Use only for cases where creation is disallowed by the benchmark threat model.

Design patterns to borrow explicitly:
- **Pact's state-change-url callback:** resolver returns a `context_additions` dict (you already do this in the committed `ResolvedCall` type) — keep this, it's exactly right.
- **k6 correlation:** extract-from-response after create. Your `_extract_response_seed_context` does this — keep it, extend it to cache URLs not just body fields.
- **factory_boy `get_or_create`:** fallback when creation collides. Mirror WASP's "if username exists, reuse" pattern.
- **Determinism:** resolvers must be deterministic under `reset_endpoint`. Document this as a contract. Do not add a random-selection constraint.
- **Versioning:** version the resolver interface (not each resolver). A single `resolver_api_version: 1` field in the site profile is enough; don't over-engineer.

**Alternative 1 (simpler, worth considering):** **Generator-time instantiation.** Phase 2 already runs against the live instance. Have Phase 2 create the adversarial resource at generation time, capture the concrete URL/ID, and bake it into the task JSON. No runtime resolver. Cost: tasks are instance-specific (can't share task JSON across hosts), but this is already true in practice — your 7/7 Phase 4 failure was caused precisely by task JSON that was instance-bound to an old host. Pros: zero new abstraction, ~0 LOC per site. Cons: regenerate task JSON on every host migration, and you lose portability across the scale-out replicas you're building for r5.8xlarge.

**Alternative 2 (avoid):** **Universal JSON lookup spec.** A single resolver that takes `{endpoint, json_path, filter}` and works for any REST API. This is what your "universal resolver" idea points at. It fails because real APIs differ in pagination, filter syntax, response envelopes, auth edge cases — the universal resolver becomes a leaky abstraction that pushes complexity into per-site config files instead of per-site Python. Net LOC savings are illusory.

**Scope re-estimate if you follow the primary recommendation:** ~800–1,200 LOC total (one shared resolver base, 5 thin per-site modules @ ~150 LOC each, mostly `create` calls), 1–1.5 days focused. Less than half of your original estimate.

## Section 5 — Citations

Directly cited:
- WASP benchmark paper: [arXiv:2504.18575](https://arxiv.org/abs/2504.18575)
- WASP code: [github.com/facebookresearch/wasp](https://github.com/facebookresearch/wasp) (especially `webarena_prompt_injections/prompt_injector.py`, `environment_setup.py`, `configs/experiment_config.raw.json`)
- WASP ICML presentation slides: [cseweb.ucsd.edu/~yuxiangw/…/WASP_presentation.pdf](https://cseweb.ucsd.edu/~yuxiangw/classes/AIsafety-2025Fall/Lectures/WASP_presentation.pdf)
- WebArena paper: [arXiv:2307.13854](https://arxiv.org/abs/2307.13854); code: [github.com/web-arena-x/webarena](https://github.com/web-arena-x/webarena); reproducibility issue: [Issue #98](https://github.com/web-arena-x/webarena/issues/98)
- WebArena-Verified: [github.com/ServiceNow/webarena-verified](https://github.com/ServiceNow/webarena-verified), [servicenow.github.io/webarena-verified](https://servicenow.github.io/webarena-verified/)
- VisualWebArena: [arXiv:2401.13649](https://arxiv.org/html/2401.13649v2), [github.com/web-arena-x/visualwebarena](https://github.com/web-arena-x/visualwebarena)
- Wu et al. "Dissecting Adversarial Robustness of Multimodal LM Agents": [arXiv:2406.12814](https://arxiv.org/abs/2406.12814); code: [github.com/ChenWu98/agent-attack](https://github.com/ChenWu98/agent-attack)
- InjecAgent (Zhan et al., ACL 2024): [arXiv:2403.02691](https://arxiv.org/abs/2403.02691), [github.com/uiuc-kang-lab/InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent)
- ST-WebAgentBench: [arXiv:2410.06703](https://arxiv.org/abs/2410.06703), [github.com/segev-shlomov/ST-WebAgentBench](https://github.com/segev-shlomov/ST-WebAgentBench)
- Mind2Web / Mind2Web-Live: [osu-nlp-group.github.io/Mind2Web](https://osu-nlp-group.github.io/Mind2Web/)
- Greshake et al. 2023: [arXiv:2302.12173](https://arxiv.org/abs/2302.12173)
- k6 correlation docs: [grafana.com/docs/k6/latest/examples/correlation-and-dynamic-data](https://grafana.com/docs/k6/latest/examples/correlation-and-dynamic-data/)
- JMeter correlation: [blazemeter.com/blog/jmeter-correlation-plugin](https://www.blazemeter.com/blog/jmeter-correlation-plugin-blazemeter)
- Pact provider states: [docs.pact.io/getting_started/provider_states](https://docs.pact.io/getting_started/provider_states), [pactflow.io/blog/injecting-values-from-provider-states](https://pactflow.io/blog/injecting-values-from-provider-states/)
- factory_boy docs: [factoryboy.readthedocs.io](https://factoryboy.readthedocs.io/en/latest/)

Searched-but-found-nothing-strong:
- A published browser-agent benchmark with a reusable seed-target resolver abstraction — does not appear to exist. WASP is the closest and uses hand-rolled per-site editors without a unified interface.
- "Constraint-based" resolver DSLs in integration testing at production scale — did not find any; the dominant patterns are correlation (extract-from-response) and factories (create-then-use).

## Orchestrator notes (added after agent returned)

**Scaffolding status when this report was written:** the agent believed the resolver scaffolding was already in place. Verified against the repo: the resolver package exists at `worldsim/seed_resolvers/` with 534 LOC total:

```
worldsim/seed_resolvers/__init__.py    29 LOC (get_resolver dispatcher)
worldsim/seed_resolvers/types.py       20 LOC (ResolvedCall)
worldsim/seed_resolvers/gitlab.py     288 LOC (project, issue, mr, mr_note, issue_note, repo_file)
worldsim/seed_resolvers/shopping.py    52 LOC (product_review)
worldsim/seed_resolvers/reddit.py      68 LOC (forum, submission, comment)
worldsim/seed_resolvers/map.py         77 LOC (node, way)
```

Pattern used: **constraint/select** (the expensive pattern the research cautions against), not create-then-use.

The dispatch branch in `worldsim/seeding.py:385-388` is in place. Preflight at line 306+ is in place. All tests pass (701/701).

**Implications for item #14:**
- Interface contract is already done — don't rebuild.
- Research says: pivot resolvers from select to create-first where possible, keep `find_or_create` as fallback, retire the richer constraint DSL elements that aren't used by the migration.
- Actual remaining work: (a) add `create` operation path in each resolver, (b) migrate `logs/phase_2/adversarial_tasks.json` from legacy `{url, body}` to `{target, body}`, (c) update Phase 2 generator to emit the new shape. Scope estimate revised from ~2500-3000 LOC down to ~800-1000 LOC.
