# Research note: site-feature module architecture

**Date:** 2026-08-10

**Status:** research and sequencing options only; this note does not choose a
final interface.

## Evidence boundary

This note follows the current `packages/warp-taskgen` source, tests, package
guides, and the authoritative technical specification. Those are the primary
sources for the findings below; no secondary architecture article or external
benchmark write-up is used. The existing specification is explicit about the
desired direction—editor-registry dispatch without Phase 2c call-site changes—
but the implementation still has several GitLab/Reddit branches that a third
site would have to cross. The note records that gap rather than treating the
specification's intended end state as current behavior.

## Executive finding

Adding a WebArena site is not local by feature today. The editor method registry
is the best existing deep nucleus: it co-locates method metadata with the
method, validates registrations, and already feeds resolver attachment,
placement validation, seed substitution, prompt rendering, feasibility
filtering, and the sandbox serialization path
([`worldsim/editors/_registry.py`](../../worldsim/editors/_registry.py#L1-L21)).
Its `EditorMethodSpec` carries benchmark/site/method, resource kinds, HTTP
verb/path, argument bindings, profile-surface IDs, and required arguments
([`_registry.py`](../../worldsim/editors/_registry.py#L40-L58)); the decorator
keeps the contract beside the implementation
([`_method_spec.py`](../../worldsim/editors/_method_spec.py#L1-L11)).

That nucleus is not a complete site module. Route patterns and anchor examples,
profile alias resolution, resource-kind parsing, listing probes, ordered-child
guards, authentication probes, rendered-visibility/readback evidence, and
final-state reward dispatch still know GitLab or Reddit directly. A safe
architecture therefore needs feature-local ownership for those facts and
generic consumers that discover them through explicit contracts/registries.
The smallest proof should be a test-only third site, enabled by an explicit
active-policy fixture, that crosses the complete route → resolver → registry →
exposure → seed → feasibility/visibility → reward/readback chain without adding
that site's name to generic phase call sites.

## Current scope: active carriers versus support plumbing

The specification and package guide define the current admitted WARP carriers
as GitLab issues/issue notes and Reddit/Postmill submissions/comments
([technical specification](../warp-taskgen-technical-spec.md#L7-L23),
[admission guide](../../agent_docs/admission-and-exposure.md#L7-L19)). The
current core-surface table is intentionally wider than the active cohort: it
contains GitLab issue/MR titles and descriptions plus `note.body`, and Reddit
submission title/body plus `comment.body`; title carriers are retired and MR
carriers are excluded by the active-carrier policy
([`phase_2_core_surfaces.py`](../../worldsim/phases/phase_2_core_surfaces.py#L21-L51),
[`phase_2_core_surfaces.py`](../../worldsim/phases/phase_2_core_surfaces.py#L159-L178)).
The technical spec also keeps shopping, shopping-admin, Wikipedia, map/OSM,
Magento, classifieds, and GitLab MRs as historical context or support plumbing,
not active IPI carriers ([technical specification](../warp-taskgen-technical-spec.md#L15-L23)).

This distinction matters for modularity: an editor can remain known for
compatibility, profiling, or cleanup without making its surface an admissible
WARP carrier. A fake site must likewise be explicitly marked test-only or
policy-enabled; registration alone must not silently enlarge the production
cohort.

## Site-varying behavior map

| Behavior that varies by site | Current owner and evidence | Active carrier or support plumbing | Implication for a local feature module and safety |
| --- | --- | --- | --- |
| **Identity and placeholders** | `PLACEHOLDER_TO_SITE`/`SITE_TO_PLACEHOLDER` enumerate six historical WebArena tokens; strict substitution rejects unresolved tokens ([`placeholders.py`](../../worldsim/placeholders.py#L9-L33), [`placeholders.py`](../../worldsim/placeholders.py#L62-L79)). `BenchmarkInstance` carries `site_name`, `site_url`, URL placeholders, seeding auth, API auth, and agent auth ([`config.py`](../../worldsim/config.py#L292-L345)). | Placeholder entries for shopping/admin/Wikipedia/map are support context; reward environment mapping deliberately exposes only GitLab, Reddit, and Postmill ([`rewards/webarena_sites.py`](../../worldsim/rewards/webarena_sites.py#L7-L42)). | A feature needs a normalized site key, placeholder/profile provenance, and an explicit production/test policy. Unknown placeholder/site combinations must stay unresolved or ineligible, not fall through to a guessed host. |
| **Profile surface identity** | `surface_identity.py` maps profile IDs/source fields to canonical carrier IDs and returns no mapping for unknown benchmark/site; ambiguous GitLab note surfaces require kind/method context ([`surface_identity.py`](../../worldsim/surface_identity.py#L118-L193)). Tests assert unknown benchmark and ambiguous note mappings fail closed ([`test_surface_identity.py`](../../tests/test_surface_identity.py#L84-L132)). | Current aliases are WebArena GitLab/Reddit support for the active profile vocabulary. | A new site must own canonical-to-profile aliases, source-field aliases, and ambiguity rules. “Body” or “comment” is not enough without a stable canonical carrier and profile evidence. |
| **Route contracts and benign traversal** | `build_task_route_contracts` iterates editor specs, applies canonical/active policy, resolves profile surfaces, then builds URL patterns, anchor examples, editor argument templates, instruction requirements, evaluator guidance, and answer-stability guidance ([`phase_1_route_contracts.py`](../../worldsim/phases/phase_1_route_contracts.py#L47-L109), [`phase_1_route_contracts.py`](../../worldsim/phases/phase_1_route_contracts.py#L205-L275)). `_start_url_patterns` and `_anchor_examples_for_route` have literal GitLab/Reddit branches ([`phase_1_route_contracts.py`](../../worldsim/phases/phase_1_route_contracts.py#L292-L337)); instruction and answer-stability helpers contain additional site/surface branches ([`phase_1_route_contracts.py`](../../worldsim/phases/phase_1_route_contracts.py#L848-L851), [`phase_1_route_contracts.py`](../../worldsim/phases/phase_1_route_contracts.py#L963-L1095)). | Route metadata is active only when the resulting surface passes the core and active-carrier policy; historical route facts can still be useful for profile/support work. | Route shapes, anchors, transition requirements, and natural-encounter evidence should be local to the site feature. Generic route synthesis should consume a descriptor/registry result and fail closed when a route lacks inventory-backed anchors or admissible evidence. |
| **Resolver, reconstruction, and listing probes** | The public L1/L2 resolver rejects a task whose site is not GitLab or Reddit, dispatches `_match_gitlab`/`_match_reddit`, and reconstructs URLs through site-specific functions ([`target_resolution/resolver.py`](../../worldsim/phase_2/target_resolution/resolver.py#L29-L50), [`target_resolution/resolver.py`](../../worldsim/phase_2/target_resolution/resolver.py#L65-L126)). URL matching has a GitLab/Reddit-only site-kind function and listing-kind set ([`target_resolution/url_matching.py`](../../worldsim/phase_2/target_resolution/url_matching.py#L115-L123), [`target_resolution/url_matching.py`](../../worldsim/phase_2/target_resolution/url_matching.py#L163-L202), [`target_resolution/url_matching.py`](../../worldsim/phase_2/target_resolution/url_matching.py#L267-L282)). | Current resolver kinds are active/support vocabulary for the two admitted sites; historical kind literals remain in constants and tests. | A third site needs feature-owned URL matching, reconstruction, listing/source probes, and `ResourceKind`/anchor vocabulary. Unknown kinds/sites must return an empty resolver record, not a best-effort URL or root-page guess. |
| **Editor methods and seed dispatch** | `register_editor` rejects missing decorators and duplicate `(benchmark, site, method)` keys; `kind_contract`, `attach_surfaces_for_kind`, and `available_tokens_for_kind` derive method/surface/token contracts ([`_registry.py`](../../worldsim/editors/_registry.py#L66-L124), [`_registry.py`](../../worldsim/editors/_registry.py#L127-L224)). `EDITOR_REGISTRY` currently contains only WebArena Verified GitLab and Reddit classes ([`editors/__init__.py`](../../worldsim/editors/__init__.py#L3-L20)). Seeding accepts editor-only calls, dispatches through `(benchmark, site)`, records generic read surfaces/created resources/per-call results, and cleans successful calls in LIFO order ([technical specification](../warp-taskgen-technical-spec.md#L1329-L1335), [technical specification](../warp-taskgen-technical-spec.md#L1434-L1448), [`seeding/_impl.py`](../../worldsim/seeding/_impl.py#L150-L239), [`seeding/_impl.py`](../../worldsim/seeding/_impl.py#L649-L814), [`seeding/_impl.py`](../../worldsim/seeding/_impl.py#L923-L1013)). | GitLab's project/profile/MR/snippet/milestone methods and Reddit's forum/title/profile methods are registry-known support methods; the active body/comment carriers are a narrower policy. | Keep the editor contract local to the feature, but require a generic `created_resource`/read-surface result and cleanup handle. Do not infer a site's IDs, URLs, or write tokens from method names in generic code. |
| **Feasibility and source-data quarantine** | The generic policy protocol is keyed by `(benchmark, site)` and owns auth self-test path, target probing, classification, source-data decision, and run bailout ([`phase_2c/policy.py`](../../worldsim/phase_2c/policy.py#L11-L100)). The WebArena policy contains GitLab/Reddit path/auth/probe branches and registers only those two sites ([`phase_2c/webarena.py`](../../worldsim/phase_2c/webarena.py#L219-L266), [`phase_2c/webarena.py`](../../worldsim/phase_2c/webarena.py#L280-L395)). Phase 2c also requires a registered editor and calls `probe_base_state` before task probes ([`phase_2/phase_2c/_impl.py`](../../worldsim/phase_2/phase_2c/_impl.py#L392-L430)). | This is admission infrastructure, not a carrier allowlist. It must quarantine stale source data/auth failures while leaving transient failures retryable. | A new feature needs a policy/test double for authenticated base-state and read-surface probes. The current compatibility helper `_task_probe_targets` directly instantiates `WebArenaFeasibilityPolicy` ([`phases/phase_2c_preflight.py`](../../worldsim/phases/phase_2c_preflight.py#L141-L142)); this is a concrete locality risk that can bypass a registered feature policy and should be covered by the vertical-slice test. |
| **Authentication and HTTP lanes** | Base editors centralize same-origin site URL, auth headers, and error taxonomy ([`editors/base.py`](../../worldsim/editors/base.py#L1-L80)). GitLab probes `/api/v4/user`; Reddit documents per-request `X-Postmill-Auto-Login` semantics and treats auth as always available when headers are supplied ([`editors/gitlab.py`](../../worldsim/editors/gitlab.py#L175-L197), [`editors/reddit.py`](../../worldsim/editors/reddit.py#L175-L198)). Shared config/auth helpers distinguish seeding API/form auth from agent storage-state/http-header/http-basic auth ([`config.py`](../../worldsim/config.py#L104-L292), [`agent_auth.py`](../../worldsim/agent_auth.py#L64-L170)). | GitLab PAT/form handling and Postmill headers are protocol-specific support for the active sites; the auth lane itself is shared safety plumbing. | A site feature should declare its protocol-specific probe and auth evidence while using central secret resolution, host scoping, and sanitized header helpers. Missing, stale, or host-mismatched credentials must remain a quarantine/fail-closed result. |
| **Render and visibility/readback** | Generic render signatures select body/title/free-text witnesses, but strict fallbacks are site-specific: GitLab issue/notes use same-origin resource probes and returned IDs ([`phase_2_render_check.py`](../../worldsim/phases/phase_2_render_check.py#L677-L794), [`phase_2_render_check.py`](../../worldsim/phases/phase_2_render_check.py#L821-L843)); Reddit comments require the exact seeded comment ID and entry-viewport visibility, not merely parent-page text ([`phase_2_render_check.py`](../../worldsim/phases/phase_2_render_check.py#L963-L985)). The admission guide makes exact discussion-region or seed-specific visibility mandatory ([`admission-and-exposure.md`](../../agent_docs/admission-and-exposure.md#L46-L66)). | These are active-carrier evidence rules. A historical renderer may remain in the tree without being an active admission path. | The feature must own selectors/JSON probes/actor or ID attribution and expose strict evidence to generic Phase 2c. A body substring on a parent page is insufficient; missing strict evidence must make the candidate ineligible. |
| **Exposure topology and ordered transitions** | Exposure contracts compose resolver output with registry specs, but mode constants and several transition/visibility guards still mention current kinds/methods ([`exposure_contract/_impl.py`](../../worldsim/phase_2/exposure_contract/_impl.py#L1-L7), [`exposure_contract/_impl.py`](../../worldsim/phase_2/exposure_contract/_impl.py#L63-L112), [`exposure_contract/_impl.py`](../../worldsim/phase_2/exposure_contract/_impl.py#L250-L277), [`exposure_contract/_impl.py`](../../worldsim/phase_2/exposure_contract/_impl.py#L527-L575), [`exposure_contract/_impl.py`](../../worldsim/phase_2/exposure_contract/_impl.py#L759-L838)). | Exposure is an admission contract, not merely seedability. Titles may remain seedable diagnostics but cannot be selected as retired carriers. | A third site must declare whether it is direct-detail, listing, or a forced created-child transition, including ordering and second-witness evidence. Registry support alone must not make a route eligible. |
| **Rewards and final-state scoring** | `_eval_final_state` first binds the expected network event, then dispatches GitLab/Reddit readback and rejects other sites ([`rewards/final_state.py`](../../worldsim/rewards/final_state.py#L27-L37), [`rewards/final_state.py`](../../worldsim/rewards/final_state.py#L73-L198)). GitLab and Reddit modules carry their exact readback, state-probe, and attribution semantics ([`rewards/final_state_webarena_verified_gitlab.py`](../../worldsim/rewards/final_state_webarena_verified_gitlab.py#L25-L118), [`rewards/final_state_webarena_verified_reddit.py`](../../worldsim/rewards/final_state_webarena_verified_reddit.py#L100-L217)). | Reward environment mapping intentionally ignores unsupported WebArena sites; historical evaluator/runners are support plumbing. | A new feature must provide a feature-local reward/readback proof or remain unsupported. Generic scoring must never credit a wrong-site event, broad scan, seeded object, or un-attributed actor. |
| **Evaluator/runners, caps, and cleanup** | Runner/environment maps retain full-WebArena historical names, while active reward mapping is restricted ([`rewards/webarena_sites.py`](../../worldsim/rewards/webarena_sites.py#L7-L42)). Phase 2c has explicit GitLab/Reddit per-replica caps plus a fallback cap ([`phase_2/phase_2c/_impl.py`](../../worldsim/phase_2/phase_2c/_impl.py#L199-L244)); cleanup and resource isolation are immediate and editor-owned ([technical specification](../warp-taskgen-technical-spec.md#L1434-L1448)). | Operational plumbing may mention sites that are not active carriers. | Keep evaluator environment selection, throughput caps, and cleanup policy separate from carrier eligibility. A test site should use an in-memory/fake host and explicit bounded cap, not accidentally inherit a production WebArena evaluator or cleanup sweep. |

### Additional seams that are easy to miss

The requested behaviors are not confined to files named `site`, `editor`, or
`reward`. A complete locality audit also has to account for these current
site-varying branches:

- **Action capability cards.** Generic action contracts currently list
  compatible sites, carrier surfaces, editor methods, benign task families,
  host evidence, and readback maturity; concrete adapters bind those contracts
  to route IDs and instruction/precondition/action-target cards
  ([`adversarial_actions/capability_contracts.py`](../../worldsim/adversarial_actions/capability_contracts.py#L48-L117),
  [`adversarial_actions/capability_adapters.py`](../../worldsim/adversarial_actions/capability_adapters.py#L33-L82)).
  A third site needs an explicit action-card opt-in. It must not inherit a
  generic mutation reward merely because its editor can write a body. The
  action-contract compiler still has site/method/seed/reward and selector
  wording branches for GitLab/Reddit, so this path is part of the migration
  inventory too ([`phase_1_contract_bound_action_api.py`](../../worldsim/phases/phase_1_contract_bound_action_api.py#L367-L410),
  [`phase_1_contract_bound_action_api.py`](../../worldsim/phases/phase_1_contract_bound_action_api.py#L900-L980)).
- **Reachability and PVPO hand-off.** Phase 2 reachability owns URL/auth/body
  witnesses and site/kind selectors; visual geometry is evidence telemetry and
  Phase 4 PVPO remains the encounter authority
  ([`phase_2_reachability.py`](../../worldsim/phases/phase_2_reachability.py#L1-L72),
  [`phase_2_reachability.py`](../../worldsim/phases/phase_2_reachability.py#L736-L860)).
  A new site needs a strict selector/readback witness or must remain
  ineligible; body text alone is not painted encounter evidence.
- **Phase 0 profile and inventory.** Profile reconstruction merges generated
  verification/data/injection evidence and current enrichment has GitLab project
  and Reddit forum-specific inventory paths. Live/empty-thread evidence must be
  provenance-bearing; stale LLM samples are not enough for a route
  ([`phase_0_recon.py`](../../worldsim/phases/phase_0_recon.py#L2122-L2169),
  [`phase_0c_reddit_enrichment.py`](../../worldsim/phases/phase_0c_reddit_enrichment.py#L39-L112),
  [`profile_validation.py`](../../worldsim/profile_validation.py#L39-L94)).
- **Auth bootstrap and text-fill defaults.** Storage-state liveness falls back
  to `True` when an editor is missing or its auth probe is unimplemented, and
  the UI-cookie liveness probe is GitLab-only
  ([`phase_0d_auth_bootstrap.py`](../../worldsim/phases/phase_0d_auth_bootstrap.py#L1514-L1566)).
  Unknown sites currently receive a generic voice exemplar fallback rather than
  failing closed ([`phase_2/text_fill/voice.py`](../../worldsim/phase_2/text_fill/voice.py#L58-L93)).
  A fake-site slice must install real test probes and an explicit voice mapping,
  or assert that generation is blocked.
- **Generation, Option A, and Phase 4 routing.** Novel-task generation admits
  any profile with route families, but one later guard special-cases only
  GitLab/Reddit; Option A normalization, orphan URL reconstruction, painted
  comment admission, issue-note anchors, and summary route variants also retain
  current-site branches. These are support gates, not safe generic defaults.
  They need feature descriptors or explicit ineligibility for a third site
  ([`phase_1_generate_new_tasks.py`](../../worldsim/phases/phase_1_generate_new_tasks.py#L221-L248),
  [`phase_1_generate_new_tasks.py`](../../worldsim/phases/phase_1_generate_new_tasks.py#L342-L360),
  [`phase_2/option_a.py`](../../worldsim/phase_2/option_a.py#L36-L109),
  [`phase_4/exposure_admission.py`](../../worldsim/phase_4/exposure_admission.py#L13-L88)).
- **Delivery-site identity.** Phase 2c resolves the mutation site from
  `delivery_channel.delivery_site`, then an editor-call site, then the task
  site; this is already needed for cross-site historical cases such as
  shopping-admin → shopping. A feature must preserve the distinction between
  the page/task site and the actual delivery/instance/reward site
  ([`phase_2/phase_2c/outcomes.py`](../../worldsim/phase_2/phase_2c/outcomes.py#L66-L88)).
- **Read-surface provenance.** Editor read surfaces intentionally include both
  host-qualified and path-only forms and record whether the URL came from an API
  response or construction. A new feature must supply the same provenance so
  cross-host replay cannot turn a guessed URL into readback evidence
  ([`editors/_read_surface.py`](../../worldsim/editors/_read_surface.py#L1-L12),
  [`editors/_read_surface.py`](../../worldsim/editors/_read_surface.py#L54-L104),
  [`test_read_surface_editors.py`](../../tests/test_read_surface_editors.py#L1-L12)).
- **Benchmark capability gate.** `benchmark_capabilities.py` separates
  WebArena Phase 2/4 from comparison-only benchmarks and rejects unknown or
  mixed benchmark metadata. Keep the third-site proof under
  `webarena_verified`; a site module must not accidentally broaden a comparison
  benchmark's Phase 2/2c support ([`benchmark_capabilities.py`](../../worldsim/benchmark_capabilities.py#L17-L53),
  [`benchmark_capabilities.py`](../../worldsim/benchmark_capabilities.py#L87-L118),
  [`test_benchmark_capabilities.py`](../../tests/test_benchmark_capabilities.py#L17-L31)).
- **Runtime auth and outcome taxonomy.** Browser Use has an explicit
  `--allow-unknown-auth` bypass and a `storage_state.per_task_refresh` mode that
  is rejected at runtime; a fake site should use an explicit `none`,
  `http_headers`, or storage-state lane and still pass auth preflight
  ([`browser_use_agent.py`](../../worldsim/browser_use_agent.py#L1542-L1591)).
  Outcome/action-attempt reporting also derives site and method labels; unknown
  mutation sites must be rejected by a feature reward adapter rather than
  falling through generic validity ([`outcome_taxonomy/_impl.py`](../../worldsim/outcome_taxonomy/_impl.py#L352-L503),
  [`rewards/action_attempt.py`](../../worldsim/rewards/action_attempt.py#L206-L222)).

## Why the editor registry is a useful nucleus—but not the whole adapter

The registry has unusually good locality today. A new editor method is
decorated in the same file as its implementation; registration validates every
declared method; the registry serializes a stable, benchmark/site-namespaced
contract; and existing consumers can ask for valid methods, attach surfaces,
or reachable tokens without importing platform code
([`_method_spec.py`](../../worldsim/editors/_method_spec.py#L84-L132),
[`_registry.py`](../../worldsim/editors/_registry.py#L66-L103),
[`_registry.py`](../../worldsim/editors/_registry.py#L173-L224),
[`_registry.py`](../../worldsim/editors/_registry.py#L245-L278)). The coverage
tests make this a concrete contract: every supported method must be decorated,
bindings must bridge LLM-facing and Python argument names, attach-surface
parity is frozen, and benchmark/site namespacing is tested
([`test_editors_registry_coverage.py`](../../tests/test_editors_registry_coverage.py#L137-L290)).

The registry deliberately does not own all of the following:

- URL grammar, route variants, listing source APIs, or anchor reconstruction;
- profile surface aliases and ambiguity rules;
- active-carrier policy (including retired title and unsupported MR methods);
- route-specific natural-encounter and ordered-child evidence;
- auth protocol probes and source-data quarantine policy;
- strict browser visibility, actor/ID attribution, and final-state readback;
- evaluator environment mapping, instance caps, or historical cleanup scripts.

Moving only `EDITOR_REGISTRY` would therefore create a false sense of
modularity. It could declare a method that placement knows about while the
resolver cannot classify its URL, the route contract cannot prove a benign
encounter, Phase 2c cannot probe its read surface, or reward code cannot prove
the persisted action. The specification's benchmark-agnosticity requirement
is stronger: canonical Phase 2c should contain no benchmark/site strings, with
knowledge in editor classes/registry and unchanged generic call sites
([technical specification](../warp-taskgen-technical-spec.md#L1471-L1479)).
The current hardcoded branches above are the migration inventory needed to make
that requirement true.

## Smallest safe vertical slice (evidence, not a final interface)

The lowest-risk proof is a **test-only fake site** on the existing
`webarena_verified` benchmark. It should not require a live WebArena service or
expand the production active-carrier cohort. Use a feature-local module and
fake/in-memory HTTP/browser fixtures to prove the hand-offs below. The names
and exact shape of a future interface remain open; the acceptance evidence is
the point.

1. **Explicit feature identity and policy.** Give the fake a normalized site
   key, one placeholder/profile fixture, one resource kind, and one body-like
   canonical carrier. Register it as test-only (or behind a named opt-in
   active-policy fixture), so the default WARP cohort still contains only
   GitLab and Reddit/Postmill. Add unknown-site and missing-placeholder
   negatives first.

2. **One editor method, fully contracted.** Implement one normal-user create
   operation in a feature-local fake editor. Decorate it with a kind, HTTP
   shape, token/selector/free-text bindings, profile-surface ID, and required
   args; register `(benchmark, fake_site)` and exercise registry serialization.
   The fake result should include a generic read-surface URL, a generic
   `created_resource` (role/kind/id/url, optional parent URL), per-call
   provenance, and an explicit cleanup handle. No generic seeding code should
   need to know the fake object's noun.

3. **Route and resolver proof.** Provide one deterministic detail route (a
   listing route can be a follow-up) with an inventory-backed anchor fixture,
   start/evaluation URL reconstruction, and profile-surface resolution. The
   generic resolver/exposure path should discover the feature's facts through a
   selected registration/descriptor seam, not a new `if fake_site` branch in a
   phase module. Assert ambiguous/unknown URLs return an empty or ineligible
   record.

4. **Exposure and active-policy proof.** Build an exposure contract from the
   resolved resource and registry method. Assert the body carrier is eligible
   only when the explicit test policy and route evidence are present; a title,
   non-core surface, missing read URL, or unsupported transition is ineligible.
   Preserve the separation between seed capability and Phase 4 topology. The
   current contract tests provide the model for these negative assertions
   ([`test_phase_2_exposure_contract.py`](../../tests/test_phase_2_exposure_contract.py#L13-L75),
   [`test_phase_2_exposure_contract.py`](../../tests/test_phase_2_exposure_contract.py#L427-L585)).

5. **Generic seed and cleanup proof.** Materialize an editor-call seed with one
   payload placeholder and benign anchor token. Run it through the existing
   seed dispatcher, verify the fake's read-surface/created-resource/per-call
   metadata, reject a phantom token or unresolved structural selector, and
   assert LIFO cleanup on a chained-call failure. Existing seed tests cover the
   desired generic evidence and rejection shape
   ([`test_seeding.py`](../../tests/test_seeding.py#L677-L735),
   [`test_seeding.py`](../../tests/test_seeding.py#L874-L1013),
   [`test_seeding.py`](../../tests/test_seeding.py#L1561-L1591)).

6. **Feasibility/auth/readback proof.** Register a feature-local feasibility
   policy or fake policy implementation. A healthy fake session must pass base
   state, auth, route, and read-surface probes; a missing/host-mismatched auth
   context, 401/403/404, or login stub must quarantine the task; transient
   timeout/5xx behavior must retain its retry classification. Use the fake
   browser/readback to require an exact payload witness bound to the created
   resource (and actor/ID when the carrier is appended). A parent-page
   substring alone must fail.

7. **Reward proof and default isolation.** Feed one matching network event and
   exact readback into a feature-local final-state evaluator or explicit local
   reward adapter; wrong site, missing event, stale seeded ID, and broad scan
   fixtures must fail. The default active reward dispatch must remain unchanged
   unless the opt-in fake policy is present. This proves scoring/readback
   locality without claiming the fake is a production WebArena evaluator.

8. **No phase call-site edits.** The slice is complete only if the generic Phase
   1/2/2c/seed/reward orchestration calls registry/contract APIs and does not
   name the fake site. If a generic branch is required, record it as the
   migration debt rather than hiding it inside a compatibility alias.

This slice demonstrates the entire safety chain while keeping the external
side effect bounded to an in-memory fixture. A live third site should be a later
slice, after its reset, auth, host isolation, evaluator/readback, and cleanup
contracts have independent evidence.

## Deletion test for locality

The architecture should have a deliberately adversarial deletion test:

1. Remove the fake feature module, its registration/fixture, and its feature
   tests. Do not change generic Phase 1, resolver, exposure, seed, Phase 2c, or
   reward modules.
2. Re-import the package and run the existing GitLab/Reddit registry, resolver,
   exposure, seeding, feasibility, render, and final-state tests. They should
   still pass.
3. Ask the generic resolver to classify a task whose site is now the removed
   key. It should return the existing empty/ineligible shape; a direct registry
   lookup should report unsupported site/method; no placeholder should resolve.
4. Run a static check such as `rg` over generic `worldsim/phase_1`,
   `worldsim/phase_2`, `worldsim/phase_2c`, `worldsim/phases`, and reward
   dispatch files. The removed site name may occur only in a test fixture or a
   generic registry API test, never in a production site branch.
5. Confirm that only the feature-local registration/contract tests fail when
   the module is absent. A broad import failure, silently eligible fallback, or
   change in active-site counts is evidence that behavior is still scattered.

This is stronger than a clean diff: it tests that the feature is removable and
that unknown input fails closed, as required by the admission and exposure
contract.

## Acceptance evidence matrix

| Contract to prove | Minimal evidence for the fake site | Existing primary-source analogue |
| --- | --- | --- |
| Identity and policy | Canonical/profile alias resolution; unknown benchmark/site and ambiguous mapping are `None`/ineligible; fake is opt-in | [`test_surface_identity.py`](../../tests/test_surface_identity.py#L27-L132), [`phase_2_core_surfaces.py`](../../worldsim/phases/phase_2_core_surfaces.py#L109-L119) |
| Editor contract | Decorator coverage, duplicate/missing-method rejection, binding alias coverage, serialized `(benchmark, site, method)` | [`test_editors_registry.py`](../../tests/test_editors_registry.py#L179-L452), [`test_editors_registry_coverage.py`](../../tests/test_editors_registry_coverage.py#L137-L290) |
| Route/resolver | Start/eval URL match, anchor reconstruction, profile route, listing/detail transition (if used), unknown URL empty record | [`test_phase_2_resolver_new_task_kinds.py`](../../tests/test_phase_2_resolver_new_task_kinds.py#L60-L237), [`target_resolution/resolver.py`](../../worldsim/phase_2/target_resolution/resolver.py#L29-L147) |
| Exposure/admission | Active body carrier eligible only with route evidence; retired/title/non-core/missing-readback and unknown site ineligible; exact contract signature | [`test_phase_2_exposure_contract.py`](../../tests/test_phase_2_exposure_contract.py#L13-L75), [`test_phase_2_exposure_contract.py`](../../tests/test_phase_2_exposure_contract.py#L610-L644) |
| Seeding | Registry dispatch, token reachability, generic created resource/read surfaces, per-call provenance, LIFO cleanup, unresolved selector rejection | [`test_seeding.py`](../../tests/test_seeding.py#L677-L735), [`test_seeding.py`](../../tests/test_seeding.py#L874-L1013), [`test_seeding.py`](../../tests/test_seeding.py#L1561-L1591) |
| Feasibility and auth | Base-state/auth/read probes, 2xx keep, auth/permission/stale-source quarantine, transient retry, registered policy | [`test_phase_2_feasibility.py`](../../tests/test_phase_2_feasibility.py#L2407-L2570), [`phase_2c/policy.py`](../../worldsim/phase_2c/policy.py#L33-L100), [`phase_2c/webarena.py`](../../worldsim/phase_2c/webarena.py#L73-L173) |
| Render/visibility | Exact body witness; appended carrier binds returned ID/actor and entry visibility; parent-only scan fails | [`test_phase_2_feasibility.py`](../../tests/test_phase_2_feasibility.py#L3331-L3467), [`test_final_state_webarena_verified_reddit.py`](../../tests/rewards/test_final_state_webarena_verified_reddit.py#L607-L799) |
| Reward/readback | Matching event plus exact persisted state passes; wrong site, missing event, wrong actor/ID, stale seeded resource, and unsupported site fail | [`test_final_state_webarena_verified_gitlab.py`](../../tests/rewards/test_final_state_webarena_verified_gitlab.py#L7-L55), [`test_final_state_webarena_verified_gitlab.py`](../../tests/rewards/test_final_state_webarena_verified_gitlab.py#L1275-L1385), [`test_final_state_webarena_verified_reddit.py`](../../tests/rewards/test_final_state_webarena_verified_reddit.py#L398-L469) |
| Locality/deletion | No generic fake-site branch; removal leaves current active sites passing and unknown fake ineligible | New test required; use the registry coverage and unknown-site tests above as the baseline |

For this research-only Markdown change, the relevant local evidence is a
source/spec truth check plus `git diff --check`. If implementation follows,
run focused pytest and Ruff first; changes to editors, seeding, Phase 2c,
render/readback, or Phase 4 use the live integration command when configured
([verification guide](../../agent_docs/verification.md#L30-L44)). The package
acceptance router remains the final focused gate (`bash
scripts/accept_taskgen.sh` from the repository root) when code is changed.

## Risks and unresolved design questions

- **Registry-only admission drift.** A method can be registered while route
  matching, profile identity, active policy, or strict readback remains absent.
  The vertical slice must require all contracts before eligibility.
- **Support leakage.** Historical editor methods, placeholders, runner maps,
  evaluator environments, and cleanup scripts can make a fake or retired site
  look production-ready. Keep “known,” “seedable,” “feasible,” and “active
  carrier” as separate semantic states.
- **Profile ambiguity.** Similar body/note fields may map to multiple profile
  surfaces. Unknown or ambiguous mappings must return no candidate rather than
  choosing the first match.
- **Route/order drift.** Listing and created-child routes can change sort order,
  pagination, or visual placement. A generic “newest” assumption is not
  encounter evidence; the feature must declare forced transition/ordering and a
  second witness where needed.
- **Readback over-credit.** Network attempts are not state success. Broad page
  scans, existing seeded objects, mutation endpoints, or a different actor must
  not satisfy a final-state reward.
- **Auth lane mismatch.** API/form seeding auth and browser agent auth have
  different lifecycles. Central helpers should continue to enforce secret
  sourcing, host scope, and header sanitization; feature code should only own
  protocol-specific probes.
- **Hardcoded token/write-key assumptions.** Current compatibility code still
  has site-named argument aliases and a finite write-token tuple. A third site
  should prove generic created-resource/per-call metadata before any alias is
  added, otherwise generic readback will silently lose the new ID.
- **Feasibility policy bypass.** The compatibility preflight helper directly
  constructs the WebArena policy for target paths even though a policy registry
  exists. A new-site test must exercise the canonical registered policy path and
  expose this bypass before shipping.
- **Placeholder/evaluator coupling.** Adding a placeholder does not imply a
  vendor evaluator environment. Keep local fake evaluation independent and
  require an explicit environment/reward registration for any live site.
- **Compatibility imports and monkeypatches.** Existing facades and patchable
  globals are part of the package contract. Move behavior behind a feature seam
  without breaking those imports, but do not create a second implementation.
- **Capacity and cleanup.** Site-specific per-replica limits and residual
  resources need explicit policy. A fallback cap or broad historical cleanup
  sweep is not evidence that a new site's operations are safe.

## Grill questions before choosing an interface

1. Is the third site test-only, support-only, or an active WARP carrier? What
   explicit policy bit makes that state visible in artifacts?
2. What is its canonical carrier ID, resource kind, profile surface ID, and
   natural benign route? Which exact evidence proves the agent reaches the
   carrier rather than its parent page?
3. Which feature owns URL matching, reconstruction, listing probes, and route
   variants? How does an unknown URL fail closed?
4. Does the editor contract describe enough read-surface, created-resource,
   cleanup, and attribution metadata without forcing generic code to learn the
   site's nouns?
5. What auth lane is used for seeding and for the browser agent? What self-test
   distinguishes stale credentials from a transient host outage?
6. If the carrier is appended or ordered, how are the exact discussion/child
   region and ordering transition proved? Is a second witness required?
7. How are returned IDs and actors bound through render and final-state reward,
   and how are benign seeded IDs excluded?
8. Does the feature policy prevent a registry-known support method from becoming
   an active carrier through fallback selection?
9. Can the feature be removed while the generic phase modules and current
   active-site tests remain unchanged? What static check demonstrates that?
10. What is the acceptance boundary before live infrastructure: fake HTTP,
    fake browser, smoke host, and then full evaluator parity?
11. Does adding a live site require a placeholder/evaluator environment map,
    reset endpoint, capacity cap, or cleanup worker, and are those support
    changes separately reviewed from carrier admission?
12. How are benchmark aliases and `(benchmark, site)` collisions normalized, and
    where is that identity recorded in fingerprints and artifacts?

## Primary references

- [`warp-taskgen-technical-spec.md`](../warp-taskgen-technical-spec.md)
- [`admission-and-exposure.md`](../../agent_docs/admission-and-exposure.md)
- [`runtime-boundaries.md`](../../agent_docs/runtime-boundaries.md)
- [`action-contracts.md`](../../agent_docs/action-contracts.md)
- [`code-organization.md`](../../agent_docs/code-organization.md)
- [`verification.md`](../../agent_docs/verification.md)
- [`worldsim/editors/`](../../worldsim/editors/), resolver, route-contract,
  exposure-contract, seeding, Phase 2c, render-check, and reward-local modules
- Focused registry, surface-identity, resolver, exposure, seeding, feasibility,
  render, and final-state tests linked in the acceptance matrix above
