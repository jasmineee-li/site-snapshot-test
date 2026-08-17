# New-site onboarding: primary-source research note (2026-08-16)

## Scope and evidence boundary

This note is a research and proposal artifact for a bounded WARP Site-onboarding
slice. It does not implement a Site, register a runtime plugin, change a
benchmark admission rule, or assert that a fake test host is a live browser
proof. “Evidence” below is either a cited primary source or a repository
file/line observation; “Inference” is the design consequence I draw from that
evidence; “Proposal” is an option for the user to accept or reject. No
percentage, throughput claim, or time-savings claim is made.

The local research branch is `codex/new-site-onboarding-research`. The initial
research pass inspected WARP at `27894b6e`; local current-state line citations
refer to that pre-cutover snapshot. The later section titled “User-reviewed
decision and bounded implementation evidence” records the approved vertical
slice added after review. Vendor repository observations below are from their
current `main` sources unless a release/tag is stated.

The safety boundary is three separate layers:

1. **Static closure** is a deterministic description/digest of typed owners for
   one `(benchmark, site, use_case)`.
2. **Active policy** is an explicit, run-scoped authorization. A static report
   cannot grant it.
3. **Live operational evidence** is host, admission, execution, visibility,
   and scoring evidence produced by the benchmark/runtime owner. It must not be
   inferred from static closure or a supplied fake observation.

That separation is already the contract in WARP’s domain vocabulary: Site
Targeting resolves deterministic routes but does not prove authentication,
reachability, visibility, admission, mutation, or scoring ([`CONTEXT.md`](../../../../CONTEXT.md#L41-L61));
ADR 0002 keeps seed, browser, exposure/admission, readback, and reward in their
existing owners ([ADR 0002](../../../../docs/adr/0002-cut-over-site-targeting-as-a-deep-module.md#L81-L91)).

## Evidence

### Local WARP contract and current seams

The repository router requires one focused short-lived worktree/PR and says to
preserve benchmark admission, exposure, visibility, scoring, readback, and
safety checks ([`AGENTS.md`](../../../../AGENTS.md#L25-L52)). The package guide
says AgentLab trees are read-only references, optional
runtime dependencies stay behind a sidecar, and the admitted WASP surfaces are
GitLab issues/comments and Reddit/Postmill posts/comments; Classifieds is a
bounded experimental canary ([`CLAUDE.md`](../../CLAUDE.md#L1-L24),
[`CLAUDE.md`](../../CLAUDE.md#L62-L70)). The technical specification states that
the normal pipeline assumes the benchmark environment is already running and
does not manage its lifecycle; the `reset_endpoint` belongs to the env-control
sidecar, not the Site URL ([technical spec](../warp-taskgen-technical-spec.md#L39-L52),
[`technical spec`](../warp-taskgen-technical-spec.md#L132-L143)). It also says
AgentLab/BrowserGym are comparison or Phase 4 sidecars and that root Taskgen
does not import AgentLab ([technical spec](../warp-taskgen-technical-spec.md#L615-L619)).

ADR 0004 makes the boundary explicit: Benchmark capability registration is not
admission; comparison-only benchmarks are not generative, executable, or
scorable; canonical WebArena task IDs remain under the vendor evaluator; and
AgentLab/vendor evaluators stay sidecar/subprocess adapters
([ADR 0004](../../../../docs/adr/0004-compose-benchmark-contracts-separately.md#L7-L11)).

The existing architecture note identifies the real onboarding graph as
identity/profile → routes → editor metadata → exposure → seed →
feasibility/visibility → readback/reward → cleanup, and says the smallest safe
proof is a test-only Site crossing that graph without a live environment
([architecture note](site-feature-module-architecture-2026-08-10.md#L19-L42),
[`architecture note`](site-feature-module-architecture-2026-08-10.md#L199-L275)).
It also warns that an editor registry is a useful nucleus, not the whole Site
adapter ([`architecture note`](site-feature-module-architecture-2026-08-10.md#L162-L197)).

The static contract types capability states as `supported`, `not_applicable`,
`unsupported`, or `missing`; `supported` and `not_applicable` both map to a
passing finding, while operational evidence has only `supported`,
`unsupported`, and `missing` ([`site_composition_contracts.py`](../../warp_taskgen/site_composition_contracts.py#L18-L38),
[`site_composition_contracts.py`](../../warp_taskgen/site_composition_contracts.py#L187-L221)).
`SiteDoctorReport` carries both `static_status` and overall `status` and only a
complete static report may be `ready` ([`site_composition_contracts.py`](../../warp_taskgen/site_composition_contracts.py#L248-L290)).
The compiler’s public docstring says it validates closure without executing
owners ([`site_composition.py`](../../warp_taskgen/site_composition.py#L1-L7),
[`site_composition.py`](../../warp_taskgen/site_composition.py#L579-L587)),
but the implementation invokes `owner.validate()` and `owner.routes()`
([`site_composition.py`](../../warp_taskgen/site_composition.py#L265-L313)),
`canonicalize_surface_id()` ([`site_composition.py`](../../warp_taskgen/site_composition.py#L314-L338)),
and `card.validate()` plus route/profile canonicalization
([`site_composition.py`](../../warp_taskgen/site_composition.py#L411-L489)).
This is a **purity/side-effect-guard gap** to close or document; it is not
evidence that an editor, evaluator, browser, or host ran.

The compiler records active policy and external operational states separately;
when those states are absent it says the static doctor does not infer them
([`site_composition.py`](../../warp_taskgen/site_composition.py#L692-L742)).
The CLI exit code, however, reads only `report.static_status`, not overall
`report.status` ([`site_doctor.py`](../../warp_taskgen/cli/site_doctor.py#L17-L18),
[`site_doctor.py`](../../warp_taskgen/cli/site_doctor.py#L153-L160)). Therefore a
static-complete report can exit `0` while the overall report is `blocked`; a
caller must inspect the serialized overall status and findings.

The current static Classifieds definition seeds with `ClassifiedsEditor` and
declares `final_state` `not_applicable` for the diagnostic use case
([`site_composition_defaults.py`](../../warp_taskgen/site_composition_defaults.py#L84-L123)).
The named live runtime composition instead seeds with
`ClassifiedsAuthenticatedEditor` ([`runtime_composition.py`](../../warp_taskgen/runtime_composition.py#L77-L101)).
Consequently the current static digest is not identity-equivalent to the live
writer/runtime composition. `not_applicable` is a pass in the static finding
model, and `ugc_reply` exempts `final_state` in its required-edge calculation;
states must therefore be defined per use case, not treated as a universal
runtime guarantee.

The test-only `proof_forum` fixture demonstrates the intended chain: route and
profile resolution, editor seed, read-surface plan, readback interpretation,
final-state evaluation, cleanup, and negative cases ([`test_site_composition.py`](../../tests/test_site_composition.py#L342-L506)).
Its fake readback accepts a caller-supplied payload only when identity tokens,
signature, and an `independent_reader_visible` flag match
([`test_site_composition.py`](../../tests/test_site_composition.py#L170-L184)).
That tests interpretation of an observation; it does **not** prove an
independent reader/browser actually produced the observation. Live evidence
must remain a separate owner and acceptance gate.

The package-boundary tests provide useful deletion and clean-room constraints:
the test Site noun cannot appear in generic Phase/reward/seeding modules, a
composition import does not mutate `SiteCatalog` or `EDITOR_REGISTRY`, removal
of a definition fails closed, and the package imports from an unrelated CWD
without bootstrapping editor registrations ([`test_composition_package.py`](../../tests/sites/test_composition_package.py#L1-L8),
[`test_composition_package.py`](../../tests/sites/test_composition_package.py#L53-L79),
[`test_composition_package.py`](../../tests/sites/test_composition_package.py#L81-L169),
[`test_composition_package.py`](../../tests/sites/test_composition_package.py#L182-L234)).
The package is `warp-taskgen` 0.1.1, requires Python 3.12+, and builds the
`warp_taskgen` package into both wheel and sdist; its `pyproject.toml` has no
`license` field, so this note makes no package-license assertion
([`pyproject.toml`](../../pyproject.toml#L1-L5), [`pyproject.toml`](../../pyproject.toml#L67-L79)).

### Primary external evidence

**BrowserGym task and benchmark contracts.** The official BrowserGym source
defines a custom task around `get_task_id`, a seeded constructor, `setup(page)`
returning goal/info, `validate(page, chat_messages)` returning reward/done/
message/info, and optional teardown ([`AbstractBrowserTask`](https://raw.githubusercontent.com/ServiceNow/BrowserGym/main/browsergym/core/src/browsergym/core/task.py)).
Its registration helper validates frozen/default kwargs and calls Gymnasium
`register` with a `browsergym/<id>` entry point
([`registration.py`](https://raw.githubusercontent.com/ServiceNow/BrowserGym/main/browsergym/core/src/browsergym/core/registration.py)).
`BrowserEnv.reset()` creates the task and runs `setup`; `close()` invokes task
teardown ([`env.py`](https://raw.githubusercontent.com/ServiceNow/BrowserGym/main/browsergym/core/src/browsergym/core/env.py)).
The benchmark base owns action-set configuration, seeds, backend preparation,
and dependency ordering ([`Benchmark`](https://raw.githubusercontent.com/ServiceNow/BrowserGym/main/browsergym/experiments/src/browsergym/experiments/benchmark/base.py)).
The current `main` source declares version `0.14.3` ([`__init__.py`](https://raw.githubusercontent.com/ServiceNow/BrowserGym/main/browsergym/core/src/browsergym/core/__init__.py));
the repository is Apache-2.0 licensed ([`LICENSE`](https://raw.githubusercontent.com/ServiceNow/BrowserGym/main/LICENSE)).
The BrowserGym ecosystem paper is published as TMLR 2025 with Expert
Certification ([official publication page](https://www.servicenow.com/research/publication/thibault-le-sellier-de-chezelles-the-tmlr2025.html),
[OpenReview record](https://openreview.net/forum?id=5298fKGmv3)), not merely a
preprint.

**AgentLab extension boundary.** AgentLab’s official README describes an
agent-development/evaluation framework over BrowserGym, supports WebArena,
WebArena-Verified, VisualWebArena, and OSWorld, and records benchmark/package/
commit information in a `Study`; it notes manual reset and reproducibility
factors such as software versions, live websites, and stochasticity
([AgentLab README](https://github.com/ServiceNow/AgentLab/blob/main/README.md)).
`make_study` accepts a benchmark and picklable/importable agent arguments and
can modify an existing benchmark ([`study.py`](https://raw.githubusercontent.com/ServiceNow/AgentLab/main/src/agentlab/experiments/study.py)).
AgentLab’s package metadata declares Python 3.11–3.12, a dynamic VCS version,
Apache-2.0, and BrowserGym/Gymnasium dependencies
([`pyproject.toml`](https://raw.githubusercontent.com/ServiceNow/AgentLab/main/pyproject.toml),
[`LICENSE`](https://raw.githubusercontent.com/ServiceNow/AgentLab/main/LICENSE)).
These are experiment/agent orchestration seams, not ownership of WARP Site
HTTP routes, reset, readback, or evaluator semantics.

**Gymnasium registry and conformance.** Gymnasium’s official registry docs say
`register()` places an `EnvSpec` in a process-global registry and `make()` loads
a previously registered environment; module imports may register IDs
([registry API](https://gymnasium.farama.org/api/registry/)). The official
`check_env()` utility checks reset/step/render/close, spaces, and API behavior
([utility API](https://gymnasium.farama.org/api/utils/)). The current release
page lists v1.3.0 and the repository is MIT licensed ([releases](https://github.com/Farama-Foundation/Gymnasium/releases),
[`LICENSE`](https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/LICENSE)).
The global registry is useful as a warning and conformance analogy; WARP must
not copy import-time registration or make discovery imply admission.

**PyPA discovery and schema references.** The PyPA entry-point specification
defines installed-distribution metadata that advertises named components to a
consumer-defined interface; the consumer chooses duplicate-name semantics and
object loading ([entry points specification](https://packaging.python.org/en/latest/specifications/entry-points/)).
It is a possible packaging seam for a future *explicitly selected* Site
provider, not permission to auto-admit an installed package. JSON Schema Draft
2020-12 (published 16 June 2022) provides Core and Validation vocabularies
([draft page](https://json-schema.org/draft/2020-12), [Core](https://json-schema.org/draft/2020-12/json-schema-core.html),
[Validation](https://json-schema.org/draft/2020-12/json-schema-validation.html)).
OpenAPI 3.1.1 (24 October 2024) is a language-agnostic HTTP interface
description with Apache-2.0 terms ([specification](https://spec.openapis.org/oas/v3.1.1.html),
[source repository](https://github.com/OAI/OpenAPI-Specification)). Both are
references for manifest/HTTP vocabulary only; neither defines WARP admission,
identity, visibility, readback, or scoring.

**Conformance and scaffolding precedents.** Kubernetes Gateway API conformance
is explicitly versioned and profile-based: reports identify the API version,
channel, implementation, supported features, and unsupported features, and a
conformant result cannot skip applicable tests
([conformance guide](https://gateway-api.sigs.k8s.io/guides/implementers-guide/),
[GEP-1709](https://gateway-api.sigs.k8s.io/geps/gep-1709/)). Terraform Plugin
Framework separates offline configuration validation and diagnostics from
acceptance tests that exercise a real API and require an isolated test account
([schema validation](https://developer.hashicorp.com/terraform/plugin/framework/handling-data/schemas),
[configuration validation](https://developer.hashicorp.com/terraform/plugin/framework/resources/validate-configuration),
[acceptance tests](https://developer.hashicorp.com/terraform/plugin/testing/acceptance-tests)).
Backstage Software Templates make generated structure reviewable and give
custom actions an explicit `supportsDryRun` contract; dry-run execution cannot
stand in for the real action
([adding templates](https://backstage.io/docs/features/software-templates/adding-templates/),
[dry-run testing](https://backstage.io/docs/features/software-templates/dry-run-testing/)).
These are useful precedents for versioned reports, actionable diagnostics, and
thin scaffolding, not reasons to copy their extension models wholesale.

The official pytest plugin guide shows the operational cost of entry-point
autoloading: `pytest11` plugins are loaded from installed distributions unless
autoload is disabled, and `--trace-config` is needed to inspect what loaded
([pytest plugin discovery](https://docs.pytest.org/en/latest/how-to/writing_plugins.html)).
pluggy can validate hook implementations and reject unknown hooks, but it still
only supplies registration and call mechanics
([pluggy API](https://pluggy.readthedocs.io/en/stable/api_reference.html)).
OpenTelemetry Semantic Conventions distinguish stable from experimental
conventions and constrain changes according to stability, illustrating why a
WARP definition format would need an explicit contract version rather than an
unversioned digest
([semantic-convention groups](https://opentelemetry.io/docs/specs/semconv/general/semantic-convention-groups/),
[versioning and stability](https://opentelemetry.io/docs/specs/otel/versioning-and-stability/)).

**WebArena, WebArena Verified, and VisualWebArena ownership.** WebArena’s
official repository is Apache-2.0 and identifies its canonical implementation
and stable dataset ([repository](https://github.com/web-arena-x/webarena),
[`LICENSE`](https://raw.githubusercontent.com/web-arena-x/webarena/main/LICENSE)).
Its environment documentation stops/removes/restarts Docker services for reset,
so reset is an environment owner concern ([environment README](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md)).
WebArena Verified is an Apache-2.0, version-controlled dataset with deterministic
evaluators over agent responses/network traces and an env-control `POST /init`
reset boundary ([repository](https://github.com/ServiceNow/webarena-verified),
[environment docs](https://servicenow.github.io/webarena-verified/dev/environments/),
[`LICENSE`](https://raw.githubusercontent.com/ServiceNow/webarena-verified/main/LICENSE)).
Its README records public release/PyPI and SEA Workshop at NeurIPS 2025 dates;
the associated paper is a 2025 workshop paper, not a NeurIPS main-track paper
([paper record](https://openreview.net/forum?id=94tlGxmqkN)). VisualWebArena is
an MIT-licensed multimodal benchmark; its evaluator source has separate string,
URL-exact, HTML-content-exact, and page-image evaluators, and its Docker/reset
scripts own service reset ([repository](https://github.com/web-arena-x/visualwebarena),
[`evaluators.py`](https://raw.githubusercontent.com/web-arena-x/visualwebarena/main/evaluation_harness/evaluators.py),
[`LICENSE`](https://raw.githubusercontent.com/web-arena-x/visualwebarena/main/LICENSE),
[reset README](https://raw.githubusercontent.com/web-arena-x/visualwebarena/main/environment_docker/README.md)).
These sources support a strict WARP rule: Site composition may report required
reset/evaluator *evidence*, but cannot own or simulate the vendor environment.

**WASP and recent benchmark work.** WASP’s official repository is archived/read-
only as of 1 July 2026, describes an isolated end-to-end web environment with
cleanup, and is predominantly CC-BY-NC 4.0 alongside VisualWebArena’s MIT
components ([repository](https://github.com/facebookresearch/wasp),
[`README.md`](https://raw.githubusercontent.com/facebookresearch/wasp/main/README.md),
[`LICENSE`](https://raw.githubusercontent.com/facebookresearch/wasp/main/LICENSE)).
The work is peer-reviewed in *NeurIPS 2025, Datasets and Benchmarks Track*,
not just an arXiv preprint ([NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1c9818387f5dd0a0bc151214660f059d-Abstract-Datasets_and_Benchmarks_Track.html));
the arXiv version is a preprint ([arXiv:2504.18575](https://arxiv.org/abs/2504.18575)).

ST-WebAgentBench’s official repository (Apache-2.0) separates policy hierarchy,
action traces, safety dimensions, and evaluator types; its README says accepted
at ICLR 2026, while the linked arXiv record remains a preprint
([repository](https://github.com/segev-shlomov/ST-WebAgentBench),
[`LICENSE`](https://raw.githubusercontent.com/segev-shlomov/ST-WebAgentBench/main/LICENSE),
[arXiv:2410.06703](https://arxiv.org/abs/2410.06703)). WebCanvas’s MIT-licensed
source decouples JavaScript event-listener evaluation from action-space
handling; its paper is an arXiv preprint ([repository](https://github.com/imeanai/webcanvas),
[`README.md`](https://raw.githubusercontent.com/imeanai/webcanvas/main/README.md),
[`LICENSE`](https://raw.githubusercontent.com/imeanai/webcanvas/main/LICENSE),
[arXiv:2406.12373](https://arxiv.org/abs/2406.12373)). OSWorld’s Apache-2.0
repository documents OSWorld-Verified and custom execution evaluators; its
widely cited paper is an arXiv preprint ([repository](https://github.com/xlang-ai/OSWorld),
[`LICENSE`](https://raw.githubusercontent.com/xlang-ai/OSWorld/main/LICENSE),
[arXiv:2404.07972](https://arxiv.org/abs/2404.07972)). These recent sources all
reinforce benchmark-local policy/evaluator ownership rather than a universal
Site registry.

## Post-cutoff precedent check (2024–2026)

This focused check uses first-party specifications, documentation, and source
repositories only. It tests six concrete WARP design questions; it is not an
argument for a general plugin framework or process-wide discovery.

### Evidence

1. **Immutable/data-only declarations separate from behavior.** Gateway API’s
   conformance-profile specification describes a profile as a static
   compilation of supported features and makes the report carry API version,
   channel, implementation version, mode, and profile results
   ([GEP-1709](https://gateway-api.sigs.k8s.io/geps/gep-1709/)). The current
   Gateway API release is v1.5.1 (14 March 2026), Apache-2.0, and the project
   explicitly points implementers to versioned conformance reports
   ([release](https://github.com/kubernetes-sigs/gateway-api/releases/tag/v1.5.1),
   [repository](https://github.com/kubernetes-sigs/gateway-api)). Terraform
   Plugin Framework documents schemas as data-shape metadata and provides
   `ValidateImplementation` unit checks, while its acceptance tests separately
   perform real plan/apply/refresh/destroy operations
   ([schemas](https://developer.hashicorp.com/terraform/plugin/framework/handling-data/schemas),
   [acceptance tests](https://developer.hashicorp.com/terraform/plugin/testing/acceptance-tests),
   [framework license](https://github.com/hashicorp/terraform-plugin-framework/blob/main/LICENSE)).
   The official pages route to Plugin Framework v1.18.x and Plugin Testing
   v1.15.x at this cutoff.
   **Verdict:** adopt a typed, immutable Site projection for static closure;
   keep mutation, reset, browser, readback collection, and scoring in existing
   executable owners.

2. **Behavior contracts as observable tests, not plugin/base-class APIs.** The
   current pytest documentation demonstrates ordinary `def test_*` functions
   and Python `assert` statements, including assertions over values and exact
   exceptions ([pytest assertions](https://docs.pytest.org/en/stable/how-to/assert.html)).
   Terraform’s schema unit-test example calls the public schema method and
   asserts diagnostics before any acceptance test is run
   ([schema unit testing](https://developer.hashicorp.com/terraform/plugin/framework/handling-data/schemas#unit-testing)).
   Pluggy’s official API is registration and hook dispatch, while pytest’s
   official plugin guide documents installed-distribution autoload and
   `--trace-config` inspection ([pluggy register](https://pluggy.readthedocs.io/en/stable/api_reference.html#pluggy.PluginManager.register),
   [pytest plugin discovery](https://docs.pytest.org/en/stable/how-to/writing_plugins.html)).
   **Verdict:** organize the WARP **Site Behavior Contract** by observable
   behavior and plain assertion/test interfaces; do not require Site classes
   to inherit a plugin base or use plugin registration to run the tests.

3. **Stable semantic owner IDs distinct from implementation paths.** OpenTelemetry
   Semantic Conventions define stable versus experimental semantic names and
   require explicit version/stability handling; the official docs render
   Semantic Conventions 1.44.0 at this cutoff, while the first-party release
   list’s latest tagged release is v1.43.0, so this note does not treat the
   generated docs number as a release tag ([semantic conventions](https://opentelemetry.io/docs/specs/semconv/),
   [stability](https://opentelemetry.io/docs/specs/otel/versioning-and-stability/),
   [version selection](https://opentelemetry.io/docs/specs/semconv/configuration/version-selection/),
   [release list](https://github.com/open-telemetry/semantic-conventions/releases)).
   Gateway reports likewise identify an implementation/project/version rather
   than a source-language import path ([reporting rules](https://gateway-api.sigs.k8s.io/geps/gep-1709/#reporting-process)).
   **Verdict:** assign each WARP owner a stable semantic ID; carry Python
   module/class paths, distribution versions, and source commits only as
   provenance. A path rename must not silently change owner identity.

4. **A separately named, versioned digest for static composition.** OCI Image
   Format v1.1.1 (3 March 2025, Apache-2.0) defines a descriptor digest as a
   content identifier over bytes and requires independent verification before
   consumption ([descriptor](https://github.com/opencontainers/image-spec/blob/v1.1.1/descriptor.md),
   [release](https://github.com/opencontainers/image-spec/releases/tag/v1.1.1),
   [license](https://github.com/opencontainers/image-spec/blob/v1.1.1/LICENSE)).
   Gateway’s reports additionally require a specific implementation release
   and reproduction instructions ([reporting process](https://gateway-api.sigs.k8s.io/geps/gep-1709/#reporting-process)).
   **Verdict:** WARP should use a separately named `SiteCompositionDigest`, hash
   a canonical static payload, and include contract/schema version plus source
   and distribution provenance. This is Site Composition integrity, not
   behavioral or live proof; do not conflate it with a Runtime Composition or
   the Definition Digest of a Run Definition.

5. **Static check reports must not imply runtime readiness or admission.**
   Gateway emits a machine-readable `ConformanceReport` with implementation,
   date, API version/channel, profile, and test results; its profile/report
   machinery is explicitly a conformance tool rather than a Kubernetes API
   ([report format](https://gateway-api.sigs.k8s.io/geps/gep-1709/#reporting-process)).
   Backstage’s official dry-run contract simulates an action and tests that the
   handler performs no action when `isDryRun` is true
   ([dry-run testing](https://backstage.io/docs/features/software-templates/dry-run-testing/)).
   Terraform makes the boundary even sharper: `validate`/schema checks are
   distinct from acceptance tests that require network access and credentials
   ([schemas](https://developer.hashicorp.com/terraform/plugin/framework/handling-data/schemas),
   [acceptance requirements](https://developer.hashicorp.com/terraform/plugin/testing/acceptance-tests#requirements-and-recommendations)).
   **Verdict:** the canonical `site composition check` command and its
   `SiteCompositionCheckReport` must say static Site Composition only. Its exit
   code must not be read as active policy, Benchmark Instance reachability,
   admission, execution, visibility, or scoring. `site doctor` may remain only
   as a compatibility alias.

6. **Synthetic implementations need test-only identity and deletion/locality
   proof.** Backstage’s dry-run tests construct a context with `isDryRun` and
   assert no side effect, while pytest’s `pytester` guidance runs test projects
   in isolation ([Backstage dry-run tests](https://backstage.io/docs/features/software-templates/dry-run-testing/),
   [pytester/plugin testing](https://docs.pytest.org/en/stable/how-to/writing_plugins.html#testing-plugins)).
   Gateway’s conformance suite can be invoked as a test CLI or library and
   writes reproducible reports, but it still tests an implementation’s real
   conformance rather than granting runtime policy
   ([GEP-1709 integration](https://gateway-api.sigs.k8s.io/geps/gep-1709/#integration)).
   **Verdict:** WARP’s `synthetic_discussion_forum` must carry an unmistakable
   test-only identity, consistently use `thread` and `comment` resource terms,
   use fake adapters with no credentials or network, and be deleted without
   changing GitLab, Reddit, Classifieds, generic Phase code, or default
   catalogs. Locality and deletion are WARP acceptance properties, not claims
   supplied by any external framework.

### Inference

- A declaration can safely name edges, owner IDs, states, provenance, and a
  versioned digest only when it never dispatches executable owner behavior.
- Observable conformance should be a feature-local test suite over public
  seams. Registration and discovery are test mechanics, not an activation
  policy or a source of live evidence.
- Stable IDs and canonical digests make reports comparable across Python path
  changes, package rebuilds, and explicit Runtime Composition selections while
  preserving provenance for audit.
- A static doctor may produce actionable missing-owner diagnostics and a green
  static status, but active policy and live evidence require separate ledgers
  and gates.

### Design impact

The six verdicts support **Option B on the constrained Option A baseline**:
keep the shallow in-repo typed Site Composition, add a feature-local
`synthetic_discussion_forum` and reusable Site Behavior Contract, make
`site composition check` the exact static command, and version its
`SiteCompositionDigest`. Do not add a generic Phase, default/global
registration, entry-point loader, or generated runtime behavior. Keep a later
live gate for Fresh Anonymous Reader/browser evidence, Benchmark Instance
reachability, Golden-State Reset, admission, execution, and scoring.

## Inference

1. BrowserGym and Gymnasium expose global registration because they are runtime
   environment frameworks. WARP’s ADRs and package-boundary tests intentionally
   require the opposite at the Taskgen root: static definitions and named
   per-run catalogs must be explicit, and import/discovery must not become
   process-wide admission. This is a safety and reproducibility boundary, not a
   claim that upstream frameworks are defective.
2. AgentLab’s `Study` and BrowserGym’s `Benchmark` are useful sidecar payloads
   (pinned commit, benchmark metadata, task setup/validation), but they cannot
   replace WARP’s Site identity, exposure/readback, cleanup, or operational
   evidence. A future adapter should serialize a neutral request and preserve
   native reward/evaluator authority.
3. PyPA entry points could make a separately packaged Site discoverable, but
   duplicate handling, package/version pinning, and explicit selection would be
   required before loading. Entry-point discovery must feed a static candidate
   list or named runtime composition; it must not silently mutate
   `SiteCatalog`, `EDITOR_REGISTRY`, or admission policy.
4. JSON Schema 2020-12 and OpenAPI 3.1.1 can describe a manifest and HTTP route
   facts, but typed Python owners remain necessary for identity, method
   behavior, readback interpretation, and cleanup. A schema-valid manifest is
   not a live proof.
5. Vendor reset/evaluator sources make the owner split concrete: env-ctrl or
   benchmark containers reset state; vendor evaluator code owns canonical task
   IDs and scoring; WARP composes only task-id-less local proof or comparison
   envelopes. A static Site digest must therefore carry provenance to these
   owners, not copy their behavior.
6. The current compiler’s owner-method calls make purity an assumption rather
   than a guarded property. Until owners are explicitly side-effect-free or
   the compiler uses a pure projection interface, the static doctor should be
   treated as deterministic validation with a side-effect risk—not as proof of
   a live action.
7. The Classifieds static/runtime class mismatch demonstrates why a digest must
   identify the exact writer factory used by the named runtime composition, or
   explicitly label the static definition diagnostic-only. A “supported” seed
   edge alone cannot establish identity equivalence.

## Proposal: bounded onboarding options

The comparison below is about an implementation change after user approval. No
option is implemented by this note.

| Handoff option | Locality and files touched | Safety and diagnostics | Reproducibility and packaging | Migration and deletion |
|---|---|---|---|---|
| **A — typed in-repo Site definition** | One feature-local definition can reference the nine existing owners and project doctor input. It becomes shallow and duplicative if it also dispatches runtime behavior or rewrites every catalog. | Good for deterministic structural closure, provided `supported` never means admitted or live. Current arbitrary owner-method calls are a purity gap. | Strongest while definitions are explicit and in-tree. The digest needs a contract version and must remain declaration evidence, not behavioral proof. | A constrained form already exists. Removing a definition must fail closed and leave unrelated definitions byte-for-byte stable. |
| **B — conformance kit only** | Keeps current explicit registrations and extracts the existing fake-forum behavior chain into a reusable, feature-local Site Behavior Contract. It should require no generic Phase edit and no new registry. | Best next evidence-producing slice: exact negative cases, deletion, comparison-only rejection, package closure, and actionable missing-owner diagnostics. It does not activate policy. | Deterministic, network-free, credential-free, and runnable against wheel/sdist installs. No new distribution mechanism. | Low migration cost. Deleting `synthetic_discussion_forum` removes all synthetic Site behavior; the Site Behavior Contract remains useful. |
| **C — minimal scaffold plus conformance kit** | Generates only a feature package, typed definition, tests, and documentation stubs; behavior remains executable Python owned by the Site. | Safe only if every generated capability starts `missing` or `unsupported` and generation never registers/adopts it. A premature template can freeze the wrong contract. | Generated output and template version would need deterministic golden tests and package-resource checks. | Moderate migration and maintenance. Revisit only if a clean-room trial shows repeated mechanical file creation—not interface understanding—is the bottleneck. |
| **D — external entry-point plugin discovery** | Provider code can live in another distribution, but discovery adds a loader, trust policy, duplicate resolution, version/resource conflicts, and installed-environment state. | Highest false-admission and hidden-global-state risk. Entry points locate objects; they do not establish WARP capability, purity, admission, or live evidence. | Requires pinned distributions, explicit allowlists, conflict rejection, contract negotiation, and provenance in the digest. | Reject now: there is no demonstrated independent-distribution requirement. Deletion can leave environment or lockfile residue. |

**Recommendation (proposal).** Choose **Option B on the existing constrained
Option A baseline**. Replace the generic static type names with the agreed
`SiteComposition`, `SiteBenchmarkComposition`, and `SiteOwnerDeclaration`
names, and keep this as a shallow, immutable composition root;
do not broaden it into a runtime adapter or default catalog. Extract a reusable
`synthetic_discussion_forum` Site Behavior Contract, add exact
`site composition check` discovery/diagnostics, and run
the clean-room comparison before considering a scaffold. Reject Option C for
now because it would stabilize templates before the conformance interface and
purity semantics are settled. Reject Option D until a real separately
distributed Site creates a concrete need and its trust, conflict, version, and
admission model is solved.

This recommendation follows the deep-module test: the definition has leverage
only when its small interface hides deterministic closure logic while behavior
stays in feature-local owners. A Site mega-adapter, plugin marketplace, or
generic capability graph would expose more coupled concepts than it hides and
would fail the locality and deletion tests.

### Behavior-level acceptance and clean-room evaluation

The later implementation target is behavior, not a class count or coverage
number. A synthetic, test-only `synthetic_discussion_forum` must satisfy all of
these:

1. No `synthetic_discussion_forum` name is added to generic Phase code; its behavior and test
   fixtures are local to one Site package.
2. The exact `(benchmark, site, use_case, carrier, action_kind)` request crosses
   canonical route/profile resolution, editor metadata and a regular-participant
   fake writer, seeding, feasibility, read-surface planning, a separately
   constructed fresh-anonymous-reader observation, exact readback, final-state
   evaluation where required, action-card validation, and idempotent cleanup.
3. Exact Resource Evidence binds mutation, readback, visibility, and cleanup to
   the same resource and parent route. Wrong Site, origin, parent, actor, body,
   signature, stale/foreign ID, ambiguous match, writer-context reuse, or
   cleanup failure fails closed.
4. Unknown, duplicate, malformed, and removed Sites fail closed. Removing the
   fixture leaves GitLab, Reddit, and the explicit Classifieds diagnostic/runtime
   POC unchanged; no fallback silently selects a default Site or composition.
5. Importing or registering a definition does not grant active policy.
   The static check accepts no policy or operational-evidence input, labels both
   ledgers `not_checked`, and reports operational readiness as `blocked` even
   when static closure is complete. A later readiness owner may evaluate those
   ledgers explicitly; the static checker cannot promote them.
6. Comparison-only Benchmarks pass only comparison ingestion. They remain
   non-generative, non-executable, non-evaluable, and non-scorable even if a Site
   or action card is present.
7. For every required edge, `site composition check` reports the exact missing owner and a
   deterministic declaration digest. It accepts and echoes carrier/action kind,
   or explicitly labels the output as broad structural diagnostics. Exit status
   must not be presented as readiness when only `static_status` is complete.
8. `supported`, `not_applicable`, `unsupported`, and `missing` remain distinct.
   `not_applicable` is compiler-derived only for an edge the exact use case does
   not require; it never satisfies a required active edge.
9. Static compilation either reads a pure data projection or invokes an
   explicitly documented deterministic validation seam. Subprocess tests reject
   environment/secret reads, network/browser calls, global mutation, sleeps,
   and import-order dependence from static validation.
10. Digest/report bytes are stable under definition ordering, import ordering,
    and `PYTHONHASHSEED`. The artifact records a contract/schema version and
    package/source provenance; a `SiteCompositionDigest` is never called behavioral
    or live proof.
11. Fake conformance uses no credentials or network. Clean installations of
    both wheel and sdist load every required package resource from an unrelated
    CWD; deliberately omitted or stale resources produce actionable failure.
12. Painted Visibility, a PVPO Encounter, Golden-State Reset, authenticated
    mutation, configured-host reachability, execution, and scoring remain a
    separate later live gate on configured sandbox infrastructure. Boolean
    self-attestation by a readback fixture cannot satisfy that gate.

The existing fake chain and negative cases are evidence that these public seams
can be crossed without a live host
([`test_site_composition.py`](../../tests/test_site_composition.py#L342-L506));
the current deletion/import tests are the starting locality oracle
([`test_composition_package.py`](../../tests/sites/test_composition_package.py#L53-L169)).

## Answers to the nine research questions

1. **Smallest typed Site contract.** The existing immutable
   `SiteDefinition`/`SiteBenchmarkBinding` baseline is close to the minimum,
   but the agreed semantic names are `SiteComposition`,
   `SiteBenchmarkComposition`, and data-only `SiteOwnerDeclaration`. Each of
   the nine edges carries state, stable owner identity, contract version, and
   provenance. It
   must not absorb authentication, browser execution, reset, mutation,
   readback collection, or scoring from their current modules.
2. **One feature-local projection.** Yes for doctor input, explicit diagnostic
   bindings, conformance-test parameters, and generated documentation. No for
   automatic activation or a complete runtime graph. Projection must be a
   deterministic view of the existing owners, not a second registry or a
   mega-adapter. The Classifieds static/live writer mismatch demonstrates why
   runtime catalogs cannot yet be inferred safely.
3. **Declared facts versus executable behavior.** Canonical Site/Benchmark/use-
   case identity, capability state, carrier/action metadata, owner/provenance
   references, contract version, and packaged-resource names can be declared.
   Route matching/reconstruction, profile resolution, editor mutation and
   cleanup, feasibility probes, read-surface construction, readback
   interpretation, reset, browser evidence, and final-state evaluation remain
   executable Python in their existing feature or benchmark owners.
4. **Generic third-Site conformance.** Yes. Parameterize the existing
   fake-forum chain over public seams and fake adapters as
   `synthetic_discussion_forum`, keep its noun out
   of generic Phase code, and add removal/import/package negatives. Passing the
   kit proves fake behavioral conformance, not active policy or a live Site.
5. **Capability-state semantics.** `supported` means the required owner is
   present and passes applicable deterministic conformance; `not_applicable`
   means the exact use case does not require the edge; `unsupported` means the
   known binding intentionally cannot provide a required capability; `missing`
   means an expected declaration, owner, or evidence item is absent or
   incomplete. Only the compiler may derive N/A, and N/A cannot satisfy an
   active required edge.
6. **Static, policy, and live separation.** Keep three ledgers and statuses.
   The `SiteCompositionDigest` covers only canonical static declarations and their
   provenance. Active policy is an explicit run-scoped allow-set. Configured-
   host, admission, execution, visibility, reset, and scoring evidence is
   separately identified and owner-produced. No status is promoted across a
   ledger by import, discovery, or doctor success.
7. **Smallest useful scaffold.** No scaffold yet. First ship and evaluate the
   Site Behavior Contract and actionable `site composition check` discovery. If the clean-room trial
   shows repeated mechanical creation is the bottleneck, a later scaffold may
   generate only a feature package, typed definition, test invocation, and
   documentation—with all behavior/evidence initially missing. It must never
   generate runtime behavior or admission.
8. **Distribution, evolution, digests, Runtime Composition.** Stay in-repo and
   explicitly selected for now. Add a contract/schema version, canonical
   normalized payload, distribution/package version, source provenance, and
   `SiteCompositionDigest`; fail closed on incompatible versions. Treat the
   digest as declaration-only. A future Run Definition should bind the exact
   selected Runtime Composition identity/digest rather than relying only on a
   name, while live evidence retains its own identity. Wheel/sdist resource
   parity and stale-resource negatives are required. Entry points remain
   deferred until independent distribution is real.
9. **Clean-room measurement.** Compare a baseline agent using the current path
   with a proposed-path agent using the kit on the same bounded fake-Site task,
   in fresh worktrees with the same model, reasoning effort, tools, budget, and
   predeclared behavioral oracle. Record prompts, commits, commands, searches,
   failures, and results. One paired trial is feasibility evidence, not a
   causal DevEx claim; repeat before claiming improvement.

## Clean-room agent evaluation plan

Freeze one task brief and oracle containing the acceptance items above. Randomly
assign or counterbalance the baseline/proposed order if more than one pair is
run. Give neither agent the other run’s transcript. Preserve complete command
and patch logs and evaluate with the same tests.

Measure at least:

- time to first valid doctor report;
- time to full fake conformance;
- repository searches and failed iterations;
- files touched and generic Phase files touched;
- explicit registry edits;
- diagnostic recovery quality (whether the next action named by the diagnostic
  fixes the actual missing owner without creating a new safety failure); and
- default-Site preservation after addition and deletion.

Record environment, starting commit, prompt, model/effort, commands, failures,
commits, and final results. Report raw outcomes and oracle disagreements. Do not
convert them to a percentage or time-saving claim without a controlled,
replicated study.

## Open questions for the user before implementation

1. Approve Option B on the constrained existing Option A baseline, or choose a
   different bounded option? The recommendation deliberately excludes a named
   runtime composition and any live host work.
2. Should static compilation consume only immutable projection data, or may it
   call a narrowly typed, audited deterministic-validation hook? The former is
   easier to police; the latter preserves more current closure checks.
3. Should `site doctor` become exact by requiring/accepting carrier and action
   kind, and should its command/exit wording distinguish “static composition
   complete” from “ready”? The current zero exit code follows static status.
4. Should `not_applicable` remain compiler-derived from a closed use-case
   requirement table, or may a Site definition request N/A with a reason that
   the compiler validates? The recommendation is compiler-derived only.
5. Should the Classifieds static writer reference be corrected in the same
   later slice, or explicitly deferred as a separate mismatch? Mixing that fix
   into the generic conformance slice increases scope.
6. What minimum package/source identity belongs in Definition Digest vNext,
   and should Runtime Composition identity binding be a prerequisite or a
   follow-up? Neither should be treated as live evidence.

## Current-state owner and branch inventory

| Concern | Current owner / exact evidence | Site-specific branch or onboarding consequence |
|---|---|---|
| Canonical vocabulary | [`CONTEXT.md`](../../../../CONTEXT.md#L41-L91), [`domain.md`](../../../../docs/agents/domain.md#L1-L36) | Runtime Composition, Site Targeting, Fresh Anonymous Reader, Exact Resource Evidence, and evidence ownership are already defined. No durable new term was found; `CONTEXT.md` should not change. |
| Static Site definition/digest | [`site_composition_contracts.py`](../../warp_taskgen/site_composition_contracts.py#L18-L184), [`site_composition.py`](../../warp_taskgen/site_composition.py#L579-L766) | Nine references are explicit; active policy and live evidence are separate. Static validation nevertheless calls owner methods despite a no-execution docstring. |
| Site Targeting | [`sites/catalog.py`](../../warp_taskgen/sites/catalog.py#L55-L224), default catalog [`sites/catalog.py`](../../warp_taskgen/sites/catalog.py#L327-L339) | Default catalog is GitLab/Reddit only. Identity and route grammar do not prove auth, reachability, visibility, mutation, admission, or scoring. |
| Editor registration | [`editors/_method_spec.py`](../../warp_taskgen/editors/_method_spec.py#L1-L142), registry [`editors/_registry.py`](../../warp_taskgen/editors/_registry.py#L40-L278), defaults [`editors/__init__.py`](../../warp_taskgen/editors/__init__.py#L8-L20) | Default registry imports GitLab/Reddit only. It is metadata, not a complete Site interface. |
| Seeding and cleanup | [`seeding/site_contracts.py`](../../warp_taskgen/seeding/site_contracts.py#L81-L322), [`seeding/_impl.py`](../../warp_taskgen/seeding/_impl.py#L191-L375) | Explicit per-run registry and strict cleanup exist; legacy validation still has a GitLab selector branch in [`seeding/validation.py`](../../warp_taskgen/seeding/validation.py#L120-L131). |
| Phase 1 route/action generation | [`phase_1_route_contracts.py`](../../warp_taskgen/phase_1/phase_1_route_contracts.py#L51-L135), [`phase_1_contract_bound_action_api.py`](../../warp_taskgen/phase_1/phase_1_contract_bound_action_api.py#L676-L681) | Current fallbacks/wording contain GitLab/Reddit assumptions. A conformance-only Site must not add another name here. |
| Phase 2c feasibility | [`phase_2/phase_2c/policy.py`](../../warp_taskgen/phase_2/phase_2c/policy.py#L115-L210) | Immutable catalog defaults to GitLab/Reddit; Classifieds has a feature-local policy. Registration is independent of Site Targeting. |
| Exposure and rendered visibility | [`exposure_contract/_impl.py`](../../warp_taskgen/phase_2/exposure_contract/_impl.py#L250-L302), [`exposure_contract/_impl.py`](../../warp_taskgen/phase_2/exposure_contract/_impl.py#L537-L585), [`phase_2_render_check.py`](../../warp_taskgen/phases/phase_2_render_check.py#L1739-L2001) | Ordered-child/effective-mode/render paths retain explicit GitLab/Reddit branches. Static composition cannot project them away or claim Painted Visibility. |
| Core surfaces | [`phase_2_core_surfaces.py`](../../warp_taskgen/phases/phase_2_core_surfaces.py#L21-L51), [`phase_2_core_surfaces.py`](../../warp_taskgen/phases/phase_2_core_surfaces.py#L109-L181) | Active core allowlist is GitLab/Reddit only and must remain unchanged. |
| Read surface/readback | [`sites/read_surface.py`](../../warp_taskgen/sites/read_surface.py#L16-L221), [`sites/readback.py`](../../warp_taskgen/sites/readback.py#L32-L216) | Same-origin planning and interpretation are typed. The current observation payload does not itself prove reader-context identity or a browser encounter. |
| Final-state evaluation | [`rewards/final_state_catalog.py`](../../warp_taskgen/rewards/final_state_catalog.py#L104-L214) | Reward-local catalog defaults to GitLab/Reddit and enforces evaluator/Benchmark identity. Diagnostic N/A must not create scoring authority. |
| Action cards | [`adversarial_actions/capability_adapters.py`](../../warp_taskgen/adversarial_actions/capability_adapters.py#L36-L137), [`adversarial_actions/catalog.py`](../../warp_taskgen/adversarial_actions/catalog.py#L33-L98) | Typed adapter/catalog remain separate owners; a definition references them but must not manufacture executable behavior. |
| Benchmark admission | [`benchmark_capabilities.py`](../../warp_taskgen/benchmark_capabilities.py#L51-L114), [`benchmark_contracts.py`](../../warp_taskgen/benchmark_contracts.py#L195-L261) | Comparison-only WASP/ST-WebAgentBench/DoomArena cannot enter WARP generation, execution, evaluation, or scoring. |
| Phase 4 safety gates | [`exposure_admission.py`](../../warp_taskgen/phase_4/exposure_admission.py#L13-L88), [`admission.py`](../../warp_taskgen/phase_4/admission.py#L255-L318) | Exposure eligibility and strict feasibility remain live admission owners, outside static composition. |
| Runtime Composition | [`runtime_composition.py`](../../warp_taskgen/runtime_composition.py#L45-L118), Run Definition [`run_definition.py`](../../warp_taskgen/run_definition.py#L53-L66) | Current object carries Site/seed/feasibility/reader-preflight/cleanup, not the complete nine-edge graph; Run Definition stores its name, not a digest. Classifieds remains explicit opt-in. |
| CLI diagnostics | [`site_doctor.py`](../../warp_taskgen/cli/site_doctor.py#L40-L86), [`site_doctor.py`](../../warp_taskgen/cli/site_doctor.py#L153-L160) | CLI omits library-supported carrier/action kind and exits from static status, so its green result is broad structural closure—not readiness. |
| Packaging/deletion | [`pyproject.toml`](../../pyproject.toml#L67-L79), [`test_composition_package.py`](../../tests/sites/test_composition_package.py#L35-L234) | Wheel/sdist include the Python package; positive clean-CWD and deletion checks exist. Non-Python resources will need explicit omission/staleness tests. |

## Research-pass validation

- Focused static/package/CLI/Classifieds slices passed: `52 passed` across
  `tests/test_site_composition.py`, `tests/sites/test_composition_package.py`,
  `tests/test_cli_site_doctor.py`, and `tests/test_classifieds_site_doctor.py`.
- The installed CLI help exposes Site, Benchmark, and use case, but not carrier
  or action kind. A Classifieds `ugc_reply` diagnostic produced static complete
  plus overall blocked; the intuitive `phase_4_execution` request remained
  statically incomplete because final-state ownership is missing.
- These are static/fake/package observations only. No credential was loaded, no
  Benchmark Host or browser was launched, and no reset, mutation, visibility,
  encounter, execution, or scoring claim was tested.
- `git diff --check` passed. The root worktree’s user-owned untracked prior
  draft was read but not overwritten, moved, staged, or deleted.

## User-reviewed decision and bounded implementation evidence

### Decision

After reviewing and grilling the options, the user selected Option B on the
constrained Option A baseline: a static, data-only Site Composition plus a
behavior-level test kit. The selected vocabulary is deliberately semantic:

- `SiteComposition`, `SiteBenchmarkComposition`, `SiteOwnerDeclaration`,
  `SiteCompositionCheckRequest`, and `SiteCompositionCheckReport` describe the
  static declaration/check boundary;
- `site_composition_digest` is an algorithm-qualified `sha256:<hex>` digest and
  remains distinct from the Run Definition's historical `definition_digest`;
- `public_reply` replaces the diagnostic-only `ugc_reply` identifier;
- `site composition check` is the canonical static CLI, while `site doctor`
  remains only a thin parser-level CLI alias with no Python compatibility
  module or API; and
- **Site Behavior Contract** names the executable fake-adapter assertions. The
  test-only Site is **`synthetic_discussion_forum`**, with parent `thread`,
  created child `comment`, carrier `comment.body`, writer method
  `create_comment`, and action kind `submit_comment`.

The user also selected compiler-derived `not_applicable`, exact carrier/action
input for `public_reply`, a direct same-PR Python-name cutover with no old
Python aliases, per-Site declaration modules, and no runtime activation or live
canary in this slice.

### Implemented boundary

The implementation keeps each built-in data-only declaration in
`warp_taskgen/site_compositions/<site>.py`; the default module only aggregates
GitLab, Reddit, and the explicit Classifieds diagnostic. Static checks read
immutable declarations and Benchmark metadata only. They do not import or call
Site, editor, seed, feasibility, readback, evaluator, browser, reset, cleanup,
admission, or scoring behavior.

`SiteOwnerDeclaration` may declare `supported`, `unsupported`, or `missing`.
Only the Host-Owned `SiteCompositionUseCaseCatalog` may derive
`not_applicable`. A complete report requires a supported registration, no
failed findings, and a supported `static_closure`; invalid reports redact
untrusted request identities and carry no digest. Comparison-only Benchmarks
still fail closed before any WARP use case is checked.

The accepted owner-contract version is explicit and incompatible versions fail
at construction. The canonical digest payload and report bind the source
package, package version, and symbolic source provenance. Static completion
still records operational readiness as `blocked` because active policy and live
evidence are not checked by this seam.

Executable fake behavior remains modular by owner under
`tests/sites/behavior_contract/`. The feature-local
`tests/sites/synthetic_discussion_forum/` slice crosses Site Targeting, Regular
Participant Writer through an immutable per-run seed registry and idempotent
cleanup, feasibility derived from the exact parent route, a separately built
anonymous-reader observation, Exact Resource Evidence, final-state evaluation,
and action-card route closure. The exact request is feature-local, editor
metadata declares all required arguments, and a body digest binds writer,
read-surface, readback, and final-state negative evidence. These tests do not
claim browser-context
identity, Painted Visibility, a PVPO Encounter, Golden-State Reset, configured
host evidence, admission, execution, or scoring.

Classifieds executable tests now obtain Site, seed, feasibility, authenticated
writer, reader-preflight, and strict-cleanup owners from the explicit
`classifieds_listing_reply_poc` Runtime Composition. Static Classifieds
declarations no longer carry or imply an executable writer object.

### Verification evidence

- The focused semantic slice passed **88 tests** across static composition,
  canonical and alias CLI behavior, Classifieds static diagnostics and fake
  runtime behavior, Runtime Composition, package boundaries, the Site Behavior
  Contract, onboarding backpressure, and acceptance-wrapper checks.
- `ruff check` and `ruff format --check` passed for all changed Python paths.
- `verify_fast.sh` passed scoped lint, full pytest collection, and the readiness
  audit after the topic worktree was populated from the lockfile.
- `verify_default.sh` passed the repository's default parallel pytest suite.
- `accept_taskgen.sh` passed locked sync, default verification, clean wheel and
  sdist builds/installs, canonical `site composition check` smoke tests for
  GitLab, Reddit, and Classifieds, installed resource checks, sidecar packaging,
  and the namespace-upgrade proof.

No credential was loaded, no network-backed Site or browser was used, and no
live Benchmark Host, mutation, reset, admission, execution, visibility, or
scoring gate was run. The clean-room comparison below reports raw onboarding
outcomes only; it is not evidence for a percentage or time-saving claim.

### Clean-room comparison results

Two independent Luna Max agents received the same bounded task in isolated git
snapshots: add `synthetic_project_discussion_forum` with thread `23`, comment
`55`, carrier `comment.body`, writer `create_comment`, action
`submit_comment`, exact static diagnostics, and the seven fake behavior-owner
checks. Neither agent could read the other snapshot. Both were prohibited from
using a network, credential, browser, live host, default activation, generic
Phase edits, or process-wide registry edits.

| Raw measure | Baseline at `27894b6e` | Proposed path |
|---|---:|---:|
| Recorded wall-clock trial interval | 10m07s | 15m42s |
| `rg` repository-search invocations | 14 | 6 |
| Files changed, including trial log | 3 | 11 |
| Generic Phase files changed | 0 | 0 |
| Explicit registry/default edits | 0 | 0 |
| Targeted fake behavior result | 12 passed | 10 passed |
| Broader recorded result | 55 relevant tests passed; collection succeeded | 78 focused tests and 175 Site tests passed |
| Isolated trial commit | `461017e` | `ca4a113`; metrics follow-up `8e7e05e` |

The file counts reflect different organization: the baseline agent created one
805-line test module plus a JSON report fixture, while the proposed agent used
ten feature-local modules and the existing plain Behavior Contract assertions.
Both preserved GitLab/Reddit defaults, comparison-only rejection, and
fail-closed unknown/foreign/malformed cases.

The recovery logs are also mixed. The baseline agent recorded five failed
iterations: the initial TDD red, a module-identity-dependent digest mismatch,
two helper/command mistakes, and Ruff findings. The proposed agent first edited
the existing synthetic fixture instead of adding a separate Site; that attempt
produced one collection failure and two failing test rounds before a temporary
green run, and the orchestrator required it to revert and restore feature
locality. Its corrected separate package then passed, after one Ruff import
ordering fix.

The requested milestone timing is not directly comparable. The baseline log
recorded command durations (`0.26s` compiler and `0.33s` targeted pytest), not
elapsed time from trial start to those milestones. The proposed log recorded a
conservative elapsed upper bound of 13m52s to a timestamped static report and
13m46s to the corrected fake suite. The paired run therefore does **not**
support a causal onboarding-improvement or time-saving claim. It does show
that both paths can satisfy the bounded task without generic Phase/default
edits, and that semantic locality still needs explicit backpressure even when
a reusable behavior test path exists.

## Source ledger (primary sources only)

Repository HEADs below were resolved with `git ls-remote` on 2026-08-16; a
branch name alone is not used as reproducibility evidence.

| Source | Current branch/version/date | License/publication status | WARP question answered |
|---|---|---|---|
| [BrowserGym repository](https://github.com/ServiceNow/BrowserGym), [task source](https://raw.githubusercontent.com/ServiceNow/BrowserGym/main/browsergym/core/src/browsergym/core/task.py), [registration source](https://raw.githubusercontent.com/ServiceNow/BrowserGym/main/browsergym/core/src/browsergym/core/registration.py) | `9e779f087de9`; source version `0.14.3` | Apache-2.0; TMLR 2025 publication ([record](https://openreview.net/forum?id=5298fKGmv3)) | Custom task setup/validate/teardown and registration boundary |
| [AgentLab repository](https://github.com/ServiceNow/AgentLab), [study source](https://raw.githubusercontent.com/ServiceNow/AgentLab/main/src/agentlab/experiments/study.py) | `cbc35a9bc0fa`; dynamic VCS version | Apache-2.0 | Experiment/agent extension and reproducibility boundary |
| [Gymnasium registry](https://gymnasium.farama.org/api/registry/), [utils/check_env](https://gymnasium.farama.org/api/utils/) | repository `368cd1cc9aae`; release page lists v1.3.0 ([releases](https://github.com/Farama-Foundation/Gymnasium/releases)) | MIT | Global registry warning and behavior-conformance analogy |
| [PyPA entry points specification](https://packaging.python.org/en/latest/specifications/entry-points/) | Spec page updated 11 Aug 2026 | Python Packaging Authority specification | Explicit discovery/duplicate handling, never default admission |
| [JSON Schema 2020-12](https://json-schema.org/draft/2020-12) | Published 16 Jun 2022 | Specification references only | Manifest vocabulary reference |
| [OpenAPI 3.1.1](https://spec.openapis.org/oas/v3.1.1.html) | 24 Oct 2024 | Apache-2.0 specification | HTTP route vocabulary reference |
| [Kubernetes Gateway API conformance](https://gateway-api.sigs.k8s.io/guides/implementers-guide/), [GEP-1709](https://gateway-api.sigs.k8s.io/geps/gep-1709/) | v1.5.1 (14 Mar 2026) | Apache-2.0; Kubernetes project specification | Versioned profiles, exact reports, supported/unsupported features, no applicable-test skipping |
| [Terraform Plugin Framework schema validation](https://developer.hashicorp.com/terraform/plugin/framework/handling-data/schemas), [acceptance testing](https://developer.hashicorp.com/terraform/plugin/testing/acceptance-tests) | Official pages route to Framework v1.18.x and Plugin Testing v1.15.x at cutoff | Framework source MPL-2.0; HashiCorp developer documentation | Offline diagnostics versus real-API acceptance boundary |
| [Backstage Software Templates](https://backstage.io/docs/features/software-templates/adding-templates/), [dry-run testing](https://backstage.io/docs/features/software-templates/dry-run-testing/) | Current official documentation | Backstage project documentation | Thin generation and explicit dry-run capability |
| [pytest plugin discovery](https://docs.pytest.org/en/latest/how-to/writing_plugins.html), [pluggy API](https://pluggy.readthedocs.io/en/stable/api_reference.html) | Current official documentation | pytest/pluggy project documentation | Entry-point autoload/debug cost and hook validation mechanics |
| [OpenTelemetry versioning and stability](https://opentelemetry.io/docs/specs/otel/versioning-and-stability/), [semantic-convention groups](https://opentelemetry.io/docs/specs/semconv/general/semantic-convention-groups/) | Current normative specification pages | OpenTelemetry specification | Stable/experimental convention evolution and explicit versioning |
| [OCI Image Format descriptor](https://github.com/opencontainers/image-spec/blob/v1.1.1/descriptor.md), [release](https://github.com/opencontainers/image-spec/releases/tag/v1.1.1) | v1.1.1 (3 Mar 2025) | Apache-2.0 | Content-addressed digest verification; bounded analogy for `SiteCompositionDigest` |
| [pytest assertions](https://docs.pytest.org/en/stable/how-to/assert.html), [pytester](https://docs.pytest.org/en/stable/how-to/writing_plugins.html#testing-plugins) | Current official docs | pytest project documentation | Plain observable assertions and isolated test projects |
| [WebArena repository](https://github.com/web-arena-x/webarena), [reset docs](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md) | `dce04686a562`; stable dataset v0.2.0 (24 Oct 2023) | Apache-2.0 | Environment reset ownership |
| [WebArena Verified repository](https://github.com/ServiceNow/webarena-verified), [env docs](https://servicenow.github.io/webarena-verified/dev/environments/) | `6473f72db5dc`; public/PyPI and SEA Workshop dates in README (2025–2026) | Apache-2.0; 2025 SEA Workshop paper ([record](https://openreview.net/forum?id=94tlGxmqkN)) | Env-ctrl reset and deterministic evaluator ownership |
| [VisualWebArena repository](https://github.com/web-arena-x/visualwebarena), [evaluator source](https://raw.githubusercontent.com/web-arena-x/visualwebarena/main/evaluation_harness/evaluators.py) | `89f5af29305c`; released 25 Jan 2024 | MIT | Evaluator router and service reset ownership |
| [WASP repository](https://github.com/facebookresearch/wasp), [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1c9818387f5dd0a0bc151214660f059d-Abstract-Datasets_and_Benchmarks_Track.html) | `ffee6f41fde7`; archived/read-only 1 Jul 2026; NeurIPS 2025 | Predominantly CC-BY-NC 4.0; peer-reviewed Datasets & Benchmarks Track; [arXiv preprint](https://arxiv.org/abs/2504.18575) | Isolated execution and cleanup model; publication status |
| [ST-WebAgentBench](https://github.com/segev-shlomov/ST-WebAgentBench) | `67f56dd7df9e`; README says accepted ICLR 2026 | Apache-2.0; linked arXiv remains a preprint ([arXiv](https://arxiv.org/abs/2410.06703)) | Policy/evaluator separation |
| [WebCanvas](https://github.com/imeanai/webcanvas) | `b9f289128614`; event-listener evaluation in v0.0.4 (26 Dec 2024) | MIT; paper is arXiv preprint ([arXiv](https://arxiv.org/abs/2406.12373)) | Decoupled evidence/evaluator design |
| [OSWorld](https://github.com/xlang-ai/OSWorld) | `091f5ef1d554`; OSWorld-Verified update 28 Jul 2025 | Apache-2.0; paper is arXiv preprint ([arXiv](https://arxiv.org/abs/2404.07972)) | Per-task setup and custom evaluator ownership |

A prior unpublished DevEx draft supplied as read-only worktree input was read;
its proposals are treated as historical input, not as primary evidence for
the claims above. The repository guides, ADRs, technical specification,
architecture note, source modules, and tests cited in this note are the current
WARP evidence set.
