# Add a Site

Use this guide to add or remove a WARP Site, define its Site Composition, or
apply the Site Behavior Contract. This procedure ends after static closure,
fake behavior, deletion, and package proof. Operational readiness remains
`blocked` because active policy and live evidence are separate gates.

The [technical specification](../docs/warp-taskgen-technical-spec.md) owns the
Site Composition contract. The [code-organization guide](./code-organization.md)
owns module boundaries. The [verification guide](./verification.md) owns the
current test commands. Use the [admission guide](./admission-and-exposure.md)
only after this offline slice passes and the requested work includes active or
live admission.

The test-only
[`synthetic_discussion_forum`](../tests/sites/synthetic_discussion_forum/)
shows one complete feature-local Site. The shared
[`behavior_contract`](../tests/sites/behavior_contract/) assertions show the
observable behavior for each owner. Reuse their public seams and negative
cases. Give the new Site its own semantic Resource Kinds, Canonical Routes,
carrier, action kind, and evidence identities.

## Exact contract

Write these values before editing code:

| Field | Required decision |
| --- | --- |
| Benchmark | One canonical Benchmark with the required Benchmark Capability |
| Site | One semantic Site name |
| Use case | One Host-Owned use case from the Site Composition catalog |
| Carrier | One exact canonical carrier when the use case requires it |
| Action kind | One exact action kind when the use case requires it |
| Parent | One Resource Kind, Canonical Route, and validated anchor set |
| Created resource | One Resource Kind and exact identity fields |
| Writer | One Regular Participant Writer method and cleanup owner |
| Reader | One Fresh Anonymous Reader observation contract |
| Scope | `test_only`, `static_diagnostic`, or separately approved live work |

The tuple `(Benchmark, Site, use case, carrier, action kind)` is the thread
through every work unit. Keep it in one feature-local source and derive test
inputs from it.

## Ordered work units

### 1. Establish the feature boundary

Choose the Site name, parent and child Resource Kinds, Canonical Route anchors,
carrier, writer method, and action kind. Resolve the Benchmark through the
Benchmark Capability catalog. A Comparison-only Benchmark stays outside WARP
generation, execution, evaluation, and scoring.

For a test-only Site, create `tests/sites/<site>/`. For a built-in static
diagnostic, create `warp_taskgen/site_compositions/<site>.py`. Keep executable
behavior with its existing owner: Site Targeting, seeding, feasibility,
readback, rewards, or action cards.

**Done when:** one feature-local source records the exact tuple and resource
identities, and no generic Phase module contains the new Site name.

### 2. Start with a data-only Site Composition

Create one `SiteComposition` and one `SiteBenchmarkComposition`. Declare all
nine owner roles:

1. `site_targeting`
2. `profile`
3. `editor_specification`
4. `regular_participant_writer`
5. `feasibility`
6. `read_surface`
7. `readback`
8. `final_state_evaluation`
9. `action_cards`

Use `missing` when an expected owner does not exist. Use `unsupported` when the
Site intentionally cannot provide a required behavior. Use `supported` only
after the matching fake behavior check passes. The Host-Owned use-case catalog
alone derives `not_applicable`.

The declaration contains semantic owner IDs, contract version `v1`, source
package identity, and non-secret symbolic provenance. It contains no
executable owner object. A built-in diagnostic is added explicitly to
`site_composition_defaults.py`; this makes the static CLI discover it and does
not change the active Site cohort.

Check the current parser before using it:

```sh
uv run warp-taskgen site composition check --help
```

For a built-in diagnostic, check the exact tuple:

```sh
uv run warp-taskgen site composition check <site> --benchmark <benchmark> --use-case <use-case> --carrier <carrier> --action-kind <action-kind> --json
```

For a test-only Site, call `check_site_composition` from its feature-local test
with the same exact request.

**Done when:** the check names the first missing or unsupported owner, carries
no digest for an invalid request, and reports operational readiness `blocked`.

### 3. Prove Site Targeting and profile resolution

Implement deterministic route matching and reconstruction for the parent
Resource Kind. Bind only the declared Site, Benchmark, same-origin Canonical
Route, and validated anchors. Resolve the profile carrier to the exact editor
method. Reject a foreign origin, wrong Site, malformed parent, wrong Resource
Kind, wrong method, and unknown carrier.

**Done when:** the Site Targeting behavior assertion resolves the exact parent
and every near-match fails closed.

### 4. Prove the Regular Participant Writer and seeding

Bind the writer through an immutable per-Run `SeedSiteRegistry`. Declare the
editor method's HTTP route, argument bindings, required arguments, Resource
Kind, and carrier. Create the child through the ordinary participant path.
Return typed write evidence for the parent ID, child ID, actor, body identity,
read surface, and editor method. Make cleanup explicit and idempotent.

The writer path rejects an admin actor and does not mutate a process-wide
editor registry.

**Done when:** the writer behavior assertion creates the exact child, returns
secret-free evidence, rejects the privileged actor, and two cleanup calls leave
the fixture clean.

### 5. Prove feasibility

Derive probe targets from the exact task evidence and Canonical Route. Keep
authentication requirements explicit. Classify supported and unsupported
probe results without selecting another Site, route, or parent as a fallback.

**Done when:** the feasibility behavior assertion admits the exact parent and
rejects a wrong Site, foreign origin, malformed route, and failed probe.

### 6. Prove Fresh Anonymous Reader and Exact Resource Evidence

Build the read-surface plan from the actual writer result. Preserve the parent
ID, child ID, actor, body identity, same-origin route, and signature. Construct
the fake Fresh Anonymous Reader observation separately from the writer. Reject
writer cookies, writer storage state, reused authentication, foreign routes,
wrong IDs, wrong actor, altered body, ambiguous matches, and missing
visibility.

This is fake readback evidence. Painted Visibility and a PVPO Encounter remain
live evidence.

**Done when:** readback verifies only the exact created child, all near-matches
fail closed, and cleanup is tied to the same writer result.

### 7. Prove final-state and action behavior

Implement final-state evaluation when the exact use case requires it. Bind the
Benchmark, Site, action kind, source event, parent, child, actor, and body
identity. Define the action card with the same Benchmark, Site, carrier,
Canonical Route, and action kind. Keep evaluator authority in the Benchmark
and reward owners.

When the use case does not require final-state evaluation, leave the Site
declaration `unsupported` or `missing` as accurate. Let the Host-Owned use-case
catalog derive `not_applicable` in the report.

**Done when:** required final state and action-card assertions pass, stale or
foreign evidence fails, and Comparison-only Benchmarks remain comparison-only.

### 8. Close static composition

Change an owner declaration to `supported` only after its fake behavior check
passes. Run the exact Site Composition request again. Treat the Site
Composition digest as declaration identity, not behavior or live proof.

**Done when:** every required owner finding passes, `static_status` is
`complete`, the digest starts with `sha256:`, and operational readiness remains
`blocked` with active policy and live evidence `not_checked`.

### 9. Apply deletion and package backpressure

Run the feature-local Site Behavior Contract, Site Composition tests, and
package-boundary tests. Prove these outcomes:

- unknown, duplicate, malformed, and removed Sites fail closed;
- deleting the Site leaves GitLab, Reddit, and the explicit Classifieds path
  unchanged;
- importing static declarations does not mutate active catalogs;
- the Site name stays out of generic Phase code;
- wheel and sdist installs contain every required static declaration;
- missing or stale packaged declarations fail with an actionable result.

Use the [verification guide](./verification.md) to select the current focused
and shipping commands.

**Done when:** focused checks and package acceptance pass from a clean state,
removing the Site removes its behavior and declaration, and no default or
generic Phase behavior changes.

## Diagnostic recovery

Use the report state to choose the next action:

| Result | Next action |
| --- | --- |
| `invalid` | Fix Site, Benchmark, use-case, duplicate, package, or version identity |
| `incomplete` | Implement or correct the exact owner named by the failed finding |
| `complete` and readiness `blocked` | Record static completion and stop this offline slice |
| Behavior assertion failure | Fix the feature-local owner that produced the observation |
| Package failure | Restore the named declaration resource or package version |

## Handoff boundary

Record the exact tuple, changed owner modules, Site Composition digest, focused
test results, package result, and remaining readiness blockers. Static and fake
completion do not authorize credentials, a Benchmark Host, a Benchmark
Instance mutation, active policy, admission, execution, scoring, Painted
Visibility, a PVPO Encounter, or Golden-State Reset.

When the approved task includes those gates, continue through the
[admission guide](./admission-and-exposure.md), the owning live procedure, and
the configured sandbox infrastructure. Otherwise, finish with operational
readiness `blocked`.

## Source-of-truth map

- Domain terms: [WARP domain model](../../../CONTEXT.md)
- Static contract: [technical specification](../docs/warp-taskgen-technical-spec.md)
- Module ownership: [code organization](./code-organization.md)
- Local checks and live-gate selection: [verification](./verification.md)
- Active and live admission: [admission and exposure](./admission-and-exposure.md)
- Complete fake example: [`synthetic_discussion_forum`](../tests/sites/synthetic_discussion_forum/)
- Reusable assertions: [Site Behavior Contract](../tests/sites/behavior_contract/)
