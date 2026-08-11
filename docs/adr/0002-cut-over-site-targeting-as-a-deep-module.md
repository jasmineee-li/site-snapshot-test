---
status: accepted
---

# Cut over Site Targeting as a deep module

## Decision

Migrate site-varying behavior one coherent capability at a time across every
active Site, beginning with a bounded, pure Site Targeting slice. An explicit
fail-closed `SiteCatalog` binds a normalized `TargetingContext`—benchmark,
profile projection, Site identity, and an explicit origin or placeholder—to a
Site-owned targeting module. Callers obtain a bound Site and use its small,
explicit route and candidate operations instead of reaching into Site
adapters. GitLab and Reddit feature modules own their route grammar, local
Resource Kinds, anchor extraction, and canonical route descriptors. Catalog
definition defects raise during construction;
malformed, foreign, or unsupported task evidence produces a structured
`TargetingFailure` rather than a guessed host or route.

Site Targeting maps deterministic task evidence to a `ResolvedTarget`. It does
not prove authentication, browser reachability, visibility, admission,
mutation, or scoring.

## ST-1 scope (deterministic evidence)

The first slice owns only the following contracts:

- the explicit catalog and immutable targeting context;
- GitLab and Reddit feature-local route descriptors and local-to-compatibility
  Resource Kind mapping;
- deterministic L1 NetworkEvent URL matching and L2 start-URL matching and
  reconstruction; and
- Phase 1 start-pattern generation delegated to the Site route descriptors.

Existing Phase 2 compatibility facades remain available and patchable while
their migrated L1/L2 paths delegate to this seam. A narrow listing-intent
compatibility call may consume GitLab route grammar for that path, but ST-1
does not claim ownership of listing probing or transition policy.

## ST-2 scope (L3 candidate materialization)

The L3 slice extends the bound Site seam with a small, typed candidate
operation. Phase 2 retains intent classification, HTTP/auth probe execution,
retry and concurrency controls, admission, encounter requirements, and
transition policy. Site Targeting owns only the deterministic parts of the
handoff:

- `TargetCandidate` accepts local or legacy Resource Kinds, probe metadata, and
  validated anchors without importing Phase 2 policy;
- `validate_probe()` rejects a Site-incoherent API/Resource Kind pair before a
  probe runs; and
- `materialize()` maps the candidate to a canonical route, reconstructs a URL
  on the bound origin, and fails closed on unknown kinds, missing anchors,
  foreign origins, or adapter errors. A caller-supplied fallback URL is never
  promoted to a target.

Active Site modules may provide source-listing route facts through the same
candidate seam. The resulting artifact may continue to expose the historical
prefixed compatibility kind; the local canonical kind remains an internal
Site contract. Tests may inject an adapter for this bounded seam, but that
does not bypass the legacy editor compatibility registry. L4 listing
expansion, browser/auth behavior, and editor or exposure contracts remain
outside ST-2.

## ST-3 scope (L4 listing-entry materialization)

The L4 slice moves only deterministic interpretation of an already-fetched
listing row behind the bound Site seam. `ListingItemCandidate` snapshots the
source listing kind, row item kind, raw payload, and optional listing evidence
URL. `materialize_listing_entry()` asks the Site adapter to validate the
source/item-kind pair and project raw row anchors, then reuses the strict
same-origin route reconstruction used by L3. It fails closed on unknown
routes, malformed rows, missing anchors, adapter errors, and relative or
foreign reconstructed URLs; evidence URLs are never promoted as fallbacks.

Phase 2 retains listing HTTP/auth/visibility probes, item ordering and
`top_n`, empty-list omission, probe-error records, concurrency, encounter and
viewport requirements, title/DOM evidence, attach-surface composition, and
the historical prefixed artifact kinds. Reddit forum expansion, editor and
admission contracts, exposure/readback, and reward behavior remain outside
this slice. `_project_item_to_record()` and the anchor reconstruction helpers
remain thin compatibility facades for one migration cycle.

## Deferred slices

The following remain separate capability migrations: L3 classifier and probe
execution policy, L4 listing/detail transitions and listing probes,
authentication and browser reachability, editors and mutation/seeding,
exposure and admission policy, visibility/readback, and reward/scoring. Legacy
facade and import deletion is also deferred until the callers and
compatibility surfaces for the relevant capability have been audited and cut
over.

## ST-4 scope (profile identity and Phase 1 route facts)

The fourth bounded slice moves only deterministic profile/surface identity and
inventory-backed Phase 1 route facts behind the bound Site. GitLab and
Reddit/Postmill feature modules implement a typed profile-route capability;
the capability exposes canonical/profile surface resolution plus route facts
for start URL patterns, anchor examples, inventory-backed-start requirements,
and route variants. A test-only adapter can be injected through the same
capability without adding a named branch to generic Phase code.

`BoundSite` binds the profile projection once and provides these operations to
Phase 1. The legacy `worldsim.surface_identity` facade and private
`phase_1_route_contracts` profile/anchor helpers remain thin delegates for one
migration cycle. The Phase 1 artifact schema, route ordering, IDs, and digest
remain unchanged because editor registration, core/active-carrier policy,
Phase 2 admissibility/exposure, instruction/editor argument templates,
answer-stability guidance, auth/browser behavior, admission, and reward policy
stay in their existing owners.

ST-4 does not move editor eligibility, exposure or admission policy, seed or
browser behavior, visibility/readback, authentication, or reward/scoring. A
Site that does not implement the profile-route capability fails closed for
profile resolution and contributes no route facts; generic callers must not
guess aliases, hosts, or inventory anchors.

## Final cutover and deletion criteria

Site Targeting is fully cut over only when all intended callers use the bound
Site interface, the superseded site-specific branches and facades for the
capability are deleted, and both active Sites pass the same conformance
contract. A test-only Site must pass that contract without adding a named
branch to generic Phase code. The deletion test must show that removing the
test Site leaves the active Sites passing, makes the removed Site fail closed,
and leaves no production site branch or fallback that depended on its module.

This capability-shaped sequencing avoids both a risky whole-Site rewrite and a
lasting dual architecture while keeping each intermediate slice honest about
what it does not yet own.
