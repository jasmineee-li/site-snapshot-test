---
status: accepted
---

# Cut over Site Targeting as a deep module

## Decision

Migrate site-varying behavior one coherent capability at a time across every
active Site, beginning with a bounded, pure Site Targeting slice. An explicit
fail-closed `SiteCatalog` binds a normalized `TargetingContext`—benchmark,
profile projection, Site identity, and an explicit origin or placeholder—to a
Site-owned targeting module. Callers obtain a bound Site and use only its
`routes()` and `resolve()` operations. GitLab and Reddit feature modules own
their route grammar, local Resource Kinds, anchor extraction, and canonical
route descriptors. Catalog definition defects raise during construction;
malformed, foreign, or unsupported task evidence produces a structured
`TargetingFailure` rather than a guessed host or route.

Site Targeting maps deterministic task evidence to a `ResolvedTarget`. It does
not prove authentication, browser reachability, visibility, admission,
mutation, or scoring.

## ST-1 scope (current slice)

The current slice owns only the following contracts:

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

## Deferred slices

The following remain separate capability migrations: L3 intent resolution, L4
listing/detail transitions and listing probes, authentication and browser
reachability, editors and mutation/seeding, exposure and admission policy,
visibility/readback, and reward/scoring. Legacy facade and import deletion is
also deferred until the callers and compatibility surfaces for the relevant
capability have been audited and cut over.

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
