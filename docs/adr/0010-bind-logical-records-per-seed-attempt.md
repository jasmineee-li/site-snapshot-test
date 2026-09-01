---
status: accepted
---

# Bind logical records per seed attempt

Generated multi-record workflows identify each declared record with a stable,
family-owned key, while physical resource IDs remain local to one seed attempt.
Seeding validates and echoes the key and normalized Benchmark in per-call result
metadata; the concrete workflow feature binds its expected records against that
attempt's exact Site, method, Resource Kind and safe identity evidence. Phase 2c
uses its binding only for admission evidence, and Phase 4 rebinds after reset
before browser execution; only the fresh Phase 4 binding may supply action and
reward anchors. Missing, duplicate, stale or mismatched evidence creates
intentional fail-closed backpressure instead of falling back to aggregate seed
metadata.

## Considered options

- Use aggregate seed tokens or the last-created resource. Rejected because the
  local proof reproduced a wrong-target binding whenever the selected record was
  not the last call.
- Treat positional `call_index` as the durable record identity. Rejected because
  inserting or reordering a setup call could silently retarget a logical record;
  the index remains useful only for within-attempt correlation.
- Require fixed physical fixture IDs. Rejected because IDs may change after
  reset/reseed and coupling tasks to one mutable world would weaken isolation.

## Consequences

The shared seeding interface gains only the declaration/result fields needed to
preserve stable identity. The first strict binder stays in the GitLab comparison
feature; no universal workflow registry or binder is introduced before a second
consumer demonstrates the same seam. Stable binding specifications participate
in existing resume identity, while ephemeral IDs stay in existing attempt
evidence rather than a new manifest. Existing workflows keep their current
compatibility behavior; aggregate fallback is forbidden only for features that
declare this selected-record contract.
