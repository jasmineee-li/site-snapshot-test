---
name: phase-4-gates
description: Model-invoked router for WARP Taskgen Phase 4 contracts. Use for PVPO or max_coverage, encounter and placement-fix routing, Needham XML, Transcript Purpose, VEA, eval-awareness iteration, reward/readback scoring, refusal-judge compatibility, trajectory artifacts, or Phase 4 verification. Use benchmark-host-ops for host lifecycle, topology, proxy, and remote-job operations.
---

# Phase 4 contracts

Use the canonical contract and verification docs; this skill contains routing only.

- Read `agent_docs/phase4-contracts.md` for PVPO, Gate 1/2, TP/VEA schemas, iterator and legacy strategy semantics, direct host-API routing, and artifact contracts.
- Read `agent_docs/verification.md` for focused tests, live gates, result audits, and required evidence.
- Read `agent_docs/trace-inspection.md` when diagnosing a trajectory, result, iterator envelope, or stop reason.
- Inspect `worldsim/phase_4/` and its tests when a contract or schema is unclear; the technical spec is the package authority.

Current routing facts:

- PVPO uses runner-owned page-surface-stable capture; there is no active dedicated PVPO browser endpoint or container.
- `max_coverage == 0` records `injection_not_encountered` and routes placement-fix; TP and VEA still run for every extractable trajectory, including zero coverage.
- Default attack effectiveness is reward/readback grounded. TP is observational except that it controls bounded default iterator continuation; VEA is report-only.
- The refusal judge and bounded `3+3+1` variant loop are legacy opt-in strategy-variation behavior.

Completion check: state the applicable contract, evidence path, and final-status/branch rule before changing code or interpreting a run.
