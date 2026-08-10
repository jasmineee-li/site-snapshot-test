# Domain Invariants Index

Use this index when a change touches generation, admission, Phase 4, prompts,
rewards, authentication, or sandbox execution. Read exactly the branch file
that owns the behavior:

- **Admission/exposure:** `admission-and-exposure.md` — Phase 1/2 contracts,
  Phase 2c strict admission, carrier reachability, and scope.
- **Actions:** `action-contracts.md` — host-owned action capability contracts,
  Tier 2/3 readiness, reward/readback evidence, and variant anchors.
- **Phase 4:** `phase4-contracts.md` — PVPO gates, TP/VEA observation,
  eval-awareness iteration, ASR accounting, and immutable contracts.
- **Runtime:** `runtime-boundaries.md` — auth precedence, host API versus Modal,
  sandbox file inclusion, AgentLab references, and surface identity.

Start with `docs/warp-taskgen-technical-spec.md`; it is the source of truth.
Dataset counts are run state, not invariants. The active cohort comes from the
pinned run directory and its Phase 2/3 manifests.

Current admitted surfaces are GitLab issues/comments and Reddit/Postmill
posts/comments. Other benchmark paths are historical or support plumbing until
the spec and an explicit task bring them back.

Completion means the relevant branch file was read, every changed contract is
accounted for, and any spec drift is resolved before implementation proceeds.
