# WARP Taskgen Agent Guide

WARP Taskgen generates and evaluates indirect-prompt-injection tasks in
sandboxed WebArena-style environments. Keep work inside repository code,
generated traces, and configured benchmark infrastructure. The admitted WASP
surfaces are GitLab issues/comments and Reddit/Postmill posts/comments.

The technical spec is authoritative: `docs/warp-taskgen-technical-spec.md`.
When code, a handoff, and the spec disagree, identify the drift and align the
implementation with the spec before changing adjacent behavior.

## First commands

- Sync (from `packages/warp-taskgen/`): `uv sync --extra dev --locked`
- One focused test (from `packages/warp-taskgen/`): `uv run pytest -x -q tests/<file>.py`
- Acceptance lanes (from the repository root): `bash scripts/accept_taskgen.sh --help`
- Ship (from the repository root): `bash scripts/accept_taskgen.sh`

## Route by task

Read the one branch document that matches the work; the index files are maps,
not invitations to preload every reference.

- **GitLab/Reddit Phase 1/2 generation, feasibility, exposure, carriers, or
  admission:**
  read `agent_docs/admission-and-exposure.md`.
- **Experimental Classifieds canary, writer/reader/reset proof, or
  `classifieds_listing_reply_poc`:** read `agent_docs/classifieds-canary.md`
  before acting; read `agent_docs/admission-and-exposure.md` too when changing
  a shared admission contract.
- **Adding or removing a Site, Site Composition, onboarding diagnostic, or
  Site Behavior Contract:** read `agent_docs/site-onboarding.md` before editing.
- **Adversarial actions, Tier 2/3 pilots, reward compilation, readback, or
  action variants:** read `agent_docs/action-contracts.md`.
- **Phase 4, PVPO, Transcript Purpose, VEA, iterator, judges, or ASR:** read
  `agent_docs/phase4-contracts.md` and then the relevant spec section.
- **Auth, Modal sandbox routing, AgentLab boundaries, or carrier identity:**
  read `agent_docs/runtime-boundaries.md`.
- **Fresh hosts, r8a jobs, proxy/locality, remote status, or rigor runs:** read
  `agent_docs/remote-runs.md` and `docs/handoffs/rigor-run-setup.md`.
- **Why a run complied, resisted, was unaware, or stopped iterating:** start at
  `agent_docs/trace-inspection.md` with `uv run warp-taskgen trace ...`.
- **Moving modules or compatibility surfaces:** read
  `agent_docs/code-organization.md` before changing imports.
- **Tests, lint, live gates, or result evidence:** read
  `agent_docs/verification.md`.
- **Generated logs, archives, or fixture promotion:** read
  `agent_docs/artifacts.md` before touching `logs/`.
- **Credentials, instance configs, or proxy tokens:** read
  `agent_docs/secrets.md`.

## Cross-cutting routes

- **Schema, CLI, or result fields:** start at the spec, then use the owning
  branch document; this router is not a second contract.
- **A new invariant:** place it in the owning branch document and add one
  pointer here; do not create parallel copies in neighboring guides.
- **A handoff or review:** link the exact run or evidence path rather than
  pasting generated output into a guide.
- **Unclear ownership:** stop at the package boundary before editing `AgentLab/`
  or generated logs. `AGENTS.md` is the cross-agent entrypoint and stays a
  symlink to this router.

## Working loop

Research the relevant spec and branch document, plan the smallest scoped
change, implement it in the canonical package source, and validate it with the
narrowest meaningful checks before broadening. The repository root owns the
topic-worktree and acceptance workflow.

Keep generated runtime output in `logs/`, preserve benign task/reward
contracts across Phase 4 variants, and use existing helpers and quiet wrappers
before introducing abstractions. Treat
`AgentLab/src/agentlab/benchmarks/redteam/{execution.py,claude_code.py}` as
read-only reference material; runtime imports from `AgentLab/` are outside the
package boundary.

Finish a task when the selected branch document has been applied, the spec and
source agree for every changed contract, the relevant validation evidence is
recorded, and no generated artifact or secret was added to the source tree. If
a branch document and the spec disagree, stop and report the conflict before
coding.
