# Code Organization

Use this when moving code, adding modules, or reviewing agent-readiness debt.

## Feature Ownership

Organize code by feature or domain, not by technical layer. A module should
answer "what behavior owns this?" before it answers "what kind of code is this?"

Current and target ownership should stay explicit:

- Phase 2 behavior includes injection generation, target resolution, exposure
  contracts, text fill, and Phase 2c admission. A package split may move this
  toward `worldsim.phase_2`, but route/exposure/admission semantics still belong
  to the Phase 2 domain.
- Phase 4 behavior includes adversarial execution, PVPO placement, postprocess
  judges, strategy variation, resume, and results. A package split may move this
  toward `worldsim.phase_4`, but Phase 4 must not own benign task eligibility.
- Phase 0c profile rigor is split by behavior: `phase_0_recon.py` remains the
  compatibility runner, `phase_0_evidence_index.py` owns neutral source indexes,
  `phase_0c_artifacts.py` owns provenance/reuse/trace artifacts, and
  `phase_0c_audit.py` owns deterministic host audits. Do not move these
  concerns into Modal sandbox setup; `modal_sandbox.py` stays an
  infrastructure-only runner.
- Seed/editor-call contracts are a shared domain used by Phase 2, Phase 4,
  seeding, and sandbox validation. If extracted, the shared package must preserve
  sandbox packaging constraints and parity tests.
- Browser Use runtime concerns belong together when that runner is split.
- Sandbox/profile/task validation has a stricter Modal runtime contract than
  ordinary host modules and should not be mechanically extracted.

Avoid `utils.py`, `helpers.py`, and global shared `types.py`. If a helper is
shared, name the domain it belongs to. Keep types next to the behavior that owns
them until multiple sibling modules need them; then move them to a feature-local
types module.

## File Size

The advisory review ceiling follows `scripts/readiness_audit.py`: 500 actual
lines is review debt and 1200 lines is urgent split debt. The 150-line floor is only a guard
against over-splitting; small files are fine for entrypoints, adapters,
protocols, `__init__` modules, and narrow local type holders.

Split at natural behavioral boundaries. Do not split a linear algorithm into
arbitrary chunks just to satisfy a number.

True exemptions should be documented in `scripts/readiness_audit.py` with a
reason. Deferred monoliths are not exemptions; keep them visible in readiness
reports until they are split.

## Tests

Keep pytest tests under top-level `tests/`, mirrored by domain where practical:

- `tests/phase_2/`
- `tests/phase_4/`
- `tests/seed_contracts/`
- `tests/browser_use/`
- `tests/sandbox_validator/`

Tests may import feature-private helpers when those helpers encode important
contracts, but they should import them from the owning feature package rather
than from compatibility wrappers once a package split exists.

## Compatibility Wrappers

When moving public or widely imported modules, keep the old import path as a
thin compatibility wrapper for one migration cycle. Wrappers may delegate and
re-export canonical names, but they should not own behavior, monkeypatch
dependencies, or contain a second implementation.

Update in-repo imports to the canonical feature package during the same change.
Remove wrappers in a follow-up cleanup once hidden consumers have had a chance
to surface.

Wrappers are temporary agent-DX debt. They are worth paying when the refactor
touches research-critical paths because they separate behavior migration from
import-path cutover. They should have tests that prove legacy imports delegate,
and a follow-up PR should remove them when the migration window closes.

## Follow-Up Debt

Do not treat all readiness debt as equally urgent or equally safe to fold into
one PR. Good sequencing reduces review risk:

- First split behavior into domain-owned modules while preserving external
  import compatibility.
- Then remove compatibility wrappers after downstream imports are moved and one
  validation cycle has had a chance to reveal hidden consumers.
- Unwind linked-context or monkeypatch-preservation mechanisms only after the
  canonical modules are stable.
- Split sandbox validation as a separate design task because it runs inside
  Modal with stdlib-only and no-project-import constraints.
- Keep other large files visible as debt instead of allowlisting them without a
  reason.

Only fold a follow-up into an active PR when it reduces review risk for that PR.
If it expands the behavioral surface under review, keep it separate.
