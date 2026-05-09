# WARP Taskgen Docs

Start here when deciding which document to trust.

## Source Of Truth

- `worldsim-v5-technical-specifcation.md` is the detailed implementation
  authority. The legacy filename typo is intentional.
- `worldsim-v5-full-specification.md` is the short current overview.
- `current_progress.md` records the current project state.
- `agent_docs/` contains operating rules for agents and runbooks.
- Top-level `README.md` is the install/run quickstart; it should mirror the
  specs but is not the behavior authority.

If docs and code disagree, update the technical specification first, then align
the overview, runbooks, and handoffs.

## Naming

The current project/distribution name is WARP Taskgen and the preferred console
script is `warp-taskgen`. The Python package, compatibility CLI, and many
artifact fields remain `worldsim`; spec filenames also keep their legacy
WorldSim-v5 names for compatibility. Do not infer that a doc is stale from the
word `worldsim` alone.

## Current Scope

WARP Taskgen is strict WASP scope:

- GitLab issues and issue comments/notes
- Reddit/Postmill submissions and comments

Shopping, Magento, Wikipedia, OpenStreetMap, classifieds, GitLab merge requests,
and title carriers are historical or support context unless the technical spec
explicitly reopens them. Support images or setup fragments for those sites do
not make them active IPI carriers.

## Current Phase 4 Rules

- Browser Use is the default runtime.
- AgentLab/BrowserGym runs through `packages/worldsim-agentlab-runner`.
- PVPO encounter is Gate 1.
- Attack effectiveness is Gate 2.
- Transcript Purpose and VEA are observational in baseline scoring.
- The default variant system is `eval-awareness-iterator`.
- Default iterator runs record an envelope for every PVPO-valid baseline row;
  Transcript Purpose controls whether rewrite iterations proceed.
- Legacy `strategy-variation` is opt-in only.
- VEA is never a branching or admission signal.

## Common Commands

Fast local checks:

```bash
bash scripts/verify_fast.sh
uv run pytest <paths> -q
uv run ruff check <paths>
```

Default top-level Phase 4 shape:

```bash
uv run warp-taskgen phase 4 \
  --instances instances.scale.json \
  --phase-4-max-workers 48 \
  --phase-4-variant-system eval-awareness-iterator \
  --phase-4-eval-awareness-max-iterations 3
```

For AgentLab parity runs, add `--runner agentlab` and validate the sidecar
artifact/PVPO/network/auth/resume gates for that host/model matrix.

## Worker Flags

For top-level Phase 4 runs:

```bash
uv run warp-taskgen phase 4 --phase-4-max-workers 48
```

Do not use `--workers 48` with top-level `warp-taskgen phase 4`; the remote
job guard rejects it.

`--workers` is only for `scripts/run_phase4_process_pool.py`, whose child
processes each run normal Phase 4 with `--phase-4-max-workers 1`.

## Handoffs

Files under `docs/handoffs/` are usually point-in-time working notes. Trust them
only when they say they are current, and cross-check against:

- `worldsim-v5-technical-specifcation.md`
- `current_progress.md`
- the relevant `agent_docs/` file
- the actual run directory or remote job status

## Partial Results

`results.partial.json` and `partial_manifest.json` are inspection artifacts.
They are useful for repair and debugging, but they are not paper-eligible
canonical outputs. For process-pool failures, `scripts/repair_process_pool_partial.py`
can combine a partial run with targeted retry runs into a repaired run that
writes `phase_4/results.json`, `phase_4/process_pool_repair_manifest.json`, and
`paper_eligible="operator_review_required"`. Paper-facing promotion still needs
operator review of the repair manifest and a completed summary.

## Updating Docs

Use simple, direct wording. Prefer current commands over historical context.
When a command is only valid for a wrapper, say that next to the command.

For every docs update, check:

- Does this change affect the technical spec?
- Does a runbook now point at a stale flag or path?
- Does a handoff need a historical or superseded note?
- Does the current status depend on a remote run or local artifact that was not
  verified?
