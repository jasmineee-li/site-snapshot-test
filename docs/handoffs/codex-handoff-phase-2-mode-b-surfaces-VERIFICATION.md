# Phase 2 Mode B surfaces — final verification runbook

This commit completes the Mode B surface admission work
(`docs/handoffs/codex-handoff-phase-2-mode-b-surfaces.md`). Six commits
landed on `feat/worldsim-v5`:

| Commit | Subject |
|--------|---------|
| `feat(phase0c): enrich gitlab agent_context with user/group handles` | Phase 0c discovery extension. |
| `test(phase1): lock in agent_context.gitlab handle threading` | Regression test for the existing per-task threading. |
| `feat(editors-gitlab): seven new methods for Mode B uncovered surfaces` | Editor methods + tokens. |
| `feat(phase2-resolver): admit six gitlab Mode B URL kinds` | Resolver patterns + disambiguator. |
| `feat(phase2-exposure): admit gitlab Mode B kinds in exposure contract` | Mode classification + selectors. |
| `test(phase2): resolver + exposure-contract coverage for Mode B kinds` | Unit tests. |

## Offline verification (already passing)

```
uv run pytest -q
```

2186 passed locally. Independent simulation against
`logs/phase_1/novel_tasks_gitlab.json` (with synthetic Phase 0c handle
lists injected into each task's `agent_context.gitlab`) shows:

- 30/30 placement-validator pass rate (`validate_generated_novel_task`).
- 100% Phase 2 admit rate (`build_exposure_contract` returns
  `eligibility.status="eligible"`).
- Kind distribution: `gitlab_user_profile` 11, `gitlab_snippets_index`
  8, `gitlab_project_labels` 5, `gitlab_project_milestone` 4,
  `gitlab_group` 2.

## Live verification (operator-driven)

These steps require a live r5 stack and Modal credentials. They are
**not** auto-executed by the implementation work; run them in order
when you're ready to ship.

### 1. Re-run Phase 0c on r5 to populate handle lists

```
uv run python -m worldsim.main phase 0c \
    --benchmark vendors/webarena-verified \
    --instances instances.smoke.json \
    --sites gitlab
```

Inspect `logs/phase_0c/AGENT_CONTEXT_gitlab.json` for the new
top-level `gitlab` block:

```json
"gitlab": {
  "user_handles": ["abisubramanya27", "agitter", "byteblaze", ...],
  "group_handles": ["a11yproject", "coding_friends", ...]
}
```

If the `gitlab` block is missing, check the Phase 0c log for
`gitlab handle enrichment for site 'gitlab' failed`. The most common
cause is the gitlab admin token failing to acquire — verify
`auth.token_generator: gitlab_pat` and credentials in
`instances.smoke.json` match the r5 deployment.

### 2. Regenerate cached Mode B novel tasks

```
uv run python -m worldsim.main phase 1 --generate-novel \
    --sites gitlab,reddit \
    --novel-tasks-per-site 30
```

The regen will pick up the new `agent_context.gitlab` block via the
existing Phase 1 attach flow. Confirm
`logs/phase_1/novel_tasks_gitlab.json` retains 30 entries and every
task carries `agent_context.gitlab.user_handles` / `group_handles`.

### 3. Run the Phase 2 admit pass

```
uv run python -m worldsim.main phase 2 \
    --benchmark vendors/webarena-verified \
    --sites gitlab,reddit \
    --feasibility-instances instances.smoke.json
```

Acceptance gates:

- `logs/phase_2/new_task_resolver_dropouts.json` carries no gitlab
  entries with `reason="unresolved_target_resource"`.
- ≥80% of `origin="new_task"` gitlab inputs admit (target: 30/30).

### 4. Live integration test pass

```
scripts/run_integration_tests.sh \
    --host-config configs/benchmark_hosts/r5.yaml --quiet
```

Per-CLAUDE.md guidance for shipping changes that touch editors and
Phase 2c. Paste any surfaced failure output into the PR description.

## Key invariants preserved

- The placement validator (`worldsim/phases/phase_1_generate_new_tasks_validation.py`)
  was not relaxed. The new admit rate comes entirely from the resolver
  and exposure-contract extending to admit kinds that were previously
  unresolvable.
- The Mode B prompt (`worldsim/prompts/generate-benign-tasks.md`) was
  not changed. The LLM's natural output for uncovered surfaces becomes
  admissible because the downstream pipeline now recognizes the URL
  shapes it generates.
- Phase 4 stratification, PVPO, Apollo / VEA classifiers, and Reddit
  remain untouched.

## Next steps after verification

Once the live admit is confirmed, the parent Mode B handoff
(`codex-handoff-phase-1-mode-b-implementation.md`) resumes at Step B
(its Commit 5) without further blockage. This handoff doc can be
moved to `docs/archive/` after Mode B ships.
