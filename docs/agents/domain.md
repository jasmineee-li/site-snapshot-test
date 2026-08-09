# Domain Docs

This is a multi-context repository.

## Before exploring

1. Read `CONTEXT-MAP.md` when it exists.
2. Read the context document that applies to the work.
3. Read relevant repository-wide ADRs under `docs/adr/`.
4. Read relevant context-specific ADRs.

If a document does not exist, continue without it. Domain documents are
created when terminology or architectural decisions need to be recorded.

## Context layout

Repository-wide map:

- `CONTEXT-MAP.md`

Repository-wide decisions:

- `docs/adr/`

WARP Taskgen context:

- `packages/warp-taskgen/CONTEXT.md`
- `packages/warp-taskgen/docs/adr/`

Evaluation-awareness context:

- `eval_awareness/CONTEXT.md`
- `eval_awareness/docs/adr/`

The root map can add more contexts when a first-party area develops a distinct
vocabulary and decision history.

## Canonical Taskgen route

`packages/warp-taskgen/` on current `origin/main` is the only writable Taskgen
source. Start one short-lived topic worktree from `origin/main`, make the
Taskgen change there, validate it from the repository root with
`bash scripts/accept_taskgen.sh`, and open one PR to `main`. Remove the
worktree after merge. The acceptance command is the same one used by CI; there
is no source snapshot or sync-back step. Package `scripts/sync_to_host.sh` is
only an operational deployment to a prepared benchmark host.

## Use glossary vocabulary

Use terms exactly as the relevant context document defines them. Do not replace
a defined term with a synonym in issue titles, specifications, test names, or
architectural proposals.

If a required term is missing, do not invent a competing term. Record the gap
for domain-modeling work.

## ADR conflicts

If proposed work conflicts with an ADR, identify the conflict in the issue or
specification. Do not silently replace the earlier decision.
