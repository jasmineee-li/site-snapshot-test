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

Taskgen changes belong in `packages/warp-taskgen/`. Validate them from the
repository root with `bash scripts/accept_taskgen.sh`, the same acceptance
command used by CI.

## Use glossary vocabulary

Use terms exactly as the relevant context document defines them. Do not replace
a defined term with a synonym in issue titles, specifications, test names, or
architectural proposals.

If a required term is missing, do not invent a competing term. Record the gap
for domain-modeling work.

## ADR conflicts

If proposed work conflicts with an ADR, identify the conflict in the issue or
specification. Do not silently replace the earlier decision.
