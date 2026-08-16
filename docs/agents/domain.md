# Domain routes

Use this file to route terminology and durable domain decisions. It is not a
second implementation workflow.

## Existing owners

- Repository-wide ubiquitous language: `CONTEXT.md`.
- Repository purpose and released vocabulary: `README.md`.
- Taskgen concepts and invariants: `packages/warp-taskgen/CLAUDE.md`,
  `packages/warp-taskgen/docs/warp-taskgen-technical-spec.md`, and the relevant
  `packages/warp-taskgen/agent_docs/` file.
- Evaluation-awareness concepts and study design:
  `eval_awareness_experiments/CLAUDE.md` and
  `eval_awareness_experiments/EXPERIMENTS.md`.
- Experiment history and result provenance:
  `eval_awareness_experiments/experiment_log.md`.
- Issue and triage vocabulary: `docs/agents/issue-tracker.md` and
  `docs/agents/triage-labels.md`.

## Vocabulary rules

- Use **WARP** in new prose. Keep `worldsim` when it names a legacy package,
  schema, environment variable, or deployed identifier.
- Use the canonical terms in `CONTEXT.md`. Owning guides and specifications
  define their behavior; they must not redefine the terms or introduce a
  competing synonym.
- When a term is ambiguous, read `CONTEXT.md` and the owning document, then
  state the ambiguity before proposing a new name.

## Decision rule

Record a resolved domain term in `CONTEXT.md`. Record a durable architecture
decision in its owning document, and link the issue or PR that motivated it.
If no existing owner fits, pause and ask for a domain-modeling decision instead
of creating another glossary.
