# Guidance Hygiene

Rules for editing the agent guidance itself: the root router, the path guides,
and the documents under `docs/agents/`.

## One source of truth

Keep each rule in exactly one document. A rule stated in a router and again in
the path guide that router points at costs context on every turn and drifts
apart over time. Place a new invariant in the guide that owns it, then add one
pointer from the router.

A safety guardrail with real-world blast radius is the deliberate exception:
sandboxed execution, secrets, and benchmark check preservation are stated in
both the root router and the path guides, because the router is the only file
loaded for every tree. Before removing one as a duplicate, confirm that every
path guide the router points at still states it.

## Discover commands from the environment

Read commands from the owning guide, the current environment, and each tool's
`--help` output. A command catalog copied into a router is a cache of something
the environment already answers, and it goes stale silently. When a command or
a path changes, update the owning guide.

## Keep a router small

A router routes; it is not a second contract. Keep the root router stable and
short so nested guidance can carry branch-specific detail. Prefer existing
specs, helpers, tests, and artifact formats over new abstractions.

## Match the reader

These documents are read by an agent on every turn, so every line is paid for
whether or not it fires. Cut a line that restates default behavior, and push
material that only some branches need behind a pointer instead of inlining it.
