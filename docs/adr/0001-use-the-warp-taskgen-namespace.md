---
status: accepted
---

# Use the WARP Taskgen namespace for active contracts

Active code and operator contracts will move to the canonical `warp_taskgen`, `warp-taskgen`, and `WARP_TASKGEN_*` names through a separately scoped, coordinated migration after the feature seams are stable. Until that migration begins, incremental feature slices may use the current `worldsim` package owner rather than create two active implementations. New schemas and artifacts created by the coordinated migration will not emit historical names; immutable published evidence and external infrastructure identifiers may retain them and remain readable through an explicit migration or normalization path. This accepts a coordinated migration cost in exchange for semantic naming, one active vocabulary, and better locality for developers and agents.
