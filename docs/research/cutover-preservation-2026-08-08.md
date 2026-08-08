# Cutover input preservation — 2026-08-08

Issue #34 freezes the source-control inputs before canonicalization. This record is
sanitized and contains no runtime output, credentials, generated host configuration,
vendor trees, or local absolute paths.

## Permanent provenance refs

| role | remote tag | peeled commit |
| --- | --- | --- |
| local authoring source including eight unpushed commits | `refs/tags/cutover-2026-08-08-authoring-local-93c7b99a` | `93c7b99adbbdc4bd4306cde73a55170320370643` |
| fetched origin/feat/worldsim-v5 tip | `refs/tags/cutover-2026-08-08-authoring-remote-9cbb0eaf` | `9cbb0eaf3f44b0e50b72ad9af8b166e187099937` |
| package snapshot/source baseline | `refs/tags/cutover-2026-08-08-snapshot-ffe2c3e8` | `ffe2c3e80974e9aab51d4281c7a6e8c0de1c8af9` |
| fetched origin/main before canonical import | `refs/tags/cutover-2026-08-08-main-pre-9eeadebf` | `9eeadebfdabc686744293584304d84cd918459a6` |

Each tag was peeled through the remote and restored in a fresh disposable worktree.
The local authoring tag preserves the eight commits not present on the fetched remote
authoring tip; both are recorded so the import can choose deliberately.

## Baseline inventory

[`cutover-baseline-inventory-2026-08-08.json`](cutover-baseline-inventory-2026-08-08.json) records path, mode, and blob for
the snapshot source, both authoring tips, and the pre-cutover `main` package. Paths are
normalized to the Taskgen root so later import comparison is direct.

## Dirty worktree decisions

| worktree | branch | decision | recovery artifact |
| --- | --- | --- | --- |
| `branch-cutover-plan` | `codex/branch-cutover-plan` | **keep** | `private-local-recovery/cutover-2026-08-08/branch-cutover-plan` |
| `engineering-skills-setup` | `codex/engineering-skills-setup` | **keep** | `private-local-recovery/cutover-2026-08-08/engineering-skills-setup` |
| `feat-multi-benchmark` | `feat/multi-benchmark` | **keep** | `private-local-recovery/cutover-2026-08-08/feat-multi-benchmark` |
| `multi-benchmark-rebased` | `multi-benchmark-rebased` | **keep** | `private-local-recovery/cutover-2026-08-08/multi-benchmark-rebased` |
| `structured-asr-parity` | `codex-structured-asr-parity` | **keep** | `private-local-recovery/cutover-2026-08-08/structured-asr-parity` |

Recovery artifacts contain only reviewed source snapshots and metadata. Paths classified
as runtime-only/generated configuration, credentials, logs, vendors, host-local state,
or local absolute paths are listed in each `record.json` but are not copied. Original
worktrees remain attached and untouched; deferred paths require owner review before any
future archive or porting action.

## Safety boundary

No existing worktree or branch was switched, cleaned, reset, pruned, or deleted. No
generated logs or secret/runtime content was committed. The local recovery root is
`private-local-recovery/cutover-2026-08-08` and is intentionally ignored/non-public.
