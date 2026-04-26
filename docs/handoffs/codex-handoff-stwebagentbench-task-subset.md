# Codex handoff: produce STWebAgentBench task subset for ICML panel

## Context (self-contained)

This repository is `feat/worldsim-v5` branch of an IPI safety evaluation benchmark project for browser agents. Track 4 of `docs/TODO-2-paper-experiments.md` plans to compare our wrapper against STWebAgentBench for the ICML CUA workshop on 4/24. Before the adapter work lands, we need a filtered task list that covers the sites we actually operate.

A recent GPT Deep Research pass (summarized in `docs/research/gpt-deep-research-2026-04-14.md`, "STWebAgentBench task list triage" section) surfaced two facts that narrow the work:

- STWebAgentBench's task inventory only covers three sites: `gitlab`, `shopping_admin`, and `suitecrm`. There is no Postmill/Reddit, Wikipedia, or OSM coverage inside the benchmark.
- We are NOT standing up SuiteCRM for the ICML scope, so our overlap is GitLab + ShoppingAdmin only.

GPT also recommended a concrete 30-task panel. This handoff asks you to produce the filtered subset from the canonical task file.

## Source

Repository: `https://github.com/segev-shlomov/ST-WebAgentBench`
Canonical task file: `stwebagentbench/test.raw.json`
Pinned commit: `f7c69df98fe67de7a587795400ca00feb9b375a4`
Flattened reference: `stwebagentbench/test.csv` @ commit `688af42c10adff8fdabdb5a687cba3dfafa62bdf`

Schema: each task object has a `sites` list (usually one site per task), an `intent` string, and a non-empty `policies` array.

## Deliverables

1. **`scripts/filter_stwebagentbench_tasks.py`** — a deterministic filter script that:
   - Takes a path to a local clone of ST-WebAgentBench as input (default: `./.cache/ST-WebAgentBench/`).
   - If the clone does not exist, clones `segev-shlomov/ST-WebAgentBench` at the pinned commit above into `.cache/`.
   - Reads `stwebagentbench/test.raw.json`.
   - Filters to 30 tasks: GitLab + ShoppingAdmin only, no SuiteCRM, non-empty `policies` on every task.
   - Writes the filtered list to `vendors/stwebagentbench-task-subset.json` (pretty-printed, stable key ordering).
   - Emits a one-line summary to stdout: `wrote N tasks: {"gitlab": A, "shopping_admin": B}`.

2. **`vendors/stwebagentbench-task-subset.json`** — the filtered output committed alongside the script.

## Recommended task selection (starting point)

GPT's recommendation (verify each task ID resolves correctly in the canonical `test.raw.json`; adjust if any ID is missing or its `sites` value does not match):

- **8 ShoppingAdmin core tasks**: IDs 77, 78, 79, 80, 81, 82, 83, 84.
- **18 GitLab tiered tasks** (6 base intents replicated across low/med/high policy-load tiers): IDs 85, 135, 185 / 90, 140, 190 / 95, 145, 195 / 100, 150, 200 / 113, 163, 213 / 123, 173, 223.
- **4 GitLab core tasks**: IDs 0, 6, 36, 42.

If any of these 30 IDs are missing, substitute the nearest-ID same-site task and note the substitution in a `# selection notes` comment block at the top of the output JSON (or in a sibling `vendors/stwebagentbench-task-subset.selection_notes.md`).

## Verification

Run these commands after the files are written:

```
jq 'length' vendors/stwebagentbench-task-subset.json
# expected: 30

jq '[.[] | .sites[]] | group_by(.) | map({site: .[0], count: length})' vendors/stwebagentbench-task-subset.json
# expected: [{"site":"gitlab","count":22},{"site":"shopping_admin","count":8}] or equivalent

jq '[.[] | select(.sites[] == "suitecrm")] | length' vendors/stwebagentbench-task-subset.json
# expected: 0

jq '[.[] | select(.policies | length == 0)] | length' vendors/stwebagentbench-task-subset.json
# expected: 0
```

Also re-run `uv run pytest` to confirm the script does not break any existing tests. If you touch Python test files, verify them specifically.

## Non-goals

- Do NOT port the STWebAgentBench adapter from `feat/multi-benchmark` into `feat/worldsim-v5`. That is separate work tracked in TODO-2 Track 4 "Migration steps (ordered)".
- Do NOT stand up SuiteCRM. It is explicitly out of scope for ICML.
- Do NOT modify the `feat/multi-benchmark` worktree or any file under `.claude/worktrees/`.
- Do NOT modify `docs/worldsim-v5-full-specification.md` or `docs/TODO-adversarial-rigor-mvp.md`.

## Operating constraints

- No commits unless the user explicitly asks.
- No pushes ever.
- No destructive git operations.
- Do not switch branches. Use `git show feat/multi-benchmark:<path>` or `git show main:<path>` for cross-branch reads.
- No em dashes in any prose output; use commas or parentheses.
- If the script needs a Python dependency not already in `pyproject.toml`, prefer `json` + `urllib` + `subprocess.run(["git", "clone", ...])` over adding a new dependency.

## Expected output to the user when done

A short summary listing: the filter script path, the output JSON path, the verified counts from the four `jq` checks above, and any ID substitutions that were needed.
