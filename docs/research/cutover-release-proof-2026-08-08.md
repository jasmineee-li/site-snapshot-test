# Taskgen cutover release proof — 2026-08-08

## Decision

**Blocked; do not merge or start issue #40.** The canonical package and local
developer boundary are green, the final source sync is recoverable, and the
rollback is clean. Issue #39 still lacks a passing live integration wrapper and
the two strictly admitted Phase 4 canaries.

The r8a host was parked after the bounded checks. Final EC2 state was `stopped`
with no `worldsim:sweep-in-progress` tag.

## Source-base identity

- Source base: `a4f79e69a03465f5d320261b7ff764fd4b7755de`.
- Root tree: `8f6a901d4c504841d9d5213342beef73a2251ea6`.
- Package tree: `901059d02239d97cac9c6962ec90db2f324dd57b`.
- Package inventory: 812 entries. SHA-256
  `75e3a60bbeae82729615fe44b6554ed8a1e1b1de74e67c2ab4a251012db2a053`
  over `mode blob relative-path\n` rows in the canonical order emitted by
  `git ls-tree -r HEAD -- packages/warp-taskgen`.
- Synced runtime source: `b7bd903aec880d3278fb3b95a38a508c1d8bebb4`,
  branch `codex/taskgen-cutover`, with `dirty=false` and Git metadata excluded
  in the remote sync stamp.

The proof document is layered after this source base and is intentionally not
included in the source-base tree identity. Its commit is recorded in issue #39;
no blocked candidate is called a final release or tagged as one.

The package tree is the final path/mode/blob manifest. The earlier four-state
reconciliation evidence remains in
`docs/research/cutover-final-manifest-2026-08-08.json`; the final tree additionally
contains the workflow switch, two host-local sync exclusions, and the active r8a
documentation correction. The workflow switch explicitly removed
`scripts/ssh_r5.sh` and `scripts/sync_warp_taskgen_snapshot.sh`.

All six public-main paths that did not require a merge remained byte-identical
to the pre-main pin; `README.md` carries the explicit r8a/public-hygiene merge.
The cutover did not change published analysis, datasets, results, or generated
logs. Package hygiene and mode checks passed in the reconciliation audit. A
final-source-base scan also passed `git diff --check 457dc542..HEAD` and found no
added local absolute paths, private-key headers, API-token patterns, or pinned
live infrastructure identifiers in `457dc542..a4f79e69`.

## Provenance and local acceptance

The five permanent annotated tags restored in disposable worktrees. Tag commit
and root-tree readback was:

| Tag suffix | Commit | Root tree |
| --- | --- | --- |
| `authoring-local-93c7b99a` | `93c7b99adbbdc4bd4306cde73a55170320370643` | `7d9de1aa3f17f9a984c0c0e4d5d56768a79cbb52` |
| `authoring-remote-9cbb0eaf` | `9cbb0eaf3f44b0e50b72ad9af8b166e187099937` | `b9b837151a1eb6b07956fe5ac8c732474e7e2b66` |
| `snapshot-ffe2c3e8` | `ffe2c3e80974e9aab51d4281c7a6e8c0de1c8af9` | `03b2e0a0c493e22e90fcf3be24eab2d863cbaa88` |
| `main-pre-9eeadebf` | `9eeadebfdabc686744293584304d84cd918459a6` | `0e8e92241cb699f850fe6968f267b8857241f1ad` |
| `imported-446316bd` | `446316bd115b7aecf4baa50e418da3e0a4958d2c` | `7293203e36a23acea992c28b8154678dcdc62da0` |

Every row is under the shared `cutover-2026-08-08-` prefix.

The imported tag remains historical and was not retargeted after the two sync
safety fixes. No final-release tag was created because the live release gate is
blocked.

At the exact branch head, `bash scripts/accept_taskgen.sh` passed locked sync,
Ruff, readiness, the default parallel pytest suite, package build, isolated
wheel install, and installed `warp-taskgen --help`.

## r8a sync and integration

The first dry-run exposed host-local vendor deletions. A second exposed local
editor/config/topology deletions. Commits `cbfea552` and `b7bd903a` added the
smallest exact exclusions and their focused assertions. The final dry-run had
no data, logs, vendors, credentials, topology, or local-config deletions. The
real sync completed and wrote the clean `b7bd903a` stamp.

The first on-host integration attempt used auto locality and was invalid: the
checkout path differed from `compose_dir_remote`, so it selected the public host
view. The valid retry used the tracked public template only as URL-shape input,
with an explicit host view; no lifecycle or deploy command consumed its dummy
AWS identifiers. The normal operator rerun must use the ignored local overlay:

```bash
bash scripts/run_integration_tests.sh \
  --host-config configs/benchmark_hosts/r8a.local.yaml \
  --host-view orchestrator \
  --verify-read-surface-urls \
  --quiet
```

It reached the intended private orchestrator bridge and completed in 72.91 seconds:
18 passed, 3 failed, 1 skipped. Both read-surface checks, all ten live
feasibility checks, and the GitLab and Reddit seed/resolver checks passed. The
three failures were the L3 Anthropic classifier checks. The configured
OpenRouter endpoint returned HTTP 404 because no endpoint matched the account's
privacy/data-policy restrictions. The host has OpenRouter auth only; no direct
Anthropic API key or Claude OAuth token is configured.

Changing an account privacy policy or substituting a model is outside this
source-control cutover. The wrapper therefore has not passed once as required.
The official-document review and authenticated small-model probes are recorded
in `docs/research/openrouter-l3-endpoint-selection-2026-08-08.md`.

## Strict cohort and Phase 4 canary

No current-source cohort is eligible. The best historical candidate has 50/50
verified Phase 2c rows and 50/50 valid Phase 3 contracts, but it was generated
on 2026-05-07 with unknown editor/dataset commits and instances digest
`a2f000aade52`. The current sync is `b7bd903a` and the current
`instances.scale.json` digest is `84bbcc137e02`. That fingerprint drift requires
fresh verification under the technical specification.

No Phase 4 canary was run. Running it against the historical cohort would
weaken strict admission and invalidate the release evidence.

## Rollback rehearsal

From the exact branch head, these commits reverted without conflicts in a
disposable worktree:

1. `a4f79e69` — active r8a documentation correction
2. `b7bd903a` — local state sync exclusions
3. `cbfea552` — vendor sync exclusion
4. `446316bd` — workflow switch
5. `457dc542` — reconciliation
6. `5ed9c27c` — mechanical import

The disposable rollback HEAD tree was
`98f180bb4c99a597c4583631494d49ba071765c7`, exactly equal to
`629a3366^{tree}`; `git diff 629a3366..HEAD` was empty inside that disposable
worktree. The worktree was then removed.

The same root acceptance command reached Ruff and readiness, then failed the
pre-cutover default suite with 24 known failures (3,476 passed, 4 skipped).
Those failures are exactly the baseline defects corrected by reconciliation:
the VEA hash pin, missing PVPO preflight import, strict encounter expectations,
Browser Use test-path pollution, and a timezone-sensitive remote-job fixture.
The cutover rollback is content-correct. On 2026-08-08 the owner approved the
canonical rollback criterion as exact frontier-tree restoration with no new
failures beyond this recorded baseline. Issue #39 was updated accordingly; no
backport to the retired authoring frontier is required.

## Required unblock

1. Configure an approved model endpoint for the existing L3 checks, then rerun
   the integration command above until the wrapper passes.
2. In a new host state directory, copy the historical Phase 1 benign and Phase
   2 adversarial inputs, run Phase 2c with `--force-reverify` against the current
   `instances.scale.json`, and run Phase 3 in the same state. Preserve the source
   and instances fingerprints.
3. Run one bounded real Phase 4 task for GitLab and one for Reddit/Postmill with
   explicit timeouts, the default `eval-awareness-iterator`, and no feasibility
   bypass. Accept artifact/invariant shape rather than ASR equality.
4. Only after all three live items pass, create a final annotated release tag, close
   #39, and begin the protected-main/post-merge proof in #40.
