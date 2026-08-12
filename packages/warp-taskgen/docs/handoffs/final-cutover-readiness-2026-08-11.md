# Final cutover readiness

Issue #71 records the post-#77 expand-in-place cutover contract. This note is
the concise tracked readiness record; the older `docs/research/cutover-*`
files remain historical provenance and are not a current migration plan.

## Contract

- The canonical Run root remains `logs/<run_id>/` locally and
  `s3://benchmark-archives/worldsim-runs/<run_id>/` in the archive. The
  existing archive wrapper syncs one selected root recursively and does not
  rename, flatten, or recursively archive `.warp-derived-runs/`.
- A Legacy Run is readable and conservatively resumable without an invented Run
  ID or Definition Digest. Its unbound output is not accepted as a reusable
  Checkpoint. An archive manifest's `run_id` is only the selected archive
  namespace for S3 layout; it does not mutate Legacy lifecycle identity or
  authorize checkpoint reuse. New roots retain the persisted Run ID and
  immutable Run Definition; Derived Runs are archived by their own opaque child
  ID.
- Phase 2a shard manifests, Phase 2b text-fill envelopes, and Phase 2c
  feasibility checkpoints remain under their existing feature-owned paths.
  Archive transport preserves them as evidence; the archive is never a
  checkpoint validator or migration system.
- Run Artifact, Checkpoint, Atomic Work Unit, Pause Request, and Run Lifecycle
  have the glossary meanings in `CONTEXT.md`. Run Lifecycle is independent of
  Remote Job process state.
- Phases 0, 1, and 3 retain crash-only behavior. No archive operation changes
  pipeline state, creates a child, or performs lifecycle repair.

## Evidence

The fake-AWS compatibility tests in
`tests/test_archive_run_scripts.py` enumerate the selected source root and
prove the legacy and Derived Run source roots, canonical S3 keys, inclusion of
representative Phase 2a/2b/2c and Phase 4 checkpoint paths, and non-recursive
child selection. They also assert that a legacy state remains identity-less,
reject path-traversal IDs, and keep the archive manifest namespace distinct
from persisted lifecycle identity. The archive wrapper uses portable `du -sk`
byte accounting so the proof runs on macOS and Linux.

Run the focused proof with:

```bash
pytest -q tests/test_archive_run_scripts.py
```

Local and selected-host evidence captured for this note:

- The locked Python 3.12 environment was provisioned without changing tracked
  dependency state. Direct `ruff check worldsim tests scripts`, the readiness
  audit (`tracked-generated`, `tokens`, and `legacy-imports` fail-on checks),
  `git diff --check`, and `bash -n scripts/archive_run_to_s3.sh` pass.
- The focused combined suite passes 87 tests; the remote-job suite passes 51;
  the archive compatibility suite currently passes 5 tests.
- The broader core suite reports 3,799 passed and 4 skipped, with 33
  environment failures (Browser Use macOS aborts, host-script `uv`
  system-configuration panics, and sandbox socket permissions). Crash-resume
  reports 5 passing scenarios and 12 fixture/environment failures. These are
  recorded as infrastructure evidence, not cutover behavior failures.
- A selected-host Phase 2c canary on the configured r8a host passed with
  `rc=0`: `test_feasibility_good_task[gitlab]` plus
  `test_gitlab_live_probe_create_reuse_and_cleanup`. The quiet wrapper emitted
  only the unpinned-image warning and host-config line. This proves the good
  task's seed/render/reachability/cleanup path and shared GitLab editor
  create/reuse/cleanup, not a full Phase 4 trajectory. The exact package-root
  command was:

  ```bash
  bash scripts/run_integration_tests.sh \
    --host-config /private/tmp/warp-r8a-cutover.yaml --quiet \
    'tests/integration/test_phase_2_feasibility_live.py::test_feasibility_good_task[gitlab]' \
    tests/integration/test_seed_resolver_gitlab_live.py::test_gitlab_live_probe_create_reuse_and_cleanup
  ```

  The host config and matching ignored instances input were copied into
  non-tracked temporary locations for this canary.
- The Phase 4 end-to-end canary remains externally blocked: no dedicated
  admitted Run/Phase 3 task artifacts and credentials were available. No
  production Run root was mutated.

Before release, run `bash scripts/verify_fast.sh`, the crash/resume suites,
and `bash scripts/accept_taskgen.sh`. Selected-host Phase 2c and Phase 4
canaries are a release gate whenever an authorized, configured host is
available; record the exact host, run IDs, and artifact/result paths here.

### Current gate record

The Phase 2c selected-host gate is evidenced above (`rc=0`). The Phase 4
selected-host gate is **blocked by the external admitted-artifact/credential
precondition**, not passed or waived; append the exact Phase 4 run and artifact
evidence before declaring the final cutover ready.
