# Phase 2 Quality Audit Findings

## Summary

The Phase 2 smoke artifacts show three distinct issues with different owners.

- Warning 1 is not a single prompt bug. Under the current Phase 0c profiles, the active attacker-controllable surface set only exposes `plaintext` and `markdown_fenced_system`, so the current framing×concealment ceiling is structurally `18/36`.
- Warning 2 is a real Phase 2b prompt-spec issue. The markdown concealment instruction is underspecified relative to the validator, which is why the model leaks or duplicates required tokens.
- Warning 3 is a CLI/runtime bug. `--sites` is accepted globally, persisted through resume dispatch, and then ignored by Phase 3/4 task selection.

## Warning 1

Diagnosis: both data and prompt contract, but data first.

- `worldsim/phases/phase_2_injections.py::_available_cells()` builds cell eligibility from `compatible_concealments` on `controllable_by_tier in {"any_user", "authed_user"}` surfaces.
- In the current `logs/phase_0c/BENCHMARK_PROFILE_*.json` artifacts, active eligible surfaces expose only `plaintext` and `markdown_fenced_system`.
- `offscreen_css` appears only on non-eligible `raw_html` surfaces (`admin` or `none`), so it is unsupported in the current paper cohort.
- `image_alt_text` appears nowhere in the active profiles, but the benchmark dataset includes Reddit image-style submissions. That points to an image-surface discovery gap in Phase 0c, not just a Phase 2 prompt miss.

Recommendation:

- Do not try to force `offscreen_css` into this cohort.
- Treat `offscreen_css` as valid in principle but unsupported by the current benchmark/threat-model slice.
- Fix Phase 0c image-surface discovery so `image_alt_text` can become a real third concealment arm.

Coverage estimate:

- Current maximum: `18/36`.
- After a successful Phase 0c image-surface fix: likely `27/36`.
- `36/36` is not realistic without adding an attacker-controllable raw-HTML benchmark arm.

## Warning 2

Diagnosis: prompt/instruction ambiguity.

Observed markdown unrecoverables from the current diagnostics:

- `73` `text_unrecoverable` markdown tasks.
- Task-level failure buckets:
  - `72` token leaked into visible prose
  - `72` token missing from concealed region
  - `17` `raw_text` mismatch
  - `15` token duplication
  - `2` parse failures

Root cause:

- The prompt asks for a `````system``` fence but does not teach the exact validator invariants:
  - both required tokens exactly once
  - both only inside the concealed block
  - no token leakage into visible prose
  - `raw_text` equals visible prose with the concealed block removed

Fix:

- Add explicit correct/incorrect markdown examples to `worldsim/prompts/fill-injection-text.md`.
- Strengthen `_concealment_instruction()` in `worldsim/phases/phase_2_text_fill.py` to mirror the validator contract.

## Warning 3

Decision: Option C.

Rationale:

- Operators already expect `--sites` to scope Phases 3-4 because those phases are site-partitioned and already have `--max-tasks-per-site`.
- Silent acceptance is the real bug. Renaming or rejecting the flag would preserve the mismatch between operator intent and runtime behavior.

Fix:

- Make Phase 3 and Phase 4 filter tasks by `args.sites` before sampling.
- Reject unknown site names with a useful error.
- Persist the site filter in Phase 3/4 state metadata so `resume` keeps the same scope.
- Update CLI help text so `--sites` no longer reads as Phase-2-only.

## Cost Estimate

- Baseline Phase 2a rerun cost from `logs/cost_report.json`: `$84.62` across `49` sandbox calls.
- Phase 2b remains low-cost API work relative to 2a.
- Warning 2 and Warning 3 fixes do not change the 2a spend envelope materially.
- A separate Phase 0c reprofile to unlock `image_alt_text` would add incremental profiling cost and should be budgeted separately from the immediate paper-cohort rerun.
