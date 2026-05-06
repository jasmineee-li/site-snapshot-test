# Upstream patches for external benchmarks

Patches we maintain against external_benchmarks/* clones. Each is a `git diff`
output reapplied via `git apply <patch>` from the relevant external_benchmarks
subdir. Apply by hand after re-cloning, or wire into `setup_benchmarks.sh`.

## `wasp_cleanup_flag_shadowing.patch`

**Target**: `external_benchmarks/wasp/webarena_prompt_injections/environment_cleanup.py`

**Bug**: The click flag `--delete-attacker-account` binds to a function param
of the same name (`delete_attacker_account`), which shadows the
module-level `delete_attacker_account()` function the same code tries to call.
Triggers `TypeError: 'bool' object is not callable` whenever the flag is
passed (i.e., when you actually want to clean up attacker gitlab accounts).

**Fix**: Rename the param to `delete_attacker_account_flag` and add an
explicit click→param mapping. Also corrects misleading help text.

**Apply**:
```bash
git -C external_benchmarks/wasp apply \
  ../../eval_awareness_experiments/patches/wasp_cleanup_flag_shadowing.patch
```

(File a PR upstream when you have a moment — facebookresearch/wasp.)
