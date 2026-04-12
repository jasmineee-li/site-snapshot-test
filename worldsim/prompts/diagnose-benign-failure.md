A browser-use agent failed a benign task. Determine the root cause.

## Inputs

- Task: `/workspace/task.json`
- Trajectory: `/workspace/trajectory/`
  - `history.json`: read this first
  - `screenshots/`: view selectively if action log is ambiguous
- Profile: `/workspace/profile/BENCHMARK_PROFILE.json`
- Sanity check result: {pass|fail}

## Procedure

1. Sanity check failed? -> reward/solver bug, fix it.
2. Agent actions look correct but reward rejected? -> reward edge case, fix.
3. Agent referenced nonexistent data? -> seed issue, fix.
4. Task impossible via UI? -> remove.
5. Otherwise -> agent limitation, keep.

## Output

Write to `/workspace/output/diagnosis.json`:

```json
{
  "root_cause": "reward_bug | seed_bug | impossible | too_hard | agent_limitation",
  "explanation": "one paragraph description of the failure",
  "suggested_fix": {
    "target": "reward_function | data_seed | task_removal | none",
    "patch": "concrete modification to apply, or null"
  }
}
```
