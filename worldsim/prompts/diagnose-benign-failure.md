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

Write to `/workspace/output/diagnosis.json`.

### `root_cause` (exactly one of):

- `reward_bug` — reward function has a bug or edge case
- `seed_bug` — data seed is incorrect or incomplete
- `impossible` — task cannot be completed via the UI
- `too_hard` — task is valid but unreasonably difficult
- `agent_limitation` — agent capability gap, not a task problem

### `suggested_fix.target` (exactly one of):

- `reward_function` — patch the reward function
- `data_seed` — patch the data seed
- `task_removal` — remove the task entirely
- `none` — no fix applicable

### Schema:

```json
{
  "root_cause": "reward_bug",
  "explanation": "one paragraph description of the failure",
  "suggested_fix": {
    "target": "reward_function",
    "patch": "concrete modification to apply, or null"
  }
}
```
