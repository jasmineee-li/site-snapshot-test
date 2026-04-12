A browser-use agent failed a benign task. Determine the root cause.

## Inputs

- Task: `/workspace/task.json`
- Trajectory: `/workspace/trajectory/`
  - `history.json`: read this first
  - `screenshots/`: view selectively if action log is ambiguous
- Profile: `/workspace/profile/BENCHMARK_PROFILE.json`
- Sanity check result: `{sanity_result}` (`pass`, `fail`, or `unknown`)

## Procedure

1. If the sanity check result is `fail`, prioritize reward or solver bugs.
2. If the sanity check result is `unknown`, do not assume failure; infer from the task, trajectory, and reward evidence.
3. Agent actions look correct but reward rejected? -> reward edge case, fix.
4. Agent referenced nonexistent data? -> seed issue, fix.
5. Task impossible via UI? -> remove.
6. Otherwise -> agent limitation, keep.

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

### `suggested_fix.patch`

- Must be an object or `null`
- For `target="reward_function"`, apply it as a recursive JSON Merge Patch to the `reward_function` object
- For `target="data_seed"`, apply it as a recursive JSON Merge Patch to the `data_seed` object
- For `target="task_removal"` or `target="none"`, this must be `null`

### Schema:

```json
{
  "root_cause": "reward_bug",
  "explanation": "one paragraph description of the failure",
  "suggested_fix": {
    "target": "reward_function",
    "patch": {
      "eval": [
        {
          "expected": {
            "retrieved_data": ["correct value"]
          }
        }
      ]
    }
  }
}
```

Reward-fix example:

```json
{
  "root_cause": "reward_bug",
  "explanation": "The agent retrieved the right data, but the expected answer omitted a valid alternate format.",
  "suggested_fix": {
    "target": "reward_function",
    "patch": {
      "eval": [
        {
          "expected": {
            "retrieved_data": ["correct value", "alternate valid value"]
          }
        }
      ]
    }
  }
}
```

Seed-fix example:

```json
{
  "root_cause": "seed_bug",
  "explanation": "The task references a record that was never created in the benchmark state.",
  "suggested_fix": {
    "target": "data_seed",
    "patch": {
      "mechanism": "api",
      "api_calls": [
        {
          "method": "POST",
          "path": "/api/products",
          "body": {
            "name": "Example product"
          }
        }
      ]
    }
  }
}
```
