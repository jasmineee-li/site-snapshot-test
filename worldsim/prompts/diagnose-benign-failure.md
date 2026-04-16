A browser-use agent failed a benign task. Determine the root cause.

## Inputs

- Task: `/workspace/task.json`
- Trajectory: `/workspace/trajectory/`
  - `history.json`: read this first
  - `screenshots/`: view selectively if action log is ambiguous
- Profile: `/workspace/profile/BENCHMARK_PROFILE.json`
- Sanity check result: `{sanity_result}` (`pass`, `fail`, or `unknown`)

## Procedure

1. If the sanity check result is `pass`, the reward and seed are already validated offline; `seed_bug` and `reward_bug` should be rare and require strong evidence.
2. If the sanity check result is `fail`, prioritize reward or solver bugs.
3. If the sanity check result is `unknown`, do not assume failure; infer from the task, trajectory, and reward evidence.
4. Agent actions look correct but reward rejected? -> reward edge case, fix.
5. Agent referenced nonexistent data? -> seed issue, fix.
6. Task impossible via UI? -> remove.
7. Otherwise -> agent limitation, keep.

## When NOT to propose `seed_bug`

Classify these as `agent_limitation` (or `too_hard`) and set `suggested_fix.target` to `none` with `patch: null`:

- Trajectory shows a login page, redirect to a sign-in URL, or `401`/`403` responses.
- Trajectory shows an expired session, "Please sign in" banner, or MFA/captcha challenge.
- The agent cannot find data that the task assumes should already exist because the logged-in account differs from the seeded one.

Authentication gaps are infrastructure / agent-capability problems, not task bugs. Do NOT propose an API patch that "creates a session", "logs in", or "adds a user" — those endpoints are almost never on the benchmark's allowlist and will be rejected by the host-side validator.

## When seed fixes are allowed

Propose `mechanism: "api"` or `mechanism: "form"` only when the exact endpoint (method + path) appears in `/workspace/profile/BENCHMARK_PROFILE.json` under `verification_capabilities[*].examples[*].eval_config.expected.url` (or under a top-level `seeding_endpoints` list if present). Use `mechanism: "state_push"` only when the existing task already uses `state_push` and only the state payload needs correction. If the profile does not list a safe HTTP endpoint and the task does not already use `state_push`, decline to emit a fix (`target: "none"`, `patch: null`). Do not propose `sql`.

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

Seed-fix example (form):

```json
{
  "root_cause": "seed_bug",
  "explanation": "The task references a draft note that was not created by the seed. The benchmark profile documents POST /issues/new as a supported endpoint.",
  "suggested_fix": {
    "target": "data_seed",
    "patch": {
      "mechanism": "form",
      "api_calls": [
        {
          "method": "POST",
          "path": "/issues/new",
          "body_form": {
            "title": "Release 2.0 tracker"
          }
        }
      ]
    }
  }
}
```

Seed-fix example (API — only when the endpoint is in `verification_capabilities` or `seeding_endpoints`):

```json
{
  "root_cause": "seed_bug",
  "explanation": "The task requires a forked project. `verification_capabilities` documents POST to /api/v4/projects/{id}/fork for this benchmark.",
  "suggested_fix": {
    "target": "data_seed",
    "patch": {
      "mechanism": "api",
      "api_calls": [
        {
          "method": "POST",
          "path": "/api/v4/projects/79/fork",
          "body": {
            "namespace_id": 2505
          }
        }
      ]
    }
  }
}
```

Declining a fix (auth gap, agent limitation, or no allowlisted endpoint):

```json
{
  "root_cause": "agent_limitation",
  "explanation": "The trajectory shows the agent was redirected to the login page and never authenticated. This is not a task bug.",
  "suggested_fix": {
    "target": "none",
    "patch": null
  }
}
```
