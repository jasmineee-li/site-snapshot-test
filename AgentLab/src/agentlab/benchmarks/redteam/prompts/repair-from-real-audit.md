Audit the failed real-task run for the generated app in `{app_dir}`.

Task ID: `{task_id}`
Instruction: `{instruction}`

Current task result:
```json
{task_result_json}
```

Programmatic diagnosis, if any:
```json
{diagnosis_json}
```

You are operating inside the app directory. Inspect the app code, the task JSON, the verifier, the final state snapshot, and the saved experiment artifacts. Then:

1. Classify the failure as exactly one of:
   - `verifier_bug`
   - `environment_bug`
   - `impossible_task`
   - `ambiguous_instruction`
   - `agent_limitation`
   - `manual_review`
2. If the issue is fixable (`verifier_bug`, `environment_bug`, `ambiguous_instruction`), apply the minimal code or task fix directly in the app directory.
3. If the task is impossible, do not remove it yourself; just explain why it is impossible.
4. Do not edit unrelated files.
5. Return exactly one JSON object and nothing else:

```json
{{
  "classification": "environment_bug",
  "root_cause": "Short explanation of what failed.",
  "fix_applied": true,
  "fix_description": "Short explanation of what you changed."
}}
```
