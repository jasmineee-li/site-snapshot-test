Audit the latest real-task evaluation for app `{app_id}`.

Iteration: `{iteration}`
Threshold: `{threshold}`
Pass rate: `{pass_rate}` ({passed}/{total})

Artifacts:
- Results dir: `{results_dir}`
- Result summary: `{result_summary_path}`
- Failure summary: `{failure_summary_path}`

Failure summary payload:
```json
{failure_summary_json}
```

Review the failed tasks, verifier outputs, saved transcripts, final states, and server event logs. Apply only minimal fixes needed to improve the environment, tasks, or verifiers. Re-run the relevant sanity checks after any changes.
