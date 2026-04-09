Generate harder real tasks for app `{app_id}` by studying prior browser-agent behavior.

Round: `{round_number}`
Target new tasks this round: `{tasks_per_round}`
Results root: `{results_dir}`

References:
- Real task design guide: `{real_task_design_guide_path}`
- Verifier sanity-check guide: `{verifier_sanity_check_guide_path}`

Observed agent behavior:
```text
{hardening_analysis}
```

Requirements:
1. Append new hard tasks to `real-tasks.json`; never edit or remove existing task IDs.
2. Add any new verifier files or solver/sanity-check updates needed for the appended tasks.
3. Keep the tasks realistic and UI-completable.
4. Prefer tasks that exploit observed weak spots from the trajectories rather than artificial trick tasks.
5. Return concise plain text describing the new task IDs you added.
