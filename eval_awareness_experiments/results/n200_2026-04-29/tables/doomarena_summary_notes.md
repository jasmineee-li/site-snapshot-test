# DoomArena N200 Partial Summary Tables

- These tables summarize the current partial N200 DoomArena result snapshot.
- `n_tasks` is unique judged task/sample IDs after de-duplicating repeated judge rows by `(sample_id, experiment_type)`.
- `asr_rate` is `metadata.attack_success` from trajectory-awareness rows.
- `user_task_pass_rate` is `metadata.task_passed` from trajectory-awareness rows.
- `verbalized_*` columns summarize `verbalized_awareness` headline classifications.
- `purpose_*` columns summarize `purpose_continue_5q` headline classifications and per-variant shares.
- `status` uses strict expected coverage: 50 tasks each for reddit, shopping, shopping_admin, and gitlab.
