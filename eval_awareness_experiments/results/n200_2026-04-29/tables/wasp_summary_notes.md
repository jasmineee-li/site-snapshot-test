# WASP N200 Summary Tables

- `asr_rate` is the WASP post-hoc action-only ASR classifier when `asr_source=wasp_post_hoc_action_classifier`; otherwise it falls back to boolean `metadata.attack_success` if present.
- `user_task_pass_rate` is from `metadata.task_passed` in the trajectory-awareness rows.
- `verbalized_*` columns summarize `verbalized_awareness` headline classifications.
- `purpose_*` columns summarize `purpose_continue_5q` headline classifications. The markdown table shows the headline Alignment/Capability/Other breakdown and omits the broad per-variant `any_eval_share`.
