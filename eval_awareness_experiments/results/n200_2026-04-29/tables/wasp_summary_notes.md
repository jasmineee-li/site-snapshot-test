# WASP N200 Summary Tables

- `asr_from_judge_metadata` is computed from `metadata.attack_success` in `trajectory_awareness_results.jsonl`. For this WASP run it is not populated, so `asr_n=0` and the value is blank.
- `stage1_attack_success_rate` is computed from AgentLab `attack_summary_info.json`. For WASP this appears to be the stage-1 placeholder/native AgentLab field, not the WASP post-hoc ASR classifier described in `wasp_post_hoc_evaluator.py`.
- `user_task_pass_rate` is from `metadata.task_passed` in the trajectory-awareness rows.
- `verbalized_*` columns summarize `verbalized_awareness` headline classifications.
- `purpose_*` columns summarize `purpose_continue_5q` headline classifications and its per-variant shares.
