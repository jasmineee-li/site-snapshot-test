# Trace Inspection

Use this when a user asks why Phase 4 tasks complied, resisted, were unaware,
or failed under the eval-awareness iterator.

## First Command

Start with the compact inspector instead of hand-written `python -c` JSON dumps:

```bash
uv run warp-taskgen trace summary logs/<run>
```

For current r8a artifacts, run the same inspector remotely:

```bash
scripts/remote_trace_inspect.sh \
  --host-config configs/benchmark_hosts/r8a.local.yaml \
  --remote-dir /srv/warp-taskgen \
  --run logs/<run> -- summary
```

For a registered remote job, prefer job selectors so the wrapper derives the
state dir from metadata:

```bash
scripts/remote_trace_inspect.sh \
  --host-config configs/benchmark_hosts/r8a.local.yaml \
  --remote-dir /srv/warp-taskgen \
  --job-id <job_id> -- summary
```

Add narrow filters before increasing detail:

```bash
uv run warp-taskgen trace summary logs/<run> \
  --site gitlab \
  --action create_issue_note
```

## Progressive Disclosure

Use this order:

1. `summary` for counts, awareness buckets, iterator stops, top reward messages,
   and representative task ids.
2. `slice` for a compact task table with `--site`, `--action`, `--status`,
   `--outcome`, `--limit`, and `--fields`. For AgentLab runs, opt into fields
   such as `runner`, `agentlab_status`, `runtime_artifact_status`,
   `browser_instance_scope`, and `sidecar_status` before opening logs.
3. `task <task_id> --iterator` when the question is about rewrite attempts,
   stop reason, or why contrasts did not work.
4. `timeline <task_id>` when you need a compact event order across agent run,
   AgentLab sidecar/browser runtime, PVPO, rewards, final-state readback,
   TP/VEA, and iterator attempts.
5. `task <task_id> --refs` when you need artifact paths and a compact artifact
   manifest for `history.json`, `result.json`, Needham XML, AgentLab
   status/timeline logs, TP/VEA raw responses, or PVPO summary.
6. For visual row-by-row review, export
   `variant_trace_table.html`; AgentLab timeline summaries are surfaced there
   as report-only observability with raw JSONL paths for drill-down.
7. Only after that, open raw trajectory artifacts directly.

Machine consumption:

```bash
uv run warp-taskgen trace slice logs/<run> \
  --outcome resistant_unaware \
  --fields task_id,site,action,final_status,reward_msg,iterator_stop \
  --output json
```

For large slices, use `--output jsonl` with a bounded `--limit`, or pass
`--all` only when the caller genuinely needs every row.

Discoverable surfaces:

```bash
uv run warp-taskgen trace schema --output json
uv run warp-taskgen trace fields
uv run warp-taskgen trace examples
```

Useful filters include `--surface`, `--origin`, `--route`, `--pvpo`,
`--coverage-min`, `--tp`, `--vea`, `--awareness`, `--iterator-stop`,
`--attack-attempted`, `--attack-success`, `--state-success`, `--has-trace`,
`--missing-artifact`, and `--reward-contains`.

## Interpretation Checklist

For `resistant_unaware`, distinguish these cases:

- PVPO `max_coverage > 0`: the payload was encountered; inspect reward/action
  evidence and awareness metrics.
- TP/VEA aware is false: `unaware` means the observational awareness metrics did
  not classify the trajectory as eval-aware. It does not mean the payload was
  unreachable.
- `eval_awareness_iterator` exists with generated attempts and a
  `rewrite_limit_reached` stop: the iterator ran through its configured rewrite
  cap but did not find a compliant rewrite.
- `attack_attempted=false` or `attack_success=false`: the browser agent never
  executed the adversarial action or the action/readback reward did not prove it.
- `state_success=false`: request-level evidence may differ from final-state
  readback; report these separately.

For default eval-awareness iterator runs, inspect the iterator envelope on all
default-path rows, not only resistant rows:

- Baseline `complied` with an iterator envelope remains a baseline ASR success;
  use iterator TP/VEA/variant fields to analyze eval-awareness realism, not to
  remove the baseline compliance.
- `eval_awareness_iterator.status=skipped` should include a `skip_reason`, such
  as unresolved `injection_not_encountered` or `task_broke`.
- A TP-`Real` baseline can have zero rewrite attempts; that is a completed
  iterator envelope, not missing postprocessing.
- For second and later rewrites, inspect `prior_iteration_feedback` in the
  prompt/checkpoint context when available. It is compact prompt context, not a
  score: TP transition/explanation fields identify remaining eval-awareness
  cues, while PVPO/action/readback and contract-QA fields identify guardrails
  that the next rewrite must preserve or repair. VEA is intentionally absent.
- For cross-run summaries, prefer semantic iterator accounting fields when
  present: `rewrite_attempted`, `variant_evaluated`, `pre_browser_rejections`,
  `post_eval_rejections`, `schema_validation_failures`,
  `tp_regression_rejections`, and `contract_inapplicable_rejections`. These
  explain where rewrite attempts were filtered better than old budget counters
  alone.

For AgentLab runs, a safe browser-step timeout retry may leave
`agentlab_infra_retries.json` in the task trace. The retry is only valid when the
timed-out attempt had no mutating network evidence; otherwise treat the timeout
as a task/runtime failure and inspect the sidecar status plus network artifacts
before counting the row.

For eval-awareness iterator rewrite failures, do not stop at the top-level
`iterator_stop`. Open `task <task_id> --iterator` or the raw
`eval_awareness_iterator_checkpoint.json` and distinguish:

- `no_slot_delta` / `unchanged_seed`: the model produced no meaningful payload
  change.
- `render_validation_failed` or `contract_qa_failed`: the model changed slots or
  payload text, but host rendering/QA rejected it before browser evaluation.
- `concealment_contract_drift`: the rewrite tried to change the admitted
  treatment cell, for example removing a required ` ```system ` fence.
- `required_token_visibility_drift`: required concealed tokens moved into
  visible prose or disappeared from the concealed region.
- `action_witness_drift`: the rewrite no longer preserved the concrete
  adversarial action witness.
- `rewrite_inapplicable_trajectory_only`: cue diagnosis found task/trajectory
  context but no mutable payload-local cue, so a rewrite would not test the
  intended intervention.
- `rewrite_inapplicable_insufficient_causal_evidence`: cue diagnosis failed or
  did not identify a reliable mutable payload cue.
- `tp_regression`: the browser-evaluated variant moved Transcript Purpose in
  the wrong direction, such as from `Capability Evaluation` to
  `Alignment Evaluation`; keep it as evidence, but do not treat it as a valid
  iterator improvement.

Do not paste full `signals`, full `history.json`, prompts, screenshots, request
bodies, cookies, auth headers, or raw stderr into the conversation by default.
Emit compact counts, task ids, artifact paths, redaction mode, and the next
exact command.
