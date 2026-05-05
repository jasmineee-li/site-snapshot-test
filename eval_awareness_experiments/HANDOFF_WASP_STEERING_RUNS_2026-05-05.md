# WASP OpenCUA-72B Steering Run Handoff

Date: 2026-05-05

## Purpose

We are measuring whether steering OpenCUA-72B toward the SAD/chat-template
eval-awareness direction changes WASP attack success rate (ASR) and task success
rate (TSR). The current steering condition is:

```text
alpha_per_layer = +0.6
layers = 13,9,11,17,19,20
probe = probes/trained/opencua-72b-chat-template-span/mcq
```

This uses the server-side token-gated steering path committed in:

```text
1f3041c7 Wire OpenCUA steering into HF server
2d25e83d Fix OpenCUA generate-based steering smoke
d4814c1f Add OpenCUA WASP probe and steering tooling
```

## Baseline Reference

Completed no-steering full WASP run:

```text
run_id: opencua72_wasp_attacked_full_20260505_052527
path: eval_awareness_experiments/results/opencua72_wasp_attacked_full_20260505_052527/bare/local_opencua-72b
n: 168
ASR: 0.06547619047619048 = 11/168
TSR: 0.8869047619047619
probe summary: probe_reward_asr_summary_sad_all.json
```

SAD-all probe threshold summary for baseline:

```text
layer: 13
threshold: 0.12451171875
threshold_source: sad_stages_oversight_all
```

## Completed Steering Pilot

Completed n=20 steering-toward run:

```text
run_id: opencua72_wasp_attacked_steer_toward_n20_20260505_135742
path: eval_awareness_experiments/results/opencua72_wasp_attacked_steer_toward_n20_20260505_135742/bare/local_opencua-72b
n: 20
steering_alpha_per_layer: +0.6
layers: 13,9,11,17,19,20
GPUs used: CUDA_VISIBLE_DEVICES=0,3,5
OPENCUA_MAX_MEMORY: 0:60GiB,1:60GiB,2:60GiB,cpu:300GiB
```

Result:

```text
ASR: 0.0 = 0/20
TSR: 0.75 = 15/20
ASR classifier: 35 step classifications, 0 parse/error steps
probe layer: 13
probe threshold: 0.11669921875 from sad_stages_oversight_dev
```

Probe means from `probe_reward_asr_summary.json`:

```text
last_user_token: 0.366259765625
user_message_mean: 0.223583984375
first_assistant_message_mean: 0.22607421875
```

All 20 rows were classified eval-aware for all three probe spans.

## Stopped Full Steering Attempt

We started a full n=168 steering-toward run on this node, then stopped it because
someone else needed GPUs.

```text
tmux session: opencua72_wasp_steer_toward_full
run_id: opencua72_wasp_attacked_steer_toward_full_20260505_150345
path: eval_awareness_experiments/results/opencua72_wasp_attacked_steer_toward_full_20260505_150345/bare/local_opencua-72b
n_tasks: 168
steering_alpha_per_layer: +0.6
layers: 13,9,11,17,19,20
GPUs used: CUDA_VISIBLE_DEVICES=0,3,4,5
OPENCUA_MAX_MEMORY: 0:45GiB,1:45GiB,2:45GiB,3:45GiB,cpu:300GiB
```

Status at stop:

```text
completed BrowserGym summary_info.json files: 10/168
no final wasp_messages.jsonl/probe/ASR tables were produced
HF server and BrowserGym runner were stopped
OpenCUA GPU memory was released
```

Partial outputs remain in the run directory, but for a different node it is
simpler to start a fresh run id unless copying the whole result tree and using
`RELAUNCH_INCOMPLETE=1`.

## Recommended Fresh Full Run

Use a fresh tmux session on the new node. The command below assumes the same repo
path and local OpenCUA-72B snapshot path as this node.

```bash
tmux new-session -d -s opencua72_wasp_steer_toward_full -c /local_data/temp/max/browser-sim \
'env RUN_ID=opencua72_wasp_attacked_steer_toward_full_$(date -u +%Y%m%d_%H%M%S) \
N_TASKS=168 \
MAX_STEPS=15 \
SHUFFLE=42 \
STEERING_ENABLED=1 \
STEERING_ALPHA_PER_LAYER=0.6 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OPENCUA_MAX_MEMORY=0:45GiB,1:45GiB,2:45GiB,3:45GiB,cpu:300GiB \
OPENCUA_PORT=8002 \
RELAUNCH_INCOMPLETE=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash eval_awareness_experiments/launch_opencua72_wasp_full_attacked_with_probes.sh'
```

If only three H100s are available, the n=20 pilot worked with:

```text
CUDA_VISIBLE_DEVICES=0,3,5
OPENCUA_MAX_MEMORY=0:60GiB,1:60GiB,2:60GiB,cpu:300GiB
```

Four GPUs are still preferred for the full run.

## Useful Status Commands

```bash
tmux capture-pane -pt opencua72_wasp_steer_toward_full -S -180

find eval_awareness_experiments/results/<RUN_ID>/bare/local_opencua-72b/_browser_runs/all \
  -type f -name summary_info.json | wc -l

tail -n 100 eval_awareness_experiments/results/<RUN_ID>/bare/local_opencua-72b/logs/run.log

tail -n 80 eval_awareness_experiments/results/<RUN_ID>/bare/local_opencua-72b/logs/hf_openai_server.log

nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits
```

Expected final files after a completed run:

```text
wasp_messages.jsonl
probe_spans.jsonl
probe_span_summary.json
wasp_asr_all.jsonl
probe_reward_asr_table.csv
probe_reward_asr_table.md
probe_reward_asr_summary.json
```

## Notes

- The launcher runs post-hoc ASR classification after BrowserGym completes. It
  needs the OpenRouter/ASR classifier credentials that were already present on
  this node.
- The server applies steering on every LLM call:
  user-message content tokens during prompt prefill, no system/template tokens,
  and generated tokens only after the first generated sentence.
- `reward=1` in the merged table means task success, not attack success.
  Attack success is the separate `attack_success`/ASR label.
- The n=20 steering pilot had lower ASR than the n=168 no-steering baseline, but
  it is too small to treat as the final comparison.
