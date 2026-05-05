# OpenCUA-72B Eval-Awareness Steering Handoff

Date: 2026-05-05

## Current Goal

We want a smoke-testable steering path for OpenCUA-72B on WASP that follows the
Hazra/Goodfire-style intervention pattern:

- Steer user message content tokens during prompt prefill.
- Do not steer system prompt tokens.
- Do not steer chat-template scaffolding or special/formatting tokens.
- During generation, leave the first generated sentence unsteered.
- After the first generated sentence, steer generation tokens.
- Apply the same steering vector simultaneously at multiple layers.

For the current eval-awareness experiment, the intended layer set is the top SAD
all-data layers:

```text
[13, 9, 11, 17, 19, 20]
```

Each layer should receive `alpha=0.6`. Probe sign is currently interpreted as:

- positive alpha: steer toward eval-awareness
- negative alpha: steer away from eval-awareness

## Code Added

The steering-specific code is in `probes/steering.py`.

New pieces:

- `TokenGatedSteeringState`
- `token_gated_steering_hook`
- `chat_role_content_token_positions`
- `generate_with_token_gated_steering`

The implementation tries to:

- identify content token positions for selected chat roles from a fully rendered
  chat template;
- exclude special tokens, whitespace-only tokens, and obvious template markers;
- apply additions only to selected prompt positions during the full prefill
  forward pass;
- apply additions to one-token cached generation forwards only after
  `generation_enabled=True`;
- support multi-layer steering with the probe vectors.

`probes/model_loader.py` was also changed so `AutoModelForCausalLM` is tried
first, even for the Qwen2.5-VL/OpenCUA model type. This matters because the HF
OpenAI server loads OpenCUA through the remote-code class
`OpenCUAForConditionalGeneration`; loading through the generic VL fallback
produced worse behavior in steering smoke tests.

## Smoke Test Results

The helper exercises the intended token gates, but generation is currently bad.

With a real saved WASP prompt from `webarena.1095`, using the OpenCUA remote-code
class and the processor-rendered chat template:

```text
hf class OpenCUAForConditionalGeneration
layers [13, 9, 11, 17, 19, 20]
rendered chars 14793
input tokens 4497
n_user_prompt_positions 3668

unsteered:              <th<|im_end|>
toward_eval_awareness:  <th<|im_end|>
away_from_eval_awareness:<th<|im_end|>
```

For all three conditions:

```text
first_sentence_end_token_index=None
generation_steering_enabled=False
```

So this is not yet demonstrating steering. It is showing that the manual
token-by-token generation path is not matching the normal model/server
generation path.

Synthetic prompts had a similar issue, usually generating only `I<|im_end|>`,
`No<|im_end|>`, or `<th<|im_end|>`.

## Important Observation

The baseline WASP transcripts from the real no-steering server run do contain
normal sentence-like assistant generations.

Earlier transcript inspection found:

- n=5 smoke run: 5 of 9 step files had sentence punctuation in assistant output;
  assistant outputs averaged roughly 3k characters.
- cancelled full partial run: 18 of 33 step files had sentence punctuation;
  assistant outputs averaged roughly 1.4k characters.

This means the bad `<th<|im_end|>` output is probably a harness/generation-path
bug, not evidence that OpenCUA cannot produce sentences on WASP.

## WASP Turn Structure

WASP/BrowserGym is multi-turn/multi-action. Steering should be applied on every
LLM call, not only the first task prompt.

For each model call:

- steer user-message content tokens in that rendered prompt;
- do not steer the system message or template scaffolding;
- let generation start normally;
- after the first generated sentence, steer generated token forwards;
- repeat the same rule for retry/correction prompts if the agent makes another
  LLM call.

## Likely Cause

The current broken path is likely caused by a mismatch between the manual cached
decoding loop and the exact generation stack used by OpenCUA in normal server
inference.

Possibilities to check:

- OpenCUA may require processor/model kwargs beyond `input_ids` for correct
  text-only generation.
- The manual loop may be bypassing model-specific generation setup that
  `model.generate(...)` performs.
- The prompt may need the exact `AutoProcessor.apply_chat_template(...)`
  encoding path, not tokenizer-only rendering/encoding. The latest smoke used
  processor-rendered prompt and input IDs, but still used manual forwards.
- The model may expect particular stopping/eos handling or generation config.
- Hooks may be fine, but the unhooked manual generation baseline is already
  broken, which should be fixed before judging steering.

## Recommended Next Steps

1. Reproduce normal output using the same locally loaded model without hooks.
   Start with `model.generate(...)` using the exact `AutoProcessor` output.

2. Once `model.generate(...)` produces normal WASP-style text, attach hooks
   without any token gating or with `alpha=0.0` and confirm output is still
   normal.

3. Move from the manual cached loop to a `generate(...)`-compatible steering
   path if possible. A practical route is:

   - use hooks during `model.generate(...)`;
   - steer selected prefill tokens by detecting the full prompt sequence length;
   - steer generation token forwards after a state flag is enabled;
   - implement first-sentence skipping using either a streamer/logits processor
     or a custom generation loop that exactly mirrors `generate(...)`.

4. If `generate(...)` cannot expose enough control, move steering into
   `models/common/hf_openai_server.py` so the same server path used by WASP can
   be instrumented directly.

5. Keep the first validation tiny:

   - one saved real WASP prompt;
   - unsteered, `alpha=+0.6`, `alpha=-0.6`;
   - layers `[13, 9, 11, 17, 19, 20]`;
   - confirm output is coherent before measuring probe/ASR.

## Overnight No-Steering WASP Run

The no-steering WASP run was relaunched in tmux:

```text
tmux session: opencua72_wasp_full_resume
run_id: opencua72_wasp_attacked_full_20260505_052527
gpus: 0,3,4,5
```

It resumed with `RELAUNCH_INCOMPLETE=1` and found 150 incomplete experiments.
This run is intentionally unsteered and uncued. The launcher will export
transcripts, run probe scoring, run WASP post-hoc ASR classification, and write a
merged probe/reward/ASR table after the benchmark run finishes.

## Local AgentLab Changes

There are local edits in the nested checkout `external_benchmarks/AgentLab`, but
that whole directory is ignored by the root repo, so the root commit/push will
not carry those files. They were left as local nested-repo changes rather than
force-tracking ignored third-party files in this repo.

Files changed there:

- `src/agentlab/experiments/exp_utils.py`: re-enabled a per-episode timeout for
  sequential runs so stuck browser tasks can be marked errored instead of
  hanging forever.
- `src/agentlab/experiments/study.py`: preserves subclassed BrowserGym env args
  such as DoomArena's attacked env args instead of coercing away extra fields.
- `src/agentlab/llm/chat_api.py`: supports `:thinking` model suffixes and
  preserves OpenRouter `message.reasoning`.
- `src/agentlab/llm/litellm_api.py`: avoids concatenating `None` content while
  preserving reasoning text.
- `src/agentlab/llm/llm_utils.py`: loads HF tokenizers with
  `trust_remote_code=True`, handles non-string HTML extraction inputs, and
  allows assistant messages to carry `reasoning`.
