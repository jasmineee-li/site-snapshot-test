# Needham-Native Transcript Metrics

This branch uses one shared Needham XML transcript contract for both third-person
eval-awareness metrics:

- `verbalized_awareness` sends the shared XML to the VEA classifier.
- `purpose_xml` sends the same shared XML to Apollo-style Transcript Purpose:
  Step 1 guesses why the interaction occurred, Step 2 classifies that guess into
  `Alignment Evaluation`, `Capability Evaluation`, `Other Evaluation`, or `Real`.

The important methodological choice is that screenshots are not part of either
metric. A screenshot can be useful for encounter detection in browser benchmarks,
but Transcript Purpose is answering a text-transcript question: does the
interaction transcript itself look like an evaluation? Mixing image realism into
that classifier would make it a different measurement.

## Serializer Contract

`eval_awareness_experiments/judges/transcript_utils.py::transcript_to_needham_xml`
is the canonical serializer. It intentionally follows the upstream Needham XML
shape:

- messages render as `<message role="...">` blocks;
- tool results may carry the upstream comma-form attribute
  `<message role="tool", function="...">`;
- assistant tool calls render inside `<tool_calls><tool_call ...>...`;
- transcript text and tool-call arguments are XML-escaped.

Hidden reasoning from local `Transcript` objects is folded into assistant message
text rather than emitted as a custom `<reasoning>` tag. This keeps the XML close
to the upstream grammar while preserving the model-output signal for VEA.

## Compatibility Notes

The older role-as-tagname XML serializer (`transcript_to_xml`) remains available
for binary/probability judges. `purpose_continue` and `purpose_continue_5q`
still measure a different replayed-chat protocol rather than the shared inline
Needham XML protocol:

- `purpose_continue` now uses Apollo Research's verbatim continue-format
  `<follow_up_question>` prompt.
- `purpose_continue_5q` runs only the five original purpose-question prompts
  (`prompt_01` through `prompt_05`) for the external benchmark. The additional
  local probe JSON files remain on disk but are not loaded by this judge.

Both continuation judges keep the refusal-aware grader for model reflections.
