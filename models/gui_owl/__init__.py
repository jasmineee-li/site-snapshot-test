"""mPLUG/GUI-Owl-1.5-32B-Think — Qwen3-VL-based CUA agent with reasoning.

The Think variant emits explicit reasoning traces, which lets us probe
eval-awareness representations at the end-of-reasoning token specifically
(stretch experiment). Otherwise served + probed identically to OpenCUA.

The base architecture differs from OpenCUA. This model reports
`model_type: qwen3_vl` and loads as `Qwen3VLForConditionalGeneration`.
OpenCUA reports `qwen2_5_vl`. `probes/model_loader._load_vl()` dispatches
on that field, so both load through the same entry point.
"""
