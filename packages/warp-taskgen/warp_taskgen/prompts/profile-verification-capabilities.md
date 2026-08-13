You are profiling the **{site_name}** site's evaluation capabilities within a web benchmark.

Files are at `/workspace/benchmark`. Write output to `/workspace/output/`.

Produce `/workspace/output/VERIFICATION_CAPABILITIES.json`.

Document every evaluation method in the benchmark's harness. For each: type,
interface (parameters and return type), constraints on usage, and 2-3 concrete
examples from existing tasks.

## JSON schema

```json
[
  {{
    "eval_type": "<eval_type_from_harness>",
    "description": "What this eval type does.",
    "interface": {{ "params": {{"<key>": "<type>"}}, "returns": "boolean" }},
    "constraints": "Any constraints on usage.",
    "examples": [{{ "task_description": "...", "eval_config": {{}} }}]
  }}
]
```

## Validation rules

- Every `eval_type` must appear in the eval harness source code.
- Each example's `eval_config` must use valid fields for its evaluator type.
