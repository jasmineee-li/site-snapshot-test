<role_and_goal>
Explore this benchmark's codebase at `/workspace/benchmark`. Map only the
top-level benchmark structure needed by later profiling steps.
</role_and_goal>

<outputs>
Produce two files:

1. `/workspace/output/BENCHMARK_MANIFEST.json`
2. `/workspace/output/BENCHMARK_MANIFEST.md`
</outputs>

<scope>
Map the top-level structure. Do NOT go deep into individual application
internals.
</scope>

<exploration_plan>
Explore by reading:

1. Top-level README, setup scripts, Docker Compose files
2. Evaluation harness entry points and task loaders
3. Docker/deployment configs to identify distinct services
4. A sample of 3-5 existing task definitions to understand the task schema
</exploration_plan>

<json_output_contract>
`BENCHMARK_MANIFEST.json` must be valid JSON with this shape:

```json
{
  "benchmark_name": "WebArena Verified",
  "sites": [
    {
      "name": "shopping",
      "stack": "Magento 2 (PHP/MySQL)",
      "source_path": "src/webarena_verified/environments/",
      "data_seeding": { "mechanism": "docker_image", "paths": [] },
      "database": {"type": "mysql"}
    }
  ],
  "evaluation": {
    "harness_paths": ["src/webarena_verified/core/evaluation/"],
    "eval_types": ["AgentResponseEvaluator", "NetworkEventEvaluator"],
    "task_definition_format": "json",
    "task_definition_paths": ["assets/dataset/webarena-verified.json"]
  },
  "reset": { "mechanism": "env_ctrl_init", "per_task": true, "estimated_seconds": 5 }
}
```
</json_output_contract>

<self_check>
Before writing the output files, verify every path referenced in the JSON exists
in the filesystem. If a path does not exist, do not include it.
</self_check>
