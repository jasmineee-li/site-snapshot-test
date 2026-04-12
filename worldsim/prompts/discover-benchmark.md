Explore this benchmark's codebase at `/workspace/benchmark`. Produce two files:

1. `/workspace/output/BENCHMARK_MANIFEST.json`
2. `/workspace/output/BENCHMARK_MANIFEST.md`

Map the top-level structure. Do NOT go deep into individual application internals.

Explore by reading:

1. Top-level README, setup scripts, Docker Compose files
2. Evaluation harness entry points and task loaders
3. Docker/deployment configs to identify distinct services
4. A sample of 3-5 existing task definitions to understand the task schema

## JSON schema

```json
{
  "benchmark_name": "WebArena Verified",
  "sites": [
    {
      "name": "shopping",
      "stack": "OpenCart (PHP)",
      "source_path": "environment_docker/shopping/",
      "data_seeding": { "mechanism": "sql", "paths": ["docker/shopping/setup/init.sql"] },
      "database": {"type": "mysql", "version": "5.7"}
    }
  ],
  "evaluation": {
    "harness_paths": ["evaluation_harness/evaluators.py"],
    "eval_types": ["url_exact_match", "html_match", "db_query_match", "string_match"],
    "task_definition_format": "json",
    "task_definition_paths": ["config_files/"]
  },
  "reset": { "mechanism": "docker_restart_and_snapshot", "per_task": true, "estimated_seconds": 20 }
}
```

Every path referenced in the JSON must exist in the filesystem. If a path does not exist, do not include it.
