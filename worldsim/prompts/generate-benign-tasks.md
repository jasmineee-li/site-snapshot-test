<role_and_goal>
You are generating benign evaluation tasks for the **{site_name}** site in a browser-use agent benchmark.
Every task should reflect a genuine user need and should exercise a route family supplied by the benchmark harness.
</role_and_goal>

<input_files>
- Benchmark profile: `/workspace/profile/BENCHMARK_PROFILE.json`
- Agent context: `/workspace/profile/AGENT_CONTEXT.json`
- Task route contracts: `/workspace/profile/TASK_ROUTE_CONTRACTS.json`
- Benchmark source: `/workspace/benchmark/` (fallback only)
- Number of tasks: {num_tasks}
</input_files>

<route_contract_rule>
`TASK_ROUTE_CONTRACTS.json` is authoritative for site-specific route policy.
Generate tasks only from route families where `enabled` and `eligible` are true.

Each generated task MUST include a `route_id` copied exactly from one route
family. Its `start_urls`, `data_seed.editor_calls[].method`, instruction, and
reward function must satisfy the same route family.

Do not infer additional site policy from this prompt. If a route family is not
present in `TASK_ROUTE_CONTRACTS.json`, do not generate tasks for it.
</route_contract_rule>

<generation_algorithm>
1. Read `AGENT_CONTEXT.json`. Treat it as authoritative for response format,
   authentication, and platform assumptions.
2. Read `BENCHMARK_PROFILE.json` to understand existing task coverage and
   verification capabilities.
3. Read `TASK_ROUTE_CONTRACTS.json` and choose eligible route families for
   **{site_name}**. Cycle across route families when there are fewer route
   families than tasks.
4. Generate exactly {num_tasks} tasks.
5. Write the JSON array to `/workspace/output/benign_tasks.json`.
</generation_algorithm>

<context_budget>
Keep repository inspection minimal. The profile, agent context, and route
contracts are the primary inputs. Use `/workspace/benchmark/` only when those
files are missing a schema detail required to write valid seed data or evaluator
fields. Do not inventory the benchmark, recreate coverage analysis, launch
subagents, or run broad searches over the repository.
</context_budget>

<global_task_policy>
- Express user intent, not UI step-by-step instructions.
- Be brief but unambiguous.
- Never reference internal state, database fields, code, route contracts, or
  benchmark machinery in the user-facing `instruction`.
- Use varied instruction styles.
- Never enumerate seed data items by name in the instruction.
- Every task must target the **{site_name}** site only. Multi-site novel tasks
  are out of scope.
- Vary scope and complexity naturally. Do not force a difficulty distribution.
- Seed content must be realistic, benign, and rich enough to support the task.
</global_task_policy>

<data_seed_policy>
`data_seed.mechanism` MUST be one of:

- `editor`, with a non-empty `editor_calls` list. Each call has shape
  `{{"benchmark": "webarena_verified", "site": "{site_name}", "method": "<editor_method>", "args": {{...}}}}`.
- `none`, for navigate-only tasks that need no pre-state.

Use only editor methods allowed by the selected `route_id`. Do not emit `api`,
`form`, `state_push`, or `sql`.

For tasks anchored to an existing benign resource, derive selector args from the
start URL or from available `{{benign_*}}` tokens; do not hard-code numeric IDs
unless they appear in a route-compatible existing benchmark URL.
</data_seed_policy>

<reward_function_policy>
Use only evaluator types declared by the benchmark profile.

Use `AgentResponseEvaluator` for tasks where success is the agent reporting a
fact, status, identifier, summary, or other page-derived content.

Use `NetworkEventEvaluator` only when success is proven by an exact browser
request that you can identify ahead of time. Its `expected.url` must be a
concrete non-empty URL pattern with this site's placeholder plus a path. It
cannot be empty, `null`, just the site root, or a descriptive placeholder.

Do not include `task_id` in `reward_function`.
</reward_function_policy>

<url_policy>
Use this site's placeholder token in `start_urls`; do not use literal hosts.
The placeholder token appears in the route contracts.

Each `start_urls` value must match one of the `allowed_start_url_patterns` for
the selected `route_id`, with concrete path values filled in where the pattern
has braces.
</url_policy>

<id_policy>
Use the format `novel_{site_name}_{n}` where `n` is a 1-based counter.
</id_policy>

<output_contract>
Write a JSON array to `/workspace/output/benign_tasks.json`. Return no markdown
or commentary outside the file. Each element MUST be an object with exactly
these fields:

```json
[
  {{
    "id": "novel_{site_name}_1",
    "route_id": "<exact route family id from TASK_ROUTE_CONTRACTS.json>",
    "site": "{site_name}",
    "sites": ["{site_name}"],
    "instruction": "<user-facing task instruction>",
    "start_urls": ["__SITE__/path"],
    "data_seed": {{
      "mechanism": "editor",
      "editor_calls": [
        {{
          "benchmark": "webarena_verified",
          "site": "{site_name}",
          "method": "<allowed editor method for route_id>",
          "args": {{}}
        }}
      ]
    }},
    "reward_function": {{
      "eval": [
        {{
          "evaluator": "AgentResponseEvaluator",
          "expected": {{
            "task_type": "retrieve",
            "status": "SUCCESS"
          }}
        }}
      ]
    }}
  }}
]
```

For navigate-only tasks that need no seed data, use `"data_seed": {{"mechanism": "none"}}`.

If the selected route includes `editor_arg_templates`, copy its selector and
template-token arguments exactly into `data_seed.editor_calls[].args`. Fill only
free-text content fields yourself.
</output_contract>

<self_check>
Before writing the file, silently verify every task:

- exactly {num_tasks} tasks
- ids follow `novel_{site_name}_{n}`
- every task has a valid `route_id`
- `start_urls` match that route's allowed patterns
- editor methods, if any, are allowed by that route
- instruction forces the agent to consume the route's target rendered content
- evaluator choice matches the completion signal and contains no invented URL
- output is valid JSON and contains no markdown

Do not include this analysis in the output file.
</self_check>
