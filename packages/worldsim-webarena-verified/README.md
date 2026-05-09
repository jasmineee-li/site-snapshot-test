# warp-taskgen-webarena-verified

Separate installable adapter that runs canonical `webarena_verified` evaluation
for WARP Taskgen in its own Python environment.

Install this package in a dedicated virtualenv. From this directory:

```bash
uv sync --locked
```

From the repo root:

```bash
uv sync --directory packages/worldsim-webarena-verified --locked
```

Then point the core WARP Taskgen process at that interpreter with
`WARP_TASKGEN_WEBARENA_EVAL_PYTHON`. The legacy
`WORLDSIM_WEBARENA_EVAL_PYTHON` name remains accepted as a compatibility alias.

WARP Taskgen uses this package as a subprocess adapter for canonical WebArena
Verified tasks that carry a benchmark `task_id`. Task-id-less novel WARP Taskgen
tasks use WARP Taskgen's own reward evaluators instead of canonical benchmark
lookup. If `WARP_TASKGEN_WEBARENA_EVAL_PYTHON` is unset, the core process
auto-detects this repo-local `.venv` before trying an in-process install.
