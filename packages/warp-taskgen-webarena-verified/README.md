# warp-taskgen-webarena-verified

Separate installable adapter that runs canonical `webarena_verified` evaluation
for WARP Taskgen in its own Python environment.

Install this package in a dedicated virtualenv. From this directory:

```bash
uv sync --locked
```

From the repo root:

```bash
uv sync --directory packages/warp-taskgen-webarena-verified --locked
```

Then point the core WARP Taskgen process at that interpreter with
`WARP_TASKGEN_WEBARENA_EVAL_PYTHON`.

WARP Taskgen uses this package as a subprocess adapter for canonical WebArena
Verified tasks that carry a benchmark `task_id`. Task-id-less novel WARP Taskgen
tasks use WARP Taskgen's own reward evaluators instead of canonical benchmark
lookup. If `WARP_TASKGEN_WEBARENA_EVAL_PYTHON` is unset, the core process
auto-detects this repo-local `.venv` before trying an in-process install.

The adapter protocol is process-local JSON: root WARP Taskgen invokes
`python -m warp_taskgen_webarena_verified.evaluate`, writes one request JSON object
to stdin, and reads one result JSON object from stdout. The console script
`warp-taskgen-webarena-eval` exposes the same protocol for smoke checks.
