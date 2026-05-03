# worldsim-webarena-verified

Separate installable adapter that runs canonical `webarena_verified` evaluation
for WorldSim in its own Python environment.

Install this package in a dedicated virtualenv:

```bash
uv sync --locked
```

Then point the core WorldSim process at that interpreter with
`WORLDSIM_WEBARENA_EVAL_PYTHON`.

WorldSim uses this package as a subprocess adapter for canonical WebArena
Verified tasks that carry a benchmark `task_id`. Task-id-less novel WorldSim
tasks use WorldSim's own reward evaluators instead of canonical benchmark
lookup. If `WORLDSIM_WEBARENA_EVAL_PYTHON` is unset, the core process
auto-detects this repo-local `.venv` before trying an in-process install.
