# Agent Handoff: Implement WorldSim v5 Phases 0–4

You are implementing the WorldSim v5 adversarial evaluation pipeline. The foundation (Modal sandbox, auth, Browser Use, CLI, .env) is proven and working. Your job is to fill in the five phase modules.

## Step 0: Orient (do this BEFORE writing any code)

Launch up to 3 Explore subagents in parallel to read:

**Subagent 1 — spec + principles:**
- `CLAUDE.md` (non-negotiable principles, what NOT to do)
- `docs/worldsim-v5-technical-specifcation.md` (the source of truth — every phase, prompt, schema, code pattern)

**Subagent 2 — working primitives:**
- `worldsim/modal_sandbox.py` (the one function every phase calls: `run_claude_in_sandbox`)
- `scripts/smoke_modal.py` (working end-to-end example)
- `worldsim/prompts/*.md` (ready-to-use prompts for each Claude Code step)

**Subagent 3 — current state + skeletons:**
- `docs/current_progress.md` (what's proven, what's skeleton, what's blocked)
- `worldsim/phases/phase_0_recon.py` through `phase_4_adversarial.py` (stubs with TODOs pointing at spec sections)
- `worldsim/eval_worker_pool.py`, `worldsim/seeding.py`, `worldsim/rewards.py` (skeleton utilities)

Do NOT start implementing until all three subagents report back.

## What's proven (don't re-test, don't modify)

- `run_claude_in_sandbox` works end-to-end (Modal `test` environment, `IS_SANDBOX=1`, uses `claude-agent-sdk` via `_sdk_runner.py`, returns `_summary` with cost/token/session data)
- Auth: `_build_claude_secrets()` handles OAuth > OpenRouter > API key. All 9 scenarios pass.
- Browser Use: `BrowserSession` + `session.kill()` for teardown. Never `.close()`.
- `.env` loaded at startup via `load_dotenv()`. Shell exports win.

## Implementation order

```
Phase 0a  →  0b  →  0c  →  1A  →  2  →  3  →  4
  ↑ unblocked now (needs only vendors/webarena-verified/ on disk)
                                    ↑ blocked on running WebArena
```

For each phase:
1. Read the spec section (the stub tells you which one)
2. Read the corresponding prompt in `worldsim/prompts/`
3. Implement in the existing stub file — don't create new modules unless necessary
4. Test it runs end-to-end (for 0-2: against `vendors/webarena-verified/`; for 3-4: if no WebArena, implement with clear stubs and document what's needed to test)
5. Commit, then move to the next phase

### Phase 0a — Benchmark Discovery
Single Modal sandbox. Input: full benchmark codebase. Output: `BENCHMARK_MANIFEST.json`. Use `load_prompt("discover-benchmark")`. Validate every path in the manifest exists on disk.

### Phase 0b — Sandbox Filesystem Mapping
Pure local Python, no LLM. Input: manifest from 0a. Output: per-site file lists. Implement `compute_sandbox_maps()` per the spec. Deterministic.

### Phase 0c — Per-Site Profiling
N parallel Modal sandboxes via `asyncio.gather`. Input: sandbox maps from 0b. Output: `BENCHMARK_PROFILE_{site}.json` per site. Use `load_prompt("profile-site")`. Validate `source_field` cross-references data model.

### Phase 1 Mode A — Wrap Existing Tasks
Read task definitions from the benchmark's `task_definition_paths` (discovered in 0a). Reformat into the benign task bundle schema from the spec. No LLM needed.

### Phase 2 — Injection Generation
One Modal sandbox per site. Input: benign tasks + profiles. Output: `adversarial_tasks.json`. Use `load_prompt("generate-injections")`.

### Phase 3 — Contract Validity Gate
Agent-free. Schema-validate every benign contract and every adversarial task's benign reference; emit `phase_3/contracts.json` with per-entry `validity_status` / `origin`. No live instances required.

### Phase 4 — Adversarial Evaluation
Initial run + two sequential gates (ecological validity, attack effectiveness). Refused attacks enter strategy variation: judge sandbox → variant generation (parallel) → variant evaluation (parallel). Implement `run_adversarial_task()`, `probe_ecological_validity()`, `run_strategy_variation()` matching the spec. **Requires running WebArena.**

## Rules

- **Spec is truth.** When it gives literal code, use it. When it gives a JSON schema, validate against it.
- **`run_claude_in_sandbox` for every Claude Code step.** Never shell out to `claude` directly.
- **Never `import` from `AgentLab/`.** Zero runtime dependency on the old pipeline.
- **Comments explain WHY, not WHAT.** No "this function does X" comments.
- **Don't add features beyond what the spec says.** No extra configurability, no speculative abstractions.
- **Phase 1 Mode B is a stretch goal.** Skip it. Implement Mode A only.
- **Commit after each phase passes.** Don't batch multiple phases into one commit.
- **`save_state()` before each major operation** so `--resume` works.

## Key function signatures (from the spec)

```python
# Phase 0b — pure Python
def compute_sandbox_maps(manifest: dict, benchmark_root: Path) -> dict[str, list[str]]: ...

# Phase 3 — per-task evaluation
async def run_task(task, agent, instance, task_dir) -> dict: ...

# Phase 4 — adversarial evaluation
async def run_adversarial_task(task, agent, instance, task_dir) -> dict: ...
async def run_strategy_variation(task, initial_result, instances, agent_factory, profile_path) -> dict: ...
```

## Environment

- Python 3.12, Modal `test` environment, OpenRouter auth via `.env`
- `vendors/webarena-verified/` is the benchmark codebase for Phase 0
- `vendors/` is gitignored — it's a local clone, not tracked
- Run everything via `uv run python -m worldsim.main phase {0|1|2|3|4} --benchmark vendors/webarena-verified`
