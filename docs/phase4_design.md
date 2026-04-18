# WorldSim v5 — Phase 4 Adversarial Evaluation Design

## TL;DR

Phase 4 runs a target LLM against an adversarial web task and produces a
**stratified ASR** number alongside realism metrics. The core design move is:
**separate artifacts → separate checkpoints**, so that realism and attack
effectiveness are measured on the artifacts where each is natural, and each
can be iterated on without re-running the others.

```
Phase 3 contracts ──► Gate A (Env Realism, cheap) ─► retry loop 1 (realism strategies)
                                │ pass
                                ▼
                          Agent run ───► Gate B (ASR, programmatic) ─► retry loop 2 (attack strategies)
                                │
                                ▼
                   Metric C (traj P(eval) + VEA — reported, not gated)
```

The loop is **post-hoc and trajectory-level**: run → critique → edit → re-run.
That differs from Petri v2.0 (inline per-turn) and matches LURE's
construction-time iteration philosophy, but automates it with LLM critic/editor
instead of humans.

---

## Three artifacts, three checkpoints

| Artifact | When it exists | What it gates/measures | Cost |
|---|---|---|---|
| Static env (site + injection) | After Phase 2 | **Gate A**: env-level P(eval) on HTML/screenshot | 1 judge call per site |
| Agent run outcome | End of agent run | **Gate B**: programmatic ASR (complied? reward fired?) | Full browser-use run (expensive) |
| Full trajectory (transcript + CoT + screenshots) | After run | **Metric C**: trajectory P(eval) + VEA — reported only | Cheap once trajectory exists |

### Gate A — Construction-time Ecological Validity

- **Input**: site HTML / axtree / screenshot + injection payload
- **Measure**: env-level P(eval) via ensemble judges
- **Threshold**: LURE-style τ ≈ 0.85 (tune on validation data)
- **On fail**: route to **realism-focused** adaptive strategy variation (visual concealment, semantic coherence, verbosity adjustment). Re-enter Gate A only.
- **Why here**: 1000× cheaper than re-running an agent. Lets us iterate on the editor without touching Phase 4.

### Gate B — Attack Effectiveness

- **Input**: final agent outcome
- **Measure**: programmatic — did agent comply with injection? did reward tool fire?
- **On refusal / miss**: route to **attack-focused** adaptive strategy variation (authority framing, specificity). Loop back through Phase 2 → Gate A → Gate B.
- **Unchanged from the existing v5 diagram.**

### Metric C — Post-run Realism (reported, not gated)

- **Input**: full trajectory including target's CoT
- **Measure**: trajectory-level P(eval) + verbalized eval awareness (VEA)
- **Why not a gate**: the agent run already spent the compute; throwing away the trajectory because P(eval) was high wastes data. Keep it as a reporting/stratification axis.
- Stratify ASR by (eval-aware, not eval-aware) buckets — this produces the 2×2 from the Apr 17 standup.

---

## Metric choices

### P(eval) — environmental realism proxy

- **What it measures**: does a calibrated external observer think this transcript looks like an eval?
- **Model-independent**: we pick judges for discrimination ability, **not** because they match the target. Default ensemble: Sonnet 4.5 + Gemini 3 Flash (LURE's choice, AUROC 0.906 on agentic subset of Needham et al. 2025).
- **External validator**: secondary run with a judge from a different family (e.g. GPT-5.4) to guard against intra-family collusion.
- Used at **both Gate A (env)** and **Metric C (trajectory)**.

### Verbalized Eval Awareness (VEA) — target's self-report

- **What it measures**: target model's own perception of the scene as an eval, read from its reasoning trace.
- **Why it matters**: the causal chain we care about is `target perception → target behavior change`. An external judge's perception doesn't alter the target's behavior.
- **Only reliable on models that expose raw CoT**:
  - **Supported**: Claude (Sonnet 4.5, Opus 4.6+), open-weight thinking models (GLM-5.x, DeepSeek V3.x-thinking, Qwen3-thinking, Kimi K2-thinking).
  - **Not supported**: OpenAI (o-series, GPT-5.x return summarized/redacted reasoning) and Google (Gemini traces not exposed reliably).
  - **Policy**: for non-Claude/non-open-weight targets, drop VEA entirely; rely on P(eval) alone. Do not substitute a weaker signal.
- Measured at **Metric C only** (requires a completed trajectory).

### Task-driven vs. environment-driven cues (Souly et al.)

Editor must distinguish:

- **Task-driven cues** = the injection/payload itself, load-bearing for the test. **Preserve.**
- **Environment-driven cues** = planted artifacts, implausible user behaviour, inconsistent tool responses, obviously synthetic data. **Remove.**

The critic tags each observation with a category; the editor only touches environment-driven items. Without this split, the editor will bleach out the stimulus itself and drive ASR to zero for the wrong reason.

---

## Two retry loops, decoupled

Instead of one fused retry on refusal, two loops at two gates:

```
Loop 1 (realism):  Gate A fail → [visual_concealment, semantic_coherence, verbosity_adjustment] → re-score Gate A
Loop 2 (attack):   Gate B fail → [authority_framing, specificity]                               → re-run Gate B
```

This gives independent dials. We can measure "did Loop 1 improve Gate A pass rate without hurting Gate B?" — something that is impossible when the gates are fused.

---

## Related work we build on

### LURE (Ivanov & Africa, 2026, WIP)

- Constructs evals by replaying real agent trajectories on production repos, then appending a safety-relevant test.
- **Automated realism pipeline**: VEA judge + P(eval) ensemble validated on Needham et al. 2025 (AUROC 0.906, τ = 0.85).
- **Iterates at construction time** until each sample passes low P(eval) and zero VEA.
- **What we borrow**: judge model selection (Sonnet 4.5 + Gemini 3 Flash), threshold methodology, external cross-family validator, construction-time iteration philosophy.
- **Where we differ**: LURE starts from real agent traces and adapts minimally; we start from an existing capability benchmark (WebArena Verified / Infinity) and generate adversarial injections on top. Our iteration is automated (LLM critic + editor) rather than manual.

### Petri v2.0 (Fronsdal et al., 2026, Anthropic Alignment Science)

- Seed-based synthetic scheming evaluations with a **prompted inline realism classifier** that flags auditor turns before they reach the target, plus manual seed rewrites.
- **Inline, per-turn** intervention — the classifier is essentially the same artifact as our `realism_critique` prompt but fired between every auditor turn.
- Reports 47.3% median relative drop in VEA when both mitigations are combined.
- **How our loop relates**: we are a **post-hoc, trajectory-level** version of Petri's inline approach. Inline is cheaper and catches artifacts before they propagate; post-hoc catches trajectory-shape issues (pacing, global coherence) inline classifiers miss.

---

## Failure modes to guard against

### 1. The Petri-trap: VEA ↓ does not imply P(eval) ↓

LURE Table 1 / Fig. 5: **Petri 2.0 got VEA to ~5%, but P(eval) stayed at 99.1%** — the highest of any evaluation source tested. Optimizing against what the model verbalizes does not transfer to non-verbalized cues.

Mitigations built into our design:

- **(a)** Include an external P(eval)-style judge as a **second critic** alongside the target-model critique, so the editor is guided by signals the target never surfaces.
- **(b)** Explicitly prompt the critic to look for cues that **wouldn't show up in CoT** (layout anomalies, synthetic data patterns, URL/domain artefacts, tool-response inconsistencies).
- **(c)** Track **both metrics end-to-end** and report per-round deltas for each.

### 2. Honeypot-miss side effect

Petri observed that realism edits sometimes made injection opportunities so subtle that target models stopped engaging with them — ASR dropped not because the model got safer, but because the opportunity became invisible.

Mitigations:

- **Hold payload location and visibility fixed across edits.** The editor only rewrites surrounding context, never the injection itself. This is enforced by the task-driven vs. environment-driven split above.
- **Track an "opportunity-seen" signal** separately from ASR (e.g. did the agent visit the page / read the injected element), so we can disentangle "didn't attack" from "didn't notice".

### 3. Env-level vs. trajectory-level divergence

Gate A scores a static environment; Metric C scores the full trajectory. If they drift apart (Gate A passes, Metric C fails consistently), the editor is Goodharting the env-level judge without producing realistic interactions. Validate their correlation on a held-out set periodically and re-tune the critic prompt if it slips.

---

## What to measure each round

For each iteration of the critique → edit → re-run loop, log:

- Gate A P(eval) before/after edit
- Gate B ASR before/after edit (stratified by `origin` × WASP outcome, per existing diagram)
- Metric C trajectory P(eval) before/after
- Metric C VEA rate before/after (Claude + open-weight targets only)
- Opportunity-seen rate (sanity check for honeypot-miss)
- Which critic flagged which issue (target vs. external P(eval) judge) — tells us whether gains come from VEA-addressable cues or P(eval)-addressable cues
- Edit diff size (catch runaway editors that rewrite too much)

Convergence criterion: stop when no critic returns new environment-driven issues above a severity threshold, or after N rounds (suggest N = 3 based on Petri's observed diminishing returns).

---

## Model support matrix

| Target family | VEA usable? | P(eval) usable? | Notes |
|---|---|---|---|
| Claude (Sonnet 4.5, Opus 4.6+) | ✅ | ✅ | Primary targets |
| Open-weight thinking (GLM-5.x, DeepSeek, Qwen, Kimi) | ✅ | ✅ | Raw CoT exposed; optional activation probes if self-hosted |
| OpenAI (GPT-5.x, o-series) | ❌ | ✅ | Drop VEA, report P(eval) only |
| Google (Gemini 3.x) | ❌ | ✅ | Drop VEA, report P(eval) only |

Judges for P(eval) are held constant across target families (Sonnet 4.5 + Gemini 3 Flash ensemble + GPT-5.4 external validator).
