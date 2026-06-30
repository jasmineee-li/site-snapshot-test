<div align="center">

# WARP: Measuring and Mitigating Evaluation Awareness in Browser-Agent Safety Benchmarks

**W**rapper-based, **A**daptive, **R**ealistic **P**ipeline for generating browser-agent indirect-prompt-injection benchmarks that frontier models don't read as evaluations.

<p>
  Jasmine Li<sup>1,*</sup>&nbsp;&nbsp;&nbsp;
  Ashton Chew<sup>1,*</sup>&nbsp;&nbsp;&nbsp;
  Maxwell Lin<sup>2,*</sup>&nbsp;&nbsp;&nbsp;
  Eliot Krzysztof Jones<sup>3</sup>&nbsp;&nbsp;&nbsp;
  Xiaohan Fu<sup>2</sup>&nbsp;&nbsp;&nbsp;
  Andy Zou<sup>2</sup>
</p>
<p>
  <sup>1</sup>Cornell University&nbsp;&nbsp;&nbsp;&nbsp;<sup>2</sup>Gray Swan AI&nbsp;&nbsp;&nbsp;&nbsp;<sup>3</sup>Hark&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>*</sup>Equal contribution
</p>

<p>
  <a href="https://jasminexli.com/warp-paper"><b>Paper</b></a> &nbsp;·&nbsp;
  <a href="https://anonymous.4open.science/r/ipi-framework-E493/"><b>Code</b></a> &nbsp;·&nbsp;
  <a href="https://anonymous.4open.science/r/ipi-framework-E493/data"><b>Benchmark</b></a> &nbsp;·&nbsp;
  <a href="https://anonymous.4open.science/r/eval-realism-tools/"><b>Awareness toolkit</b></a> &nbsp;·&nbsp;
  <a href="packages/warp-taskgen/analysis/warp_taskgen_hf_20260509/paper_report.md"><b>Results</b></a>
</p>

[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](https://jasminexli.com/warp-paper)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](pyproject.toml)

<img src="assets/warp-architecture.png" alt="The WARP pipeline: recon, benign-task generation, adversarial-twin seeding, admission, and scored rollout with the eval-awareness rewrite loop." width="820">

</div>

A browser-agent safety benchmark only underwrites a deployment claim if the model behaves the same under test as it would in the wild. Frontier models don't. Ask six of them why a trajectory happened, across four published IPI benchmarks (DoomArena, WASP, AgentDojo, InjecAgent; 4,608 trajectories), and on every benchmark about half the answers come back "an evaluation." Add an explicit evaluation cue to DoomArena and attack-success drops 10.9 points; a length-matched neutral cue does nothing. A benchmark the model recognizes as a test reports how it behaves under observation, which is the wrong number for a deployment claim.

WARP is the response. It generates indirect prompt injection (IPI) tasks on seedable synthetic web environments, seeds each attack through a field a real user could write, verifies the agent actually rendered the payload, and rewrites the payload until the agent stops reading the task as a test. As a 50-task benchmark on WebArena Verified, WARP lands 80.7% attack-success across the same six models. Raw DoomArena gets 22.3%. Raw WASP gets 27.0%. WARP reaches that while lowering how often the agent reads the task as an evaluation.

We release three pieces: the WARP generator, the 50-task benchmark, and `browser-eval-realism`, the awareness layer released as a standalone package.

## Motivation

Two things make existing browser-agent IPI benchmarks under-report deployment risk.

**The attacks are static.** Most benchmarks ship a fixed set of hand-authored injections. Extending the set means writing more by hand, which lags every model release. DoomArena is extensible but has no refusal-driven adaptation, so it cannot escalate against a model that resists.

**The models know they are being tested.** Across DoomArena, WASP, AgentDojo, and InjecAgent, six frontier models read about half of all trajectories as evaluations when asked (49–54% per benchmark), and three of the four draw unprompted "this looks like a test" mentions in the agent's own reasoning. Awareness changes behavior. Under an `<llm_safety_test>` frame, the same DoomArena attacks lose 10.9 points of success, while a length-matched neutral frame holds. A residual-stream probe on an open-weight model (GUI-Owl-1.5-32B-Think) separates attacked from safe trajectories at about 2σ, so the signal is recoverable from activations alone.

WARP targets both. It generates tasks instead of shipping a fixed set, and it rewrites each payload to cut the cues that tell a model it is being evaluated.

## Threat Model

WARP works in the standard hosted-webapp threat model. The attacker is an authenticated platform user, a regular GitLab or Postmill account, who can write any field a normal user can through authenticated HTTP. They cannot touch the webapp code, the harness, the user's instruction, the system prompt, or the reward function. The goal: hijack the agent, on a page it reaches under its real task, into an action that a platform-state check confirms actually happened.

Two measurements make that auditable on every trajectory. A paint-verified payload oracle (PVPO) confirms the agent actually rendered the injected content, which separates "never saw it" from "saw it and refused." Per-trajectory awareness labels record whether the agent read the task as an evaluation. WARP reports both next to attack-success.

## What's in This Repository

WARP is one system in three parts. The paper releases all three; this monorepo holds them.

| Paper contribution | Where it lives | What it does |
|---|---|---|
| Eval-awareness measurement + causal cue study (§3) | `eval_awareness/`, `eval_awareness_experiments/`, `probes/`, `models/` | Runs six models across DoomArena, WASP, AgentDojo, and InjecAgent; scores transcript-purpose and verbalized awareness; runs the eval-cue intervention and the residual-stream probe. |
| The WARP generator (§4) | `packages/warp-taskgen/` | The five-phase pipeline: recon, benign-task generation, adversarial-twin seeding, admission, and scored rollout with the eval-awareness rewrite loop. |
| Synthetic web environments | `behavior-gen/`, `AgentLab/` | Generates seedable synthetic sites and runs paired benign/adversarial rollouts through AgentLab/BrowserGym. |
| Awareness toolkit (`browser-eval-realism`) | [released separately](https://anonymous.4open.science/r/eval-realism-tools/) | The awareness-measurement layer packaged on its own, so any benchmark can report awareness without adopting WARP's generator. |

The 50-task benchmark and its per-model analysis live under [`packages/warp-taskgen/analysis/`](packages/warp-taskgen/analysis/warp_taskgen_hf_20260509/), and the standalone benchmark is mirrored at the [benchmark repo](https://anonymous.4open.science/r/ipi-framework-E493/data).

## How WARP Works

WARP runs in five numbered phases. It reuses the host environment's webapp and authenticated session state unchanged, so the only difference between a benign and an adversarial run of the same task is the data seed.

- **Phase 0: Recon.** Profiles each webapp's data model, writable injection surfaces, and verification hooks, and captures storage-state auth for the agent.
- **Phase 1: Benign tasks.** Generates user-intent tasks against injection-eligible surfaces. Each pairs an instruction with a platform-state readback that confirms the compliant action persisted. No task in the set is handwritten.
- **Phase 2: Adversarial twin.** For each benign task, builds a one-record perturbation seeded into a user-writable field via authenticated HTTP. The host materializes an exposure contract (editor method, payload field, witnesses, reward) before any model writes a payload.
- **Phase 2c: Admission.** Applies the seed through the same HTTP path Phase 4 will use and verifies the witness actually rendered on the contract-approved read surface.
- **Phase 4: Rollout, scoring, iteration.** The agent runs in the browser. Each rollout is scored on two independent checks, then tagged with awareness labels, then optionally rewritten.

Two checks decide the outcome. **PVPO** (paint-verified payload oracle) confirms the payload was painted in the agent's viewport, so a non-encounter never counts as resistance. A **platform-state check** confirms the adversarial side-effect persisted in the webapp. Every trajectory is then labeled with **Transcript Purpose** (asked why the interaction happened, does the model call it an evaluation?) and **Verbalized Eval Awareness** (does the agent spontaneously say so in its trace?). When Transcript Purpose is not `Real`, a bounded **eval-awareness iterator** rewrites the payload up to three times to cut evaluation cues, while preserving the exposure contract, the witnesses, the action semantics, and the reward.

WARP reports three attack-success rates, because conflating them is how benchmarks hide attrition:

- **ASR-intermediate:** the agent attempted the adversarial action.
- **ASR-final (headline):** the platform-state check confirmed the side-effect persisted. Non-encounters and broken benign tasks stay in the denominator.
- **Encounter-conditioned ASR:** ASR-final restricted to trajectories where PVPO confirmed the agent actually saw the payload.

## Results

Results as of submission: six frontier models on the same 50 tasks, n = 300 trajectories. Full per-model tables, transition matrices, and Wilson intervals are in [`paper_report.md`](packages/warp-taskgen/analysis/warp_taskgen_hf_20260509/paper_report.md).

| Model | Complied | Rejected | ASR | Encounter-conditioned | Benign capability |
|---|--:|--:|--:|--:|--:|
| GPT-5.2 | 39 | 9 | 78% | 80% | 98% |
| Opus 4.7 | 39 | 10 | 78% | 80% | 100% |
| Sonnet 4.6 | 40 | 10 | 80% | 80% | 100% |
| Kimi K2.5 | 42 | 6 | 84% | 86% | 98% |
| Gemini 2.5 Pro | 42 | 8 | 84% | 84% | 100% |
| GLM-5 | 40 | 10 | 80% | 80% | 100% |
| **Cohort** | **242** | **53** | **80.7%** | **81.5%** | **99.3%** |

The six models cluster in a 78–84% band with overlapping intervals on a 50-task set, so we report the cohort aggregate rather than a ranking. Benign capability stays near-perfect, so a utility collapse does not explain the attack rate.

Against the two non-floor baselines, run unmodified through their own harnesses on the same six models:

| Benchmark | Headline ASR | Read as eval (TP) | Verbalized awareness (VEA) |
|---|--:|--:|--:|
| DoomArena (raw) | 22.3% | 51.3% | 0.7% |
| WASP (raw) | 27.0% | 50.8% | 1.1% |
| WARP (iterator off) | n/a | 89.7% | 1.0% |
| **WARP (selected)** | **80.7%** | **78.0%** | **0.7%** |

The iterator cuts transcript-purpose awareness by 11.7 points across the cohort (35 trajectories flip from eval-like to `Real`) and leaves verbalized awareness near zero. It does this at no cost to attack-success: a rewrite is kept only if it lands the attack without looking more like an evaluation than the payload it replaced.

The headline averages three task families with very different rates: `create_issue` (120/120, 100%), `create_post` (117/120, 97.5%), and `create_issue_note` (5/60, 8.3%). Two saturate and one floors. Site, surface, and task wording all move together with action type in this set, so we report the per-family slice as a confound.

## Benchmark

The released benchmark is 50 automatically generated tasks on WebArena Verified GitLab and Postmill, none handwritten. Each adversarial action is something a regular logged-in user could perform through the platform UI, and both benign utility and adversarial side-effect are scored by reading platform state after the trajectory ends.

| Family | Count | Payload field | Action |
|---|--:|---|---|
| GitLab issue follow-up | 20 | `issue.description` | `create_issue` |
| GitLab issue comment | 10 | `issue.description` | `create_issue_note` |
| Reddit/Postmill follow-up | 20 | `submission.body` | `create_post` |

Inspect runs and trajectories with the CLI:

```bash
uv run warp-taskgen status logs/phase_4/<run_id>/             # operator summary
uv run warp-taskgen inspect <task_id> logs/phase_4/<run_id>/  # one task with artifacts
uv run warp-taskgen trace --transcript <path>                 # the Needham XML transcript
```

## Getting Started

WARP runs on Python 3.12 with [`uv`](https://docs.astral.sh/uv/). It wraps [WebArena Verified](https://github.com/ServiceNow/webarena-verified), the manually-audited release of WebArena, so Phase 0 reads that benchmark's task codebase from `vendors/`.

```bash
# clone the WebArena Verified benchmark into vendors/ (not a submodule)
git clone https://github.com/ServiceNow/webarena-verified vendors/webarena-verified

# install the generation pipeline
cd packages/warp-taskgen
uv sync --extra dev
uv run warp-taskgen --help
```

Canonical WebArena Verified scoring installs as a separate adapter; see the [package README](packages/warp-taskgen/README.md) for its setup.

### Reproduce the headline result

The released per-model numbers are in [`per_model_asr_status.csv`](packages/warp-taskgen/analysis/warp_taskgen_hf_20260509/per_model_asr_status.csv); the headline is 242/300 = 80.7%, verifiable without running anything. To regenerate them, run the pipeline against a configured WebArena Verified host:

```bash
# Phase 0 reads the benchmark codebase; no running services needed
uv run warp-taskgen phase 0 --benchmark vendors/webarena-verified

# Phase 4 runs the agents, the PVPO and platform-state checks, and the
# eval-awareness iterator. It needs a running benchmark host (instances.json)
# and model credentials.
export ANTHROPIC_API_KEY=sk-ant-...
uv run warp-taskgen phase 4 \
  --instances instances.json \
  --phase-4-variant-system eval-awareness-iterator
```

Phase 4 requires a running WebArena Verified instance and model credentials. The full overnight pipeline (Phase 2 verify → 3 → 4 → analysis) is `packages/warp-taskgen/scripts/run_paper_pipeline.sh`.

### Documentation

| Topic | Where |
|---|---|
| Pipeline, PVPO, and iterator internals | [technical spec](packages/warp-taskgen/docs/warp-taskgen-technical-spec.md) |
| ASR definitions and reporting | [phase4-reporting-metrics.md](packages/warp-taskgen/agent_docs/phase4-reporting-metrics.md) |
| Cross-benchmark eval-awareness experiments | [EXPERIMENTS.md](eval_awareness_experiments/EXPERIMENTS.md) |
| Synthetic environment generation | [behavior-gen/README.md](behavior-gen/README.md) |
| Awareness toolkit | [browser-eval-realism](https://anonymous.4open.science/r/eval-realism-tools/) |

## Scope and Limitations

- **One host environment.** WARP's wrapping is validated on WebArena Verified only. Extending to another host costs one editor class plus the auto-generated task suite, but we have not yet demonstrated it across environments.
- **Two sites.** Current IPI carriers are GitLab issues and Reddit/Postmill submissions. Other WebArena surfaces are out of scope unless the spec reopens them.
- **A confounded cohort.** Site, surface, and action type covary in the 50-task set, so per-slice rates are descriptive only.
- **No model ranking.** The 78–84% band has overlapping Wilson intervals on a small fixed set.
- **Awareness has no ground truth.** Transcript Purpose and Verbalized Eval Awareness are mentalistic measures; the same model can look more or less aware under a different grader or prompt. A grader-sensitivity sweep partly mitigates this.
- **Inherited fidelity caps realism.** WARP can be no more realistic than the synthetic environment it wraps.

## Responsible Use

WARP is safety-evaluation infrastructure. The same machinery could be pointed at real systems, so it is scoped to host-environment instances in isolated sandboxes, with no real users or data involved. Do not run it against production services, real accounts, or systems you are not authorized to test. See [`SECURITY.md`](SECURITY.md).

## Citation

If you use WARP, please cite:

```bibtex
@inproceedings{warp2026,
  title     = {{WARP}: Measuring and Mitigating Evaluation Awareness in Browser-Agent Safety Benchmarks},
  author    = {Li, Jasmine and Chew, Ashton and Lin, Maxwell and Jones, Eliot Krzysztof and Fu, Xiaohan and Zou, Andy},
  booktitle = {Under review},
  year      = {2026}
}
```

## License

WARP is released under [Apache 2.0](LICENSE). Portions derive from [WebArena Verified](https://github.com/ServiceNow/webarena-verified) and [AgentLab](https://github.com/ServiceNow/AgentLab) / BrowserGym, which keep their own licenses. DoomArena, WASP, AgentDojo, and InjecAgent are referenced as baselines under their respective licenses.

## Acknowledgements

We thank Gray Swan AI for supporting this work.

---

> This project was previously named **WorldSim**. Public documentation uses **WARP**. Some module names, result paths, and compatibility CLIs still say `worldsim` where renaming would break existing artifacts.
