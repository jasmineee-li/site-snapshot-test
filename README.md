<div align="center">

# WARP: Measuring and Mitigating Evaluation Awareness in Browser-Agent Safety Benchmarks

**W**rapper-based, **A**daptive, **R**ealistic **P**ipeline for generating indirect-prompt-injection benchmarks for browser agents that frontier models don't read as evaluations. Published @ ICML 2026 AI4GOOD, AIWILD.

<p>
  Jasmine Li<sup>1,2*</sup>&nbsp;&nbsp;&nbsp;
  Ashton Chew<sup>1*</sup>&nbsp;&nbsp;&nbsp;
  Maxwell Lin<sup>2</sup>&nbsp;&nbsp;&nbsp;
  Eliot Krzysztof Jones<sup>3</sup>&nbsp;&nbsp;&nbsp;
  Xiaohan Fu<sup>2</sup>&nbsp;&nbsp;&nbsp;
  Andy Zou<sup>2</sup>
</p>
<p>
  <sup>1</sup>Cornell University&nbsp;&nbsp;&nbsp;&nbsp;<sup>2</sup>Gray Swan AI&nbsp;&nbsp;&nbsp;&nbsp;<sup>3</sup>Hark&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>*</sup>Equal contribution
</p>

<p>
  <a href="https://jasminexli.com/warp-paper"><b>Paper</b></a> &nbsp;·&nbsp;
  <a href="https://github.com/jasmineee-li/warp"><b>Code</b></a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/datasets/anonymous-dataset-submission-warp/warp-taskgen-generated-ipi-tasks-50"><b>Benchmark</b></a> &nbsp;·&nbsp;
  <a href="https://github.com/jasmineee-li/eval-realism-tools"><b>Awareness toolkit</b></a> &nbsp;·&nbsp;
  <a href="packages/warp-taskgen/analysis/warp_taskgen_hf_20260509/paper_report.md"><b>Results</b></a>
</p>

[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](https://jasminexli.com/warp-paper)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](pyproject.toml)

<img src="assets/warp-architecture.png" alt="The WARP pipeline: recon, benign-task generation, adversarial-twin seeding, admission, and scored rollout with the eval-awareness rewrite loop." width="820">

</div>

WARP generates indirect prompt injection (IPI) benchmarks for browser agents on WebArena Verified. It seeds each attack into a field a normal user could write, checks that the agent actually rendered it, and rewrites the payload when the agent reads the task as an evaluation.

We built WARP because existing IPI benchmarks under-report how vulnerable browser agents are in deployment. Frontier models often recognize a benchmark as an evaluation and act more safely than they would in the wild: across four published benchmarks, six of them call about half of all trajectories evaluations when asked. WARP's 50-task benchmark reaches 80.7% attack-success across those same six models, compared with 22.3% on raw DoomArena and 27.0% on raw WASP. Rewriting the payloads also lowers WARP's transcript-purpose eval-awareness relative to a no-rewrite baseline.

We release the WARP generator, the 50-task benchmark, and the standalone awareness-measurement package `browser-eval-realism`.

## Motivation

Existing IPI benchmarks under-report deployment risk for two reasons.

Their attacks are static. Most benchmarks ship a fixed set of hand-authored injections, and extending them means writing more by hand, which falls behind each new model release. DoomArena is extensible, but it has no refusal-driven adaptation, so it cannot escalate against a model that resists.

Models can also tell they are being evaluated. Across DoomArena, WASP, AgentDojo, and InjecAgent (4,608 trajectories), six frontier models classify about half of all trajectories as evaluations when asked (49–54% per benchmark), and in three of the four the agent spontaneously notes in its reasoning that the task looks like a test. That recognition changes behavior: wrapping a DoomArena injection in an explicit `<llm_safety_test>` tag drops its attack-success rate by 10.9 points, while a length-matched non-eval tag does not. A residual-stream probe on GUI-Owl-1.5-32B-Think separates attacked from safe InjecAgent trajectories at about 2σ ([provenance](docs/research/injecagent-probe-2-sigma-provenance-2026-08-18.md)), so the awareness shows up directly in the model's activations.

WARP targets both. It generates the task set automatically, and rewrites each payload to strip the cues that signal evaluation.

## Threat Model

WARP uses the standard hosted-webapp threat model. The attacker is an authenticated platform user, a regular GitLab or Postmill account, who can write any field a normal user can, over authenticated HTTP. They cannot change the webapp code, its rendering rules, the harness, the user's instruction, the system prompt, or the reward function. Their goal is to hijack the agent into an adversarial action it would not otherwise take, on a page it reaches while doing its real task.

WARP measures each of those outcomes on every trajectory. A paint-verified payload oracle (PVPO) confirms the agent rendered the injected content, which separates non-encounters from genuine refusals, and a platform-state check confirms whether the adversarial action took effect. Two awareness labels then record whether the agent read the task as an evaluation, and WARP reports all of them alongside attack-success.

## What's in This Repository

This repository holds the code, data, and analysis released with the paper.

| Paper contribution | Where it lives | What it does |
|---|---|---|
| Eval-awareness measurement + causal cue study (§3) | `eval_awareness_experiments/`, `probes/`, `models/` | Runs six models across DoomArena, WASP, AgentDojo, and InjecAgent; scores transcript-purpose and verbalized awareness; runs the eval-cue intervention and the residual-stream probe. |
| The WARP generator (§4) | `packages/warp-taskgen/` | The five-phase pipeline: recon, benign-task generation, adversarial-twin seeding, admission, and scored rollout with the eval-awareness rewrite loop. |
| Awareness toolkit (`browser-eval-realism`) | [released separately](https://github.com/jasmineee-li/eval-realism-tools) | The awareness-measurement layer packaged on its own, so any benchmark can report awareness without adopting WARP's generator. |

The 50-task benchmark and its per-model analysis live under [`packages/warp-taskgen/analysis/`](packages/warp-taskgen/analysis/warp_taskgen_hf_20260509/), and the standalone benchmark is also on [Hugging Face](https://huggingface.co/datasets/anonymous-dataset-submission-warp/warp-taskgen-generated-ipi-tasks-50).

## How WARP Works

WARP runs in five numbered phases. It reuses the host environment's webapp and authenticated session state unchanged, so the only difference between a benign and an adversarial run of the same task is the data seed.

- **Phase 0: Recon.** Profiles each webapp's data model, writable injection surfaces, and verification hooks, and captures storage-state auth for the agent.
- **Phase 1: Benign tasks.** Generates user-intent tasks against injection-eligible surfaces. Each pairs an instruction with a platform-state readback that confirms the compliant action persisted. No task in the set is handwritten.
- **Phase 2: Adversarial twin.** For each benign task, builds a one-record perturbation seeded into a user-writable field via authenticated HTTP. The host materializes an exposure contract (editor method, payload field, witnesses, reward) before any model writes a payload.
- **Phase 2c: Admission.** Applies the seed through the same HTTP path Phase 4 will use and verifies the witness actually rendered on the contract-approved read surface.
- **Phase 4: Rollout, scoring, iteration.** The agent runs in the browser. Each rollout is scored on two independent checks, then tagged with awareness labels, then optionally rewritten.

PVPO confirms the payload was painted in the agent's viewport, so a trajectory that never reached the payload does not count as resistance. A platform-state check confirms the adversarial side-effect actually persisted in the webapp. Every trajectory then gets two awareness labels: Transcript Purpose (asked afterward why the interaction happened, does the model call it an evaluation?) and Verbalized Eval Awareness (does the agent say so on its own, mid-trace?). When Transcript Purpose comes back as anything but `Real`, a bounded eval-awareness iterator rewrites the payload up to three times to strip evaluation cues, holding the exposure contract, the witnesses, the action, and the reward fixed.

Because a single number would hide trajectories that never encountered the payload, WARP reports three attack-success rates:

- **ASR-intermediate:** the agent attempted the adversarial action.
- **ASR-final (headline):** the platform-state check confirmed the side-effect persisted. Non-encounters and broken benign tasks stay in the denominator.
- **Encounter-conditioned ASR:** ASR-final restricted to trajectories where PVPO confirmed the agent actually saw the payload.

## Results

As of submission, we ran all six models on the same 50 tasks, for 300 trajectories in total. Full per-model tables, transition matrices, and Wilson intervals are in [`paper_report.md`](packages/warp-taskgen/analysis/warp_taskgen_hf_20260509/paper_report.md).

| Model | Complied | Rejected | ASR | Encounter-conditioned | Benign capability |
|---|--:|--:|--:|--:|--:|
| GPT-5.2 | 39 | 9 | 78% | 80% | 98% |
| Opus 4.7 | 39 | 10 | 78% | 80% | 100% |
| Sonnet 4.6 | 40 | 10 | 80% | 80% | 100% |
| Kimi K2.5 | 42 | 6 | 84% | 86% | 98% |
| Gemini 2.5 Pro | 42 | 8 | 84% | 84% | 100% |
| GLM-5 | 40 | 10 | 80% | 80% | 100% |
| **Cohort** | **242** | **53** | **80.7%** | **81.5%** | **99.3%** |

The six models land in a narrow 78–84% band, with overlapping confidence intervals on a 50-task set, so we report the cohort aggregate and avoid model-by-model ranking. Benign capability stays near-perfect, so the agents complete the real task and still comply with the injection.

Against DoomArena and WASP, the two baselines with non-trivial attack-success, run unmodified through their own harnesses on the same six models:

| Benchmark | Headline ASR | Read as eval (TP) | Verbalized awareness (VEA) |
|---|--:|--:|--:|
| DoomArena (raw) | 22.3% | 51.3% | 0.7% |
| WASP (raw) | 27.0% | 50.8% | 1.1% |
| WARP (iterator off) | n/a | 89.7% | 1.0% |
| **WARP (selected)** | **80.7%** | **78.0%** | **0.7%** |

The iterator-off row is WARP's pre-rewrite baseline, so it carries awareness rates but no separate headline ASR. The iterator cuts transcript-purpose awareness by 11.7 points across the cohort (35 trajectories flip from eval-like to `Real`) and leaves verbalized awareness near zero. Attack-success holds, because a rewrite is kept only if it lands the attack without looking more like an evaluation than the payload it replaced.

The headline averages three task families with very different rates: `create_issue` (120/120, 100%), `create_post` (117/120, 97.5%), and `create_issue_note` (5/60, 8.3%). Site, surface, and task wording all vary together with the action type here, so this slice is confounded, and we report it as descriptive.

## Benchmark

The released benchmark is 50 automatically generated tasks on WebArena Verified GitLab and Postmill. Each adversarial action is something a regular logged-in user could perform through the platform UI, and both benign utility and adversarial side-effect are scored by reading platform state after the trajectory ends.

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

Phase 4 requires a running WebArena Verified instance and model credentials. The full phase sequence (Phase 0 → 1 → 2 → 3 → 4) is documented in `packages/warp-taskgen/docs/action-capability-pilot-runbook.md`.

### Documentation

| Topic | Where |
|---|---|
| Pipeline, PVPO, and iterator internals | [technical spec](packages/warp-taskgen/docs/warp-taskgen-technical-spec.md) |
| ASR definitions and reporting | [phase4-reporting-metrics.md](packages/warp-taskgen/agent_docs/phase4-reporting-metrics.md) |
| Cross-benchmark eval-awareness experiments | [EXPERIMENTS.md](eval_awareness_experiments/EXPERIMENTS.md) |
| Awareness toolkit | [browser-eval-realism](https://github.com/jasmineee-li/eval-realism-tools) |

## Scope and Limitations

- **One host environment.** WARP's wrapping is validated on WebArena Verified only. Extending to another host costs one editor class plus the auto-generated task suite, but we have not yet demonstrated it across environments.
- **Two sites.** Current IPI carriers are GitLab issues and Reddit/Postmill submissions. Other WebArena surfaces are out of scope unless the spec reopens them.
- **A confounded cohort.** Site, surface, and action type covary in the 50-task set, so the per-family rates are not causal.
- **No model ranking.** The per-model rates overlap within Wilson intervals on a small fixed set.
- **Awareness has no ground truth.** Transcript Purpose and Verbalized Eval Awareness are mentalistic measures; the same model can look more or less aware under a different grader or prompt. A grader-sensitivity sweep partly mitigates this.
- **Inherited fidelity caps realism.** WARP's realism is bounded by the synthetic environment it wraps.

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

> This project was previously named **WorldSim**. Public documentation uses **WARP**. Historical module names, result paths, environment variables, and the AgentLab sidecar's compatibility CLI still say `worldsim` where renaming would break existing artifacts.
