# Web-agent benchmark discovery (snapshot 2026-08-30)

## Scope and date discipline

This is a read-only literature/repository discovery pass for possible WARP
cross-benchmark adapters. The comparison target is a benchmark with a
self-hostable stateful web environment, ordinary participant-writable content,
independent rendered exposure to another participant, deterministic reset,
state-based grading, and preferably benign/adversarial variants of the same
workflow. A benchmark need not satisfy every property to be useful as a
comparator; missing properties are recorded explicitly. The snapshot date is
2026-08-30. I searched for changes from 2026-02-17 through 2026-08-30 (the
requested post-cutoff interval), while retaining older candidates when they
remain technically useful.

I keep four dates separate:

* **First paper:** first arXiv submission (or the paper's stated first
  publication when no arXiv record was found).
* **Publication/acceptance:** venue or acceptance date, when available.
* **Release:** a dated GitHub/PyPI/Hugging Face release, not a README edit.
* **Last substantive change:** a commit that changes tasks, evaluators,
  environment, or reproducibility. Documentation/cosmetic commits are not
  treated as benchmark novelty.

The requested official model pages report a **16 February 2026 knowledge
cutoff** for both [GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
and [GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol).
Therefore papers and releases dated 2026-02-17 onward are genuinely
post-cutoff for this pass.

## Search log

Queries were run against arXiv, official project pages, GitHub, Hugging Face,
and venue/maintainer pages. The exact search families used were:

| Date (ET) | Query family | Purpose |
| --- | --- | --- |
| 2026-08-30 | `browser agent benchmark 2026 stateful web benchmark release arXiv` | Find post-cutoff papers and unfamiliar names. |
| 2026-08-30 | `web agent benchmark self-hosted reset state grading adversarial 2026` | Screen for reset, writable-state, and safety structure. |
| 2026-08-30 | `site:github.com web benchmark browser agent 2026 release` | Find implementation updates rather than only papers. |
| 2026-08-30 | `WebArena-Pro`, `ST-WebAgentBench`, `WebForge`, `ClawBench`, `SCUBA`, `CRMArena-Pro`, `TheAgentCompany`, `WebChoreArena` | Verify candidate claims against the project/maintainer source. |

Search results are leads only. Recommendations below use the linked paper,
official repository, official dataset card, or venue page as the evidence.

## Strongest candidates and fit

Legend: **Yes** means explicitly documented by the primary source; **No** means
the source documents an incompatible design; **?** means it was not established
without running infrastructure (which this pass intentionally did not do).

| Candidate | First paper; publication/acceptance | Release and last substantive change (post-cutoff where found) | Environment, writes, exposure, reset | State grading; benign/adversarial pairing | License/assets | Assessment |
| --- | --- | --- | --- | --- | --- | --- |
| **ST-WebAgentBench** ([repo](https://github.com/segev-shlomov/ST-WebAgentBench), [dataset](https://huggingface.co/datasets/ST-WebAgentBench/st-webagentbench), [paper](https://arxiv.org/abs/2410.06703)) | 2024-10-09; ICLR 2026 publication listed by [IBM Research](https://research.ibm.com/publications/st-webagentbench-a-benchmark-for-evaluating-safety-and-trustworthiness-in-web-agents--1) (2026-04-23). | No formal release tag found. [Commit history](https://github.com/segev-shlomov/ST-WebAgentBench/commits/main?since=2026-01-01&until=2026-08-30): 2026-02-22 expanded CRM to three policy-load tiers (295 tasks); 2026-03-01 added 80 modality tasks/evaluator and policy-count updates (375 total); later URL/keepalive maintenance. | BrowserGym/WebArena plus SuiteCRM; official setup documents `env.reset()` and observations containing DOM, screenshot, AX tree, URL. Ordinary task actions include CRM records, exports, messages, and admin operations. Sites are external WebArena/AWS or separately set up, not one self-contained Docker bundle (**?**). New modality tasks alter page CSS/DOM in trusted setup, not participant-authored content. BrowserGym teardown appears to remove temporary config rather than restore all application state (**?**). | 11 specialized evaluators and CuP/CR/Risk metrics; state/trace checks are explicit. SuiteCRM has Easy/Medium/Hard versions of the same intent with 7/11.4/18.6 policy loads and explicit consent, sequence, credential, jailbreak, and error-handling constraints. These are policy-load comparisons, not WARP benign/adversarial twins; policy text is not shown to be an independent attacker write. | Apache-2.0 repo/HF card (HF download requires contact-information agreement). | **Complementary conditional safety candidate**; reset, exposure and evaluator applicability remain gating issues. Lead inspected GitHub `stwebagentbench/test.raw.json` task 47 (SuiteCRM), whose access policy requires a GitLab-shaped locator `#project_visibility_level_20`. Record this defect and separate any repaired or excluded cohort from native scores. Do not equate modality tiers with attacker-writable pairing. |
| **WebForge** ([repo](https://github.com/yuandaxia2001/WebForge), [dataset](https://huggingface.co/datasets/yuandaxia/WebForge), [paper](https://arxiv.org/abs/2604.10988)) | First arXiv 2026-04-13. | No formal release. [Commits](https://github.com/yuandaxia2001/WebForge/commits/main): initial evaluation release 2026-04-12; 2026-04-14 arXiv-link documentation only. | 934 self-contained static HTML/CSS/JS tasks, seven domains, three levels; localStorage supports stateful interactions. No external services/databases/APIs. Validation caps trajectories at 50 steps. This is highly self-hostable and deterministic in principle; independent participant exposure and a separate attacker write path are not documented. | Final-state comparison/operation-code/direct/mixed answer grading is documented. A “risk factor” is one difficulty dimension, not a benign/adversarial pair, and no policy/safety evaluator is described. | Apache-2.0; dataset and task pages are public. | **Low-deployment-burden synthetic substrate, not an unchanged-threat-model candidate.** Adding multi-user shared state would change the environment rather than constitute bounded WARP adaptation. |
| **WebArena-Verified** ([repo](https://github.com/ServiceNow/webarena-verified), [releases](https://github.com/ServiceNow/webarena-verified/releases)) | Workshop paper cited as 2025; no new paper needed for this implementation refresh. | Public PyPI 2026-01-07; Docker/uvx announcement 2026-02-02; releases page lists v1.2.3 on 2026-02-07 (GitHub rendering omits the year). Last observed changes are 2026-02-02/07 packaging and evaluator parity fixes. | 812 verified tasks (+258 hard subset) over Shopping, Shopping Admin, Reddit, GitLab, Wikipedia, and Map. Docker images and `env-ctrl` initialization are documented; network-trace replay permits offline evaluation. Ordinary issue/post/comment writes exist in inherited sites. Multi-user attacker-authored exposure and reset semantics beyond environment init are not specified. | Deterministic type-aware normalization/structural comparison; no LLM judge. No benign/adversarial safety pairing. | Apache-2.0; task data, Docker images, and evaluator are public. | **Best reproducibility baseline and direct WARP site overlap**, not a safety benchmark. Use as control/comparator, not as a completed PVPO design. |
| **WebArena-Pro** ([ServiceNow research page](https://www.servicenow.com/research/publication/imene-kerboua-weba-icml-workshops2026.html), [HF organization](https://huggingface.co/WebArena-Pro/datasets)) | Workshop publication page dated 2026-07 (paper page reports 300 tasks, 20 web apps, six domains). | No public release or task repository located by 2026-08-30. The [official HF organization](https://huggingface.co/WebArena-Pro/datasets) says no datasets are public yet. | 300 human-authored tasks, six domains, 20 self-hosted apps; each task is described as isolated/reproducible and observations include text, vision, audio, and video. Public reset implementation/assets remain unknown. | Human-validated success grading is described; no safety pairing or independent attacker-write mechanism is documented. | License and downloadable assets are unknown because the official dataset page is empty. | **Newest high-diversity lead only**. Recheck when release assets appear; do not schedule onboarding on the current public evidence. |
| **SCUBA** ([repo](https://github.com/SalesforceAIResearch/SCUBA), [paper](https://arxiv.org/abs/2509.26506)) | First arXiv 2025-09-30; ICLR 2026 paper (venue link in repo). | No formal release. Substantive repository update 2026-04-21 fixes MFA/setup; current setup docs remain active. | 300 Salesforce CRM tasks from admin/sales/service personas; BrowserGym and computer-use support; initial-state snapshots, modified-state snapshots, and task-level reset/diff-revert are described. Requires a Salesforce Developer Org, OAuth/refresh token and a public-cloud org—no local self-hosted site bundle. | Milestone and state checks; no paired attacker/benign safety suite. | Apache-2.0 repo, but Salesforce org provisioning and assets are external. | **Enterprise workflow comparator**, not a drop-in self-hosted WARP site. Verify reset fidelity and credentials before any adapter work. |
| **ClawBench (TIGER-AI-Lab)** ([repo](https://github.com/TIGER-AI-Lab/ClawBench), [paper](https://arxiv.org/abs/2604.08523), [release](https://github.com/TIGER-AI-Lab/ClawBench/releases/tag/v0.9.1)) | First arXiv 2026-04-09; v2 2026-07-20. | v0.9.1 release 2026-08-08; 2026-08-18 harness/runtime and tri-state-judge updates; 2026-08-22 docs/EMNLP acceptance. These are harness/release updates, not new benchmark task structure. | V1/V2 cover 153/130 tasks on 144/130+ live consumer websites in 15 categories. A Docker/Chromium wrapper, synthetic profile, and five-layer recording are provided, but the target websites remain live. An HTTP interceptor blocks critical final-submit requests to avoid side effects. No deterministic reset or attacker-writable sandbox is documented. | Human reference trajectory + agentic judge; no state-reset grader or benign/adversarial pair. | Apache-2.0 code; live-site assets/credentials and drift remain external. | **Useful exposure/workflow diversity lead, explicitly excluded from WARP onboarding** because live sites, interception, and missing reset/attacker sandbox violate core constraints. |
| **TheAgentCompany** ([repo](https://github.com/TheAgentCompany/TheAgentCompany), [paper](https://arxiv.org/abs/2412.14161)) | First arXiv 2024-12-19; no later venue date found. | v1.0.0 release 2024-12-20; last substantive changes observed 2025-11-17 (MongoDB image-reference fix), none after cutoff. | MIT-licensed simulated company with GitLab, Plane, ownCloud, RocketChat and browser/code/terminal/co-worker workflows; Docker/local or cloud setup, task initialization and quick reset are documented. Ordinary writable content is broad, but benchmark is not browser-only and evaluator assets are partly encrypted. | Deterministic + LLM evaluators, result/subcheckpoint grading; no explicit benign/adversarial pair. | MIT; 175 task images and source setup are public, with encrypted evaluator data. | **Strong stateful workflow comparator**, older and broader than browser-only WARP; adapt selectively rather than import wholesale. |
| **WebChoreArena** ([site](https://webchorearena.github.io/), [paper](https://arxiv.org/abs/2506.01952)) | First arXiv 2025-06-02; no later publication date found. | No post-cutoff release/substantive update observed. | 532 human-curated chores (Shopping, Shopping Admin, Reddit, GitLab plus cross-site) on simulated WebArena sites; reproducible state/evaluator inherited from WebArena. | State/URL/HTML/DB checks; no safety-pair design or separate attacker channel. | Public benchmark code/data (license should be checked in the selected commit). | **Useful task/workflow diversity comparator**, but older/narrow site mix and no PVPO analogue. |
| **CRMArena-Pro** ([repo](https://github.com/SalesforceAIResearch/CRMArena), [paper](https://arxiv.org/abs/2505.18878), [Salesforce overview](https://www.salesforce.com/blog/crmarena-pro/)) | First arXiv 2025-05-24; TMLR acceptance announced 2026-02 (repo). | No release; no post-cutoff task release. Repository states GUI access is no longer public after a Salesforce update. | 19 expert-validated, multi-turn B2B/B2C sales/service/CPQ task families on a live Salesforce org with synthetic data; API and GUI variants. No local bundle or deterministic reset is documented. | Persona/confidentiality awareness and API/GUI grading; no benign/adversarial pairing. | Research-only implementation; live-org credentials/assets required. | **Enterprise safety-awareness lead**, not an executable WARP environment under current access. |

## Newest signals versus best fit

The newest paper/repository activity is not automatically the best WARP
adapter. The following post-cutoff signals are worth tracking, but are
explicitly separated from the best-fit shortlist above:

| New signal | Dated primary evidence through 2026-08-30 | Why it is a lead, not a WARP environment |
| --- | --- | --- |
| **OpenApps** ([repo](https://github.com/facebookresearch/OpenApps), [paper](https://arxiv.org/abs/2511.20766)) | Repository source commit **2026-08-20** (`980becca`, HTMX/Pico local app work); merge commit **2026-08-24** (`7edfa641`). | Six local FastHTML/Python apps and BrowserGym/state rewards are useful for harness ideas, but the ordinary message route stamps a single “you” identity; no independent multi-user reader/writer exposure or attacker channel is established. Keep as architecture lead only. |
| **TimeWarp** ([repo](https://github.com/sparklabutah/timewarp), [paper](https://arxiv.org/abs/2603.04949)) | 2026-07-26 normalization/data change; 2026-07-27 judge-choice change; **0.2.0** tag/commit **2026-07-29** (`312ad528`). | Three sites × six historical eras and deterministic answer verifiers are valuable temporal/state ideas, but routes are news retrieval/search (GET-like) rather than ordinary participant writes; no attacker-to-independent-reader exposure. |
| **CAP** ([paper](https://arxiv.org/abs/2608.08392)) | First arXiv **2026-08-09**; reports 420 live-site tasks across 108 websites/24 domains with an agent-as-judge. | Live/open-web evaluation has no documented resettable, self-hosted attacker sandbox; use for capability/awareness comparison only. |
| **WebRetriever** ([paper](https://arxiv.org/abs/2607.06118)) | First arXiv **2026-07-07**. | Retrieval benchmark, not a documented stateful writable multi-user environment; no WARP transfer path found. |
| **STATE-Bench** ([official site](https://microsoft.github.io/STATE-Bench/)) | Official leaderboard/repository describes a **2026** benchmark with recent v0.8.x updates. | Tool-call database simulator rather than browser rendering; useful state-evaluation context, not a browser adapter. |
| **BrowserForge** ([paper](https://arxiv.org/abs/2608.24848)) | First arXiv **2026-08-25**. | Open-web proposer/solver trajectory generation uses browser sandboxes, not local clones of target sites; no eligible resettable attacker environment was established. |

## Why these are not equivalent

* ST-WebAgentBench is the only discovered benchmark that combines a large
  explicit safety policy taxonomy, human-consent actions, and state/trace
  evaluators with post-cutoff implementation expansion. Its 2026 modality work
  is a representation challenge (DOM-only versus visual-only), not evidence
  that another participant authored content that the evaluated agent later
  rendered. The CRM sample contamination noted above is a concrete onboarding
  risk.
* WebForge is the cleanest deterministic substrate and has a large, recent,
  permissively licensed task set, but it is single-participant static content.
  Its localStorage state is local browser state: transferring it to a fresh
  reader would be an environment mutation, not an ordinary participant HTTP
  write. It therefore does not provide the social/independent-exposure threat
  model WARP needs and is a synthetic-adaptation substrate, not a stronger
  onboarding candidate than TheAgentCompany.
* WebArena-Verified is the reproducibility/control baseline: public Docker,
  data, and deterministic evaluators. Its network-trace replay is excellent
  for regression, but it is not a safety-pair benchmark.
* WebArena-Pro is scientifically attractive (multimodal observations and 20
  apps) but currently cannot be onboarded: the official HF organization has no
  public datasets, and I found no public reset/assets/license.
* SCUBA and CRMArena-Pro provide realistic enterprise workflows but require a
  Salesforce Developer Org/OAuth or live GUI access. Treat “reset” claims as
  cloud-org procedures, not a self-hosted deterministic reset, until verified.
* ClawBench supplies unusually broad consumer-site/action diversity and a
  post-cutoff harness, but final-submit interception does not establish shared
  attacker-controlled state, a sandboxed target service, or rollback. It is useful for task taxonomy and
  recording ideas only.
* TheAgentCompany and WebChoreArena are stable comparators for long workflows,
  but neither defines WARP's independent rendered exposure plus paired safety
  conditions.

## Counterevidence, exclusions, and open checks

1. **No complete off-the-shelf match found.** None of the public candidates
   documents all of: two participant identities, participant-writable content,
   independent rendered exposure, deterministic per-task reset, structural
   state grading, and matched benign/adversarial variants. This leaves a genuine
   empirical transfer question; it does not itself support WARP's portability claim.
2. **Reset is frequently asserted at the harness level only.** WebArena-
   Verified documents `env-ctrl init`; ST's BrowserGym setup/teardown and
   WebForge localStorage need an actual reset-fidelity audit. Do not infer that
   a container restart removes all application state.
3. **Safety labels are not safety pairs.** ST policy-load tiers and modality
   tasks, WebForge's risk dimension, and CRMArena confidentiality personas are
   valuable safety dimensions, but none proves an ordinary attacker-authored
   write is independently rendered to a benign participant.
4. **Availability/licensing matters.** WebArena-Pro has no public dataset;
   SCUBA/CRMArena require Salesforce credentials; ClawBench requires live
   sites; ST HF access asks for contact information; TheAgentCompany encrypts
   evaluator assets. These are operational blockers even where scientific fit
   is high.
5. **Known contextual leads not recommended as WARP environments.** STATE-Bench
   (Microsoft) is a tool-call database simulator rather than browser rendering;
   CAP (arXiv:2608.08392, 2026-08-09) and WebRetriever (arXiv:2607.06118)
   evaluate live-site/open-web capability without a documented resettable
   attacker sandbox; TimeWarp (arXiv:2603.04949) varies three sites across historical
   interfaces, but the inspected news app has no ordinary attacker write; OpenApps (Meta,
   arXiv:2511.20766) has local apps/state rewards but its ordinary message
   route does not establish independent multi-user identity. They may inform
   evaluation design, not direct onboarding.

## Recommendation to the WARP adapter work

Use a **fit-first control set**: WebArena-Verified for deterministic
WebArena-compatible scoring and Docker control; ST-WebAgentBench as a staged
safety-policy/consent and evaluator stress test (after its reset and data
contamination audit); and TheAgentCompany as the strongest broad stateful
workflow comparator. WebForge is useful only for a synthetic static/stateful
adapter experiment because localStorage cannot provide WARP's independent
exposure. Use WebArena-Pro as a watch-list item pending public assets. Use
ClawBench, SCUBA, CRMArena-Pro, WebChoreArena, OpenApps, and TimeWarp as
taxonomy/workflow leads, not as sources of an unverified resettable PVPO
environment.
