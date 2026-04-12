# WebArena-Infinity

**Source:** https://webarena.dev/webarena-infinity/  
**GitHub:** https://github.com/web-arena-x/webarena-infinity
**Title:** *WebArena-Infinity: Generating Browser Environments with Verifiable Tasks at Scale*  
**Author:** Shuyan Zhou  
**Date:** March 2026

---

## 1. What WebArena-Infinity Is

WebArena-Infinity is a system for **automatically generating realistic browser environments and verifiable tasks at scale** for training and evaluating browser-use agents.

### Core motivation
Creating realistic web environments is usually expensive and manual because it requires:
- sandbox creation
- data curation
- task design
- programmatic verifier writing
- engineering expertise

### Main claim
The project aims to automate this workflow using a **multi-agent pipeline** that starts from static artifacts such as:
- recorded workflows
- user manuals

### Reported outcomes
- Each environment can be generated in **under 10 hours**
- Cost is reported as **under $100 per environment**
- Process is **highly parallelizable**
- Initial release includes:
  - **10 environments**
  - **1,260 verifiable tasks**
  - **2,070 successful trajectories**

---

## 2. High-Level Contribution

The paper proposes a pipeline that automatically produces:

1. **High-authenticity web environments**
2. **Functionally correct implementations**
3. **Verifiable realistic tasks**
4. **Progressively harder tasks through self-challenging**
5. **RL-friendly environments with fast reset and replication**

A key idea is that **coding agents have privileged access to source code and execution artifacts**, allowing them to:
- inspect environment implementations
- debug issues
- modify task logic
- iteratively improve both environments and verifiers

---

## 3. System Overview

The approach uses two agent types:

### 3.1 Coding agent
Responsible for:
- generating web applications
- creating tasks
- writing verifiers
- repairing bugs
- auditing failures
- increasing task difficulty

### 3.2 Browser-use agent
Responsible for:
- executing tasks through the front-end UI like an end user

---

## 4. Pipeline Stages

The full pipeline has **5 stages**.

### Stage 1: Web environment generation
The coding agent generates a web app conditioned on static artifacts such as:
- user manuals
- recorded workflows

### Stage 2: Functional correctness task generation
The coding agent creates:
- simple functional tasks
- corresponding programmatic verifiers

The browser-use agent executes these tasks to test whether the site actually works through the UI.

### Stage 3: Realistic task generation
The coding agent generates more realistic, difficulty-graded tasks and associated verifiers.

### Stage 4: Self-challenging / hardening
The coding agent reviews prior agent trajectories and failure patterns, then synthesizes harder tasks that target capability limits.

### Stage 5: Regression testing
A final regression pass checks that later changes did not break earlier functionality.

---

## 5. Web Environment Generation

### Main design choice
Environment generation is **conditioned on real application artifacts**, especially **product user manuals**.

### Why manuals are used
Manuals provide:
- core concepts
- workflow descriptions
- UI layout hints
- interaction patterns

Example kind of instruction from manuals:
- “click the top-right button to access settings”

### Scope of generation
The system selects:
- **10 functional sections**
- from **8 applications**
- each corresponding to a coherent subset of product functionality

Example:
- GitLab planning and tracking module

---

## 6. Functional Correctness Verification

### Goal
Check whether generated websites actually work when interacted with through the front-end.

### Method
The coding agent generates:
- basic functionality tasks
- corresponding verifiers

Then the browser-use agent executes the tasks.

### Interpretation
These tasks are intentionally simple, so a capable agent should complete them reliably. Reportedly:
- average pass rate is **above 90%**

This stage is used as a proxy for:
- environment functional correctness

---

## 7. Auditing Loop

After task execution, a separate coding agent acts as an **auditing agent**.

### What the auditing agent does
- reviews recorded trajectories
- focuses especially on failures
- determines whether each failure comes from:
  - an environment bug
  - a verifier issue
  - browser-agent limitations
- fixes confirmed environment bugs
- repeats until remaining failures are judged to mostly come from agent limitations

### Example from GitLab Plan & Track
A task asked the agent to delete an issue:
- **Instruction:** delete the issue titled *“Markdown preview renders incorrectly with nested code blocks”*
- The verifier checked backend state to confirm the issue no longer existed
- The backend supported deletion, but the browser-use agent failed because the delete button was missing in the UI
- The auditing agent fixed this by adding the missing delete button to the HTML

### Importance
The audit loop is not just error analysis:
- it directly improves environment correctness
- it improves task validity
- it improves verifier reliability

---

## 8. Realistic Task Generation and Hardening

### Realistic tasks
This stage generates tasks intended to be:
- realistic
- difficulty-graded
- naturally grounded in the environment

### Hardening
Even with prompts about difficulty, coding agents tend to produce tasks that are too easy.

To address this, WebArena-Infinity adds a **hardening step**:
- inspect prior agent execution logs
- identify current capability frontier
- generate harder tasks over multiple rounds

### What makes hardened tasks harder
Not just longer action chains, but:
- implicit prerequisites
- conditional reasoning
- domain-specific constraints
- naturally arising multi-step dependencies

### Example difficulty progressions

#### Career portal
- Easy: Save a Microsoft internship listing
- Medium: Leave a comment on a user’s post
- Hard: RSVP to all employer-hosted upcoming events and save each employer’s unsaved active jobs

#### Prescription management
- Easy: Approve a refill request
- Medium: Approve the refill and modify the sig
- Hard: Update allergy status, discontinue one medication, and prescribe a new treatment with detailed constraints

#### Presentation editor
- Easy: Change one slide transition
- Medium: Replace every dissolve transition
- Hard: Change default template, default transition, timing/easing, and slide number formatting across matching slides

#### Digital wallet
- Easy: Stop weekly digest emails
- Medium: Save all current food and drink cashback offers
- Hard: Transfer funds, split them, convert part to EUR, and invest part in Bitcoin

---

## 9. Configurations

### Coding agent
- **Claude Code**
- **Opus 4.6**
- high-effort setting with planning mode
- equipped with a frontend-design skill

### Browser-use agent
- **Gemini-3-Flash** with the **Browser Use** harness

### Additional evaluated agents
- **Kimi-K2.5**
- **Qwen-3.5-Plus**

### Note from authors
A stronger browser-use agent may improve generation quality, but at the cost of:
- greater computation
- slower iteration

---

## 10. Key Design Choices

### 10.1 Text-based interaction as a first-class citizen

The environments are intentionally designed to work well with browser-use agents that rely mainly on **textual observations**, such as:
- accessibility trees
- DOM-exposed UI structures

#### Rationale
Language models are described as more robust and generalizable when operating on textual representations.

#### Practical implications
The environments encourage:
- accessibility best practices
- browser-native UI elements
- DOM-visible interactions like popups and dropdowns

#### Additional note
The browser-use agent is run **without vision input** in this setup for:
- speed
- efficiency

---

### 10.2 Enhancing verifier reliability

Verifier correctness is critical, especially for RL training.

Two guardrails are used:

#### (1) Programmatic sanity check
A privileged program directly manipulates backend state and then runs the same verifier used in evaluation.

Interpretation:
- if the verifier fails here, the verifier is likely wrong

#### (2) Automatic auditing
The auditing agent analyzes failed trajectories and determines whether failures come from:
- environment bugs
- verifier bugs
- impossible tasks
- browser-agent limitations

---

### 10.3 Standalone verifiers

Verifiers are designed to be **standalone** through an exposed API.

#### Benefit
This separates:
- **task execution** from
- **task verification**

#### Claimed advantage over prior work
Unlike benchmarks such as WebArena that rely on browser-based inspection of rendered pages, this design avoids expensive browser-side validation and is claimed to be:
- more efficient
- more reliable

---

## 11. Environment Protocol

The environments are designed to be:
- lightweight
- easily replicable
- easy to reset
- suitable for parallel RL training

### Implementation style
Each environment is:
- a self-contained directory
- static HTML/CSS/JS
- served by a lightweight Python HTTP server
- no database
- no external dependencies

### Central design idea
**The browser owns all application state.**

On every user interaction:
- the browser serializes its full state
- pushes it to the server
- the server stores only the latest copy in memory

The server acts as a **passive state mirror**.

### Why this matters
This allows:
- verifiers to query state directly without touching the UI
- clean separation between acting and verifying
- very fast resets

### Reset mechanism
- the first state push is stored as an immutable seed snapshot
- reset restores that snapshot in memory
- the browser is notified via **Server-Sent Events (SSE)** to reload

### Minimal API
- `PUT /api/state` → browser pushes state after mutations
- `GET /api/state` → verifiers read current state
- `POST /api/reset` → restore seed snapshot and notify browser
- `GET /api/events` → browser listens for reset events via SSE

### Scaling implication
Replication is easy:
- copy files
- launch separate processes on different ports

This makes parallel RL rollouts straightforward.

---

## 12. Environment and Task Statistics

### Initial release summary

| Environment | Domain | Source | Tasks | Any-Agent Success |
|---|---|---:|---:|---:|
| Career Exploration | Careers | Handshake | 200 | 144 |
| Project Planning | DevOps | GitLab | 140 | 93 |
| Accounting & Invoicing | Finance | Xero | 120 | 100 |
| Personal Wallet | Finance | PayPal | 140 | 133 |
| Clinical Records | Healthcare | Elation EHR | 120 | 106 |
| Prescription Management | Healthcare | Elation EHR | 120 | 102 |
| Email Management | Productivity | Gmail | 60 | 55 |
| Accounts & Contacts | Productivity | Gmail | 120 | 89 |
| Email Client | Productivity | Superhuman | 120 | 68 |
| Account Settings | Project Mgmt | Linear | 120 | 100 |
| **Total** |  |  | **1,260** | **990** |

### Meaning of “Any-Agent Success”
This is reported as:
- the number of tasks for which at least one agent succeeded

The authors present it as a conservative lower bound on:
- task validity
- environment quality

---

## 13. Filtering Criteria

### Quality gate
Environments with **functional test pass rate below 50%** are excluded from release.

### Why
The authors found functional task success rate to be a strong indicator of:
- environment quality
- usability
- correctness

### Interpretation
This acts as a simple automated filtering heuristic without heavy manual inspection.

---

## 14. Performance of Browser-Use Agents

### Agents evaluated
- Gemini-3-Flash with Browser Use
- Qwen-3.5-Plus
- Kimi-K2.5

### Overall results across 1,260 real tasks
- **Gemini-3-Flash + Browser Use:** 69.3% (873 / 1260)
- **Qwen-3.5-Plus:** 49.1% (619 / 1260)
- **Kimi-K2.5:** 45.9% (578 / 1260)

### Additional observations
- Performance varies significantly by environment
- PayPal My Wallet is the easiest environment for all agents
- Kimi-K2.5 underperforms Qwen overall, but does better on Gmail-related apps

### Comparison to other benchmarks
The authors note these numbers are lower than reported scores on established benchmarks.

Examples cited:
- Kimi-K2.5:
  - 58.9% on WebArena
  - 63.3% on OSWorld-verified
  - 45.9% here
- Qwen-3.5-Plus:
  - 62.2% on OSWorld-verified
  - 49.1% here

### Claimed implication
Automatically generated environments can still be meaningfully difficult, not trivially easy.

---

## 15. Ablation Studies

### 15.1 Richer source artifacts improve environment quality

The authors compare environments generated from:
- full documentation
- only brief high-level descriptions

#### Metrics examined
- functional complexity
- task diversity
- feature authenticity

#### Reported findings
Using full documentation generally leads to:
- more capabilities implemented
- more diverse tasks
- better fidelity to the real product
- less feature hallucination
- better coverage of documented features

#### Noted exception
GitLab Plan & Track showed weaker gains, attributed to the structure of its documentation.

---

### 15.2 Auditing improves functionality and verifier reliability

The evaluate–audit loop showed monotonic improvement in functional-task pass rates across tested environments.

#### Example
- Xero Invoicing reportedly improved from **72% to 100%**

#### Error category breakdown
The auditing reports suggest failures were attributed roughly as:
- **45%** environment/task-side issues
  - application bugs
  - verifier bugs
  - impossible tasks
- **55%** browser-agent limitations

#### Claimed takeaway
Auditing is important not only for fixing apps but for preventing incorrect reward signals in downstream training.

---

### 15.3 Self-challenging increases task difficulty

The self-challenging stage generates tasks harder than the original “hard” tasks.

#### Reported effect
For Gemini-3-Flash with Browser Use:
- original hard tasks: **54.2%**
- hardened tasks: **41.9%**

#### Interpretation
Using prior trajectories helps the system discover weaknesses beyond what initial generation alone can identify.

---

### 15.4 Limited functional regression across iterations

A concern with iterative generation is that later edits may break earlier functionality.

#### Reported result
In most environments, functional correctness stays close to initial levels, but some environments regress significantly.

| App | Initial | Final | Delta |
|---|---:|---:|---:|
| Elation Clinical | 100.0% | 98.2% | -1.8pp |
| Xero Invoicing | 98.3% | 94.8% | -3.4pp |
| GitLab Plan & Track | 98.2% | 98.2% | +0.0pp |
| PayPal My Wallet | 96.4% | 96.4% | +0.0pp |
| Superhuman | 93.3% | 73.3% | -20.0pp |
| Elation Prescriptions | 87.5% | 89.3% | +1.8pp |
| Elation Patient Comm. | 75.9% | 32.8% | -43.1pp |

#### Interpretation
Most environments remain stable, but some deteriorate materially. Functional pass rate is therefore useful as an automated quality gate.

---

## 16. Why This Matters

WebArena-Infinity is positioned as a way to make browser-agent benchmarking and RL data generation much more scalable by replacing manual benchmark construction with an automated pipeline.

### Main practical advantages claimed
- lower cost
- faster environment creation
- scalable parallel generation
- verifiable reward signals
- RL-friendly reset and replication
- realistic, difficulty-graded tasks
- reduced need for expensive manual environment engineering

---

## 17. Key Ideas in One Page

### Problem
Manual construction of realistic browser environments and verifiable tasks is slow, expensive, and hard to scale.

### Solution
Use:
- a coding agent to generate apps, tasks, and verifiers
- a browser-use agent to execute tasks
- an audit loop to repair bugs
- a self-challenging loop to generate harder tasks

### Technical enablers
- conditioning on manuals and workflows
- standalone API-based verifiers
- browser-owned application state
- lightweight resettable environment protocol

### Main result
A fully automated pipeline can generate environments that are:
- realistic
- verifiable
- challenging for current browser-use agents
- suitable for large-scale training and evaluation

---

## 18. Limitations / Caveats Implied by the Text

The source text also suggests several practical caveats:

- generated environments are **not bug-free**
- later iterations can introduce regressions
- environment quality still varies
- stronger browser-use agents may improve results but increase cost
- task validity is estimated partly through indirect signals such as any-agent success and functional pass rates

---

## 19. Citation

```bibtex
@article{zhou2026wainf,
  title   = "WebArena-Infinity: Generating Browser Environments with Verifiable Tasks at Scale",
  author  = "Shuyan Zhou",
  journal = "shuyanzhou.com",
  year    = "2026",
  month   = "March",
  url     = "https://webarena.dev/webarena-infinity/"
}
````

---

## 20. References Mentioned in the Source

* Zhou et al. *WebArena: A Realistic Web Environment for Building Autonomous Agents.* ICLR 2024
* Xie et al. *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments.* NeurIPS 2024
* Koh et al. *VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks.* ACL 2024
* Xu et al. *TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks.* ICLR 2025
* Browser Use. *Enable AI to Access the Web.* GitHub 2024
* Kimi Team. *Kimi K2.5: Visual Agentic Intelligence.* arXiv:2602.02276, 2026
* Qwen Team. *Qwen3.5: Towards Native Multimodal Agents.* 2026
