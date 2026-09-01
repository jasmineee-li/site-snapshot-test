# Infrastructure and integration feasibility

**Research date:** 2026-08-30 (ET)
**Scope:** read-only source and documentation review for two promising
browser-benchmark paths. No setup, install, browser run, credential use,
container start, evaluator run, or runtime validation was performed. Commands
in this note are proposed future sandbox smoke plans, not test results.

The source pins used here are [WorkArena
`a772230a94cf1caf4166b8ead3983f3b3786455b`](https://github.com/ServiceNow/WorkArena/tree/a772230a94cf1caf4166b8ead3983f3b3786455b),
[BrowserGym
`9e779f087de9a65668b6974d11f9ce9816026e96`](https://github.com/ServiceNow/BrowserGym/tree/9e779f087de9a65668b6974d11f9ce9816026e96),
and [TheAgentCompany
`98b68ef82a47690c316f42fddb05baafaab56851`](https://github.com/TheAgentCompany/TheAgentCompany/tree/98b68ef82a47690c316f42fddb05baafaab56851).
The WorkArena pin is the 2026-02-03 merge that hides context menus (a
substantive exploit-prevention maintenance change); the BrowserGym pin is the
2026-03-17 TimeWarp/setup clarification; TAC's last substantive source change
in this review was 2025-11-17. These dates are maintenance provenance, not a
claim that the projects are continuously maintained.

## Decision summary

| Candidate | WARP admission decision | Useful part | Blocking evidence |
| --- | --- | --- | --- |
| WorkArena + BrowserGym | **Do not admit as a WARP Site.** Keep BrowserGym as a possible generic runner substrate only. | Mature browser observations/actions, screenshots, videos, and trajectory artifacts. | ServiceNow is a remote/gated instance with credentials; the base task gives the participant `admin` by default and setup/evaluators use privileged REST. No independent fresh-anonymous reader or WARP-style attacker-authored exposure is documented. Reset is task cleanup, not a database snapshot. |
| TheAgentCompany (TAC) | **Conditional exploratory candidate.** Admit only a deliberately reduced, browser-only task subset after exposure and reset canaries pass. | Self-hosted multi-app state (GitLab, Plane, ownCloud, RocketChat), cross-app workflows, task images with init/evaluator/checkpoints. | The benchmark documents username/password for every service, uses a host-network Docker control plane with an unauthenticated reset API, and includes shell/code/NPC/LLM paths. A public GitLab/Plane read route and ordinary-user writer are not established. Existing evaluators use fixed fixtures and privileged APIs. |

This is a feasibility decision, not a safety result. A source claim marked
**documented** below is directly visible in the pinned source. **Inferred** is
an integration implication. **Unknown** means a future disposable sandbox
canary is required. **Counterevidence** is a source behavior that fails a WARP
gate or weakens a proposed claim.

## 1. WorkArena and BrowserGym

### What the source actually implements

* WorkArena tasks subclass `AbstractServiceNowTask`. Its constructor defaults
  `user_roles=["admin"]`; `setup()` creates a new user through an admin table
  API, copies credentials into the task instance, and logs in with those
  credentials. Teardown deletes the generated user through the same privileged
  API. See [`tasks/base.py`](https://github.com/ServiceNow/WorkArena/blob/a772230a94cf1caf4166b8ead3983f3b3786455b/src/browsergym/workarena/tasks/base.py).
* `SNowInstance` resolves a remote developer/pool instance from environment
  credentials or a gated Hugging Face instance pool. WorkArena's README and
  BrowserGym's [WorkArena setup guide](https://raw.githubusercontent.com/ServiceNow/BrowserGym/9e779f087de9a65668b6974d11f9ce9816026e96/docs/src/environments/workarena.rst)
  require an instance URL/password, `browsergym-workarena`, the WorkArena
  installer, and Chromium. The instance and install code are
  [`instance.py`](https://github.com/ServiceNow/WorkArena/blob/a772230a94cf1caf4166b8ead3983f3b3786455b/src/browsergym/workarena/instance.py)
  and [`install.py`](https://github.com/ServiceNow/WorkArena/blob/a772230a94cf1caf4166b8ead3983f3b3786455b/src/browsergym/workarena/install.py).
* WorkArena task setup and validation routinely call authenticated
  [`table_api_call`](https://github.com/ServiceNow/WorkArena/blob/a772230a94cf1caf4166b8ead3983f3b3786455b/src/browsergym/workarena/api/utils.py)
  rather than observing a public rendered resource. Form validators query the
  expected record and generally check only expected fields; extra fields are
  not rejected. The compositional delete validator rewards any run in which
  the target row is gone, even if unrelated rows were also deleted
  ([`delete_record.py`](https://github.com/ServiceNow/WorkArena/blob/a772230a94cf1caf4166b8ead3983f3b3786455b/src/browsergym/workarena/tasks/compositional/delete_record.py)).
  WorkArena issue [#155](https://github.com/ServiceNow/WorkArena/issues/155)
  and draft PR [#156](https://github.com/ServiceNow/WorkArena/pull/156) call
  out these validator weaknesses; the draft had not been shown as tested on a
  live instance.
* BrowserGym `BrowserEnv.reset()` creates a fresh browser context and calls
  task setup; it does not restore the ServiceNow database. BrowserGym's
  experiment loop records action/observation/error/timestamp artifacts and can
  save screenshots and videos ([`env.py`](https://github.com/ServiceNow/BrowserGym/blob/9e779f087de9a65668b6974d11f9ce9816026e96/browsergym/core/src/browsergym/core/env.py),
  [`loop.py`](https://github.com/ServiceNow/BrowserGym/blob/9e779f087de9a65668b6974d11f9ce9816026e96/browsergym/experiments/src/browsergym/experiments/loop.py)).
* BrowserGym's high-level action set is rich enough for forms, lists and
  navigation, but its base action executor executes supplied Python with
  `exec` and explicitly warns that this is unsafe
  ([`action/base.py`](https://github.com/ServiceNow/BrowserGym/blob/9e779f087de9a65668b6974d11f9ce9816026e96/browsergym/core/src/browsergym/core/action/base.py),
  [`action/highlevel.py`](https://github.com/ServiceNow/BrowserGym/blob/9e779f087de9a65668b6974d11f9ce9816026e96/browsergym/core/src/browsergym/core/action/highlevel.py)).
  A WARP agent must receive an allow-listed browser action grammar, never this
  arbitrary Python surface.

### WARP gate matrix

| Gate | Documented evidence | Inference / counterevidence | Status |
| --- | --- | --- | --- |
| Ordinary participant attacker | A task can request non-admin roles, but the base default is `admin` and creation uses an admin API. | The default participant is privileged; role-specific tasks need a new contract proving the requested role can perform the ordinary write. | **Blocked** until a regular-role writer is demonstrated. |
| Fresh anonymous reader | Evaluators and setup use authenticated ServiceNow API/login. No source describes an anonymous route. | A remote ServiceNow instance may expose some public pages, but this cannot be inferred from API checks. | **Unknown; fail closed** until an independent context with no storage state renders the exact resource. |
| Exact rendered identity | Validators retain record IDs/sys_ids and query fields. | That is privileged state evidence, not a second participant seeing the exact actor/body on a rendered route. | **Missing** for WARP exposure. |
| Isolation and reset | Browser context reset and task teardown delete the generated user; task classes may delete their own fixtures. | No DB snapshot or whole-instance golden-state restore is documented. Shared remote instances and incomplete cleanup can contaminate runs. | **Insufficient** for a WARP golden-state gate. |
| State grading | Many tasks perform deterministic table/API checks. | Expected-field-only validators and the delete flaw do not reject all unintended side effects. | **Requires a WARP wrapper and negative tests.** |
| Authoring | New Python task classes implement `setup_goal`, browser start, `validate`, and `teardown`. | The authoring model is ServiceNow-specific and assumes admin setup APIs. | **Reusable only as a separate adapter.** |
| Debug/traces | BrowserGym captures observations, actions, errors, screenshots, videos, and trajectories. | Good DX does not supply exposure/reset evidence. | **Reusable.** |
| Action richness and containment | Scroll, fill, select, click, drag, tabs/navigation and related actions are available. | `exec`-based execution is an attacker escape surface. | **Reusable only behind strict WARP action mediation.** |
| Resources/dependencies | Gated HF instances, a ServiceNow developer/pool instance, instance credentials, WorkArena installer, Playwright/Chromium, and external network are required. | Not a local Docker image or deterministic self-contained benchmark. | **Operational blocker.** |
| License/maintenance | WorkArena and BrowserGym are Apache-2.0; pins above provide dated provenance. | Remote service and gated data remain external dependencies. | **Source license identified; service/data rights and deployment remain separate.** |

### Future sandbox smoke plan (not run)

Keep instance credentials in host-only approved configuration. Run the
following only on a disposable benchmark host, never in an agent-visible
container or WARP ordinary-reader context:

```sh
# Placeholders only; do not put a real password in a task, fixture, or log.
export SNOW_INSTANCE_URL='https://<disposable-instance>'
export SNOW_INSTANCE_UNAME='admin'
export SNOW_INSTANCE_PWD='<host-secret>'
workarena-install --instance-url "$SNOW_INSTANCE_URL" \
  --instance-password "$SNOW_INSTANCE_PWD"
playwright install chromium
```

Then, using the documented BrowserGym API, instantiate one task and call
`env.reset(seed=0)` in a host harness. Do not call arbitrary Python actions.
The acceptance sequence must be:

1. Host setup creates one uniquely identified fixture with the admin observer
   and provisions a separate regular participant account.
2. The regular participant performs one ordinary UI write; the write result
   records parent ID, child ID, actor, body identity, and canonical route.
3. A *new* browser context with `storage_state=None` visits that exact route.
   Read only HTTP/rendered DOM/screenshot evidence. If it redirects to login,
   hides the child, changes actor/body, or requires the writer cookie, stop and
   mark the Site inadmissible. Do not substitute a logged-in reader or a root
   page.
4. Run `env.reset(seed=1)` and independently verify, with the host-only
   observer, that the fixture and any unrelated sentinel return to the golden
   baseline. Repeat with two task seeds. Context reset alone is not proof.
5. Exercise a create/edit/delete validator with an unrelated extra mutation.
   Preserve the native score and document its blind spots; a separate WARP
   safety predicate must detect the scoped unintended effect. Do not silently
   repair native validators and compare the resulting score as unchanged.

**Integration estimate (explicitly an estimate):** a BrowserGym observation /
trace adapter is small-to-medium (roughly 2–5 engineering days, excluding a
live instance), but direct WorkArena admission is large and uncertain (roughly
several weeks for isolated instance provisioning, ordinary-role semantics,
anonymous exposure, reset, and a WARP evaluator). These numbers are planning
estimates, not measured effort.

## 2. TheAgentCompany (TAC)

### Deploy, reset, task, and evaluator source

* TAC's [server setup guide](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/docs/SETUP.md)
  requires Docker/Compose, `curl`, host networking, access to the Docker
  socket, and at least 30 GB free disk (the guide gives an EC2 `t3.2xlarge`
  as a reference). `servers/setup.sh` pulls many pinned images, starts an API
  server with host networking, mounts `/var/run/docker.sock`, and waits for
  health checks for GitLab, Plane, ownCloud, and RocketChat
  ([setup source](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/servers/setup.sh)).
* The API server binds Flask to `0.0.0.0:2999`. Its reset routes have no
  authentication in the pinned source and return HTTP 202; Plane/RocketChat resets launch asynchronously, while
  other routes wait for their reset command. The routes and health checks are visible in
  [`api-server.py`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/servers/api-server/api-server.py)
  and its [README](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/servers/api-server/README.md).
  This control plane must be on a host-only network; exposing port 2999 to the
  agent would grant a reset-all-services primitive.
* A task image contains `task.md`, `dependencies.yml`, `populate_data.py`,
  `evaluator.py`, checkpoints, and a Dockerfile. `/utils/init.sh` adds the
  synthetic `the-agent-company.com` host entry, calls `/utils/reset.sh`, then
  runs pre-init, optional NPC, data-population, and post-init hooks
  ([`init.sh`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/base_image/init.sh),
  [`reset.sh`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/base_image/reset.sh)).
  Reset posts to `api/reset-*` and polls health for up to 15 minutes; health is
  not proof that task data equals a golden snapshot.
* The evaluation guide says every service requires username/password and that
  baseline agents are logged into all services. It also says many tasks need an
  environment LLM for NPCs or grading and that evaluator files are encrypted
  ([`docs/EVALUATION.md`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/docs/EVALUATION.md)).
  The public guide's example uses `--output_path`, while the pinned
  [`/utils/eval.py`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/base_image/eval.py)
  parser accepts `--result_path`; an adapter must pin and smoke the actual
  parser rather than copy the documentation typo. No credential or decryption
  fixture value is reproduced here.
* The strongest structural candidate is the original cross-app
  `pm-update-plane-issue-from-gitlab-status` task. Its instruction makes the
  agent inspect two GitLab issues and update corresponding Plane state/cycle
  values ([`task.md`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/pm-update-plane-issue-from-gitlab-status/task.md)).
  The upstream evaluator checks trajectory URLs, then looks up fixed issue,
  cycle, and completed-state records through authenticated API clients and
  awards checkpoint/bonus points
  ([`evaluator.py`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/pm-update-plane-issue-from-gitlab-status/evaluator.py)).
  This is useful state/action structure, but not a WARP reader: IDs and actor
  attribution are not the writer-created child, and trajectory URL presence is
  not proof of painted visibility.
* TAC is MIT-licensed. Its task images and setup source are public, but the
  encrypted evaluator payload, optional LLM/NPC services, and multi-container
  host control are operational dependencies. The source pin predates the
  requested post-cutoff window; do not call it a new 2026 release.

### WARP gate matrix

| Gate | Documented evidence | Inference / counterevidence | Status |
| --- | --- | --- | --- |
| Ordinary participant attacker | Task instructions assume authenticated services; the server README lists service credentials (omitted here). | TAC baseline is a logged-in company user, often with broad access. No regular-participant writer lifecycle or role matrix is documented. | **Unknown; host-provision a least-privilege writer first.** |
| Fresh anonymous reader | The evaluation guide states all services require username/password. | That strongly conflicts with the WARP fresh-anonymous gate, but source inspection does not prove that every GitLab project/issue route rejects anonymous access. Do not infer either outcome. | **Unknown; exact GitLab canary is mandatory.** |
| Exact rendered identity | RocketChat and GitLab tasks contain ordinary content and checkpoint state. | Existing evaluators use fixed names/API lookups and may accept trajectory URL visits; they do not prove a fresh reader saw the new writer-attributed child. | **Missing until a writer-created ID/actor/body is rendered and captured.** |
| Isolation/reset | `/utils/reset.sh` resets each dependency named in `dependencies.yml` and waits for health; task init repopulates fixtures. | API resets are unauthenticated and destructive, with some asynchronous routes; reset health does not prove no residual rows/comments. | **Conditional; isolate control plane and add exact-state proof.** |
| State grading | Checkpoints and deterministic/LLM graders are part of each task image. | Fixed fixture IDs, trajectory-only checkpoints, encrypted code, and optional LLM graders do not satisfy a WARP host-owned exact-state contract. | **Requires candidate-specific evaluator.** |
| Browser-only scope | Services expose web UIs and cross-app tasks. | Task images also include shell/code/terminal workflows, NPC/co-worker interactions, file transfers, and arbitrary methods; browser-only task subset is not enumerated. | **Unknown; select and enforce a strict subset.** |
| Action richness | GitLab/Plane/RocketChat workflows exercise navigation, forms, comments, state/cycle changes, and messages. | The benchmark permits ways of working that bypass the browser. | **Potentially rich after mediation.** |
| Debug/traces | Evaluation accepts an optional trajectory and task containers retain task artifacts. | Format/coverage are task-specific; encrypted evaluator prevents an examinee from inspecting grading. | **Reusable only after a WARP trace schema adapter.** |
| Resources/dependencies | 30+ GB, many images/databases, host networking, Docker socket, optional LiteLLM/NPC services; Mac/Windows host-network caveats are documented. | A reset API on a host interface plus Docker socket is a high-impact attacker surface. | **Large isolation/DX risk.** |
| License/maintenance | MIT source; pinned task/evaluator files are public, encrypted evaluator is shipped in images. | Last substantive source change found was pre-cutoff. | **Source license identified; asset rights, maintenance and evaluator opacity remain risks.** |

### Conditional TAC candidate and safe smoke plan (not run)

The PM GitLab→Plane task is a useful *conditional* candidate only if the
existing task's carrier can be preserved. A new public project or a changed
visibility setting would be a different threat-model variant, not a harmless
configuration tweak. First inspect the existing seeded project anonymously;
if it is private/login-gated, stop or label the variant explicitly rather than
silently making it public.

Proposed future disposable-host sequence:

```sh
# Host-only control plane; use the pinned source and private Docker network.
git clone https://github.com/TheAgentCompany/TheAgentCompany.git tac-src
cd tac-src
git checkout 98b68ef82a47690c316f42fddb05baafaab56851
bash servers/setup.sh

# Host-only readiness checks; never route :2999 into the agent container.
curl -fsS http://127.0.0.1:2999/api/healthcheck/gitlab
curl -fsS http://127.0.0.1:2999/api/healthcheck/plane
```

For a selected pinned task image, initialize it with the documented
`SERVER_HOSTNAME`/`/utils/init.sh` flow in an isolated task container. The
agent container must have only the selected app origins, no `:2999`, no Docker
socket, no host filesystem, no service-admin credentials, no evaluator key,
and no LLM/NPC control channels. Use BrowserGym-style allow-listed browser
actions; reject shell, Python, direct REST, and NPC/API shortcuts.

The acceptance canary is deliberately ordered:

1. Host setup creates a disposable task instance and one least-privilege
   regular writer account. Keep reset/evaluator/admin credentials in host
   memory or approved local configuration only.
2. The writer performs one ordinary UI write to a chosen GitLab project (or a
   separately named task variant) and records the exact parent ID, child ID,
   actor identity, normalized body, canonical same-origin route, and expected
   Plane target.
3. A fresh reader browser context is created with no storage state, cookies,
   credentials, or writer session. It visits the exact GitLab child route and
   must render the expected identity/body. A login redirect, missing content,
   changed actor/body, or root-page-only visibility is a hard failure.
4. The host then runs the pure browser cross-app workflow (read GitLab status,
   update the matching Plane issue/cycle) and captures BrowserGym/WARP action,
   screenshot, DOM/AXTree, URL, and network traces. Exclude TAC tasks that need
   shell, terminal, file system, NPC, or API shortcuts.
5. Grade exact state through a new host-owned observer keyed to the writer's
   IDs. Check the intended Plane state/cycle and assert that an extra unrelated
   mutation causes failure. Do not use the upstream fixed-fixture API evaluator
   as the fresh reader; it is a post-run privileged grader only.
6. Invoke the control-plane reset from the host and wait for both health and
   exact baseline sentinels. A 202 response or healthy service is not reset
   proof. Repeat the writer/read/cross-app/reset sequence twice and verify no
   writer child, unrelated mutation, cookie, or task-specific fixture remains.

The reset API itself is a destructive capability. Firewall it to the host
control namespace and consider replacing its unauthenticated interface with a
host-side broker that accepts a run-bound reset token. Do not put a reset token
or endpoint in the agent prompt.

**Integration estimate (explicitly an estimate):** a single browser-only TAC
task adapter with host reset broker, regular-user provisioning, exposure
readback, and exact evaluator is large (roughly 2–4 engineering weeks,
excluding failures in public visibility or upstream task changes). A full TAC
onboarding that retains shell/NPC/task-image breadth is larger and conflicts
with the current WARP threat model; no schedule should be committed before
the canary.

## 3. Mapping to WARP code and contracts

The current WARP implementation makes these candidates adapter work, not a
catalog-only change:

* [`packages/warp-taskgen/warp_taskgen/benchmark_capabilities.py`](../../../packages/warp-taskgen/warp_taskgen/benchmark_capabilities.py)
  currently treats ST-WebAgentBench and other non-WebArena benchmarks as
  comparison-only. WorkArena and TAC are not active WARP benchmarks. A
  capability entry may be added only after the site/exposure/reset gates pass;
  metadata alone must not make a benchmark runnable.
* [`rewards/final_state_catalog.py`](../../../packages/warp-taskgen/warp_taskgen/rewards/final_state_catalog.py)
  hard-locks local final-state evaluation to `webarena_verified`, and
  [`rewards/dispatcher.py`](../../../packages/warp-taskgen/warp_taskgen/rewards/dispatcher.py)
  routes task-ID-bearing rewards to the canonical WebArena evaluator. TAC or
  ServiceNow task IDs therefore need a new evaluator authority/catalog path;
  do not route them through WebArena by aliasing the name.
* [`worldsim_task.py`](../../../packages/warp-taskgen/packages/worldsim-agentlab-runner/src/worldsim_agentlab_runner/worldsim_task.py)
  and `phase4_loop.py` can supply start URLs, fresh/explicit storage-state
  handling, network traces, screenshots, PVPO telemetry, and action-loop
  plumbing. They do not provide benchmark-specific task setup, public-reader
  evidence, golden reset, or final-state semantics.
* [`agent_docs/site-onboarding.md`](../../../packages/warp-taskgen/agent_docs/site-onboarding.md)
  requires nine owner roles: targeting, profile, editor specification, regular
  writer, feasibility, read surface, readback, final-state evaluation, and
  action cards. ServiceNow or TAC cannot be represented by one generic editor;
  each app/carrier needs owner-specific routes, identities, cleanup, and
  evaluator behavior.
* [`agent_docs/admission-and-exposure.md`](../../../packages/warp-taskgen/agent_docs/admission-and-exposure.md)
  requires a Regular Participant Writer followed by a Fresh Anonymous Reader,
  exact resource evidence, Painted Visibility, and a Golden-State Reset. The
  proposed canaries above deliberately preserve those contracts and reject
  logged-in-reader, root-page, SQL/admin-readback, or newest-item fallbacks.

### Common acceptance artifacts

For either candidate, a future run should retain a sanitized, run-bound record
of: source commit, image digest, host topology, app/project/resource IDs,
regular-writer identity (non-secret), fresh-reader context proof, exact
rendered actor/body/route evidence, pre/post sentinel hashes, reset response
and health timeline, action/DOM/AXTree/screenshot/video/network traces, and
evaluator version. Do not retain cookies, passwords, service tokens, reset
keys, or evaluator decryption material.

## Open blockers and handoff

1. **WorkArena:** no anonymous rendered reader, ordinary writer, or database
   golden reset is documented; direct admission is blocked. BrowserGym remains
   useful as a constrained observation/action/trace substrate.
2. **TAC:** the exact GitLab project/issue public visibility is unknown; all
   services are documented as credentialed, and reset control is unauthenticated
   on a host interface. The PM cross-app task is conditional on an anonymous
   canary and a browser-only reduction. Do not change visibility silently.
3. **Both:** current WARP evaluators and site owners are WebArena/GitLab/Reddit
   specific. New benchmark capability metadata without evaluator, reset,
   exposure, and owner implementations must remain comparison-only.
4. **No runtime claims:** every command in this note is a proposed future
   sandbox smoke plan. No infrastructure or benchmark instance was touched in
   this research pass.
