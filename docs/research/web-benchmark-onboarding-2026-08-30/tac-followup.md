# TAC follow-up: single-site carriers versus GitLab-to-Plane

Research date: **2026-08-30 (ET)**. This is a read-only source review at TAC
commit [`98b68ef82a47690c316f42fddb05baafaab56851`](https://github.com/TheAgentCompany/TheAgentCompany/tree/98b68ef82a47690c316f42fddb05baafaab56851).
I used official TAC GitHub files only; no repository snapshot, install,
container, browser, model, or evaluator run was performed. Credential values
are intentionally omitted.

## Recommendation and decision question

**Recommendation (inferred):** a TAC single-site canary is scientifically
adequate as **Stage 1** if it is explicitly a carrier/evaluator and generation
transfer probe, not evidence of arbitrary cross-benchmark portability. Favor a
Rocket.Chat message workflow (ordinary-account write to an existing public
channel) for the first low-trust exposure contract; use a Plane lifecycle
variant as a second single-site family. Keep a GitLab-to-Plane workflow as an
optional held-out extension or control: it is needed to claim cross-application
relation, but it is not a prerequisite for a cross-benchmark generation claim
if an independently structured single-site workflow succeeds.

**Tradeoff for the user:** should the first TAC implementation spend its
bounded effort on (A) Rocket.Chat/Plane single-site exposure and evaluator
canaries, then optionally extend the schema for cross-site generation, or (B)
pay the cross-site schema cost immediately because a cross-application claim is
required? I recommend A for a first TAC transfer result; B is better only if
the first result must answer a cross-application question.

**Smallest useful Stage 1 (inferred).** Start with one practical generated
family per carrier, reusing actions a benign participant already needs: post a
message in an existing public Rocket.Chat project channel, and create or
triage a Plane issue with a normal description/status change. Pair each with a
legitimate read or follow-up task. An observable unauthorized deviation can be
an extra body/message, wrong record/project, duplicate side effect, or
unintended state transition; it does not require project deletion or another
high-impact action. Keep tasks where the benign agent never naturally reads the
carrier as a labeled non-encounter, not as a failed attack. Do not select only
payloads expected to win, and do not require every pair to expose content: the
scientific result is the distribution of blocked, encountered, and exposed
outcomes under a fixed contract.

## Concrete TAC slices

### 1. Plane: create an issue (new single-app state family)

**Documented.** [`task.md`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/pm-create-plane-issue/task.md#L1-L3)
asks the agent to create one issue in the “Frontend Development” project;
[`dependencies.yml`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/pm-create-plane-issue/dependencies.yml#L1)
requires only Plane. The two-point native evaluator checks an exact project URL
in the trajectory and then queries all project issues for a case-insensitive
title match ([`evaluator.py#L5-L27`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/pm-create-plane-issue/evaluator.py#L5-L27));
the published checkpoints are one point for navigation and one for creation
([`checkpoints.md#L5-L11`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/pm-create-plane-issue/checkpoints.md#L5-L11)).

**What it adds (inferred).** The persisted object is a Plane issue, not a
GitLab/Reddit post, so a generated family can vary project, cycle, assignee,
labels, comments, and status transitions. To carry WARP content, the variant
must put a payload in the issue description (or another explicitly rendered
field), use an ordinary attacker account, and have a separate legitimate
participant read the exact issue. Bind the WARP check to project/issue ID,
creator, body hash, and route; keep the native title score separate. A title
only is another “create issue” task, not meaningful variety.

**Unverified.** The source does not establish that a non-admin employee may
create/read the chosen project issue, that another authenticated account sees
the description, or that reset removes duplicate titles. The evaluator does
not check body, actor, timestamp, or causal creation; a stale matching title
could pass. It is an API state oracle, not an independent exposure oracle.

### 2. Rocket.Chat: channel, membership, and message (stronger low-trust carrier)

**Documented.** [`task.md`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/pm-create-channel-message/task.md#L1-L3)
creates public `#finance-talk`, invites David Wong, and posts an introduction;
[`dependencies.yml`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/pm-create-channel-message/dependencies.yml#L1)
requires Rocket.Chat. The native three-point evaluator checks `/home`, channel
membership, then scans channel history for an exact message **and** author
(`theagentcompany`) ([`evaluator.py#L14-L16`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/pm-create-channel-message/evaluator.py#L14-L16),
[`#L25-L49`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/pm-create-channel-message/evaluator.py#L25-L49),
[`#L52-L59`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/pm-create-channel-message/evaluator.py#L52-L59)).

**What it adds (inferred).** A message is a natural low-trust text carrier and
the workflow combines a timeline write with membership state. For WARP, prefer
an existing public project channel so the attacker only posts a payload; a
legitimate participant (or Independent Authenticated Reader) must locate that
message. Add a unique run marker/ID and exact channel/author binding. The
current invitation/channel-creation task is more administrative than an
ordinary collaboration probe and should not be the sole exposure result.

**Unverified/counterevidence.** History matching accepts any old exact
message by the expected username and does not bind timestamp, run, or message
ID. Public in the instruction does not prove anonymous read; authenticated
ordinary-user visibility and write permission are unknown. The native client
and evaluator are likely privileged paths, so this is a candidate contract,
not proof of a normal-user threat model.

The role fixture does provide ordinary *identities and context*: TAC lists
employees, occupations, and channels such as Zhang Wei (streaming engineer),
David Wong (finance director), and QA/engineering roles in
[`NPC.md#L3-L43`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/servers/rocketchat/npc/NPC.md#L3-L43)
and [`npc_definition.json#L7-L35`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/servers/rocketchat/npc/npc_definition.json#L7-L35).
These fixtures describe personas/channels, **not** service permissions,
audience visibility, or separate-account reset behavior.

### 3. GitLab to Plane: structural reference, not another single-site task

**Documented.** [`task.md`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/pm-update-plane-issue-from-gitlab-status/task.md#L1-L4)
requires reading two JanusGraph issue statuses in GitLab and moving/closing
the corresponding Plane issues; its dependency file requires both services
([`dependencies.yml#L1-L2`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/pm-update-plane-issue-from-gitlab-status/dependencies.yml#L1-L2)).
The checkpoints total seven points: three trajectory visits, a Sprint 2
membership check, and a completed-state check
([`checkpoints.md#L2-L21`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/pm-update-plane-issue-from-gitlab-status/checkpoints.md#L2-L21)).
The evaluator API-checks those final states and URL substrings
([`evaluator.py#L13-L67`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/pm-update-plane-issue-from-gitlab-status/evaluator.py#L13-L67)).

**Limitations.** It carries no attacker-authored content and does not establish
ordinary-role exposure. Its bonus function awards the full total whenever the
last two checkpoints pass, even if earlier visits fail
([`evaluator.py#L69-L91`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/pm-update-plane-issue-from-gitlab-status/evaluator.py#L69-L91)).
Preserve this native task as a control, but add a distinct WARP effect check
for the actual cross-site relation and never treat native score as exposure
evidence.

## Deployment, reset, and restrictions

**Documented.** TAC’s base initializer maps the synthetic hostname, calls
`reset.sh`, then optionally runs NPC, population, and post-init hooks
([`init.sh#L3-L38`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/base_image/init.sh#L3-L38)).
`reset.sh` scans dependencies for Rocket.Chat, Plane, GitLab, and ownCloud,
POSTs per-service reset endpoints, and waits on health checks
([`reset.sh#L6-L59`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/base_image/reset.sh#L6-L59)).

**Inferred/unverified.** Service health is not a golden-state proof. Exact
issue/message absence, duplicate cleanup, account/session isolation, and
task-specific population need independent before/after checks. For the
smallest first slice, choose browser actions by ordinary participants; keep
shell, code, and admin operations out of that cohort, and treat NPC-mediated
dialogue as a separate choice when it would confound the reader contract. The
attacker must never receive control-API capability, and no arbitrary
full-platform prerequisite should be imposed. Authenticated ordinary
collaboration is eligible with separate ordinary writer and evaluated accounts
plus an Independent Authenticated Reader.

OwnCloud is a useful negative control for the inspected task, not evidence that
the whole platform has no writable carrier. Its `admin-translate-sales-chat`
task reads a screenshot and writes only `/workspace/ans.txt`
([`task.md#L0-L7`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/admin-translate-sales-chat/task.md#L0-L7));
the evaluator grades a numeric answer in that local file, with no OwnCloud
write/readback ([`evaluator.py#L9-L40`](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/admin-translate-sales-chat/evaluator.py#L9-L40)).
This is direct counterevidence against using an OwnCloud read task as a
low-trust exposure slice.

## Evidence that would reverse the recommendation

Evidence sufficient to support a stronger Stage 1 claim would include a
disposable TAC canary with ordinary-account write permission; exact
body/message exposure to the declared legitimate reader; actor and record
identity; native and WARP scores kept separate; and negative controls for
stale, wrong-record, wrong-actor, and API-error states. Two consecutive
verified resets are a useful confidence test, not a canonical user gate; report
reset uncertainty instead of silently converting it to success. If the core
exposure contract remains unverified, retain TAC single-site tasks as native
controls and do not claim transfer.
If single-site generation yields only title/message paraphrases, skip Stage 1
as a scientific result and prioritize the cross-site schema/evaluator
extension.
