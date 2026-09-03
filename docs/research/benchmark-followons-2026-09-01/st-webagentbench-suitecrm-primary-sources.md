# ST-WebAgentBench and SuiteCRM: comparison boundary and readiness gates

Research date: **2026-09-01** (America/New_York). The accepted WARP research
uses 2026-08-30 as its baseline. I checked the current WARP `origin/main`
(`577bb40c`, including PR #206's GitLab tracer) and the upstream
ST-WebAgentBench `main` ref. No benchmark code, browser, model, infrastructure,
or live Run was executed for this note.

## Source freeze and current upstream state

The current ST-WebAgentBench `main` ref still resolves to
`67f56dd7df9eca1646c9e49407b087e950aa1e77`; `git ls-remote` was checked on
2026-09-01 and the upstream history query shows no commit in the
2026-08-30--2026-09-01 window ([commit](https://github.com/segev-shlomov/ST-WebAgentBench/commit/67f56dd7df9eca1646c9e49407b087e950aa1e77),
[post-baseline history](https://github.com/segev-shlomov/ST-WebAgentBench/commits/main?since=2026-08-30&until=2026-09-02)). Thus the accepted
2026-08-30 pin remains current; this is a source check, not a runtime result.

The pinned README describes 375 enterprise tasks (197 GitLab, 8
ShoppingAdmin, and 170 SuiteCRM), 3,057 policy instances, six safety
dimensions, and native CR/CuP-style reporting ([README at the pin](https://github.com/segev-shlomov/ST-WebAgentBench/blob/67f56dd7df9eca1646c9e49407b087e950aa1e77/README.md)). The old arXiv
preprint reports an earlier 222-task/646-policy release and must not be used
for current counts ([preprint](https://arxiv.org/abs/2410.06703)). The current
README is the source for dataset arithmetic; neither source is evidence that a
WARP-generated task has run.

SuiteCRM is a 3-tier set plus 80 modality variants. The modality tasks use
trusted harness JavaScript/CSS setup before observation; that is fixture
manipulation, not attacker-controlled WARP seed data. It should be excluded
from an initial WARP-generated slice (or retained only as a clearly labeled
native control) ([README modality description](https://github.com/segev-shlomov/ST-WebAgentBench/blob/67f56dd7df9eca1646c9e49407b087e950aa1e77/README.md#modality-aware-tasks),
[task setup](https://github.com/segev-shlomov/ST-WebAgentBench/blob/67f56dd7df9eca1646c9e49407b087e950aa1e77/browsergym/stwebagentbench/src/browsergym/stwebagentbench/task.py#L85-L126)).

The source Compose file uses mutable `public.ecr.aws/bitnami/mariadb:11.4`
and `public.ecr.aws/bitnami/suitecrm:8` images, `linux/amd64`, port 8080,
named persistent volumes, and passwordless-development settings. The setup
README requires a manual `demo_data.sql` import and a browser login; neither
source documents a Golden-State Reset ([Compose](https://github.com/segev-shlomov/ST-WebAgentBench/blob/67f56dd7df9eca1646c9e49407b087e950aa1e77/suitecrm_setup/docker-compose.yaml),
[setup README](https://github.com/segev-shlomov/ST-WebAgentBench/blob/67f56dd7df9eca1646c9e49407b087e950aa1e77/suitecrm_setup/README.md),
[initialization script](https://github.com/segev-shlomov/ST-WebAgentBench/blob/67f56dd7df9eca1646c9e49407b087e950aa1e77/suitecrm_setup/init-db/setup.sh)). The fixture itself says it inserts ten records per several tables, but the pinned SQL has visible count/relationship drift (for example, fewer users/accounts/cases than the header and a relation referring to `Hooli` without a matching account row); it also uses `NOW()`/`UUID()`. Counts, relations, IDs, and timestamps therefore need to be measured after a controlled import, not inferred from the header ([fixture](https://github.com/segev-shlomov/ST-WebAgentBench/blob/67f56dd7df9eca1646c9e49407b087e950aa1e77/suitecrm_setup/init-db/demo_data.sql)).

## What WARP can ingest now

WARP already registers `stwebagentbench` with only the
`comparison_ingestion` capability, `comparison_runner` Phase 4 mode, native
`capability` outcome mode, and `comparison_runner` evaluator authority
([catalog](../../../packages/warp-taskgen/warp_taskgen/benchmark_capabilities.py#L87-L95)). The
sidecar package depends on AgentLab only; ST-WebAgentBench is not a WARP
generation or execution dependency ([sidecar package](../../../packages/warp-taskgen/packages/worldsim-agentlab-runner/pyproject.toml#L1-L24)). Its
benchmark configuration is looked up from the sidecar's installed
`bgym.DEFAULT_BENCHMARKS`; an ST-specific adapter/registration is still a
separate implementation question ([configuration hook](../../../packages/warp-taskgen/packages/worldsim-agentlab-runner/src/worldsim_agentlab_runner/cli.py#L130-L163)).

The current boundary is deliberately narrow: native AgentLab JSON becomes an
immutable `ComparisonRecord`; `native_reward` is checked against native
`summary_info.cum_reward`; WARP-only fields are rejected; and the result is
written as `comparison_result.json`, not WARP `result.json`
([ingestion module](../../../packages/warp-taskgen/warp_taskgen/comparison_ingestion.py#L1-L7),
[record and validation](../../../packages/warp-taskgen/warp_taskgen/comparison_ingestion.py#L168-L218),
[native-only checks](../../../packages/warp-taskgen/warp_taskgen/comparison_ingestion.py#L322-L367)). The AgentLab runner clears stale
comparison/WARP sentinels and invokes only the configured reset hook before a
sidecar call; it does not supply SuiteCRM reset semantics
([runner boundary](../../../packages/warp-taskgen/warp_taskgen/runners/agentlab.py#L1907-L1937)). Focused tests cover catalog outcome modes and envelope isolation,
not a live ST or SuiteCRM run ([test](../../../packages/warp-taskgen/tests/test_comparison_ingestion.py#L19-L73)).

### Claims supported by comparison-only ingestion

After a real native run with healthy provenance, it can support:

* native ST task completion/reward and policy-compliance measurements, with
  the benchmark's own capability outcome vocabulary;
* per-task, per-policy, and (where the task metadata supports it) tier or
  modality diagnostics, including explicit native error/timeout records; and
* an auditable statement that a particular AgentLab payload was ingested under
  `comparison_runner` authority, with its model, versions, steps, and artifact
  references.

These are comparison baselines or diagnostics. They require evaluator-health
and fixture checks before aggregating rates. A source pin or hash establishes
identity/compatibility; it does not prove that an evaluator saw the intended
record or that an application state changed.

### Claims it cannot support

Comparison-only ingestion cannot establish WARP attack success rate, content
propagation or unauthorized extra artifacts, exact Resource Evidence, painted
visibility, current-attempt binding, PVPO, WARP generated-task novelty,
cross-Site transfer, or a WARP Phase 4 score. Native policy text and native
task success must remain separate from WARP safety/effect attribution. A
native run also cannot prove reset determinism unless the reset and readback
artifacts are independently captured.

## Conditional SuiteCRM gates

The following are the minimum gates before proposing a WARP-generated,
executed, and scored SuiteCRM slice. Each gate names the failure it prevents,
the smallest sufficient check, and the observation that would falsify
readiness.

| Gate | Evidence required | Failure prevented | Smallest sufficient check | Readiness falsifier |
| --- | --- | --- | --- | --- |
| Ordinary-role access and authentication | Two controlled non-admin identities: an ordinary writer and a legitimate independent reader; pinned app URL and credentials loaded out-of-band; denied admin-only route/action recorded. | Treating the bootstrap `user`/`bitnami` credential or an admin account as an ordinary participant. | Login separately as writer and reader; create/read one non-sensitive fixture through the normal UI; attempt one admin-only operation and capture denial. | Only the bootstrap/admin identity works, or reader access depends on sharing writer session/cookies. |
| Exact authorized exposure | A concrete record/field/relationship, audience, route, and expected positive/negative readback; authorization is checked with the independent reader. | Calling a consent request, source SQL row, or URL reachability “exposure” without proving the intended audience can see exactly the intended content. | Writer seeds one unique marker; reader independently verifies marker and parent relation; reader also checks a negative/unauthorized marker. | Positive readback is absent/ambiguous, appears only to writer, or unauthorized reader sees it. |
| Fixture integrity and readback | Immutable fixture manifest captured after import (IDs, parent links, actor, timestamps, counts) plus UI/API readback using the audience-appropriate path. | Relying on the SQL header or silently accepting drift (`Hooli` relation, short tables, UUID/time variability). | Import once in an isolated instance; record actual rows/relations; run one positive and one negative query/readback. | Counts/relations differ across repeat imports without explanation, or the evaluator target is not represented in the rendered readback. |
| Evaluator applicability | Static audit and known-positive/negative fixtures for each selected task/evaluator; stale/missing selectors and evaluator exceptions reported as errors, not safe outcomes. | Native evaluator silently grading a wrong Site selector (task 47's GitLab-shaped `#project_visibility_level_20`), dormant policy, or exception as success/safety. | Run evaluator logic against one known-positive and one known-negative fixture in a controlled harness; inspect every report/error flag. | Any selected evaluator is site-mismatched, exception-prone, or produces only dormant/unavailable reports. |
| Golden-State Reset and isolation | Restore both database and app volumes (or an equivalent full snapshot) before each attempt; verify parent/child presence and absence, login state, and control-plane isolation. | Cross-run contamination, stale records, and false propagation/extra-artifact attribution. | Two sequential twin attempts from the same frozen baseline: first mutates one unique record; reset; second proves the mutation is absent before starting. | Reset only removes a temp config, uses `env.reset()` without app-state proof, or the second twin observes first-run state. |
| WARP payload boundary | Ordinary UI/API task payloads and explicit adversarial seed rules; no trusted setup-script JavaScript/CSS in the attack channel; native control labels retained. | Mistaking ST modality setup (`page.evaluate`) or native policy text for WARP attacker content. | Generate one ordinary-role task with no setup script; inspect serialized payload and initial observation for only allowed seed channels. | The candidate relies on harness DOM injection, trusted policy injection, or a seed channel not represented in WARP provenance. |
| Runtime/provenance | Isolated Benchmark Instance, exact source/image/config digests, model/agent trace, reset/readback artifacts, and separate native `comparison_result.json`. | Reporting focused tests, generated corpus count, or a hash as a completed live experiment/paper result. | One serialized dry-run-shaped artifact set with explicit status, then a real Run only after all prior gates pass. | Missing instance identity, mutable image/config, absent reset/readback artifacts, or conflated native and WARP result files. |

These gates follow the accepted WARP baseline: all 170 CRM tasks currently
declare login and no reset; BrowserGym setup/teardown authenticates and removes
only a temporary configuration file; and evaluator exceptions can produce zero
or unavailable native outcomes ([accepted baseline](https://github.com/segev-shlomov/ST-WebAgentBench/blob/67f56dd7df9eca1646c9e49407b087e950aa1e77/browsergym/stwebagentbench/src/browsergym/stwebagentbench/task.py#L85-L160),
[WARP source audit](../web-benchmark-onboarding-2026-08-30.md#L124-L148)). SuiteCRM `ui_login` supports a normal form login, but the upstream automatic-login helper does not include SuiteCRM, so storage-state renewal cannot be assumed ([UI login](https://github.com/segev-shlomov/ST-WebAgentBench/blob/67f56dd7df9eca1646c9e49407b087e950aa1e77/browsergym/stwebagentbench/src/browsergym/stwebagentbench/instance.py#L74-L120),
[auto-login helper](https://github.com/segev-shlomov/ST-WebAgentBench/blob/67f56dd7df9eca1646c9e49407b087e950aa1e77/stwebagentbench/browser_env/auto_login.py#L22-L30)).

## Counterfactual decision and sequencing

**What would make SuiteCRM ready now:** all gates above pass on a pinned,
isolated stack; ordinary writer and independent reader are proven; one small,
non-modality WARP-generated task has exact positive and negative readback; a
full snapshot restore makes two attempts state-identical before the task; and
the native evaluator's known-positive/negative behavior is documented without
site-mismatched or dormant checks. At that point, a tiny generated slice can
be proposed as a WARP experiment, while native ST metrics remain a labeled
comparison control.

**What would falsify readiness:** any missing independent reader, reliance on
administrator/bootstrap credentials, inability to restore app/database state,
fixture count/relation drift that cannot be explained, evaluator mismatch or
exception that is treated as safe, or trusted setup/policy injection in the
WARP attack channel. A successful native task, a merged adapter, a focused
ingestion test, or a generated-task count cannot repair any of these gaps.

Therefore the near-term follow-on should be **comparison-only ingestion and
offline evaluator/fixture audit**, not SuiteCRM onboarding. Native comparison
work can proceed in parallel with Plane feature research because it does not
change Site or WARP action contracts. A conditional SuiteCRM generated slice
comes only after the gates and an isolated serialized Run; modality variants,
cross-Site generation, and a generic multi-Site/evaluator framework stay
deferred. This keeps the scientific claim proportional to the evidence and
avoids importing ST's trusted setup or reset assumptions into WARP.

## Source/evidence ledger

* **Source identity:** upstream ref `67f...1e77` and WARP `origin/main`
  `577bb40c`; no post-2026-08-30 ST commit observed.
* **Static source evidence:** README, task/evaluator/instance Python, Compose,
  setup script, and SQL fixture at the pinned ref; WARP capability and
  ingestion source above.
* **Focused test evidence:** WARP comparison-ingestion tests only; no ST
  browser/evaluator execution was performed.
* **Live sandbox evidence:** none in this note.
* **Corpus/evaluation evidence:** task counts are source arithmetic; no WARP
  generated/admitted corpus or completed experimental Run is claimed.
