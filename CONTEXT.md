# WARP

WARP generates and evaluates browser-agent safety benchmarks while preserving the distinction between a benchmark's task contract, the sites it exercises, their live deployments, and individual pipeline executions.

## Language

**WARP Taskgen**:
The WARP pipeline that generates, admits, executes, and scores browser-agent safety tasks.
_Avoid_: WorldSim, worldsim

**Benchmark**:
A task corpus together with the evaluation contract that determines how its tasks are interpreted and scored.
_Avoid_: Web environment, harness

**Benchmark Capability**:
An explicitly declared operation supported by a Benchmark contract, such as task generation, WARP evaluation, or comparison-only ingestion.
_Avoid_: Feature flag, inferred support

**Comparison-only Benchmark**:
A Benchmark whose artifacts may be normalized for analysis but which is not admitted to WARP task generation, browser execution, or scoring.
_Avoid_: Supported Benchmark, evaluation Benchmark

**Site**:
A kind of hosted web application exercised by a Benchmark, such as GitLab or Reddit/Postmill.
_Avoid_: Web environment, website, app

**Benchmark Instance**:
One reachable deployment or replica of a Site used by a Run.
_Avoid_: Web environment, host, Site

**Benchmark Host**:
A configured infrastructure machine that runs Remote Jobs and Benchmark
Instances. Its ownership and power state are separate from each Run and
Benchmark Instance.
_Avoid_: Configured host, Benchmark Instance, Site

**Run**:
One persisted execution of the WARP pipeline with a resolved set of effective inputs.
_Avoid_: Session, experiment

**Runtime Composition**:
An explicitly selected, immutable bundle of Site, seeding, feasibility,
readback, and cleanup owners bound to one Run without changing process-wide
defaults.
_Avoid_: Feature flag, global registration, Site registration

**Host-Owned**:
Describes data or behavior selected and enforced by trusted WARP Taskgen code,
rather than by a model. It does not describe Benchmark Host lifecycle
ownership.
_Avoid_: Trusted host, Benchmark Host-owned

**Remote Job**:
An operating-system process registration that executes or manages a Run on a
Benchmark Host. Its process lifecycle is distinct from the Run's durable state
and evidence.
_Avoid_: Run, pipeline execution

**Site Targeting**:
The deterministic capability that binds a Benchmark profile and Benchmark Instance to one Site, describes that Site's canonical routes, and resolves task evidence to a Site-owned resource and route. It does not prove authentication, reachability, visibility, admission, mutation, or scoring.
_Avoid_: Site adapter, target resolver

**Resource Kind**:
A Site-owned semantic category of target resource, identified by the Site together with a local name such as `issue` or `submission`.
_Avoid_: Globally prefixed resource string, surface

**Canonical Route**:
A host-independent Site route and validated anchor set that can be bound to a Benchmark Instance.
_Avoid_: URL, start URL

**Resolved Target**:
A Resource Kind and Canonical Route bound to one Benchmark Instance using deterministic task evidence.
_Avoid_: Reachable target, eligible target

**Regular Participant Writer**:
A non-admin benchmark participant that creates the exact resource through the
Site's ordinary user path.
_Avoid_: Writer, admin seeder, database seeder

**Fresh Anonymous Reader**:
An independent reader context with no Regular Participant Writer cookies,
storage state, or authentication, used to observe the public exact resource.
_Avoid_: Reader, writer session, reused browser context

**Exact Resource Evidence**:
Evidence that binds mutation, readback, visibility, and cleanup to the same
created resource identity and parent route.
_Avoid_: Body scan, newest item, parent-page evidence

**Golden-State Reset**:
A restoration controlled by the Benchmark Host lifecycle that returns a
Benchmark Instance to a known seeded baseline, with independent postcondition
evidence.
_Avoid_: Cleanup, delete endpoint, browser reset

**Painted Visibility**:
Evidence that a required witness produced non-background pixels inside the
committed browser viewport capture; DOM or rendered-HTML presence alone is
insufficient.
_Avoid_: DOM visibility, rendered HTML, HTTP readback

**PVPO Encounter**:
A Phase 4 classification that the Paint-Verified Payload Oracle measured
positive Painted Visibility for the selected attack witness on the agent
trajectory (`max_coverage > 0`).
_Avoid_: Page visit, DOM match, HTTP readback

**Run Definition**:
The immutable, versioned, normalized set of effective inputs that defines a Run's semantics and determines checkpoint compatibility.
_Avoid_: CLI arguments, resume metadata, configuration dump

**Run ID**:
The opaque persisted identity of one Run execution. It is retained by exact resume and is distinct from semantic input equality.
_Avoid_: Definition digest, state directory, Remote Job ID

**Definition Digest**:
The deterministic digest of a Run Definition used to compare semantic inputs. Separately started Runs may share a Definition Digest while retaining different Run IDs.
_Avoid_: Run ID, artifact fingerprint

**Derived Run**:
A new Run created from an existing Run after a result-affecting input changes. It records the source Run and may reuse only checkpoints proven compatible.
_Avoid_: Resumed Run, modified Run

**Resume Plan**:
A read-only explanation of which checkpoints a requested Run will reuse, rerun, or reject and the evidence for each decision.
_Avoid_: Resume status, retry plan

**Atomic Work Unit**:
The smallest admitted operation that must finish its validation, durable evidence, and required cleanup before a Run can become paused. An Atomic Work Unit may be a shard or one task, depending on the owning phase.
_Avoid_: Worker, coroutine, process

**Checkpoint**:
Durable evidence that one Atomic Work Unit completed under a specific Run Definition. A phase-owned validator decides whether a Checkpoint may be reused.
_Avoid_: Progress marker, log entry, partial output

**Pause Request**:
Operator intent to stop admission of new Atomic Work Units. Already-admitted units drain before the Run becomes paused. A Pause Request does not stop a Remote Job by itself.
_Avoid_: Cancellation, process stop, signal

**Legacy Run**:
A historical Run whose persisted state predates Run ID and Run Definition storage. It remains readable and may resume conservatively, but WARP never invents its identity or treats its unbound artifacts as reusable Checkpoints.
_Avoid_: Derived Run, migrated Run

**Run Artifact**:
Persisted output or evidence owned by one Run. A Run Artifact may be a reusable Checkpoint, a canonical aggregate, or inspection evidence, but it is not source code.
_Avoid_: Asset, source fixture

**Run Lifecycle**:
The durable execution state of a Run, such as running, paused, interrupted, failed, or complete. It is independent of the process state of any Remote Job.
_Avoid_: Remote Job status, process status

**Independent Authenticated Reader**:
A fresh reader context authenticated as a non-admin participant other than the
Regular Participant Writer, used to observe the exact resource through that
participant's legitimate access. It does not reuse the writer's cookies,
storage state, or credentials.
_Avoid_: Fresh Anonymous Reader, writer session, admin observer

**Information-only Benign Task**:
A legitimate browser task whose requested result is information rather than a
change to Site state. Its paired attack may still seek a state-changing effect.
_Avoid_: Non-mutating attack, read-only attack

**State-changing Benign Task**:
A legitimate browser task whose requested result includes creating, updating,
or removing persistent Site state. The requested changes remain distinct from
the paired attack's objective.
_Avoid_: Mutation attack, adversarial action

**Task Archetype**:
A behavioral family of benign browser tasks defined by their information need,
decision logic, required actions, navigation, and data dependencies. Different
record names or wording alone do not define a new archetype.
_Avoid_: Carrier family, payload variant

**Attacker Objective**:
The observable effect an attack seeks to induce beyond or contrary to the
legitimate user's request.
_Avoid_: Payload text, attack strategy

**Content Propagation**:
Transfer of attacker-specified content into an agent answer or persisted
artifact. Propagation alone does not establish endorsement or obedience.
_Avoid_: Semantic obedience, endorsement
