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

**Run**:
One persisted execution of the WARP pipeline with a resolved set of effective inputs.
_Avoid_: Session, experiment

**Remote Job**:
An operating-system process registration that executes or manages a Run on a configured host. Its process lifecycle is distinct from the Run's durable state and evidence.
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
