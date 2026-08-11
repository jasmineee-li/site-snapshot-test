---
status: accepted
---

# Compose Benchmark contracts separately from Sites and runtimes

Benchmark task and evaluation behavior will be modeled independently from Site behavior, browser runners, and Benchmark Instance reset or control. The immutable `worldsim.benchmark_capabilities.BenchmarkCatalog` is the small, fail-closed Benchmark Contract ledger: it normalizes aliases, resolves one canonical entry, and explicitly declares `phase_1_generation`, `phase_2_generation`, `phase_2_feasibility`, `phase_4_execution`, `warp_evaluation`, or `comparison_ingestion` separately from runner support. Registration alone never implies admission to another capability; comparison-only entries are not WARP-generative, executable, or scorable.

The ledger also records evaluator authority. WebArena Verified tasks with a canonical `task_id` remain under the vendor evaluator subprocess, while task-id-less novel WARP tasks use WARP-local evaluators; comparison-only entries declare `comparison_runner` authority for a future ingestion adapter. Unknown or mixed benchmark metadata and an evaluator shape with no declared authority fail closed. This metadata seam does not implement comparison ingestion, evaluator subprocess behavior, Site protocols, auth/browser/reset runners, or reward/scoring logic; those remain owned by their existing modules and adapters. AgentLab and vendor evaluator runtimes remain sidecar or subprocess adapters so the root WARP Taskgen package does not import optional runtime dependencies. This accepts explicit orchestration composition in exchange for independent variation, runtime isolation, and preventing a generic environment adapter from accumulating unrelated behavior.
