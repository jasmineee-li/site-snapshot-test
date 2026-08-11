---
status: accepted
---

# Compose Benchmark contracts separately from Sites and runtimes

Benchmark task and evaluation behavior will be modeled independently from Site behavior, browser runners, and Benchmark Instance reset or control. An explicit fail-closed Benchmark catalog will declare capabilities such as task generation, WARP evaluation, and comparison-only ingestion; registration alone will never imply admission to another capability. The first cutover will model the WebArena Verified task and authoritative evaluator contract while preserving its evaluator subprocess, and comparison-only Benchmarks will remain ineligible for WARP generation, execution, and scoring. AgentLab and vendor evaluator runtimes will remain sidecar or subprocess adapters so the root WARP Taskgen package does not import optional runtime dependencies. This accepts explicit orchestration composition in exchange for independent variation, runtime isolation, and preventing a generic environment adapter from accumulating unrelated behavior.
