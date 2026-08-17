---
status: accepted
---

# Model pairwise runs as a capability

An eval-awareness experiment declares exactly one way of consuming samples, and the shared base class carries only what every way needs. `BaseExperiment` owns the model, the output directory, and website content loading. `PerSampleExperiment` owns the abstract `run_sample` together with the `run` driver that fans it out across samples and formats. `PairwiseExperiment` owns the abstract `run_pairs` and leaves pairing, counterbalancing, and resume to the implementing experiment, because its unit of work spans two samples. An experiment that needs a third shape gets a third capability class beside these two, which keeps the shared base free of methods that only some implementors can answer.

The CLI registry in `eval_awareness_experiments/run.py` maps each experiment name to a class whose declared type is one of the two capabilities, and the runner selects a driver by asking which capability the constructed experiment has. Registering an experiment is therefore the whole of adding one: its driver follows from the base class it chose, and a class declaring neither capability is rejected at the point of registration. This also keeps the registry's constructor call checked against a real signature.

The evidence for separating the two is what a single base class made possible. The pairwise experiment supplied a `run_sample` stub to satisfy an abstract contract it did not participate in, and inherited a `run` that opens the experiment's results file for writing and emits zero rows — the same file `run_pairs` reads to resume. The dispatch branch kept that path unreached, so no recorded run was affected. That branch selected the pairwise experiment by its concrete class, which left the hazard one registration of a second pairwise experiment away. Selecting on capability retires the stub and the inherited driver together, and a second pairwise experiment reaches its own driver by registration alone.

Per-experiment configuration stays outside this decision. The runner still passes constructor extras as an untyped keyword mapping and still selects the judge-name extra by experiment name; whether the capability classes should accept configuration directly is open and tracked separately. This decision governs run shape only. It changes no experiment's behavior, no results schema, and no output location.
