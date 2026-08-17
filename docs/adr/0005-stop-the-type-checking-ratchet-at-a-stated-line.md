---
status: accepted
---

# Stop the type-checking ratchet at a stated line

The eval-awareness tree does not target `mypy --strict`. It targets the two global tiers already configured — the `warn_*` family, then `check_untyped_defs` and `strict_equality` — plus per-module `disallow_*` on a named durable set, and stops there. `eval_awareness_experiments/`, `probes/` and `scripts/` are research code with no external consumers, where `--strict` taxes a throwaway analysis script at the same rate as durable infrastructure. Recording the stop line here keeps a later contributor from reading mypy's adoption ladder as an obligation to climb it to the top, and from deleting the tiers already in place for want of a stated destination.

Membership of the durable set is measured rather than argued: a module imported by ten or more other first-party modules gets `disallow_untyped_defs` through a `[[tool.mypy.overrides]]` entry. Annotating a widely imported module improves checking in every module that imports it, which a rarely imported one cannot. Measured on this commit: `types` 20 importers, `llm` 15, `system_prompt_frame` 10, `injection_modifier` 10. A new module joins when its count reaches the threshold.

The mypy version pin and its review trigger are deliberately not recorded here. A pin is edited whenever a release lands, which is the opposite of an accepted decision, and its rationale belongs beside the pin in `pyproject.toml` where `.github/dependabot.yml` makes it self-enforcing.
