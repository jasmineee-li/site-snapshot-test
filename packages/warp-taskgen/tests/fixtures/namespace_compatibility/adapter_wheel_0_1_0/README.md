# Namespace adapter 0.1.0 fixture

This synthetic wheel models only the packaging surface shipped by the
adapter-bearing WARP Taskgen 0.1.0 release at commit `29a58753`: the
`warp_taskgen` distribution, the `worldsim` import adapter, and both core
console scripts. Its metadata mirrors that commit's relevant
`pyproject.toml` fields, while the minimal modules preserve the import and
console behavior needed by the upgrade proof. It contains no Taskgen runtime
data, credentials, or benchmark artifacts.

The package-proof lane builds this fixture locally, installs it with ordinary
pip, and upgrades to the current wheel. The proof succeeds only when pip uses
the version transition and wheel RECORD ownership to remove the old package
and console without force/reinstall flags. Change this fixture only when the
historical 0.1.0 packaging claim or the upgrade boundary changes.
