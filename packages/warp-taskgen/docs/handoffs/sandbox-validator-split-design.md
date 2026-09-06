# Sandbox Validator Split Design

> **ACTIVE DESIGN NOTE.** This is the current safe split plan for
> `worldsim/_sandbox_validator.py`; preserve sandbox stdlib-only behavior and
> Modal staging contracts.

This is the next reviewable slice for the `worldsim/_sandbox_validator.py`
large-file debt. Do not split it as part of the transition-wrapper cleanup.
The validator has a stricter runtime contract than normal `worldsim` modules,
so a mechanical package extraction would change behavior or Modal packaging.

## Current Contract

- `worldsim/modal_sandbox.py` stages `worldsim/_sandbox_validator.py` into every
  Modal sandbox as `/workspace/_validate.py`.
- The validator runs inside the sandbox with stdlib only. It must not import
  `worldsim`, editor classes, `requests`, or any project dependency.
- It exits with code `0` for valid artifacts and `1` for invalid artifacts,
  printing a JSON result to stdout. Existing subcommands and result shapes are
  part of the sandbox prompt contract.
- `worldsim/modal_sandbox.py::_write_registry_snapshot()` serializes
  `worldsim.editors._registry.serialize_registry()` once on the host and stages
  the snapshot as `/workspace/_editor_registry.json`.
- `worldsim/_sandbox_validator.py` reads that JSON snapshot directly. The
  in-sandbox validator must keep treating the registry as data, not as a host
  package import.

## Preferred Slice

Create a domain-owned `worldsim/sandbox_validation/` package only if the modules
remain stdlib-only and can be staged explicitly into the sandbox. Two safe
approaches are acceptable:

1. Generate a standalone `/workspace/_validate.py` source file from the package
   at image-build time.
2. Stage the package files alongside `/workspace/_validate.py` and keep imports
   limited to those staged stdlib-only modules.

Either approach should preserve the public CLI surface first, then move one
cohesive validation domain at a time. Avoid generic `utils.py` or shared global
types; keep schema-specific validation next to the artifact it validates.

## Parity Tests

Add host-side parity coverage before moving behavior:

- Compare sandbox data-seed validation against `worldsim.seeding.validate_data_seed`
  for API, SQL, editor calls, deprecated mechanisms, and rejected methods.
- Compare self-contained adversarial seed checks against
  `worldsim.seed_contracts` helpers, including template placeholder counts,
  benign-token binding, finalized HTTP seed contracts, and discriminating
  payload validation.
- Exercise editor registry snapshot behavior by writing a temporary
  `_editor_registry.json` and invoking the validator code as it runs in the
  sandbox.
- Keep the existing `tests/sandbox_validator/` cases green throughout the
  split.

## Acceptance Criteria

- No `worldsim` imports execute inside `/workspace/_validate.py`.
- Modal file routing changes are limited to explicit staging of the standalone
  generated validator or stdlib-only validation package files.
- Validator subcommands, exit codes, JSON output shape, and schema semantics are
  unchanged.
- No artifact schema, feasibility, PVPO, judge, reward, auth precedence, or
  WASP-scope behavior changes are included in this slice.
