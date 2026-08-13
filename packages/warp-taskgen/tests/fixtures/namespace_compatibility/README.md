# Namespace compatibility fixture

This fixture is a secret-free **synthetic schema projection** of the historical
`worldsim` Run Artifact contract used by the #136 readback parity tests. It is
not a copy of an archived run and does not claim to reproduce live task
content. It intentionally keeps the historical
schema identifiers and artifact-relative paths while replacing host paths,
cookies, prompts, and credentials with placeholders or deterministic hashes.

- owner: WARP Taskgen compatibility evidence
- provenance: field names, schema identifiers, and artifact layout are derived
  from the selected-host Run Artifact shape recorded in
  `docs/handoffs/final-cutover-readiness-2026-08-11.md`; identifiers and
  topology values are deterministic fixture values; producing implementation
  commit: `ecfe19ae523bbeea510c1ffd773a1b8b2ae7e82a`
- content hash: sorted JSON-file SHA-256 manifest digest
  `e2023380f23f251b544e111099ad46629072d81efd599e448390a8b42214dec6`
- purpose: prove canonical and legacy readers expose one Run identity,
  Definition Digest, lifecycle state, result summary, checkpoint metadata, and
  artifact locations
- regeneration: update the fixture only from a read-only, secret-free
  projection of an archived run; run
  `pytest -q tests/test_namespace_compatibility_evidence.py`
- retention: tracked as a small schema fixture; raw traces and host output are
  not promoted here
