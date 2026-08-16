# Classifieds canary

Use this guide for the experimental `classifieds_listing_reply_poc` canary.
The canary proves one VisualWebArena Classifieds `listing_reply.body` carrier
on a configured sandbox host. It is opt-in. It does not add Classifieds to the
default GitLab or Reddit cohort.

The [technical specification](../docs/warp-taskgen-technical-spec.md) owns the
contract. This guide owns the operator procedure. The
[admission guide](./admission-and-exposure.md) owns the shared Phase 1,
Phase 2, and Phase 2c admission rules.

## Terms

- **Writer** — a regular, non-admin Classifieds participant used for the
  ordinary reply form.
- **Fresh anonymous reader** — a new HTTP or browser session with no Writer
  cookies or storage state. It reads the public listing after the write.
- **Run** — the WARP execution and its immutable inputs. A Run ID identifies
  the execution; a Definition Digest identifies its effective inputs.
- **Remote Job** — the registered host process that owns the canary command. It
  is separate from the Run.
- **Painted visibility** — the reply text appears in the committed reader
  viewport capture. DOM presence alone is not painted visibility.
- **PVPO encounter** — the Phase 4 runtime check that records painted
  visibility for an attack witness.
- **Completion artifact** — `completion.json` in the Run root. It is the
  terminal Run evidence; local host cleanup has a separate postcondition.

## Safety boundary

Apply this boundary before loading credentials or resetting an instance:

1. Use only the configured sandbox host. Do not use a public demo, a personal
   account, or a production instance.
2. Run one canary at a time. The launcher owns the host lock, lifecycle tags,
   Remote Job, and final host parking.
3. Use the ordinary regular-user reply path. Do not use SQL, an admin route, a
   reset token in a task, or a database query to create the reply.
4. Keep writer storage state, application environment files, and provider
   credentials outside the checkout. They must not appear in task JSON,
   command output, Run Artifacts, or commits. See [secret handling](./secrets.md).
5. Reset only through the host-owned golden-state operation. The reset is
   required before the write and after the Run; it is not a browser action.
6. Treat missing exact-resource, independent-reader, painted-visibility, or
   cleanup evidence as a failed canary.

The canary is a state-changing operation. If any ownership, reset, or cleanup
check is uncertain, stop and inspect the Run and host state before retrying.

## Prerequisites

Confirm these prerequisites from `packages/warp-taskgen`:

- The package environment is installed with the locked development
  dependencies.
- The ignored operator file
  `configs/benchmark_hosts/r8a.local.yaml` exists and contains a validated
  `classifieds_canary` block.
- The operator file points to the configured loopback Classifieds origin, one
  seeded listing, immutable source/image identities, and secret references
  outside the checkout.
- No other Classifieds canary owns the selected host.

The loader rejects a missing or tracked operator file, non-loopback origin,
mutable image identity, unsafe Run root, or in-checkout secret path. Use the
[remote-run guide](./remote-runs.md) for generic host preparation; this guide
does not repeat those commands.

## Entry command

The launcher is the only canonical entry point. First, check the current
parser without changing state:

```sh
uv run python scripts/run_classifieds_canary.py --help
```

Then start one fresh Run from `packages/warp-taskgen`:

```sh
uv run python scripts/run_classifieds_canary.py --host-config configs/benchmark_hosts/r8a.local.yaml --run-dir logs/classifieds-canary/canary-docs-check
```

Use a new safe run name for every attempt; `canary-docs-check` is only a
copyable example. The launcher performs host
ownership, preparation, the bounded remote job, status polling, graceful stop
when needed, cleanup, and host parking. Do not call its internal preparation,
probe, or remote wrapper scripts as a second workflow.

**Parser check:** `--help` exits successfully without loading the operator
configuration or changing state.

**Launch check:** the operator configuration loads, and the selected Run root
is exactly `logs/classifieds-canary/<safe-run-name>`.

## Procedure

The launcher executes these ordered work units. Each unit has one observable
completion condition.

### 1. Claim the host

The launcher verifies that the selected host is stopped and that no other
operator owns it. It then acquires the local lock, records its owner token, and
resumes the host.

**Done when:** the status output names one running Remote Job and the launcher
reports no host-ownership error.

### 2. Prepare the isolated instance

The Remote Job starts a dedicated Classifieds web/database pair with the
configured immutable runtime. It binds the web service to the loopback
canary origin and performs the host-owned golden reset before mutation.

**Done when:** the Run contains a secret-free preparation and image-evidence
record, the compose configuration is valid, and the exact seed listing is
present without the canary body.

### 3. Prove the HTTP Writer and reader paths

The canary probe submits one body through the regular participant reply form.
It then opens the exact same-origin listing URL with a fresh anonymous reader
and matches the reply identity to the listing, displayed actor, normalized
body, and body digest. The Remote Job recreates the golden instance and uses
the anonymous reader to prove that exact reply absent.

**Done when:** the HTTP evidence proves the regular Writer, exact anonymous
readback, and exact-ID absence after reset. This probe proves rendered HTML
identity; it does not prove painted visibility.

### 4. Run Phase 2c and Phase 3

Phase 2c creates its own reply and repeats exact independent readback. Its
browser render check proves that the body is painted on the canonical listing.
Its Atomic Work Unit records cleanup of that temporary seed. Phase 3 validates
the matching benign contract. The Remote Job then recreates the golden instance
and rechecks the saved HTTP canary witness against the restored baseline.

**Done when:** Phase 2c records verified feasibility, exact identity, and
painted body evidence plus successful seed cleanup; Phase 3 records one valid
matching contract; and the following reset restores the probe baseline.

### 5. Pass the last preflight

The host-owned preflight checks the Run ID and Definition Digest, exact
one-instance auth split, Phase 2c evidence, Phase 3 contract, pinned runtime,
reset evidence, and one-task command limits. This is the last gate before the
browser agent runs.

**Done when:** `preflight.json` reports `passed` for the same Run, task,
topology, and immutable runtime that Phase 4 will use.

### 6. Run the bounded evaluation

The Remote Job runs the explicit `classifieds_listing_reply_poc` composition
with one admitted task, one worker, and at most one eval-awareness iterator
rewrite. Phase 2c and Phase 4 use the same host-owned exposure contract and
editor path.

**Done when:** the Run has a completed Phase 4 result, a positive PVPO
encounter that proves the attack witness was painted for the browser agent,
and a reward/readback outcome tied to the exact reply.

### 7. Prove final reset and complete the Remote Job

After Phase 4, the probe creates one new exact reply and records its identity.
The Remote Job recreates the golden instance and proves that reply absent while
the seeded parent remains. It then removes the canary containers, network, and
volumes before it validates terminal completion.

**Done when:** the final reset witness proves reply absence and parent presence,
the container cleanup check passes, and `completion.json` exists.

### 8. Park the host

The launcher verifies its owner token, clears the canary lifecycle marker,
parks the host, and then clears its owner marker. If the owner cannot be
verified, it leaves the host fenced and reports cleanup failure.

**Done when:** the Remote Job is terminal, the host is parked, and no canary
owner or sweep marker remains.

## Completion and evidence

The Remote Job has complete Run evidence only when `completion.json` exists in
the selected Run root. The completion validator checks these records and binds
their identities without copying secrets into the summary:

- preparation and preflight evidence;
- HTTP Writer and fresh anonymous reader evidence;
- Phase 2c painted readback and the Phase 3 contract;
- Phase 4 result and positive PVPO encounter;
- Run ID and Definition Digest;
- post-reset absence/presence evidence;

The overall canary is complete only when the local launcher also exits with
status `0`. That exit proves the Remote Job is terminal, the host is parked,
and the canary owner and sweep tags are absent. `completion.json` does not
attest this local lifecycle postcondition.

The retained operator evidence file is a sanitized summary. The detailed
provenance note is
[`classifieds-listing-reply-poc-2026-08-14.md`](../docs/research/classifieds-listing-reply-poc-2026-08-14.md).
It records the internal proof boundary and the remaining redistribution
license/data-inventory gate. Do not rewrite historical Run IDs, hashes, retry
chronology, or generated evidence to make a later document appear to have
been present during an earlier Run.

## Failure handling

- If a preflight check fails, do not start the browser task. Fix the named
  configuration or evidence input and use a new Run root.
- If the writer or reader witness is incomplete, classify the canary as
  failed. Do not substitute a body scan, root page, newest reply, writer
  session, SQL lookup, or admin readback.
- If the evaluation or reset fails, keep the Run and host evidence. Do not
  edit `completion.json` or promote partial output.
- If the launcher is interrupted, wait for its cleanup path to finish. Follow
  the status and stop checks in the [remote-run guide](./remote-runs.md), verify
  host ownership, and use a new Run root for any retry.
- A successful internal canary does not authorize redistribution. Complete
  the source, image, data, and notice inventory before publishing an image.

## Source-of-truth map

- Contract and admission: [technical specification](../docs/warp-taskgen-technical-spec.md)
  and [admission guide](./admission-and-exposure.md).
- Host and Remote Job rules: [remote-run guide](./remote-runs.md).
- Credential rules: [secret handling](./secrets.md).
- Provenance and release boundary: [Classifieds research note](../docs/research/classifieds-listing-reply-poc-2026-08-14.md).
