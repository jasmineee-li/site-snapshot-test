# Wave-1 Platform Manifest And Manual Authoring

This guide turns TODO 09 into the concrete operator workflow for the current
pilot platform and the later broader shared-app rollout.

Use this alongside:

- [09-TODO-design-platform-consolidation-user-manuals-and-multi-behavior-apps.md](/Users/ashtonchew/projects/browser-sim/docs/09-TODO-design-platform-consolidation-user-manuals-and-multi-behavior-apps.md)
- [architecture.md](/Users/ashtonchew/projects/browser-sim/docs/architecture.md)
- [download-manual.md](/Users/ashtonchew/projects/browser-sim/vendors/webarena-infinity/docs/download-manual.md)

## Scope

The active rollout is an Amazon pilot, not the older 5-platform wave proposed
earlier in TODO 09.

Current pilot target:

- Amazon home page
- product detail page
- reviews on the product detail page
- one simple action on the product detail page, recommended default:
  `add to cart`

Deferred platform candidates:

- Gmail
- ZocDoc
- Airbnb
- AutoTrader
- Zillow

The authoritative active membership is not the narrative table in the design
doc. It is the committed [platform_manifest.json](/Users/ashtonchew/projects/browser-sim/platform_manifest.json).

## What "Defer" And "Reject" Mean

These are curation decisions, not generation prompts.

- `include`: the behavior is mapped into the active shared app and appears in
  `platform_manifest.json`.
- `defer`: the behavior stays in the validated benchmark corpus but is omitted
  from the current `platform_manifest.json`. It is a candidate for a later app
  wave.
- `reject`: the behavior is intentionally excluded from the current app family
  because it would require unsupported product scope, unsupported domains, or a
  non-manuals-backed workflow.

In practice:

- `defer` means "do not put this behavior in the current manifest yet."
- `reject` means "do not put this behavior in this app, even temporarily."

Examples:

- An Amazon behavior that only needs home page browsing, product detail, and
  reviews can be `include`.
- An Amazon behavior that requires cart, checkout, returns, or order history
  before we intentionally add those surfaces should usually be `defer`.
- An Amazon behavior that only works if we invent undocumented marketplace,
  fraud-reporting, or seller tooling should be `reject` for the pilot app.

The generation prompt does not make this decision. The curator does, by
choosing what enters `platform_manifest.json`.

## Platform Curation Workflow

### 1. Start from the validated behavior corpus

List the candidate `behavior_id`s you want to consider for the current app.

### 2. Choose the app boundary the WAI way

Pick a coherent manuals-backed application scope first. Do not start from
attack themes.

Use this test:

- one docs corpus
- one shared benign seed
- one coherent feature area
- no invented unsupported workflows

If a broad platform needs to split by feature area, split it. WAI does this
regularly. It does not force one monolithic app per brand.

Examples from WAI:

- Gmail: `gmail-accounts-and-contacts`, `gmail-organize-and-manage`
- Figma: `figma-slides`, `figma-text-and-typography`
- Elation: `elation-patient-communication`, `elation-prescriptions`,
  `elation-clinical-records`

For the Amazon pilot, start with one app boundary only if it stays narrow. If
even Amazon becomes too broad, split by feature area before generation.

### 3. Map each included behavior

Every included behavior needs:

- `behavior_id`
- `entry_route`
- `allowed_routes`
- `domain_bindings`
- optional `mapping_rationale`

Use the narrowest routing/domain scope that still supports the documented safe
workflow.

### 4. Defer or reject anything that widens the app silently

If a behavior cannot be supported without inventing UI, adding undocumented
features, or treating a second real app as fully available, do not force-fit
it into the current manifest.

### 5. Validate before generation

Run:

```bash
cd /Users/ashtonchew/projects/browser-sim/AgentLab
uv run agentlab-redteam-validate-platform-manifest \
  --platform-manifest /Users/ashtonchew/projects/browser-sim/platform_manifest.json \
  --benchmark-file /path/to/behaviors.json
```

This checks:

- duplicate or missing `behavior_id` mappings
- required per-behavior routing fields
- `domain_bindings` structure
- docs-path existence

## WAI-Style Manual Scraping Workflow

The source method is [download-manual.md](/Users/ashtonchew/projects/browser-sim/vendors/webarena-infinity/docs/download-manual.md).

### 1. Create `urls.txt`

For the current pilot, create:

- `apps/user-manuals/amazon/urls.txt`

This file should contain help-center entry points only. Do not seed with API,
SDK, CLI, or developer docs unless the page is truly GUI-user-facing.

### 2. Scrape GUI docs only

Follow the WAI constraints:

- crawl from `urls.txt`
- keep only GUI usage docs
- strip nav, footers, scripts, and chrome
- save markdown under `apps/user-manuals/{platform}/{feature-area}/`
- include a `Source:` header in every file

### 2A. Fallback when the site blocks automated fetches

Some sites, including Amazon, may block simple HTTP fetches or obvious bot
traffic. Do not try to brute-force around that with more aggressive scraping.

Use this fallback instead:

1. Open the help pages in a real browser session.
2. If needed, pass any human verification manually.
3. Save the relevant pages locally as HTML or PDF under:
   `apps/user-manuals/{platform}/raw/`
4. Then run Claude Code against those local raw files to convert and clean them
   into the committed markdown corpus under
   `apps/user-manuals/{platform}/{feature-area}/`.

This still satisfies the goal of a committed manual corpus. The difference is
that Claude Code is operating on local captured source pages rather than trying
to fetch the site directly.

### 3. Keep the formatting strict

The output needs:

- one paragraph per line
- no 80-column hard wrapping
- fenced code blocks
- no PUA characters
- working relative internal links when practical

### 4. Run post-processing checks

Use the WAI checks described in `download-manual.md`:

- line-wrap detection
- PUA scan
- structural spot-check across 3-5 files per directory
- paragraph unwrap pass if needed

### 5. Commit the markdown snapshot

The committed manual corpus is what generation reads. Do not treat scraping as
an ephemeral pre-step.

### 6. Point the platform manifest at the committed corpus

Set each app's `docs_path` to the committed platform manual directory.

Example:

```json
{
  "app_id": "amazon-home-pdp",
  "platform": "amazon",
  "docs_path": "apps/user-manuals/amazon",
  "behaviors": []
}
```

### 7. Let generation capture `docs_snapshot`

The controller computes and persists `docs_snapshot` into
`apps/{app_id}/app_manifest.json`. Resume is fail-closed if the manual corpus
changes after generation starts.

## Recommended Authoring Order

1. Scrape and validate manuals for Amazon.
2. Draft the Amazon pilot app entry in `platform_manifest.json`.
3. Curate included Amazon behaviors only.
4. Defer the rest by omission from the current manifest.
5. Validate the manifest.
6. Generate the shared app.

## Non-Goals

This guide does not require us to preserve old generated artifacts. There are no
checked-in generated shared-app artifacts in the repo today, so current curation
should optimize for a clean contract, not backward compatibility.
