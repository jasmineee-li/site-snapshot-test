# Classifieds listing-reply POC: provenance and proof boundary

**Date:** 2026-08-14
**Status:** research-only provenance note; this is not a live-environment approval

The scope follows [#147](https://github.com/jasmineee-li/warp/issues/147) and
the narrower acceptance contract in [#149](https://github.com/jasmineee-li/warp/issues/149):
one `visualwebarena` Classifieds listing body reply, created by a regular
participant and proven by an exact independent reader.

## Executive conclusion

VisualWebArena Classifieds is a credible real-site candidate for the narrow
`classifieds`/`listing_reply.body` proof in issue [#149](https://github.com/jasmineee-li/warp/issues/149).
It is a self-hosted, sandboxable OSClass marketplace with a documented regular
user comment workflow, deterministic seed/reset material, and a database model
that has an exact comment primary key and user foreign key. The optimized source
tree at inspected commit `2916a5b2c5ae295aa3c38909f5a98afa93443fa1` is the
smallest useful source starting point for a proof of concept. The Zoo’s exact
runtime SQL source pin is the earlier optimized commit
`fb33fea4b701a4eef502488d06267368b9104e90`; its published web/database image
digests are recorded below. The Zoo is a useful packaging and topology
reference, not a reason to add a Zoo-wide integration layer.

The upstream benchmark already proves only a weaker property: a logged-in agent
can submit a comment and the parent page contains the title, displayed author,
and body. Its evaluator does not prove a stable reply ID, a backend actor ID, or
visibility from a fresh independent reader. Those are the exact missing facts
that WARP must prove before this site can be admitted to a live canary. No public
demo host should be used for evaluation.

MUZZLE is now publicly packaged on Zenodo and explicitly includes
`classifieds.zoo` task specifications plus The Zoo, so it is useful provenance
for how recent red-teaming work exercised this site. Its record has no declared
license in the Rights field, and its archive contains a larger dependency graph;
it should not become a WARP dependency for this POC.

This note records verified facts, WARP inferences, and unresolved gates
separately. Sources were read without downloading or executing benchmark images,
archives, or red-teaming artifacts. Secret values shown by upstream examples are
intentionally omitted.

## Pinned primary sources

| Artifact | Exact revision or immutable identifier | What it establishes | License/provenance status |
| --- | --- | --- | --- |
| [VisualWebArena repository](https://github.com/web-arena-x/visualwebarena/tree/89f5af29305c3d1e9f97ce4421462060a70c9a03) | Git commit `89f5af29305c3d1e9f97ce4421462060a70c9a03` (main at research time) | Task corpus, evaluator, Docker instructions, benchmark configuration | Repository README identifies the project as MIT licensed; verify the exact distribution/third-party notice set before redistribution. |
| [VWA Classifieds task corpus](https://raw.githubusercontent.com/web-arena-x/visualwebarena/89f5af29305c3d1e9f97ce4421462060a70c9a03/config_files/vwa/test_classifieds.raw.json) | Same VWA commit | Official comment-task intents and `program_html` checks | Task data is source-controlled; image/data provenance still needs an explicit inventory for a packaged image. |
| [VWA evaluator](https://raw.githubusercontent.com/web-arena-x/visualwebarena/89f5af29305c3d1e9f97ce4421462060a70c9a03/evaluation_harness/evaluators.py) | Same VWA commit | `program_html`/selector/string evaluator behavior | Evaluator behavior is MIT-project source, not a proof of WARP’s stronger readback contract. |
| [Optimized Classifieds source](https://github.com/bgrins/vwa_classifieds_optimized/tree/2916a5b2c5ae295aa3c38909f5a98afa93443fa1) | Git commit `2916a5b2c5ae295aa3c38909f5a98afa93443fa1` (main at research time) | OSClass app, comment model, browser form/controller, reset controller, Docker build | Top-level `LICENSE` is Apache-2.0. The tree vendors Osclass/PHP components with their own notices; a complete inventory remains a gate. |
| [The Zoo’s Classifieds SQL pin](https://github.com/bgrins/the_zoo/blob/90f2bfed01a3ae7bd73a0fb16fb1eaa406705b52/core/mysql/Dockerfile#L11-L13) | Optimized source commit `fb33fea4b701a4eef502488d06267368b9104e90` | The Zoo’s MySQL image downloads `classifieds_import.sql` and `classifieds_restore.sql` from this exact source revision; this is the runtime data pin, while `2916…` is the inspected successor | Source license/inventory is the same optimized-tree obligation above. |
| [Classifieds Docker archive](https://archive.org/download/classifieds_docker_compose) and [file manifest](https://archive.org/download/classifieds_docker_compose/classifieds_docker_compose_files.xml) | `classifieds_docker_compose.zip`: 25,366,023 bytes; MD5 `cf4fe746f22efa4e6102ac08fe76d4db`; SHA-1 `b351be6468feeeb0b9b261dfc0ad9e8c2966d718`; CRC32 `b5acea88` | Public archive named by the VWA Docker README; item metadata says it was added 2024-05-28 and describes the VWA Classifieds environment | Internet Archive metadata supplies hashes and uploader metadata, but no license field. Do not treat the archive as license-cleared. |
| [The Zoo repository](https://github.com/bgrins/the_zoo/tree/90f2bfed01a3ae7bd73a0fb16fb1eaa406705b52) | Git commit `90f2bfed01a3ae7bd73a0fb16fb1eaa406705b52` (main at research time) | `.zoo` proxy/DNS topology, personas, reset-on-restart convention, and `classifieds.zoo` registration | Top-level repository license is Apache-2.0; its app images and bundled data retain separate provenance obligations. |
| [The Zoo compose](https://raw.githubusercontent.com/bgrins/the_zoo/90f2bfed01a3ae7bd73a0fb16fb1eaa406705b52/docker-compose.yaml) | Same Zoo commit | `vwa-classifieds` service wiring and image reference | Source compose uses mutable `:latest` tags and public example configuration; runtime must substitute the recorded OCI digests below. |
| Published Optimized Classifieds images (OCI manifests resolved 2026-08-14) | Web index `sha256:a75c98c1383e125def1149e538175c0ca30a26a205fb9c1e2e3b5394a2d3984a`; Linux/amd64 manifest `sha256:b2df5018c66bb48ce1828bef3f28171b90c4c74027ab0a3611d93cbb7a4509f7`; DB index `sha256:b5b16ce11efe9417f17acf198708ad2a321ae765abf1e3fa0d56efddccaba635`; Linux/amd64 manifest `sha256:70c327b1c16aac0b17c05fd59ef79f6930a1389e9da45b04d33cd47f327b7d1a` | Immutable image identities corresponding to the published GHCR web/database images used by the Zoo packaging | These digests close the tag-mutability bookkeeping gap, but a WARP canary must still verify the pulled manifests and record the source/image pair together. |
| [Mozilla’s The Zoo paper](https://research.mozilla.org/files/2026/06/from_the_wild_web_to_the_zoo.pdf) | June 2026 published PDF | Classifieds scale, golden-state reset design, `.zoo` proxy, and regular/admin persona split | Research description, not a substitute for inspecting the exact instance image and license inventory. |
| [MUZZLE artifact record](https://zenodo.org/records/20399450) | Zenodo record `20399450`, version v1, published 2026-05-27; `muzzle.zip` MD5 `2308f2a4fc9d46efdf72d9c8f302ad35` (29.3 MB) | Public package says it contains MUZZLE, `classifieds.zoo` scenario specs, and The Zoo with seed/reset logic | Zenodo Rights/License is blank. Treat as provenance only until package contents, submodules, and licenses are audited. |
| [MUZZLE USENIX page](https://www.usenix.org/conference/usenixsecurity26/presentation/syros) and [paper](https://arxiv.org/abs/2602.09222) | USENIX Security ’26 / arXiv record | Primary description of the red-teaming study; confirms the artifact is a research package, not a WARP site adapter | No claim here that MUZZLE’s task specs satisfy WARP’s exact-ID/independent-reader proof. |

## Verified facts

### 1. The benchmark and site are real, self-hosted, and task-relevant

The VisualWebArena project page describes 910 tasks across Classifieds,
Shopping, and Reddit, and calls Classifieds a new environment with real-world
data ([project page](https://jykoh.com/vwa#L19-L24)). It explicitly warns that
the linked public demo hosts are for demonstration only and recommends setting
up a local copy for evaluation ([project page](https://jykoh.com/vwa#L35-L39)).
The pinned VWA README names a `CLASSIFIEDS` host and a
`CLASSIFIEDS_RESET_TOKEN` configuration value, and its Docker README points to
the public Docker archive, a seeded SQL dump, and a POST reset operation. This
is enough to establish a reproducible *candidate*, not a configured WARP host.

The official corpus marks comment tasks as `sites: ["classifieds"]`,
`require_login: true`, and uses a Classifieds storage state. Tasks 28–30 use
the normal intent “add a comment” with a title and body. Their evaluators visit
the exact seeded listing URL and require the `.comments_list h3` text to include
the requested title and the displayed user label, then require the
`.comments_list` text to include the requested body. Tasks 31–33 similarly
post an offer and check only that the offer text appears in the comments list.
Examples are in the pinned [task corpus](https://raw.githubusercontent.com/web-arena-x/visualwebarena/89f5af29305c3d1e9f97ce4421462060a70c9a03/config_files/vwa/test_classifieds.raw.json)
around task IDs 28–33.

The VWA evaluator routes `program_html` to selector/JavaScript extraction and
then applies `must_include`, `must_exclude`, or exact string checks. Its
implementation does not query a resource primary key, a user primary key, or a
second browser identity ([pinned evaluator](https://raw.githubusercontent.com/web-arena-x/visualwebarena/89f5af29305c3d1e9f97ce4421462060a70c9a03/evaluation_harness/evaluators.py#L318-L451)).

### 2. The ordinary browser path is a normal-user comment submission

At the pinned optimized source, the Sigma item page renders a same-origin POST
form with `action=add_comment`, `page=item`, the listing `id`, and an optional
`replyId`. A logged-in browser receives hidden author name/email fields from the
session; the form includes title and body inputs and a submit button. The
reader-visible comments list renders a title, “by” display name, and body in
`.comments_list` ([item template](https://github.com/bgrins/vwa_classifieds_optimized/blob/2916a5b2c5ae295aa3c38909f5a98afa93443fa1/myapp/oc-content/themes/sigma/item.php#L150-L326)).

The exact Zoo-pinned source commit `fb33fea4b701a4eef502488d06267368b9104e90`
adds one important proof surface to that ordinary-reader page: when the seeded
preferences have `enable_comment_reply=1` and an open
`comment_reply_user_type=''`, each top-level comment renders a
`.comment-reply[data-id]` link whose `data-id` is `osc_comment_id()`. The same
seed sets `moderate_comments=-1` (new comments are active) and
`reg_user_post_comments=0` (anonymous comments are allowed; WARP must still
require its authenticated regular-user session). See the pinned [item
template](https://github.com/bgrins/vwa_classifieds_optimized/blob/fb33fea4b701a4eef502488d06267368b9104e90/myapp/oc-content/themes/sigma/item.php#L150-L205)
and [seed preference row](https://raw.githubusercontent.com/bgrins/vwa_classifieds_optimized/fb33fea4b701a4eef502488d06267368b9104e90/mysql-baked/classifieds_import.sql).
This makes a stable reply ID observable by an independent ordinary reader:
match the exact actor/body/signature within one comment block, then read that
block’s `data-id`; no DB/admin/newest heuristic is needed. The upstream VWA
evaluator still does not perform this exact-ID readback.

The item controller checks CSRF, calls `ItemActions::add_comment()`, maps
approved/moderation/validation outcomes, and redirects back to the item URL
([controller](https://github.com/bgrins/vwa_classifieds_optimized/blob/2916a5b2c5ae295aa3c38909f5a98afa93443fa1/myapp/oc-includes/osclass/controller/item.php#L624-L736)).
The action reads `userId` from the authenticated session, not from a caller
supplied user ID. It sanitizes title/body, builds a row with
`fk_i_item_id`, `s_title`, `s_body`, and `fk_i_user_id`, inserts it, obtains the
database `insertedId`, and returns only a numeric status to the controller
([action](https://github.com/bgrins/vwa_classifieds_optimized/blob/2916a5b2c5ae295aa3c38909f5a98afa93443fa1/myapp/oc-includes/osclass/ItemActions.php#L1480-L1670)).

This has two important consequences:

* the browser path is a legitimate regular-user mutation, but it does not
  return the newly created comment ID in the redirect;
* the displayed author name is request/form data, while the authenticated
  `userId` is the stronger actor identity. A WARP proof must not treat a
  rendered “by …” string as sufficient attribution.

### 3. The application has an exact stable comment/reply data model

The pinned `ItemComment` model maps `t_item_comment.pk_i_id` as its primary key
and includes `fk_i_item_id`, `fk_i_user_id`, `fk_i_reply_id`, title, body, active,
and enabled fields. It provides exact lookup by primary key, item, reply, and
author, and the normal public listing query filters to active/enabled rows
([model](https://github.com/bgrins/vwa_classifieds_optimized/blob/2916a5b2c5ae295aa3c38909f5a98afa93443fa1/myapp/oc-includes/osclass/model/ItemComment.php#L55-L180)).
The database therefore has the ingredients for a stable reply ID and actor
attribution. At the Zoo-pinned source, the ordinary reader also exposes the
comment primary key as `data-id` on the reply link, so a fresh browser context
can perform the exact ID readback without SQL or admin access. The upstream
browser evaluator does not expose or check that field; WARP must correlate the
reader’s ID to the exact parent listing and authenticated writer evidence.

Top-level comments are the narrowest POC: leave `replyId` empty and create one
comment attached to the seeded listing. The application also supports one-level
nested comment replies, but that is a separate surface and should not be
silently included in `listing_reply.body`.

### 4. Reset and package material exist, but are operationally privileged

The VWA Docker instructions point to `classifieds_docker_compose.zip`, the
`osclass_craigslist.sql` seed, and a POST reset route. The optimized repository
adds a baked MySQL image that restores a golden state on DB-container restart,
while retaining a reset controller for compatibility ([optimized README](https://github.com/bgrins/vwa_classifieds_optimized/blob/2916a5b2c5ae295aa3c38909f5a98afa93443fa1/README.md)).

The pinned reset controller accepts POST only, compares a supplied token with
`RESET_TOKEN`, executes the SQL restore through `exec`, and echoes the complete
MySQL command. The command includes the database password in its response
([reset controller](https://github.com/bgrins/vwa_classifieds_optimized/blob/2916a5b2c5ae295aa3c38909f5a98afa93443fa1/myapp/oc-includes/osclass/controller/reset.php#L14-L53)).
This must remain an instance-side privileged operation, outside the browser
editor path; WARP diagnostics must never log the response or token.

The Zoo repository describes a `.zoo` proxy/DNS environment, regular/admin
personas, and MySQL/PostgreSQL reset-on-restart. The Mozilla paper explains its
golden-state snapshot approach: the DB directory is restored from an image-held
tar archive at container start. The Zoo’s pinned compose registers
`classifieds.zoo`, and its `core/mysql/Dockerfile` pins the Optimized Classifieds
SQL to `fb33fea4b701a4eef502488d06267368b9104e90`. The compose file still names
mutable `:latest` tags, but the published web/database OCI index and amd64
manifest digests are recorded in the source table. A canary must use those
digests (or a separately hashed local build) and retain the source/image pair.

### 5. MUZZLE is publicly available, but not a drop-in WARP dependency

The Zenodo v1 record says `muzzle.zip` contains the complete MUZZLE pipeline,
task specifications for `gitea.zoo`, `postmill.zoo`, `classifieds.zoo`, and
cross-application scenarios, plus The Zoo and its database seeding/reset logic.
The record provides an archive MD5 and DOI but leaves Rights/License blank
([Zenodo record](https://zenodo.org/records/20399450#L20-L41)). The USENIX page
and arXiv paper describe the research framework and its adaptive red-teaming
purpose; neither claims that the Classifieds scenarios provide WARP’s exact
comment-ID, actor-ID, or independent-reader evidence.

The artifact is therefore useful for provenance and comparison (“recent work
did exercise `classifieds.zoo`”), but adopting it would add submodules, agent
scaffolds, registry images, and a license audit to a POC whose question is much
narrower. Keep it out of the first WARP implementation unless a later experiment
needs the MUZZLE attack pipeline itself.

## WARP inference (not upstream guarantees)

These are the minimal conclusions I would carry into issue #149’s implementation:

1. **Use explicit identities.** Keep benchmark `visualwebarena`, site
   `classifieds`, route `classifieds.listing`, action `create_listing_reply`,
   and only `listing_reply.body` as separate fields. The VWA corpus’s
   `require_login` and storage state justify a regular-user lane; they do not
   justify admin, SQL, or reset access from the editor.
2. **Seed one known parent and record its exact ID.** Resolve the seeded listing
   before the action and carry its exact same-origin URL/ID through creation and
   readback. Do not select “latest,” scan for a matching body, or infer a parent
   from a broad listing page.
3. **Prove creation with two witnesses.** The mutation witness is the normal
   browser POST/redirect and status. In the Zoo-pinned build, the ordinary reader
   exposes the inserted `pk_i_id` as `.comment-reply[data-id]`; the readback
   witness must match that ID to the exact listing, actor/body/title signature,
   active/enabled visibility, and same-origin URL within one comment block. Then
   open a fresh independent reader context and prove the public reader sees that
   exact row.
4. **Treat backend identity as authoritative.** `fk_i_user_id` from the
   authenticated session is the actor identity; displayed `s_author_name` is
   presentation and request data. The reader-visible ID is sufficient for
   stable-resource proof, but the canary must still correlate the displayed
   actor/signature to the known authenticated writer. If that attribution or
   exact ID is missing, remain ineligible rather than falling back to body text.
5. **Keep reset outside the editor.** Instance setup/reset should be a
   secret-ref-backed harness operation. After reset, assert that the seed parent
   exists and the created row is gone; do not pass reset tokens through browser
   task/editor arguments.
6. **Pin the runtime before a canary.** Use the Zoo SQL source pin
   `fb33fea4b701a4eef502488d06267368b9104e90` together with the recorded web and
   database OCI index/amd64 manifest digests, or record equivalent digests for a
   local build. The source compose’s `:latest` tags are not sufficient evidence
   by themselves.
7. **Keep the proof semantic-only.** Credit only the exact body action and its
   exact final-state proof. Do not import generic plugin abstractions, broad
   marketplace operations, nested replies, or MUZZLE orchestration into this
   vertical slice.

## Unresolved blockers and live-canary gates

1. **Image identity is recorded but not live-verified.** The web/database OCI
   index and Linux/amd64 manifest digests are now pinned in the source table,
   and the Zoo SQL source is pinned to `fb33fea4b701a4eef502488d06267368b9104e90`.
   The source compose still refers to `:latest`, so the live harness must pull
   by digest (or verify a local build) and record that source/image pair.
2. **License/data inventory is incomplete.** VWA, optimized source, and The Zoo
   expose top-level MIT/Apache-2.0 licenses, but the Internet Archive archive has
   no license field and the optimized tree vendors Osclass/PHP code plus a large
   image/database corpus. Complete a notice and data-provenance inventory before
   publishing or redistributing an image.
3. **Separate-session canary and reset are unverified here.** The pinned seed
   makes comments active and exposes a stable `data-id`, but no run in this
   research note has demonstrated writer login/CSRF, a regular-user POST,
   exact ID/actor readback from a fresh independent reader, or reset restoration.
   A configured sandbox must prove all of those; the public VWA demo is
   explicitly not an evaluation host.
4. **Stable ID is available but independent attribution is not in upstream
   scoring.** The app inserts a stable primary key and the ordinary reader
   exposes it as `data-id`, but the redirect and `.comments_list` evaluator do
   not check it or prove the actor. The #149 canary must still use separate
   participant/reader sessions and correlate the exact actor/body/signature.
5. **Reset is a secret-bearing privileged endpoint.** The reset controller’s
   command echo can leak DB credentials. Configure a private instance route,
   inject the token from approved secret storage, redact all responses, and
   verify deterministic post-reset state.
6. **MUZZLE artifact rights and mutable dependencies are unresolved.** Its
   Zenodo package is discoverable and hashable, but the record has no license and
   describes a multi-component package. It should remain research provenance,
   not be bundled into WARP, until separately audited.
7. **No live independent-reader result exists yet.** Until the bounded canary
   produces an artifact containing the parent ID, reply ID, actor ID, body
   signature, visibility evidence, source revision/image digest, reset proof, and
   sanitized logs, the Classifieds policy must remain experimental/opt-in.

## Bounded proof sequence

The smallest useful next experiment is deliberately one task and one worker:

1. Verify the source SHA, archive SHA-1/MD5, image build/digest, license
   inventory, and secret references without placing secrets in task data.
2. Run the existing fake conformance tests for explicit benchmark/site/action
   identities and negative cases (wrong site, missing parent, missing actor,
   newest/latest-only evidence, and reset invoked through the editor).
3. On a configured sandbox, reset outside the editor, resolve one seeded listing,
   submit one body as the documented regular user, and capture the status.
4. Open a separate reader context with no writer session and verify exact
   listing/reply/actor/body/signature/visibility, including the comment’s
   `.comment-reply[data-id]`, on the same-origin surface.
5. Reset again and prove the created ID is absent while the seed parent remains;
   record a sanitized manifest and fail closed on any missing witness.

This sequence demonstrates the requested vertical slice without making
Classifieds active by default and without creating a general plugin ecosystem.

## Deterministic implementation evidence

The local POC implements the static and fake-host portion of that sequence. It
does not promote the Site or claim the configured-host canary:

- The installed Site doctor reports `static_status=complete` for
  `visualwebarena` / `classifieds` / `ugc_reply`, with definition digest
  `9e4e31ac9532da9f40fa262288d1bad99e00757c21e38ea7acdc5ebbd8c10902`.
  Its overall status remains `blocked` because active-policy, configured-host,
  admission, execution, and scoring evidence are deliberately absent.
- The corrected feature-focused smoke matrix passed 189 tests, including the
  generic seed dispatcher, exact HTML-derived readback, body-digest mismatch,
  exact-body visibility, redirect, login-form, and wrong-listing negative cases.
- The complete package test run passed 4,103 tests, skipped four, and deselected
  41 opt-in tests when run with the host facilities required by the repository's
  lifecycle checks.
- `scripts/verify_fast.sh --skip-collect` passed. The package-proof acceptance
  lane built and installed both the wheel and source distribution in isolated
  environments, executed the installed Classifieds doctor, retained the
  AgentLab sidecar smoke, and passed the ordinary 0.1.0-to-0.1.1 package upgrade
  cleanup.

These results prove deterministic composition, packaging, exact fake-host form
and readback behavior, and regression safety. They do not prove that the pinned
Classifieds image is reachable, that two real browser sessions observe the same
reply, or that reset restores the configured instance. Those remain the bounded
live gate above.
