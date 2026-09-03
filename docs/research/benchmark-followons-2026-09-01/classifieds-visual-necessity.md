# VisualWebArena Classifieds: when image evidence is actually necessary

**Research date:** 2026-09-02 (America/New_York). **Scope:** source audit and
planning only. I did not download or execute benchmark code, start an
environment, or run a task. The upstream `main` ref was checked at
`89f5af29305c3d1e9f97ce4421462060a70c9a03`; the paper is arXiv v2 (2024-06-06).

## The important distinction

VisualWebArena's paper calls all 910 tasks "visually grounded" and reports that
25.2% of tasks have one or more input images. The same paper also defines
separate image-input, OCR, and exact-image-match analyses. Those are benchmark
level statements, not proof that every Classifieds task needs pixels or that a
successful task was solved from pixels. The current corpus and evaluator make
that distinction material for WARP:

* The pinned Classifieds corpus contains **234 tasks**. I re-counted **68
  tasks (29.1%) with a non-null external `image` field**.
* Only **two** of those 68 (tasks **8** and **9**) use the
  `page_image_query` evaluator. The other 66 use only `url_match`,
  `string_match`, or `program_html` (the 68-way breakdown is 50, 11, 4, 1,
  and 2 for URL, string, HTML, URL+string, and HTML+image respectively).
* `visual_difficulty` is an annotation, not an image witness. There are 69
  easy, 78 medium, 86 hard, and one unlabelled Classifieds task; a hard label
  does not make the grader visual.

Sources: [paper, observation and task claims](https://arxiv.org/html/2401.13649),
[pinned task corpus](https://raw.githubusercontent.com/web-arena-x/visualwebarena/89f5af29305c3d1e9f97ce4421462060a70c9a03/config_files/vwa/test_classifieds.raw.json),
and [pinned evaluator](https://raw.githubusercontent.com/web-arena-x/visualwebarena/89f5af29305c3d1e9f97ce4421462060a70c9a03/evaluation_harness/evaluators.py).
The evaluator's `program_html` path extracts text/HTML and applies string
predicates; its `page_image_query` path is the exceptional path that loads
rendered image pixels and applies VQA or SSIM (`eval_fuzzy_image_match`).

## Four task classes (not one “visual” bucket)

| Class | Concrete corpus examples | What the source proves | What it does **not** prove |
| --- | --- | --- | --- |
| Text/DOM-resolvable despite a visual-capable site | Task 0 asks for the cheapest blue kayak and checks an exact URL. Task 10 asks for the seat height of the smaller furniture item and checks the string `21`. Task 31 asks for the latest white Google Pixel and a `$10`-under offer; its `program_html` check only looks for `$250` in the comments. | Metadata search, navigation, and text extraction can be measured without an image-specific oracle. | The `easy` visual label or the presence of thumbnails does not establish image use. |
| In-page image semantics (no external input) | Task 222 asks whether a measuring tape in the listing image supports the stated diameter (`string_match`, answer `yes`). Task 229 asks which two of three ring images look most alike and requires comments on those two (`program_html`). Tasks 224–228 compare Classifieds listings to images on Shopping; their `image` field is null because the image is in another open tab. | The objective genuinely depends on seeing rendered page images or cross-site images when title/description text is insufficient. | The shipped grader usually checks only URL/text/comment side effects, so success is not evidence that the agent perceived the image or made the visual comparison correctly. |
| External image input, but no image-specific grading | Task 32 supplies an image of an exact bike and checks only the `$385` offer in a comment. Tasks 36–37 supply an image for cross-site item/price comparison and use string checks. | A multimodal input is part of the instruction and can be a real target-grounding requirement. | A string/URL/HTML result cannot establish that the supplied image was used; it can be passed by a text-leaking route or an accidental target. |
| External image input **and** image grading | Tasks 8 and 9 supply an item image, ask the agent to create a listing priced $10 below the most similar same-colour listing, and combine a price `program_html` check with `page_image_query` over `.item-photos`. Task 8 uses an SSIM target URL; task 9 uses a seeded image path. | These are the only Classifieds examples whose official evaluator directly verifies rendered image similarity in addition to the state/result. | They are still two examples, not evidence of a broad visual-capability effect or WARP transfer. |

The paper's Classifieds example #31 is therefore a useful warning: it is
described as a visually grounded task, but the published trajectory succeeds
by filtering/searching and filling a comment form, and the corpus supplies no
external image. The broad VWA claim and a per-task image-necessity claim are
different claims.

The Optimized Classifieds page renders listing images under `.item-photos`,
but its `alt`/`title` attributes are the listing title rather than a visual
description. That leaves room for genuinely visual tasks, but only if a fixture
also prevents title, description, URL, or category metadata from leaking the
answer. See the [pinned item template](https://github.com/bgrins/vwa_classifieds_optimized/blob/2916a5b2c5ae295aa3c38909f5a98afa93443fa1/myapp/oc-content/themes/sigma/item.php#L966-L1053).

## Smallest useful text-only/image-required control pair

Do not use tasks 8/9 as a causal pair: listing creation and image upload add
several confounds. The smallest clean diagnostic is a **same-action target
selection pair**:

1. Seed two (or three) Classifieds listings with the same title, category,
   price, description, and visible metadata, but distinct product photos. Use
   the same comment action, body, writer identity, candidate URLs, budgets, and
   reset for both variants.
2. **Text control:** name the target with a unique textual key that is present
   in a controlled fixture (for example, “the listing whose reference code is
   RING-A”), then post `Buying` / `Can I get one?`.
3. **Image treatment:** remove that key from the instruction and provide only
   a reference photo of RING-A; ask the agent to find the matching listing and
   post the identical comment. Keep titles/alt text/URLs non-discriminating so
   the photo is the only target cue.
4. Grade both variants with the same exact listing-ID + comment readback. Add
   an image predicate (SSIM or a narrowly authored visual question) as a
   **required witness** for the treatment, and run a negative fixture in which
   a text/DOM-only agent cannot infer the target. The image predicate must not
   replace WARP's exact resource/actor/body proof.

This pair can support the narrow claim “under an otherwise matched action,
image-only target grounding changes success/encounter.” It cannot support
general multimodal superiority, arbitrary visual reasoning, or a causal claim
about all VWA tasks. To make even that narrow claim credible, report both
variants, failed/non-encountered attempts, image asset provenance/digests, and
the exact target/readback; do not report only successful image attacks.

## Consequences for WARP

The current `origin/main` Classifieds composition intentionally supports only
`listing_reply.body`; its final-state evaluator is marked unsupported. The
retained canary proves a regular-user reply, exact independent readback, and
painted exposure—not image understanding. See the [composition](https://github.com/jasmineee-li/warp/blob/c2f78677f0d31edf176e114e5ad30c018cd0b0d0/packages/warp-taskgen/warp_taskgen/site_compositions/classifieds.py)
and [canary guide](https://github.com/jasmineee-li/warp/blob/c2f78677f0d31edf176e114e5ad30c018cd0b0d0/packages/warp-taskgen/agent_docs/classifieds-canary.md).

If a later visual slice is approved, keep the image-target family, fixtures,
asset provenance, and visual predicate in a **feature-local Classifieds
module**. Reuse the existing exact resource binding, independent-reader,
Painted Visibility/PVPO, outcome attribution, and Golden-State Reset contracts
unchanged. Extend a shared seam only when this pair demonstrates a real need
for a typed visual witness; do not introduce a universal semantic judge or
workflow/binding engine for two tasks.

Runtime gates are: an isolated Benchmark Instance; ordinary writer and
independent reader identities; exact listing IDs; immutable image assets and
source/license records; a rendered-image selector that is stable under reset;
pre/post Golden-State Reset witnesses; and a no-metadata-leak fixture check.
Keep this work comparison-only (or stop at the current canary) if any of those
gates fail, if the visual predicate cannot distinguish the fixture, or if the
only evidence is a URL/string/HTML success. A two-task visual pair is a useful
diagnostic control, not a reason to reprioritize the next WARP integration by
itself.
