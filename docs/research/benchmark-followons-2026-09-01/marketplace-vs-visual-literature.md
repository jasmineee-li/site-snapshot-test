# Marketplace breadth versus visual grounding

Research baseline: **2026-08-30 (ET)**.  Follow-on source review: **2026-09-02
(ET)**.  This note answers one bounded question: what can a new marketplace
Site claim about domain breadth, and what additional evidence is needed before
calling its tasks visually grounded?  It is source inspection only; no WARP
integration, browser/model run, live sandbox, corpus admission, or paper
result was produced.

## Short answer

Site breadth and visual necessity are independent factors.

* **Breadth** means that a benchmark covers a materially different application
  genre, state model, or user objective.  WebArena deliberately chose four
  categories (e-commerce, discussion forums, collaborative development, and
  content management) and released 812 long-horizon tasks.  WorkArena then
  isolated enterprise knowledge work (33 atomic tasks, 19,912 instances), and
  WorkArena++ composed those atoms into 682 workflow tasks.  These sources
  support claims about domain/workflow diversity and scale; none proves that
  pixels are necessary.
* **Visual necessity** means that the task's discriminating fact is absent from
  text/DOM/accessibility observations and must be inferred from an image or
  spatial rendering.  VisualWebArena intentionally makes this claim: its 910
  tasks are split across Classifieds, Shopping, and Reddit, and the paper says
  each requires visual understanding.  Its results show GPT-4V + Set-of-Marks
  (SoM) at 16.37% overall versus text-only GPT-4 at 7.25%, but this is a
  modality/model comparison, not a controlled removal of the visual signal.
  WorkArena's same-model screenshot comparison instead gives GPT-4o-V 41.8%
  versus text GPT-4o 42.7% on WorkArena, demonstrating that visual access alone
  is not a necessity claim.

For WARP, a Classifieds slice should first claim a bounded marketplace/content
workflow and exposure behavior.  Add a separate visual-grounding arm only when
the task has a paired text/visual control and the visual cue is demonstrably
not leaked through title, description, alt text, URL, or evaluator metadata.

## Re-measured primary sources (as of 2026-09-02)

The merged files, rather than issue/PR proposals, are the count authority:

| Source | Current primary count | What the count means |
| --- | ---: | --- |
| [WebArena paper](https://arxiv.org/html/2307.13854) and [canonical task file](https://raw.githubusercontent.com/web-arena-x/webarena/refs/heads/main/config_files/test.raw.json) | **812** (`task_id` 0–811) | One release covering the four application categories plus utility/knowledge sites; the paper's functional-correctness claim, not a visual claim. |
| [VisualWebArena Classifieds task file](https://raw.githubusercontent.com/web-arena-x/visualwebarena/refs/heads/main/config_files/vwa/test_classifieds.raw.json) | **234** (`task_id` 0–233) | New Classifieds environment; posting/searching/commenting and visual matching are mixed in one site slice. |
| [VisualWebArena Reddit task file](https://raw.githubusercontent.com/web-arena-x/visualwebarena/refs/heads/main/config_files/vwa/test_reddit.raw.json) | **210** (`task_id` 0–209) | Image-rich forum tasks. |
| [VisualWebArena Shopping task file](https://raw.githubusercontent.com/web-arena-x/visualwebarena/refs/heads/main/config_files/vwa/test_shopping.raw.json) | **466** (`task_id` 0–465) | Product-image and shopping tasks. |
| [VisualWebArena paper](https://arxiv.org/html/2401.13649) | **910 total** (234 + 210 + 466) | 314 intent templates, average 2.9 instances/template; 46 (5.1%) intentionally unachievable.  Several instances are multi-Site, so per-file counts do not form three disjoint semantic families. |
| [WorkArena paper](https://arxiv.org/html/2403.07718) and [current repository README](https://raw.githubusercontent.com/ServiceNow/WorkArena/master/README.md) | **33 tasks / 19,912 instances** | Atomic enterprise UI coverage, mostly evaluated with AXTree; screenshot support is optional. |
| [WorkArena++ paper](https://arxiv.org/html/2407.05291) | **682 tasks** | Composed planning/reasoning/retrieval tasks; each task samples thousands of configurations, and ten visual brand themes are a diversity intervention, not proof that pixels are required. |

The current upstream state matters for reproducibility.  WebArena `main` still
publishes the 812-task file, while annotation/evaluator corrections remain in
open proposals such as [PR #274](https://github.com/web-arena-x/webarena/pull/274)
(nine `shopping_admin` annotation errors reported 2026-08-27).  VisualWebArena
`main` publishes the files above; its visual/evaluator updates remain open in
[PR #67](https://github.com/web-arena-x/visualwebarena/pull/67) and its
documentation/package work in [PR #81](https://github.com/web-arena-x/visualwebarena/pull/81)
and [PR #86](https://github.com/web-arena-x/visualwebarena/pull/86).  Do not
silently use an unmerged proposal as a current benchmark result.

## What the papers actually establish

### WebArena: domain breadth and functional state, not visual necessity

The WebArena paper describes four self-hosted applications from distinct
categories, imports sampled real-world content, and provides deterministic
reset scripts.  Its 812 intents are judged by functional state rather than an
exact action sequence.  The observation space intentionally supports screenshot,
DOM, and accessibility-tree modes; the canonical README's walkthrough uses
`observation_type="accessibility_tree"`.  Thus WebArena is a useful precedent
for a bounded Site/world and exact readback, but a successful AXTree agent does
not disprove its domain-breadth claim and the benchmark never claims all tasks
need pixels.

### VisualWebArena: explicit visual-task construction, with a confounded baseline

VisualWebArena adds a Classifieds environment with 65,955 listings and says all
910 tasks require visual understanding; examples include selecting a green
polo whose color is not explicit in text, exact image matching, OCR, and spatial
disambiguation.  Its evaluator includes textual predicates, VQA, image SSIM,
and final-page locators.  The task files make the distinction concrete: some
Classifieds tasks have `visual_difficulty: easy` and URL/string evaluators,
while others have `page_image_query`, image inputs, or comments explaining that
the relevant fact appears only in an image.

The reported results are useful but not causal evidence of visual necessity:
text-only GPT-4 (AXTree) scores 7.25% overall, GPT-4V with screenshot +
captions 15.05%, and GPT-4V with SoM 16.37%; Classifieds improves from 12.38%
to 17.14% with SoM.  The comparison changes the model/input stack and SoM's
element-ID grounding, so the uplift can include model capability, navigation,
or action-space effects.  The paper's subset table further shows that 74.8%
of tasks have no input image, even though the authors label every task visually
grounded; “visually grounded” therefore includes page/layout/image semantics,
not only user-supplied image inputs.

### WorkArena/WorkArena++: visual diversity is a separate intervention

WorkArena exposes HTML/AXTree/screenshot and explicitly lets researchers run
text-only or vision-augmented agents.  Its same-codebase ablation reports
WorkArena 42.7% for GPT-4o text and 41.8% for GPT-4o-V with SoM screenshot;
the authors call the multimodal difference minor.  WorkArena++ adds ten
fictitious brands with different colors/logos and randomizes the theme, but its
main observation remains AXTree and its contribution is compositional planning,
retrieval, reasoning, and task isolation.  This is direct primary evidence that
visual *diversity* (rendering style) can improve realism without making a task
visually necessary.

## Implication for a WARP Classifieds follow-on

### Breadth arm (recommended first)

Use one generated Classifieds family whose semantic predicate and exposure
carrier are text/state based: e.g., inspect a finite set of listings, decide
which satisfy a category/price/status predicate, and optionally post exactly one
regular-user reply containing the WARP payload.  This can support:

* marketplace/UGC Site breadth beyond GitLab, Postmill, and Rocket.Chat;
* finite-set reasoning and information-only versus state-changing behavior; and
* exposure outcomes (encounter, propagation, incorrect conclusion,
  wrong-target action, unauthorized extra artifact) with WARP's existing
  evidence, binding, and reset contracts.

It cannot support a claim that a multimodal agent needs vision, that image
matching is solved, or that all marketplace tasks are visually grounded.

### Optional visual arm (only after the breadth contract)

Add a paired family where the target listing is identified by a visual-only fact,
then perform the same bounded read/reply action.  A concrete example is “find
the listing whose photograph shows a red bicycle with a front basket and post
the prescribed question”; the listing title/description/category must not
encode those discriminators.  Keep a matched text-only control (same records,
same payload, same action and reset) and a visual-cue-masked control.

The visual arm needs additional evidence: preserve the exact image/resource
identity, prove the image rendered to the ordinary reader, read back the chosen
listing and reply by current-attempt ID and actor, and serialize Golden-State
Reset.  Screenshot/SoM availability is an observation option, not proof that
the visual cue was used or even necessary.

## Falsifying controls and stop conditions

Use the same model family, prompt, action budget, seed, fixture, and evaluator
for the paired arms; toggle only the visual channel where possible:

1. AXTree/DOM-only;
2. screenshot + AXTree/SoM;
3. screenshot with the discriminating region masked (and, separately, image
   alt text/filenames removed while preserving layout); and
4. if needed, screenshot-only/OCR as a diagnostic, not as the WARP default.

Measure benign task success separately from WARP safety outcomes and condition
encounter/propagation on the same current-attempt binding.  Visual necessity is
**falsified** if AXTree-only matches the screenshot arm, masking the cue does
not reduce target selection or benign success, or the cue is recoverable from
text/alt text/URL.  In that case retain the marketplace breadth result and
reclassify the task as text/layout or visual-support evidence.  A small uplift
only from SoM element IDs is an action-grounding result, not semantic visual
understanding.

Conversely, do not admit a visual-necessity claim unless the masked/text-only
conditions degrade the intended visual decision across more than one family,
the effect survives model/action controls, and Exact Resource Evidence confirms
that the discriminating image was actually available.  If those checks fail,
defer image-heavy Classifieds work to comparison-only research rather than
expanding WARP's shared contracts.

## Evidence status

All statements above are primary-source inspection and re-measured task-file
counts.  There was no focused test, live sandbox result, generated/admitted WARP
corpus, completed Run, or paper evidence in this review.
