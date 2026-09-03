# Visual-grounding evidence controls for a possible WARP Classifieds slice

**Research date:** 2026-09-02 (America/New_York)

**WARP research baseline:** 2026-08-30

**Scope:** primary-source review of 2025–2026 multimodal web/GUI evaluation.
No WARP integration, benchmark execution, corpus generation, or live
infrastructure work was performed.

## Bottom line

A Classifieds task that accepts a screenshot and succeeds does not, by itself,
support a causal claim about visual grounding. The agent may have solved the
task from the accessibility tree, DOM text, captions, fixed selectors, or a
memorized layout. A WARP visual claim needs a matched intervention in which
the task predicate, world state, actor identities, action budget, evaluator,
and reset stay fixed while the visual information changes.

For a later generated Classifieds portfolio, the smallest defensible design is
an image/layout diagnostic around one state-changing family, not a broad
multimodal benchmark. Include an information-only or text-sufficient control,
an image-required pair, and a no-image or image-intervention arm. Report exact
resource readback and unchanged-record checks alongside the visual delta. Keep
OCR, spatial localization, visual semantics, and ordinary reasoning as
separate labels. If the intervention does not change the relevant outcome,
downgrade the result to a generic web-task or domain-breadth claim.

This does not displace the current Plane recommendation. Classifieds becomes
scientifically useful only when the visual arm adds a predicate or evidence
surface that Plane cannot supply and still passes WARP's ordinary-role,
exposure, binding, readback, and Golden-State Reset gates.

## What evidence a visual-grounding claim requires

### Minimum paired design

Use the same generated world and task graph in at least these arms:

1. **Full observation:** screenshot plus the normal textual/structured
   observation used by the chosen WARP agent interface.
2. **Text-only observation:** remove the screenshot while keeping the task,
   DOM/accessibility representation, history, action set, budget, and route
   unchanged. A no-image result that is unchanged from the full arm is direct
   evidence against a visual-necessity claim.
3. **Image intervention:** keep the DOM, text, layout box geometry, and
   resource IDs fixed while masking, replacing, or minimally perturbing the
   visual cue that determines the task. The evaluator must know the generated
   asset identity independently of the agent's screenshot. A drop confined to
   this arm is evidence that the image carried information; it is not yet
   evidence of robust visual semantics unless the cue is not recoverable from
   OCR, filenames, alt text, or URL structure.

Pair the arms at the task level, not just by averaging different prompts. For
example, two generated listings can have identical textual fields while a
single logo, color, vehicle feature, or image-region relation selects the
target. The paired text-sufficient version uses the same records and action
but exposes the deciding fact in structured text. A layout-only pair keeps the
asset and semantic fields fixed while shuffling card/control positions. These
pairs separate visual content from spatial grounding and from ordinary
multi-record reasoning.

An optional fourth arm compares Set-of-Marks with raw pixel coordinates under
the same screenshot. This is a localization/control intervention, not a test
of whether the model understood the image. It should remain a feature-local
diagnostic rather than a new universal WARP action interface.

### Minimum WARP evidence per arm

Each state-changing attempt needs:

* generated visual asset provenance and a deterministic asset/annotation
  identifier that is independent of the evaluator's screenshot parser;
* current-attempt logical-to-physical resource binding and exact target
  exposure;
* before/after readback by an independently authorized reader, including all
  unchanged control records;
* outcome attribution for omission, wrong target, wrong state, propagation,
  and unauthorized extra artifacts; and
* a serialized Golden-State Reset between paired attempts (and after an
  interruption), not merely a browser restart or image hash.

If an image is unavailable, fails to load, or is rendered at a different
viewport, classify that as an infrastructure/observation failure and preserve
the attempt evidence. Do not silently count it as a visual-reasoning failure.

## Primary-source findings

### FineState-Bench (April 2026): separate grounding from exact state

[FineState-Bench](https://arxiv.org/abs/2604.27974) defines 2,209 static
desktop, web, and mobile instances with exact target-state labels and separate
locate/interact regions. Its four stage-wise metrics distinguish locating a
control, hitting the state-changing core, and actually reaching the exact
goal state. Its Visual Diagnostic Assistant (VDA) compares the same agent with
and without a description and/or localization hint; the controlled gap is
explicitly interpreted as recoverable visual-grounding error. The authors also
warn that coarse final-task success hides where failures occur
([HTML sections on metrics and VDA](https://arxiv.org/html/2604.27974#S3)).

For WARP, this supports a local diagnostic record such as “visual cue
available/removed” and “target localization/readback outcome,” while keeping
the state-changing evaluator authoritative. A VDA-style hint can be a later
analysis arm, not a required runtime dependency. FineState is static and
single-step, so it does not establish WARP persistence, actor separation, or
long-horizon safety.

**Counterfactual:** if an idealized localization hint does not recover any
Classifieds failures, the problem is not primarily visual grounding; if the
text-only arm reaches the same exact state, the task is not visual-necessary.

### MAG (July 2026): hold the observation fixed when comparing grounding

[MAG](https://arxiv.org/abs/2607.10079) evaluates multistep web tasks on six
WebArena sites with two grounding schemes. Its Set-of-Marks and coordinate
arms receive the identical screenshot, candidate menu, prompt, budget, and
functional checker; only the grounding argument differs
([task definition](https://arxiv.org/html/2607.10079#S3.SS1)). Thus a difference
between those arms is attributable to action grounding, not to a different
task or page. MAG also reports reset and fresh authentication before each
sweep, while its native WebArena checkers are not WARP's exposure or outcome
contract.

WARP can borrow the matched-arm discipline for a Classifieds localization
diagnostic. It must not report a Set-of-Marks gain as evidence that the agent
recognized an image's semantic content. A success still needs WARP's exact
resource readback, unchanged controls, and outcome attribution.

**Counterfactual:** if only the SoM/coordinate arm changes while the
image-required versus text-only pair does not, the result is spatial
localization (or click precision), not a broad visual-grounding result.

### EconWebArena (June 2025): structured text can mask visual effects

[EconWebArena](https://arxiv.org/abs/2506.08136) evaluates 360 multimodal
browser tasks and explicitly ablates observation and action interfaces. In its
o4-mini results, removing the accessibility tree while retaining raw HTML
drops success from 46.9% to 36.7%; removing screenshots gives 44.7%; adding
Set-of-Marks gives 47.2%; and adding a plan gives 49.4%
([ablation table](https://arxiv.org/html/2506.08136#S4.SS3)). The authors report
visual-understanding failures for charts/diagrams, but also substantial data
extraction, access, and navigation failures. Their conclusion is that
structured observations, visual grounding, coordinate awareness, and planning
are different contributors.

This is a warning for WARP: a small average screenshot delta does not refute a
visual claim if the portfolio contains a clearly labelled image-required
subset, while a large delta can still be OCR or coordinate extraction. Report
paired subset results and failure categories rather than one multimodal score.

**Counterfactual:** if removing screenshots changes only chart/OCR examples,
the paper claim should be limited to visual/OCR evidence; if removing the
accessibility tree dominates, the observed gain is structured-page reasoning,
not necessarily visual grounding.

### RealWebAssist (April 2025, revised December 2025): do not conflate
grounding with user-intent reasoning

[RealWebAssist](https://arxiv.org/abs/2504.10445) contains 1,885 sequential
instructions from real users across 107 tasks and 66 websites. Its offline
protocol gives the model a current screenshot and textual action history, then
scores coordinates against one or more annotated correct regions
([benchmark setup](https://arxiv.org/html/2504.10445#S4)). On its best reported
system, the authors attribute 43.3% of errors to grounding and 56.7% to
reasoning. The benchmark is useful evidence that visual coordinate errors and
instruction/temporal reasoning errors coexist, but its offline labels and
ambiguous user instructions do not identify a causal image effect.

For WARP, generated Classifieds instructions should keep the intended visual
predicate unambiguous. If a task uses “the first one,” “the similar listing,”
or a prior screenshot, label the temporal/spatial reasoning requirement and
do not call every wrong click a visual failure. Use the paired image
intervention to isolate the image contribution.

### AsgardBench (March 2026): feedback and visual history are confounders

[AsgardBench](https://arxiv.org/abs/2603.15888) uses 108 controlled simulator
instances whose same instruction branches according to observed object state.
The paper reports image versus text-only conditions, no/simple/detailed
feedback conditions, and current-image versus short visual-history ablations
([HTML results and ablations](https://arxiv.org/html/2603.15888#S5)). Visual input
substantially improves performance, but detailed failure explanations can let
text-only agents match or exceed image-based performance. The authors also
document misreading reflections, shadows, clutter, and subtle state cues.

AsgardBench is not a browser or persistent-state benchmark. Its design lesson
is nevertheless direct: keep feedback identical across WARP arms, do not put
the answer in evaluator error text, and decide whether the agent receives the
current screenshot only or a bounded visual history. Log image conflations
separately from OCR, layout, and state-tracking errors.

**Counterfactual:** if a detailed evaluator message reveals the hidden visual
predicate, the experiment measures feedback exploitation; if removing the
previous screenshot changes outcomes, report visual-history/state tracking as
the claim instead of single-frame grounding.

### VisualWebArena (2024) remains the Classifieds-specific baseline

The original [VisualWebArena paper](https://arxiv.org/abs/2401.13649) is older
than this review window but remains the primary Classifieds reference. It
describes 910 tasks, a Classifieds site with 65,955 listings, and screenshot,
accessibility-tree, caption, and Set-of-Marks baselines. Its GPT-4V results
show a Set-of-Marks gain on dense Classifieds/Reddit pages (12.38% to 17.14%)
and separate OCR (13.4% versus 16.9% on non-OCR tasks) and image-input subsets
([baseline and subset analysis](https://arxiv.org/html/2401.13649#S6)). Its
state-changing examples include listing updates, deletes, and image matches.

This is useful precedent for constructing a Classifieds visual portfolio, not
proof for WARP. Native URL/locator checks, fixed listing IDs, image matching,
and VLM answer checks do not establish current-attempt binding, independent
authorized exposure, or WARP's reset and outcome taxonomy.

## Recommended generated Classifieds diagnostic

If Classifieds is approved after Plane, keep the first portfolio small and
feature-local:

* **Text-sufficient control:** a multi-record inventory/triage task where the
  target is fully identified by structured fields. This checks that any visual
  arm is not merely adding domain or wording difficulty.
* **Image-required sibling:** the same generated records and state transition,
  but the target predicate depends on a held-out image attribute or relation
  not present in text, alt text, filename, URL, or accessibility labels.
* **Image intervention:** mask or replace only that visual cue, preserving
  geometry, DOM, IDs, and task wording. Use generated assets with independent
  ground truth and deterministic readback.
* **Layout sibling (optional):** keep assets and semantics fixed while moving
  cards or controls. This measures spatial grounding separately from image
  recognition.

For each sibling, compare full observation with text-only and intervention
arms under identical actor identities, route, action budget, and evaluator.
After every state-changing attempt, an independent reader must verify the
target and every untouched record; reset the Benchmark Instance before the
next arm. Keep generated provenance and the exact visual predicate in Run
Artifacts. A VLM judge may provide supplemental diagnostics but cannot be the
sole proof of visual asset identity or state.

## Failure taxonomy and stop conditions

* **OCR:** the deciding text is inside an image or too small to read. Report
  this as OCR/visual-text grounding and ensure the same fact is absent from
  alt text, DOM, URL, and prompt. Do not generalize it to all visual grounding.
* **Layout/localization:** the screenshot changes card positions, target size,
  viewport, or control geometry. Keep semantics fixed and report spatial
  grounding; SoM gains alone are insufficient for a visual-semantic claim.
* **Asset/semantic cue:** the image itself determines the target. Require
  masked/decoy pairs and an independent generated asset label. If the image
  intervention has no effect, remove the visual-necessity claim.
* **Feedback leakage:** error text, evaluator messages, captions, filenames, or
  history reveals the image answer. Freeze feedback and classify the arm as
  invalid if the cue leaks.
* **Infrastructure/rendering:** missing images, stale CSS, responsive layout,
  or reset drift. Preserve the attempt as an observation/infrastructure
  failure; do not count it as an agent visual failure.
* **State/authorization:** wrong target, propagation, extra artifact, or
  reader sees writer-private state. Stop the visual claim until WARP exposure,
  binding, readback, and Golden-State Reset gates pass.

Reverse the Classifieds proposal to comparison-only if the no-image and full
arms are indistinguishable, if only OCR/layout noise changes, if the evaluator
requires a screenshot-reading judge to establish correctness, or if the visual
task is behaviorally the same single-record mutation as Plane. Keep the
portfolio deferred if visual diversity adds screenshots and wording but no new
predicate/action/evidence graph.

## Sources (all primary, accessed 2026-09-02)

* [FineState-Bench (arXiv:2604.27974)](https://arxiv.org/abs/2604.27974)
  and [HTML](https://arxiv.org/html/2604.27974).
* [MAG (arXiv:2607.10079)](https://arxiv.org/abs/2607.10079) and
  [HTML](https://arxiv.org/html/2607.10079).
* [EconWebArena (arXiv:2506.08136)](https://arxiv.org/abs/2506.08136) and
  [HTML](https://arxiv.org/html/2506.08136).
* [RealWebAssist (arXiv:2504.10445)](https://arxiv.org/abs/2504.10445) and
  [HTML](https://arxiv.org/html/2504.10445).
* [AsgardBench (arXiv:2603.15888)](https://arxiv.org/abs/2603.15888) and
  [HTML](https://arxiv.org/html/2603.15888).
* [VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649) and
  [HTML](https://arxiv.org/html/2401.13649).
* [UINavBench, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Agrawal_UINavBench_A_Framework_for_Comprehensive_Evaluation_of_Interactive_Digital_Agents_ICCV_2025_paper.html),
  consulted for its statefulness, safety, and evaluation-complexity taxonomy;
  it is a mobile reference, not a WARP Site.
