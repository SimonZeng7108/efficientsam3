# Stage 3 Brainstorm and Development Notes

This document is a critical planning note for EfficientSAM3 Stage 3.
It is based on the current repository state, the existing Stage 1 pipeline,
the EfficientSAM3 fork under `sam3/`, and the updated upstream SAM3.1 code in
`original_sam3/sam3`.

The goal here is not to describe an ideal paper plan. The goal is to identify
what is most likely to work in this codebase with the current models and
weights.

## Executive Summary

My current view is:

1. The student text encoder already works well because it is a relatively clean
   interface replacement. It only needs to output a compatible token memory and
   attention mask for the rest of SAM3.
2. The student image encoder underperforms because Stage 1 only teaches it to
   mimic raw teacher trunk features, while the deployed model depends on a much
   larger prompt-conditioned stack: FPN neck, transformer, geometry encoder,
   scoring heads, and segmentation decoder.
3. A full end-to-end Stage 3 fine-tune on noisy pseudo-labeled SA-1B is not the
   first thing I would do. It is too easy to destroy the parts that already
   work.
4. The most realistic path is a staged recovery plan:
   - first repair the visual interface,
   - then align prompt-conditioned behavior,
   - then expand with a carefully filtered SA-1B data engine.
5. Before serious Stage 3 experiments, the repository needs cleanup: the
   current Stage 3 configs reference a `stage3` Python package that is not
   actually present as runnable source, and `pyproject.toml` does not package
   `stage3` at all.

## What The Repo Already Tells Us

### 1. Stage 1 image distillation is feature regression only

The current image teacher path in Stage 1 extracts the teacher output from:

- `backbone.vision_backbone.trunk(x)[-1]`

and the training loss is just masked MSE plus masked cosine similarity on that
feature map.

This means the image student is *not* trained to preserve:

- prompt-conditioned mask behavior,
- geometry prompt alignment,
- compatibility with the frozen FPN neck,
- compatibility with the transformer decoder,
- compatibility with the scoring and segmentation heads.

So even if the student matches the teacher trunk feature map well enough under
an embedding loss, that does not imply it is a good drop-in replacement for the
full SAM3 image pipeline.

### 2. Stage 1 text distillation is a cleaner replacement problem

The text student replaces the language backbone more directly:

- tokenize text,
- run a MobileCLIP-style transformer,
- project to SAM3 `d_model=256`,
- return token memory and attention mask.

This is a much cleaner contract than the visual side. The rest of the model can
often tolerate modest text embedding drift as long as the token memory space is
roughly aligned.

This matches the empirical observation in this repo: the student text encoder
merged into the original SAM3 stack works much better than the student image
encoder merged into the original SAM3 stack.

### 3. The image student is also being asked to do a harder job

The visual side has several extra disadvantages:

- The image student configs appear to train from scratch in Stage 1. They do
  not point to a pretrained checkpoint by default.
- The student image adapter adds its own projection head inside the visual
  trunk, including BatchNorm. That creates another learned interface whose
  running statistics and feature distribution must line up with the frozen SAM3
  neck and heads.
- The visual backbone affects almost everything downstream: grounding,
  segmentation, geometry interaction, and video memory.

So the visual replacement is both structurally more sensitive and trained under
weaker supervision.

### 4. The repo already contains clues that vision-only Stage 3 has been hard

The Stage 3 Hydra configs under `sam3/sam3/train/configs/stage3/mixed` already
encode several useful observations:

- there are dedicated "vision only" experiments,
- there is a separate experiment that unfreezes the FPN neck,
- there is a "very low LR" experiment whose comment explicitly says the
  pre-finetune baseline outperforms the existing Stage 3 runs,
- full segmentation fine-tuning is extremely memory heavy.

This is consistent with the current diagnosis: the issue is not just "train
longer". The issue is that the current image student enters the full SAM3 stack
through a brittle frozen interface.

## Why The Student Text Encoder Works Better Than The Student Image Encoder

This is the most important question.

### Reason 1: text is a self-contained interface, image is not

The text encoder only needs to produce a usable token sequence in the right
space. The visual encoder has to supply the entire scene representation used by
every downstream module.

If the text encoder is slightly off, the model may still recover because the
rest of the SAM3 stack is strong and the text branch mainly conditions the
decoder.

If the image encoder is off, everything is off:

- object localization,
- mask quality,
- prompt grounding,
- presence scores,
- geometry fusion,
- memory features for later video extensions.

### Reason 2: Stage 1 image loss does not train promptability

Stage 1 image training optimizes a raw feature imitation objective, not the
actual downstream prompt-conditioned task.

That means the student can become good at matching a frozen teacher feature map
in a global sense while still being bad at preserving the precise local
structure that SAM3 uses for:

- points,
- boxes,
- grounding queries,
- mask boundaries,
- presence prediction.

### Reason 3: the frozen teacher neck is likely a bottleneck

The current image weight conversion mainly replaces:

- `detector.backbone.vision_backbone.trunk.*`

while keeping the downstream neck and decoder stack from the teacher.

That is convenient, but it means the student image trunk must feed a frozen
projection stack that was tuned for teacher-ViT feature statistics.

This is exactly why the existing `seg_vision_fpn` config is one of the more
sensible ideas in the repo. If the neck is part of the mismatch, freezing it is
the wrong bias.

### Reason 4: the visual student probably needs prompt-conditioned distillation

The text student can work without prompt-conditioned supervision because the
interface is simple.

The image student probably cannot. To make it truly deployable, Stage 3 needs
to train it against downstream prompt-conditioned targets, not just against raw
backbone embeddings.

### Reason 5: BatchNorm in the image adapter may be a secondary stability issue

The visual student adapter includes BatchNorm in the projection head. That is
not necessarily wrong, but it makes the merged checkpoint more sensitive to:

- small effective batch size,
- distributed BN behavior,
- running-stat mismatch at evaluation time.

I do not think this is the main problem, but it is a plausible secondary source
of instability.

## Critical View On A SA-1B Data Engine

### Short answer

Yes, a SA-1B data engine can help Stage 3, but only if it is used carefully.
If you try to pseudo-label the whole of SA-1B and treat it like clean text-mask
supervision, I do not think that will work well.

### Why it could help

SA-1B gives you enormous mask diversity and strong geometry supervision. That is
exactly what the weak image student needs.

If the data engine can attach high-quality text prompts to a subset of SA-1B,
you get large-scale prompt-conditioned training data that is much closer to your
deployment setting than plain feature imitation.

### Why it can easily fail

SA-1B does not come with reliable language annotations. So the data engine has
to create them. That creates several risks:

- caption noise,
- ambiguous noun phrases,
- masks that correspond to parts, stuff, or visually unclear regions,
- prompts that do not reproduce the mask when re-run through SAM3,
- long-tail phrases that are linguistically valid but operationally useless.

If this noisy pseudo-text becomes the main Stage 3 signal, it can damage the
model more than it helps.

### What I would trust

I would trust a SA-1B data engine mainly for two things:

1. Geometry-first training without text.
2. High-confidence text pseudo-labels on a filtered subset, not the full data.

### What I would not trust

I would not trust:

- full-data end-to-end fine-tuning on raw pseudo-text SA-1B,
- using pseudo-text SA-1B as the dominant Stage 3 dataset at the start,
- using caption-style text instead of short, referable noun phrases.

## What I Think Will Actually Work

## Phase 0: Make The Codebase Stage 3 Ready

Before training, the repo needs basic cleanup.

### A. Turn Stage 3 into a real package

Right now the configs refer to objects such as:

- `stage3.model.build_stage3_model`
- `stage3.data.mixed_text_mask_dataset.Stage3MixedTextMaskDataset`
- `stage3.transforms.geometry_sampling.AddGeometricQueries`

but the top-level `stage3/` directory is effectively empty and `pyproject.toml`
only packages:

- `stage1*`
- `sam3*`

So the current Stage 3 configuration surface is more complete than the actual
Stage 3 source tree.

I would fix this first, either by:

1. making `stage3/` a real Python package and adding it to packaging, or
2. moving Stage 3 code under the installed `sam3` package.

### B. Keep the upstream SAM3.1 sync clean

Use `original_sam3/sam3` as the upstream reference and keep EfficientSAM3
changes as isolated as possible.

The current `sam3/sam3/model_builder.py` mixes:

- upstream SAM3 runtime logic,
- EfficientSAM3 image builder logic,
- student text builder logic,
- SAM3.1 multiplex logic.

That works for experimentation, but it will get harder to re-sync with upstream
if Stage 3 adds even more custom paths in the same file.

My preferred direction is:

- keep upstream-like builders thin,
- move EfficientSAM3-specific stage3 wrappers into dedicated modules,
- keep checkpoint loading and freezing policy centralized in one place.

### C. Fix interface inconsistencies before video Stage 3

The efficient video builder path looks suspicious today: the efficient visual
backbone helper returns a `Sam3TriViTDetNeck`, while the generic video builder
still wraps it with the non-tri VL backbone path. That should be treated as a
repo-readiness issue before any serious video or memory-stage experiments.

Even if Stage 3 starts with image-only fine-tuning, the codebase should not
quietly carry a broken efficient video path into later stages.

## Phase 1: Repair The Visual Interface Before Full Stage 3

This is the highest-confidence step.

### Recommendation

Start with a merged EfficientSAM3 checkpoint and train:

- student image trunk,
- student image projection head,
- FPN neck,

while freezing at least:

- text encoder,
- geometry encoder,
- most of the transformer,
- dot-product scoring,
- segmentation decoder at first.

This is basically the direction hinted by the existing `seg_vision_fpn` config,
and I think it is the correct first repair move.

### Why this is likely to work

It addresses the most obvious mismatch:

- Stage 1 trained the trunk to imitate teacher features,
- inference expects those features to flow through a neck tuned for teacher
  statistics,
- so unfreezing the neck is the minimum structural adaptation needed.

### Data for this phase

Do not start with pseudo-labeled SA-1B text.

Start with trusted data that already has good masks and reasonably good labels:

- COCO
- LVIS
- RefCOCO
- ODINW
- RF100
- existing in-repo text annotation mixes

If memory is tight, use a short detection-only warm-up first, but do not stop
there. The endpoint must be segmentation-aware.

### Critical detail

Use low learning rates for the vision path. The current configs already suggest
that the original higher LR was destructive.

## Phase 2: Geometry-First SA-1B Fine-Tuning Before Text Data Engine Expansion

This is the step I think is most underappreciated.

You do not need a text data engine to use SA-1B productively.
SA-1B already gives you masks, which means you can create:

- points,
- boxes,
- mask prompts,
- prompt perturbations,
- positive/negative click sequences.

That is exactly the supervision needed to recover a weak image student.

### Recommendation

Run a geometry-only or geometry-dominant fine-tuning stage on SA-1B before
large-scale pseudo-text training.

Trainable modules should still be conservative:

- image trunk,
- image adapter head,
- FPN neck,
- optionally a small part of the prompt-conditioning stack later.

### Why this matters

If the visual student still cannot support point/box/mask prompting reliably,
then adding noisy text labels on top will not solve the root problem.

Geometry-first tuning is a much cleaner way to restore promptability.

## Phase 3: Add Prompt-Conditioned Distillation, Not Just Supervised Loss

If Stage 3 only uses standard detection/segmentation losses, it may still drift
away from SAM3-style behavior.

I think the visual student needs a stronger teacher-guided Stage 3 objective.

### Recommended teacher signals

Use the original SAM3.1 model as an online or offline teacher for at least one
of these:

- FPN feature distillation,
- decoder query distillation,
- mask-logit distillation,
- box/presence distillation,
- text-conditioned mask consistency.

This is much more faithful to the real deployment objective than Stage 1 raw
trunk regression alone.

### Why this is important

The whole problem with the image student is that raw feature similarity was not
enough. Stage 3 should therefore distill behavior at the level where the failure
actually shows up.

## Phase 4: Add A SA-1B Text Data Engine Cautiously

Only after the image path has recovered should SA-1B text pseudo-labeling
become a major component.

### Best-use version of the data engine

The data engine should generate short, operational prompts, not general
captions.

Examples of good targets:

- `dog`
- `red car`
- `person`
- `tree branch`

Examples of bad targets:

- full sentence descriptions,
- vague scene summaries,
- prompts that do not uniquely map back to the mask.

### Minimum filtering I would require

For each SA-1B mask, keep a pseudo-label only if:

1. a phrase is generated for the mask crop,
2. SAM3.1 can reproduce the mask from that phrase with high overlap,
3. the phrase is not too generic or too ambiguous relative to nearby masks,
4. the sample passes confidence and stability thresholds,
5. the text length and tokenization stay inside the intended context budget.

### Dataset policy

Do not dump the whole pseudo-labeled SA-1B set into training at full weight.

Instead:

1. start with a high-confidence subset,
2. mix it with trusted labeled datasets,
3. upweight or expand it only if held-out metrics improve.

My default would be to start with a relatively small but clean subset and use it
as a minority fraction of the Stage 3 mixture.

## Should The Student Text Encoder Be Trained In Stage 3?

Yes, but not at the beginning.

### My recommendation

At the start of Stage 3:

- freeze the student text encoder,
- focus on repairing the image path.

The user observation already says the student text encoder works nearly as well
as the teacher when merged into the SAM3 stack. That means it is not the
highest-risk component.

Once the image path stabilizes, then unfreeze the text encoder with a lower LR
for joint polishing.

### Diagnostic variant worth running

For clarity, compare these two checkpoints early:

1. student image + teacher text + teacher rest
2. student image + student text + teacher rest

If those are close, freeze student text and stop worrying about it early.
If teacher text is materially better, use teacher text temporarily during the
visual repair phase and swap the student text back in later.

## What I Would Not Do First

I would not start Stage 3 with any of the following:

- immediate full-model end-to-end fine-tuning on noisy pseudo-labeled SA-1B,
- large-LR vision-only fine-tuning with the neck frozen,
- treating Stage 1 feature regression as sufficient proof that the image student
  is ready for end-to-end deployment,
- mixing video-stage objectives into Stage 3 before the image path is repaired,
- relying on detection-only training as the main Stage 3 solution.

## Recommended Experiment Order

If the goal is to maximize the chance of a real win, I would run experiments in
this order:

1. Confirm the baseline gap with a clean evaluation matrix:
   - teacher image + teacher text
   - teacher image + student text
   - student image + teacher text
   - student image + student text
2. Run visual repair with low LR:
   - train image trunk + adapter head + FPN neck
   - freeze text and the rest
3. Run geometry-first tuning on SA-1B using synthetic point/box/mask prompts.
4. Add prompt-conditioned teacher distillation on top of the repair stage.
5. Add a filtered SA-1B text data engine subset into the mixed training pool.
6. Only then try conservative joint fine-tuning of both student encoders.

## Concrete Criteria For Success

I would promote a Stage 3 recipe only if it passes all of these:

1. It beats the merged pre-finetune checkpoint, not just the previous Stage 3
   run.
2. It improves the student-image checkpoint without regressing the already-good
   student-text behavior.
3. It is stable across at least a few checkpoints and not just a single lucky
   epoch.
4. It improves prompt-conditioned mask quality, not only detection metrics.
5. It remains compatible with the intended efficient runtime path.

## Bottom Line

I do think Stage 3 can work, but I do not think the first winning version is:

- "just fine-tune the whole efficient model on pseudo-labeled SA-1B".

The most likely winning version is:

- repair the visual interface first,
- use geometry-first SA-1B supervision before large-scale text pseudo-labeling,
- unfreeze the FPN neck early,
- use prompt-conditioned teacher distillation,
- bring the data engine in gradually and only with strong filtering.

If there is one central conclusion from the current repo state, it is this:

- the text student is already close enough,
- the image student is not failing because it is small,
- it is failing because the current training objective and integration contract
  are not strong enough for a full SAM3 drop-in replacement.