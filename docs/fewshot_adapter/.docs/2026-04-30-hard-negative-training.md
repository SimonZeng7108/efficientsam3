# Hard Negative Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow pure-background false positives to enter the next training round as no-object hard negative samples.

**Architecture:** Add a `TrainingSample` layer above `Annotation`, update sampling to produce positive and negative image-level samples, extend SAM3 batch construction to support zero-box targets, and update the native loop / visualization / docs to consume training samples.

**Tech Stack:** Python dataclasses, existing JSON IO, SAM3 `BatchedFindTarget`, pytest.

---

### Task 1: Training Sample Data Model

**Files:**
- Modify: `fewshot_adapter/data/models.py`
- Modify: `fewshot_adapter/data/json_io.py`
- Modify: `fewshot_adapter/data/__init__.py`
- Test: `tests/fewshot_adapter/test_training_samples.py`

- [ ] **Step 1: Write failing tests**

Tests must assert that positive samples and negative samples can be saved and loaded, and that negative samples preserve `image_id`, `label`, and `reason`.

- [ ] **Step 2: Implement `TrainingSample` and JSON IO**

Add `TrainingSample`, `save_training_samples`, and `load_training_samples`.

### Task 2: Sampling And Batch Construction

**Files:**
- Modify: `fewshot_adapter/data/sampling.py`
- Modify: `fewshot_adapter/data/sam3_batch.py`
- Test: `tests/fewshot_adapter/test_trainset.py`
- Test: `tests/fewshot_adapter/test_sam3_batch.py`

- [ ] **Step 1: Write failing tests**

Tests must assert that pure-background selected false positives create negative samples, labeled-image false positives add same-label GT, and negative samples are grouped with zero annotations.

- [ ] **Step 2: Implement sampling helpers**

Add helpers that convert annotations to positive training samples and update selected errors into positive or negative samples.

- [ ] **Step 3: Implement zero-box batch support**

Add `build_sam3_training_batch_from_samples` so negative samples load images and create targets with `num_boxes=0`.

### Task 3: Native Loop And Visualization

**Files:**
- Modify: `fewshot_adapter/native/trainer.py`
- Modify: `fewshot_adapter/visualization/round_outputs.py`
- Test: `tests/fewshot_adapter/test_sam3_native_loop.py`
- Test: `tests/fewshot_adapter/test_visualization_outputs.py`

- [ ] **Step 1: Write failing tests**

Tests must assert the trainer helper adds background false positives as negative samples and visualization writes negative training input images.

- [ ] **Step 2: Wire native loop**

Use training samples for `current_train`, write `train_round_0.json` / `next_train.json` with training-sample JSON, and summarize positive / negative counts.

- [ ] **Step 3: Update visualization**

Draw positive samples with GT and negative samples with `NEGATIVE no-object` text only.

### Task 4: Documentation And Verification

**Files:**
- Modify: `docs/fewshot_gpu_validation_guide.md`
- Modify: `docs/few_shot_adapter_detection_design.md`
- Modify: `AGENT_CONTEXT.md`

- [ ] **Step 1: Document hard negative outputs**

Explain that pure background false positives now enter `next_train.json` as no-object samples and appear in `train_inputs/`.

- [ ] **Step 2: Verify**

Run `python -m pytest tests\fewshot_adapter -q`, `python -m compileall -q fewshot_adapter`, and `git diff --check`.
