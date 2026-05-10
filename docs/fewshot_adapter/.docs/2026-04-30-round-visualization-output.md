# Round Visualization Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-round visualization images for training inputs, error samples, and all prediction results.

**Architecture:** Create a focused visualization module that reads original images from `image_map`, groups annotations/predictions/errors, draws HBB/polygon overlays with Pillow, and writes images inside each `round_xx/` directory. The native training loop calls this module after JSON outputs are produced, so visualization remains a side-effect of each completed round and does not affect model training.

**Tech Stack:** Python, Pillow, existing `Annotation` / `Prediction` / `ErrorItem` dataclasses, pytest.

---

### Task 1: Visualization Renderer

**Files:**
- Create: `fewshot_adapter/visualization/__init__.py`
- Create: `fewshot_adapter/visualization/round_outputs.py`
- Test: `tests/fewshot_adapter/test_visualization_outputs.py`

- [ ] **Step 1: Write failing tests**

Add tests that create two temporary images, one GT polygon, one prediction, and one error item. Assert `train_inputs`, `errors_vis`, and `predictions_vis` images are generated under a fake `round_00/` directory.

- [ ] **Step 2: Verify tests fail**

Run: `python -m pytest tests\fewshot_adapter\test_visualization_outputs.py -q`

Expected: import failure because `fewshot_adapter.visualization` does not exist.

- [ ] **Step 3: Implement renderer**

Implement `render_round_visualizations(...)` with these responsibilities:

- Create `train_inputs/`, `errors_vis/`, and `predictions_vis/`.
- Render current training annotations grouped by image.
- Render all images in `image_map` for prediction review, including images with no predictions.
- Render only images referenced by `errors`.
- Use green for GT, red for predictions, and yellow text labels.

- [ ] **Step 4: Verify renderer tests pass**

Run: `python -m pytest tests\fewshot_adapter\test_visualization_outputs.py -q`

Expected: all tests pass.

### Task 2: Native Loop Integration

**Files:**
- Modify: `fewshot_adapter/native/trainer.py`
- Test: `tests/fewshot_adapter/test_sam3_native_loop.py`

- [ ] **Step 1: Write failing integration test**

Patch heavy native model calls in the existing loop test and assert each `round_xx/` summary contains paths for `train_inputs`, `errors_vis`, and `predictions_vis`.

- [ ] **Step 2: Verify integration test fails**

Run: `python -m pytest tests\fewshot_adapter\test_sam3_native_loop.py -q`

Expected: failure because summaries do not include visualization paths and directories are not generated.

- [ ] **Step 3: Call renderer from `run_native_fewshot_loop`**

After `predictions.json`, `errors.json`, and `next_train.json` are written, call `render_round_visualizations(...)` and add the three visualization directory paths to the round summary.

- [ ] **Step 4: Verify integration test passes**

Run: `python -m pytest tests\fewshot_adapter\test_sam3_native_loop.py -q`

Expected: all tests pass.

### Task 3: Documentation And Regression

**Files:**
- Modify: `docs/fewshot_gpu_validation_guide.md`
- Modify: `AGENT_CONTEXT.md`

- [ ] **Step 1: Document outputs**

Update the GPU validation guide so users know where to find `train_inputs/`, `errors_vis/`, and `predictions_vis/`.

- [ ] **Step 2: Run full few-shot test suite**

Run: `python -m pytest tests\fewshot_adapter -q`

Expected: all tests pass.

- [ ] **Step 3: Run syntax check**

Run: `python -m compileall -q fewshot_adapter`

Expected: exit code 0.
