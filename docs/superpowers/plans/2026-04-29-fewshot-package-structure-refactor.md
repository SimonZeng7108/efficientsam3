# Fewshot Package Structure Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `fewshot_adapter` 按 data / geometry / evaluation / native / cli / utils 分包，并为核心能力提供类化入口。

**Architecture:** 通过移动现有模块保持算法不变，在每个领域模块内新增轻量类封装，同时保留旧命令入口作为薄 wrapper。测试同步迁移到新导入路径，并保留 CLI 兼容测试。

**Tech Stack:** Python 3.12, pytest, EfficientSAM3/SAM3, PyTorch lazy import.

---

### Task 1: Data Package

**Files:**
- Move: `annotations.py` -> `data/models.py`
- Move: `datatrain.py` -> `data/datatrain.py`
- Move: `io.py` -> `data/json_io.py`
- Move: `initial_train_set.py` and `trainset.py` -> `data/sampling.py`
- Move: `sam3_batch.py` -> `data/sam3_batch.py`

- [ ] Add `DataTrainDataset`, `AnnotationJsonIO`, `InitialTrainSelector`, `TrainSetUpdater`, `Sam3BatchBuilder`.
- [ ] Update tests to import from `fewshot_adapter.data.*`.

### Task 2: Geometry and Evaluation Packages

**Files:**
- Move: `geometry.py` -> `geometry/ops.py`
- Move: `matching.py` -> `evaluation/matching.py`

- [ ] Add `GeometryOps`, `DetectionMatcher`, `ErrorSelector`.
- [ ] Update all imports.

### Task 3: Native Package

**Files:**
- Move: `sam3_native_adapter.py` -> `native/adapter.py`
- Move: `sam3_native_loss.py` -> `native/loss.py`
- Move: `sam3_native_predictor.py` -> `native/predictor.py`
- Move: `sam3_native_loop.py` -> `native/trainer.py`
- Move: `torch_utils.py` -> `utils/torch.py`

- [ ] Add `NativeLossFactory`, `NativePredictor`, `NativeFewShotTrainer`.
- [ ] Update lazy torch imports and native trainer imports.

### Task 4: CLI Package and Compatibility

**Files:**
- Move implementation CLI into `cli/convert_datatrain.py` and `cli/train_native.py`.
- Keep root `convert_datatrain.py` and `train_native_efficientsam3_fewshot.py` as wrappers.

- [ ] Ensure old `python -m` commands still work.

### Task 5: Verification

- [ ] Run `python -m pytest tests/fewshot_adapter -q`.
- [ ] Run `python -m compileall -q fewshot_adapter`.
- [ ] Check no missing local imports.
