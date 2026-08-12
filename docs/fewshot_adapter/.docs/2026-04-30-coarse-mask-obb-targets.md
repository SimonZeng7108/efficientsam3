# Coarse Mask OBB Targets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 DataTrain 的 HBB/OBB/Polygon 标注可以生成粗 mask target，训练 SAM3 原生 mask loss，并在推理阶段优先由预测 mask 拟合真实旋转 OBB。

**Architecture:** 数据层新增独立 mask 栅格化工具；SAM3 batch 在 `LOSS.USE_MASKS=true` 时填充 `BatchedFindTarget.segments/is_valid_segment`；loss 层接入官方 `Masks` loss；预测层在存在 `pred_masks` 时把二值 mask 转 polygon/OBB，否则回退到 angle=0 HBB 兼容字段。

**Tech Stack:** Python、Pillow、NumPy、PyTorch/SAM3 原生 loss、pytest。

---

### Task 1: 数据层粗 Mask Target

**Files:**
- Create: `fewshot_adapter/data/masks.py`
- Modify: `fewshot_adapter/data/sam3_batch.py`
- Modify: `fewshot_adapter/data/__init__.py`
- Test: `tests/fewshot_adapter/test_sam3_batch.py`

- [ ] **Step 1: Write failing tests**

新增测试：`annotation_to_mask()` 能把 polygon 栅格化成指定分辨率的 bool mask；`_build_find_target(..., include_masks=True)` 会生成与 packed boxes 对齐的 `segments` 和 `is_valid_segment`。

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests\fewshot_adapter\test_sam3_batch.py -q`

Expected: FAIL because `fewshot_adapter.data.masks` and `include_masks` do not exist yet.

- [ ] **Step 3: Implement minimal code**

用 Pillow `ImageDraw.polygon` 按原图尺寸到训练分辨率比例缩放点坐标，返回 `np.ndarray(dtype=bool)`；SAM3 batch 在 `include_masks=True` 时用 `torch.from_numpy(np.stack(...))` 构造 `(N,H,W)` mask。

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests\fewshot_adapter\test_sam3_batch.py -q`

Expected: PASS。

### Task 2: 原生 Mask Loss 接入

**Files:**
- Modify: `fewshot_adapter/native/loss.py`
- Modify: `fewshot_adapter/native/trainer.py`
- Test: `tests/fewshot_adapter/test_sam3_native_loss.py`
- Test: `tests/fewshot_adapter/test_sam3_native_trainer.py`

- [ ] **Step 1: Write failing tests**

新增测试：`build_native_loss(NativeLossConfig(use_masks=True))` 会把官方 `Masks` loss 加入 `loss_fns_find`；`LOSS.USE_MASKS=true` 且 `MODEL.ENABLE_SEGMENTATION=false` 时训练入口应给出清晰错误。

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests\fewshot_adapter\test_sam3_native_loss.py tests\fewshot_adapter\test_sam3_native_trainer.py -q`

Expected: FAIL because mask loss is currently rejected and trainer has no segmentation guard.

- [ ] **Step 3: Implement minimal code**

移除 `USE_MASKS` 拒绝逻辑；导入官方 `Masks`；训练轮次根据 loss 配置给 batch builder 传 `include_masks`；闭环启动时校验 mask loss 需要 segmentation head。

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests\fewshot_adapter\test_sam3_native_loss.py tests\fewshot_adapter\test_sam3_native_trainer.py -q`

Expected: PASS。

### Task 3: 推理 Mask 转 OBB

**Files:**
- Modify: `fewshot_adapter/native/predictor.py`
- Test: `tests/fewshot_adapter/test_sam3_native_predictor.py`

- [ ] **Step 1: Write failing tests**

新增测试：当输出含 `pred_masks` 时，预测结果的 OBB 来自 mask 的最小外接旋转矩形，而不是 HBB 的 angle=0 占位字段。

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests\fewshot_adapter\test_sam3_native_predictor.py -q`

Expected: FAIL because predictor currently ignores `pred_masks`。

- [ ] **Step 3: Implement minimal code**

把 mask logits/binary mask 归一到原图尺寸，提取所有前景像素中心点，复用 `polygon_to_obb()` 拟合旋转框；mask 为空时回退 HBB OBB。

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests\fewshot_adapter\test_sam3_native_predictor.py -q`

Expected: PASS。

### Task 4: 配置与中文文档同步

**Files:**
- Modify: `fewshot_adapter/configs/efficient_sam3_efficientvit_s_fewshot.yaml`
- Modify: `docs/fewshot_gpu_validation_guide.md`
- Modify: `docs/few_shot_adapter_detection_design.md`
- Modify: `docs/superpowers/specs/2026-04-29-native-efficientsam3-fewshot-design.md`
- Modify: `docs/superpowers/specs/2026-04-30-fewshot-yaml-config-design.md`
- Modify: `AGENT_CONTEXT.md`

- [ ] **Step 1: Update docs**

说明 `LOSS.USE_MASKS=true` 时必须同时打开 `MODEL.ENABLE_SEGMENTATION=true`；粗 mask 来自原始 HBB/OBB/Polygon；真正 OBB 首版来自预测 mask 后处理。

- [ ] **Step 2: Verify docs do not contain stale warnings**

Run: `Get-ChildItem -Path docs,fewshot_adapter -Recurse -File | Select-String -Pattern '尚未生成 mask target','USE_MASKS.*必须保持 false','angle=0 的 OBB 基线'`

Expected: 只保留说明“无 mask 输出时会回退 angle=0”的语句，不再说 mask target 未实现。

### Task 5: Final Verification

**Files:**
- All touched files

- [ ] **Step 1: Run targeted tests**

Run: `python -m pytest tests\fewshot_adapter\test_sam3_batch.py tests\fewshot_adapter\test_sam3_native_loss.py tests\fewshot_adapter\test_sam3_native_predictor.py -q`

- [ ] **Step 2: Run package tests**

Run: `python -m pytest tests\fewshot_adapter -q`

- [ ] **Step 3: Run compile and diff checks**

Run: `python -m compileall -q fewshot_adapter`

Run: `git diff --check`

- [ ] **Step 4: Commit and push**

Commit message: `Add coarse mask targets for native training`
