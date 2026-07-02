# Native EfficientSAM3 Few-Shot Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 fewshot_adapter 的主训练闭环重构为 EfficientSAM3 原生 decoder + 少量 task prompt / adapter 微调。

**Architecture:** 新主线直接加载 `build_efficientsam3_image_model` 得到完整 `Sam3Image`，冻结大部分权重，只训练任务级 prompt/adapters，并复用 SAM3 原生 matcher/loss 与 `pred_logits/pred_boxes/pred_masks`。旧 proposal/外置 head 流程已从代码中清理，不再作为默认产品验证路径。

**Current package note:** 本计划最初按平铺文件编写，当前代码已重构为分包结构。后续实现以 `fewshot_adapter/data/`、`fewshot_adapter/evaluation/`、`fewshot_adapter/geometry/`、`fewshot_adapter/native/`、`fewshot_adapter/cli/` 为准。

**Tech Stack:** Python 3.12, PyTorch, PIL, EfficientSAM3/SAM3 package, pytest.

---

### Task 1: SAM3 Batch Conversion

**Files:**
- Create: `fewshot_adapter/data/sam3_batch.py`
- Test: `tests/fewshot_adapter/test_sam3_batch.py`

- [ ] **Step 1: Write tests for annotation grouping and normalized boxes**

```python
from fewshot_adapter.data.models import Annotation, HBB
from fewshot_adapter.data.sam3_batch import group_annotations_by_image, hbb_to_cxcywh_norm

def test_group_annotations_by_image_keeps_order():
    anns = [
        Annotation("b.jpg", "b1", "target", "hbb", hbb=HBB(0, 0, 10, 10)),
        Annotation("a.jpg", "a1", "target", "hbb", hbb=HBB(5, 5, 15, 25)),
    ]
    grouped = group_annotations_by_image(anns)
    assert list(grouped) == ["b.jpg", "a.jpg"]
    assert grouped["a.jpg"][0].object_id == "a1"

def test_hbb_to_cxcywh_norm_converts_pixels():
    assert hbb_to_cxcywh_norm(HBB(10, 20, 30, 60), width=100, height=200) == (
        0.2,
        0.2,
        0.2,
        0.2,
    )
```

- [ ] **Step 2: Implement conversion helpers**

Create pure-Python helpers first so tests pass without PyTorch. The torch-dependent batch builder must import torch lazily.

- [ ] **Step 3: Add `build_sam3_training_batch`**

Build images, empty `FindStage`, and `BatchedFindTarget` lazily when torch/SAM3 are installed. Use Chinese comments to explain each tensor shape.

### Task 2: Native Adapter Wrapper

**Files:**
- Create: `fewshot_adapter/native/adapter.py`
- Test: `tests/fewshot_adapter/test_sam3_native_adapter.py`

- [ ] **Step 1: Test trainable name filtering without SAM3**

```python
from fewshot_adapter.native.adapter import NativeAdapterConfig, should_train_parameter

def test_should_train_parameter_matches_prompt_and_dot_prod():
    cfg = NativeAdapterConfig(train_dot_prod_scoring=True)
    assert should_train_parameter("task_prompt_tokens", cfg)
    assert should_train_parameter("model.dot_prod_scoring.prompt_proj.weight", cfg)
    assert not should_train_parameter("model.backbone.vision_backbone.trunk.weight", cfg)
```

- [ ] **Step 2: Implement config and prompt adapter**

Define `NativeAdapterConfig`, `TaskPromptAdapter`, `should_train_parameter`, `freeze_for_fewshot`.

- [ ] **Step 3: Implement wrapper forward path**

`NativeEfficientSAM3FewShotModel` receives a SAM3 model, injects empty language features, injects task prompt tokens through `_encode_prompt(..., visual_prompt_embed=..., encode_text=False)`, then calls native encoder/decoder/segmentation head.

### Task 3: Native Loss

**Files:**
- Create: `fewshot_adapter/native/loss.py`
- Test: `tests/fewshot_adapter/test_sam3_native_loss.py`

- [ ] **Step 1: Test missing torch error**

```python
import pytest
from fewshot_adapter.native.loss import build_native_loss

def test_build_native_loss_requires_torch(monkeypatch):
    monkeypatch.setattr("fewshot_adapter.utils.torch.require_torch", lambda: (_ for _ in ()).throw(ModuleNotFoundError("PyTorch is required")))
    with pytest.raises(ModuleNotFoundError, match="PyTorch is required"):
        build_native_loss()
```

- [ ] **Step 2: Wrap SAM3 loss components**

Build `BinaryHungarianMatcherV2`, `IABCEMdetr`, `Boxes`, optional `Masks`, and `Sam3LossWrapper` with local normalization.

### Task 4: Native Predictor

**Files:**
- Create: `fewshot_adapter/native/predictor.py`
- Test: `tests/fewshot_adapter/test_sam3_native_predictor.py`

- [ ] **Step 1: Test normalized boxes to predictions**

```python
from fewshot_adapter.native.predictor import tensor_box_to_hbb

def test_tensor_box_to_hbb_converts_cxcywh_to_pixels():
    assert tensor_box_to_hbb([0.5, 0.5, 0.25, 0.5], width=200, height=100) == (75.0, 25.0, 125.0, 75.0)
```

- [ ] **Step 2: Implement prediction postprocess**

Convert native outputs to `Prediction` objects with HBB, optional mask path, optional OBB from mask/polygon later.

### Task 5: Native Closed Loop CLI

**Files:**
- Create: `fewshot_adapter/native/trainer.py`
- Create: `fewshot_adapter/train_native_efficientsam3_fewshot.py`
- Modify: `fewshot_adapter/__init__.py`
- Modify: `docs/few_shot_adapter_detection_design.md`
- Test: `tests/fewshot_adapter/test_train_native_efficientsam3_fewshot_cli.py`

- [ ] **Step 1: Test CLI fails clearly without torch**

Run the CLI with small JSON files and assert it exits non-zero with “PyTorch is required”.

- [ ] **Step 2: Implement loop**

Create initial train set, train adapter for N steps, infer all images, build error queue, add selected GT, save adapter checkpoint and round summary.

- [ ] **Step 3: Export public symbols**

Add new native modules to `fewshot_adapter/__init__.py`.

### Task 6: Verification

**Files:**
- Modify only if tests expose gaps.

- [ ] **Step 1: Run focused tests**

Run: `python -m pytest tests/fewshot_adapter/test_sam3_batch.py tests/fewshot_adapter/test_sam3_native_adapter.py tests/fewshot_adapter/test_sam3_native_predictor.py tests/fewshot_adapter/test_train_native_efficientsam3_fewshot_cli.py -q`

- [ ] **Step 2: Run full fewshot tests**

Run: `python -m pytest tests/fewshot_adapter -q`

- [ ] **Step 3: Summarize remaining GPU-only validation**

If local torch/checkpoint is unavailable, state that real training forward/backward remains for the user's GPU verification.
