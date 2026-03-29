# Stage 3 Experiment Log — ES-TV-M + MobileCLIP-S1

This document records the full development and validation of the Stage 3
fine-tuning pipeline for EfficientSAM3. It is intended as a handoff document
so that any developer (human or LLM) can pick up where this left off.

## Goal

Fine-tune Stage 1 distilled student encoders (image + text) jointly inside the
full SAM3 image model. The decoder, geometry encoder, scoring head, and all
video/tracker modules remain frozen.

## Test Configuration

| Component | Value |
|-----------|-------|
| Image student | **ES-TV-M** — TinyViT-11M (`tinyvit_11m`) |
| Text student | **MobileCLIP-S1** (`mobileclip_s1`, context_length=16, pos_embed_table_size=16) |
| Image student checkpoint | `output/stage1_image_2p/es_tv_m/ckpt_epoch_49.pth` |
| Text student checkpoint | `output/stage1_text/mobileclip_s1_5dataset_ctx16_fixed/ckpt_epoch_79.pth` |
| SAM3 teacher checkpoint | `sam3_checkpoints/sam3.pt` |
| Merged checkpoint | `output/efficient_sam3_stage3_es_tv_m_mobileclip_s1_ctx16.pth` (551 MB) |

---

## What Was Done

### 1. Merged Stage 1 Student Encoders with SAM3

**Script:** `stage1/convert_both_encoders_weights_stage1.py`

```bash
python stage1/convert_both_encoders_weights_stage1.py \
  --image-student-ckpt output/stage1_image_2p/es_tv_m/ckpt_epoch_49.pth \
  --text-student-ckpt  output/stage1_text/mobileclip_s1_5dataset_ctx16_fixed/ckpt_epoch_79.pth \
  --sam3-ckpt          sam3_checkpoints/sam3.pt \
  --output             output/efficient_sam3_stage3_es_tv_m_mobileclip_s1_ctx16.pth \
  --image-model-name   tinyvit_11m \
  --text-model-name    mobileclip_s1 \
  --text-context-length        16 \
  --text-pos-embed-table-size  16
```

**Result:** `Teacher params: Replaced=715, Skipped=0, Appended=750`

The merged checkpoint has **1201 keys** with `detector.*` and `tracker.*`
prefixes, comprising:

| Prefix | Keys | Source |
|--------|------|--------|
| `detector.backbone.vision_backbone.trunk.model.*` | ~300 | Student image encoder (TinyViT-11M) |
| `detector.backbone.language_backbone.*` | ~151 | Student text encoder (MobileCLIP-S1) |
| `detector.backbone.vision_backbone.convs.*` | ~44 | FPN neck (from SAM3, unchanged) |
| `detector.geometry_encoder.*` | 76 | Geometry encoder (from SAM3, frozen) |
| `detector.transformer.*` | 283 | Decoder (from SAM3, frozen) |
| `detector.dot_prod_scoring.*` | 10 | Scoring head (from SAM3, frozen) |
| `detector.segmentation_head.*` | 28 | Segmentation head (from SAM3, frozen) |
| `tracker.*` | 309 | Video/memory bank (from SAM3, frozen) |

### 2. GPU Sanity Check

**Script:** `stage3/sanity_check_gpu.py` via `scripts/sanity_check_gpu.sh`

**Slurm job:** `es3_sanity_gpu` — single GPU, ~10 min.

**Verified:**

- Model loads on GPU (122M total, 84.0M trainable, 38.2M frozen).
- Freeze state correct: only `backbone.vision_backbone.trunk.*` (300 params)
  and `backbone.language_backbone.*` (151 params) are trainable. Zero gradient
  leakage to frozen modules.
- Vision encoder produces 3 FPN levels: `[1,256,288,288]`, `[1,256,144,144]`,
  `[1,256,72,72]`.
- Text encoder produces `[16,2,256]` language features with `d_model=256`.
- Cosine similarity with SAM3 teacher: non-trivial but reasonable (expected —
  student is distilled, not identical).

### 3. Smoketest Training (2 epochs)

**Config:** `sam3/sam3/train/configs/stage3/mixed/stage3_mixed_smoketest_train.yaml`

**Slurm job:** `es3_stage3_smoketest` (ID: 3274278) — 4 GPUs, completed in 20 min.

**Settings:** 2 epochs, ~500 samples per source, `train_batch_size=2`,
`gradient_accumulation_steps=1`, validation every epoch.

**Results:**

| Metric | Epoch 0 | Epoch 1 | Trend |
|--------|---------|---------|-------|
| Train loss (`all_loss`) | 136.49 | 118.76 | -13.0% |
| Val loss (`all_loss`) | 2.68 | 2.59 | -3.5% |
| Val CE F1 | 0.949 | 0.964 | +1.6% |
| Val presence accuracy | 99.78% | 99.78% | stable |
| Val bbox loss | 0.0114 | 0.0106 | -7.0% |
| Val GIoU loss | 0.0896 | 0.0856 | -4.4% |

GPU memory: peaked at ~57 GB on H200, average ~20 GB during training.

Note: train loss is much higher than val loss because training includes
one-to-many (o2m) auxiliary losses that dominate the total. Validation only
reports one-to-one losses.

### 4. Checkpoint Integrity Verification

**Script:** `stage3/verify_checkpoint_integrity.py` via
`scripts/verify_checkpoint_integrity.sh`

**Slurm job:** `es3_verify_ckpt` (ID: 3277625).

**Check 1 — Merged checkpoint vs model state_dict:**
- Merged has 1201 keys; model (with `enable_segmentation=False`) expects 842.
- All 842 model keys present in merged with correct shapes.
- 359 extra merged keys are `backbone.vision_backbone.sam2_convs.*` and
  `tracker.*` — video components not used by the image model, harmlessly
  ignored on load (`strict=False`).

**Check 2 — Trainer checkpoint keys:**
- Smoketest trainer saved 451 keys: 300 vision + 151 text, 0 unexpected.
- All are a proper subset of the merged checkpoint.

**Check 3 — Frozen weights bitwise identical to SAM3 after training + merge-back:**
- **750 frozen keys checked: all 750 bitwise identical to SAM3 teacher.**
- 0 differences, 0 missing.
- Frozen modules verified: `geometry_encoder.*`, `transformer.*` (decoder),
  `dot_prod_scoring.*`, `segmentation_head.*`, FPN convolutions.

---

## What Stage 3 Finetunes vs What Is Frozen

### Finetuned (trainable, gradients flow)

| Module | Prefix | Params |
|--------|--------|--------|
| Student image encoder | `backbone.vision_backbone.trunk.*` | ~10.55M |
| Student text encoder + projector | `backbone.language_backbone.*` | ~63.56M |

### Frozen (no gradients, kept in eval mode)

| Module | Prefix | Status |
|--------|--------|--------|
| FPN neck convolutions | `backbone.vision_backbone.convs.*` | Frozen, in merged ckpt |
| Geometry encoder | `geometry_encoder.*` | Frozen, in merged ckpt |
| Transformer decoder | `transformer.*` | Frozen, in merged ckpt |
| Dot-product scoring | `dot_prod_scoring.*` | Frozen, in merged ckpt |
| Segmentation head | `segmentation_head.*` | Frozen, in merged ckpt (disabled at runtime) |
| Video tracker | `tracker.*` (309 keys) | Frozen, in merged ckpt, **not loaded by image model** |
| SAM mask decoder (video) | `tracker.sam_mask_decoder.*` | Frozen, in merged ckpt, not loaded |
| SAM prompt encoder (video) | `tracker.sam_prompt_encoder.*` | Frozen, in merged ckpt, not loaded |
| Memory backbone | `tracker.maskmem_backbone.*` | Frozen, in merged ckpt, not loaded |
| Object pointer projections | `tracker.obj_ptr_proj.*` | Frozen, in merged ckpt, not loaded |

**Key point:** The video/tracker modules (memory bank, SAM mask decoder, SAM
prompt encoder, object pointer projections — 309 keys total) **are present in
the merged checkpoint** and pass through the merge-back step unchanged. They
are **not loaded or used during Stage 3 training** (which is image-only), but
they are preserved in the checkpoint so the final eval checkpoint can be used
for video inference too. Stage 3 training runs the image path only
(`build_efficientsam3_image_model`), so the tracker keys are simply ignored
during loading and training, and survive intact in the output.

---

## Dataset Audit

### Confirmed Ready for Stage 3 Training

These datasets have **text annotations (category names) + images + bounding
boxes + masks**, and are wired into the Stage 3 training configs:

| Dataset | Path | Format | Text Type | Train Samples | Val Samples |
|---------|------|--------|-----------|---------------|-------------|
| **COCO** | `data/coco` | COCO JSON | Category names (80 classes) | ~118k images | ~5k images |
| **LVIS** | `data/lvis` | COCO JSON + synonyms | Category names + synonyms (1203 classes) | ~100k images | ~19.8k images |
| **ODINW** | `data/odinw` | Roboflow COCO JSON | Category names per subset | ~44 subsets | ~44 subsets |
| **RF100-VL** | `data/rf100-vl` | Roboflow COCO JSON | Category names per subset | 100 subsets | 100 subsets |

Note: ODINW and RF100-VL subsets vary in mask quality. The Stage 3 config uses
`require_masks: true` to filter out samples without valid masks.

### Partially Ready

| Dataset | Path | Issue |
|---------|------|-------|
| **RefCOCO** | `data/refcoco` | Parquet download incomplete — HF metadata present but `data/*.parquet` files missing. Config includes it as optional. |
| **DAVIS + text** | `data/davis` + `data/davis_text_annotations` | Video frames + referring expressions. Not COCO JSON format. Would need a custom dataloader to use in Stage 3. |
| **SA-Co Gold** | `data/sa-co` | Rich `text_input` + segmentation GT. Currently used for **evaluation only**. Would need a dataloader adapter. |
| **SA-V + text** | `data/sa-v` + `data/sa-v-text` | Video + text prompts. Video-oriented, not single-image COCO format. |

### Not Ready

| Dataset | Path | Issue |
|---------|------|-------|
| **PhraseCut** | `data/phrasecut` | Directory exists but appears empty or placeholder. |
| **SA-1B pseudo labels** | `data/sa1b_stage3_pseudo_*` | Experimental VLM-generated labels. Most labels are empty/rejected. Not usable until label quality improves. |
| **SA-1B raw** | `data/SA-1B-2p` | Mask-only (no text). Useful for Stage 1 distillation, not Stage 3 text-grounded training. |

### Not Text-Paired (Stage 1/2 only)

| Dataset | Path | Purpose |
|---------|------|---------|
| **SA-1B** | `data/sa-1b`, `data/SA-1B-2p` | Image encoder distillation (Stage 1) |
| **Recap-DataComp-1B** | via `data/download_datacomp.py` | Text encoder distillation (Stage 1) |
| **SA-V** | `data/sa-v` | Video tracking (Stage 2) |
| **LVOS** | `data/lvos` | Video object segmentation |
| **YouTube-VOS** | `data/ytvos` | Video object segmentation |

---

## Files Created / Modified

### New Files (Stage 3 pipeline)

| File | Purpose |
|------|---------|
| `stage3/model.py` | `build_stage3_model()` — freeze/unfreeze + train mode patching |
| `stage3/train_stage3.py` | Hydra entry point for training |
| `stage3/data/mixed_text_mask_dataset.py` | Multi-source dataset loader |
| `stage3/data/sa1b_pseudo_labels.py` | SA-1B pseudo-label dataset (experimental) |
| `stage3/sanity_check_gpu.py` | GPU model verification |
| `stage3/sanity_check_merge.py` | CPU key-alignment check |
| `stage3/merge_stage3_checkpoint_for_eval.py` | Merge trained encoders back for eval |
| `stage3/verify_checkpoint_integrity.py` | Full checkpoint integrity verification |
| `stage3/scripts/train_stage3_mixed.sh` | Local training launcher |
| `scripts/train_stage3_h200.sh` | Slurm launcher: H200 training |
| `scripts/train_stage3_smoketest.sh` | Slurm launcher: smoke test |
| `scripts/sanity_check_gpu.sh` | Slurm launcher: GPU sanity check |
| `scripts/verify_checkpoint_integrity.sh` | Slurm launcher: checkpoint verification |

### New Configs

| Config | Purpose |
|--------|---------|
| `sam3/sam3/train/configs/stage3/mixed/stage3_mixed_local_train.yaml` | Base config (all paths, hyperparameters, data sources, losses) |
| `sam3/sam3/train/configs/stage3/mixed/stage3_mixed_smoketest_train.yaml` | 2-epoch smoke test |
| `sam3/sam3/train/configs/stage3/mixed/stage3_mixed_h200_30ep_train.yaml` | H200 production (30 epochs, batch=4) |
| `sam3/sam3/train/configs/stage3/mixed/stage3_mixed_stable_30ep_train.yaml` | Stable 30-epoch run |
| `sam3/sam3/train/configs/stage3/mixed/stage3_mixed_h200_30ep_epochckpt_train.yaml` | H200, COCO+LVIS only |
| `sam3/sam3/train/configs/stage3/mixed/stage3_mixed_h200_30ep_fullft_train.yaml` | Full-model fine-tuning variant |

---

## Errors Encountered and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'` | Typo in sanity check script | Changed `total_mem` to `total_memory` |
| `AttributeError: 'str' object has no attribute 'shape'` | `forward_image()` returns dict, not tuple | Fixed unpacking to use dict keys (`"backbone_fpn"`, etc.) |
| `torch.OutOfMemoryError` (90 GB used) | `train_batch_size=4` + `gradient_accumulation_steps=2` too large | Reduced to `batch_size=2`, `grad_accum=1` |
| `AssertionError: Expected a list of batches, got dict` | Trainer expects list when `gradient_accumulation_steps > 1` | Set `gradient_accumulation_steps=1` |

---

## How to Continue: Full Training

### Immediate next step

```bash
sbatch scripts/train_stage3_h200.sh
```

This launches 30-epoch training with `stage3_mixed_h200_30ep_train.yaml`
(batch=4, 4 GPUs, ~24h estimated).

### After training

```bash
# Merge for evaluation
python stage3/merge_stage3_checkpoint_for_eval.py \
  --stage3-ckpt  output/stage3/mixed_h200_30ep_es_tv_m_mc_s1_ctx16/checkpoints/checkpoint.pt \
  --base-ckpt    output/efficient_sam3_stage3_es_tv_m_mobileclip_s1_ctx16.pth \
  --output       output/efficient_sam3_stage3_finetuned_es_tv_m_mc_s1.pth

# Verify integrity
python stage3/verify_checkpoint_integrity.py \
  --trainer-ckpt output/stage3/mixed_h200_30ep_es_tv_m_mc_s1_ctx16/checkpoints/checkpoint.pt
```

### Running other encoder combinations

Repeat the pipeline with different `--image-model-name` / `--text-model-name`
in the merge step, and update `backbone_type` / `model_name` /
`text_encoder_type` in the config. See `README_stage3.md` for the full table.

### Possible improvements

- **Download RefCOCO parquets** to add free-form referring text to training.
- **Enable segmentation losses** (`enable_segmentation: true`) for mask-level
  supervision — requires more GPU memory (~45 GB/sample).
- **Add SA-Co Gold** as a training source by writing a dataloader adapter.
- **Try full-model fine-tuning** with `stage3_mixed_h200_30ep_fullft_train.yaml`
  (unfreezes decoder too; heavier, needs batch_size=1).

---

## Geometry-Aware Fine-Tuning

### Motivation

The initial Stage 3 pipeline trained encoders with text-only prompts. While
each text query implicitly carried a GT bounding box processed by the frozen
geometry encoder, there were no "geometry-only" training signals. The frozen
geometry encoder uses cross-attention between box/point prompts and image FPN
features — if the student image encoder's features drift from the original
distribution, geometry-conditioned detection degrades at inference time.

Inspiration from SAM2 (iterative point/box refinement on GT masks) and EdgeSAM
(knowledge distillation with teacher mask decoder) guided the design.

### Approach

Instead of replacing text queries or modifying the model architecture, we
**augment** the training data with geometry-only queries:

1. **`AddGeometricQueries` transform** (`stage3/transforms/geometry_sampling.py`):
   With probability `geo_prob` (default 0.5), selects up to `max_geo_queries`
   objects with valid masks and creates `FindQueryLoaded` entries with
   `query_text="geometric"`. These target a single object each.

2. **`RandomGeometricInputsAPI`** (existing SAM3 transform): Recognizes queries
   with `query_text="geometric"` and samples a noised bounding box from the
   target object's GT mask (`resample_box_from_mask=true`, `box_chance=1.0`).

3. **Data pipeline integration**: `DecodeRle` is placed before the geometry
   transforms to ensure masks are available as tensors. `load_segmentation: true`
   is set in the dataset config, while `with_seg_masks: false` in the collator
   prevents masks from being sent to the GPU.

4. **No architecture changes**: The frozen geometry encoder's cross-attention
   over FPN features provides gradient signal back through the trainable image
   encoder, ensuring the FPN feature space stays compatible with the geometry
   pathway.

### Key design decisions

- **Box-only prompts**: `SAM3Image.forward` ignores `input_points` and only
  processes `input_boxes`, so we use `box_chance=1.0` for geometry queries.
- **Reduced query count**: The decoder's RPB (Relative Positional Bias) matrix
  scales quadratically with total queries. We reduced `max_find_queries_per_img`
  from 64 to 40 and capped `max_geo_queries` at 4 to keep peak memory under
  50GB on H200 (95GB).
- **`geo_prob=0.5`**: Half of training batches get geometry-only queries, so
  text-grounded training signal remains dominant.

### OOM Debugging

The first geometry smoketest used `max_find_queries_per_img=64` + `max_geo_queries=8`
(up to 72 queries total). This OOM'd at 92.55GB in the decoder's `_get_rpb_matrix`
— a 12.5% query count increase caused a ~39% memory spike due to quadratic
attention scaling. Reducing to 40+4=44 max queries brought peak memory to 49GB.

### Geometry Smoketest Results

**Config:** `stage3_mixed_geo_smoketest_train.yaml`
**Slurm job:** `es3_geo_smoke` (ID: 3278054) — 4 GPUs, completed in 19 min.
**Settings:** 2 epochs, ~500 samples per source, `batch_size=2`,
`max_find_queries_per_img=40`, `max_geo_queries=4`, `geo_prob=0.5`.

| Metric | Epoch 0 | Epoch 1 | Trend |
|--------|---------|---------|-------|
| Train loss (`all_loss`) | 155.37 | 142.52 | -8.3% |
| Val loss (`all_loss`) | 3.67 | 3.53 | -3.8% |
| Val CE F1 | 0.963 | 0.973 | +1.0% |
| Val presence accuracy | 99.56% | 99.56% | stable |
| Val bbox loss | 0.0136 | 0.0130 | -4.4% |
| Val GIoU loss | 0.0849 | 0.0807 | -4.9% |

Peak GPU memory: 49 GB on H200 (safely under 95 GB limit).

### Production Training

**Config:** `stage3_mixed_geo_h200_train.yaml`
**Slurm job:** `es3_geo_1n` (ID: 3278232) — 1 node, 4 GPUs, 24h time limit.

Multi-node (4 nodes, 16 GPUs) was attempted first but blocked by the cluster's
`AssocGrpMemMinutes` limit. Fell back to single-node with automatic checkpoint
resume.

**Data:** COCO (30K) + LVIS train (25K) + LVIS val-as-train (10K) + ODINW all
splits (30K) + RF100 all splits (15K) ≈ 110K samples. RefCOCO skipped due to
schema mismatch (not critical).

**Estimated timing:** 9125 steps/epoch × ~3.5s/step ≈ 8.8h/epoch. The 24h
window yields ~2.7 epochs. `max_data_epochs=10` set as safety margin.

**Early observations (step 0-110/9125):**
- Loss decreasing from 290 → 139
- Peak memory: 45GB
- No errors; training stable

### New Files for Geometry Finetuning

| File | Purpose |
|------|---------|
| `stage3/transforms/__init__.py` | Package init |
| `stage3/transforms/geometry_sampling.py` | `AddGeometricQueries` transform |
| `stage3/train_stage3_srun.py` | `srun`-based DDP entry point (maps SLURM→PyTorch env vars) |
| `scripts/train_stage3_geo_smoketest.sh` | Slurm launcher: geometry smoketest |
| `scripts/train_stage3_geo_h200_multinode.sh` | Slurm launcher: 4-node multi-node training |
| `scripts/train_stage3_geo_h200_1node.sh` | Slurm launcher: 1-node fallback |

### New Configs for Geometry Finetuning

| Config | Epochs | Batch | Max queries | Geo queries | Notes |
|--------|--------|-------|-------------|-------------|-------|
| `stage3_mixed_geo_smoketest_train.yaml` | 2 | 2 | 40 | 4 | Quick validation, ~500 samples |
| `stage3_mixed_geo_h200_train.yaml` | 10 | 2 | 40 | 4 | Production, all datasets, 24h |

### How to Resume / Continue

The `train_stage3_srun.py` entry point automatically detects
`checkpoint_dir/checkpoint.pt` and resumes from it. To continue training:

```bash
# Single node (auto-resume from last checkpoint):
sbatch scripts/train_stage3_geo_h200_1node.sh

# Multi-node (if cluster allocation permits):
sbatch scripts/train_stage3_geo_h200_multinode.sh
```

### After Training — Merge and Evaluate

```bash
# Merge for evaluation
python stage3/merge_stage3_checkpoint_for_eval.py \
  --stage3-ckpt  output/stage3/geo_h200_es_tv_m_mc_s1_ctx16/checkpoints/checkpoint.pt \
  --base-ckpt    output/efficient_sam3_stage3_es_tv_m_mobileclip_s1_ctx16.pth \
  --output       output/efficient_sam3_stage3_geo_finetuned_es_tv_m_mc_s1.pth

# Run SA-CO evaluation
python sam3/scripts/eval/gold/eval_efficientsam3_all_subsets.py \
  --checkpoint output/efficient_sam3_stage3_geo_finetuned_es_tv_m_mc_s1.pth \
  ...
```
