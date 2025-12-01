# Stage 2 — Temporal Memory Distillation

Stage 2 distills SAM3’s temporal memory bank into the Hybrid Memory Module used by EfficientSAM3. We freeze the Stage‑1 student encoder + text backbone, replace the SAM3 memory encoder / transformer with an EdgeTAM‑style hybrid perceiver and EfficientTAM attention, and train the new modules to match the teacher’s memory-conditioned features.

## Overview
- **Teacher**: Official SAM3 tracker (ViT-Huge backbone, full memory bank)
- **Student**: Stage‑1 EfficientSAM3 encoder + Hybrid Memory + Efficient Attention
- **Goal**: Reproduce the teacher’s memory-conditioned features while compressing the number of memory tokens by >10×

## Prerequisites
1. Conda env from the root `README.md`
2. Stage‑1 merged checkpoint (image + text), e.g. `output/efficient_sam3_stage1_merged.pt`
3. SA‑V frames reorganised under `data/sa-v/extracted_frames` (see Dataset Prep below)

## Step 0: Download and Reorganize SA-V Dataset

We use the SA-V dataset for training. You can start with a small 1% subset for sanity checks.

### 1. Download SA-V Data
Use the provided helper script to download the data.

**For the 1% Sanity Subset (Recommended for testing):**
```bash
bash data/download_sa_v.sh data/sa-v-1p.txt data/sa-v/raw 4
```

**For the Full SA-V Dataset:**
```bash
bash data/download_sa_v.sh data/sa-v.txt data/sa-v/raw 8
```

### 2. Reorganize and Extract Frames
The raw data comes as tar archives. We need to extract frames and annotations into a standard directory structure.

```bash
python data/reorg_sav.py \
  --source_dir data/sa-v/raw \
  --output_dir data/sa-v/extracted_frames \
  --workers 8
```

The resulting structure will look like:
```text
data/sa-v/extracted_frames/
  video_001/
    00000.jpg
    00001.jpg
    ...
    annotations.json
```

---

## Step 1 — Extract Teacher Backbone Features

We pre-compute the teacher's backbone features to save time during training. This extracts multi-scale FPN features (`f0/f1/f2`) and positional encodings for every frame.

### Single GPU (Debug/Testing)
```bash
bash stage2/scripts/extract_features.sh \
  TEACHER_CHECKPOINT=sam3_checkpoints/sam3.pt \
  DATASET_PATH=data/sa-v/extracted_frames \
  OUTPUT_DIR=output/stage2_features \
  GPUS=1
```

### Multi-GPU (Single Node, e.g. 4 or 8 GPUs)
```bash
# 4 GPUs
bash stage2/scripts/extract_features.sh \
  TEACHER_CHECKPOINT=sam3_checkpoints/sam3.pt \
  DATASET_PATH=data/sa-v/extracted_frames \
  OUTPUT_DIR=output/stage2_features \
  GPUS=4

# 8 GPUs
bash stage2/scripts/extract_features.sh \
  TEACHER_CHECKPOINT=sam3_checkpoints/sam3.pt \
  DATASET_PATH=data/sa-v/extracted_frames \
  OUTPUT_DIR=output/stage2_features \
  GPUS=8
```

---

## Step 2 — Cache Teacher Memory Embeddings

We replay the teacher tracker over the cached features to generate the "gold standard" memory-conditioned features. We use **teacher forcing** here: the teacher uses its own ground-truth-like masks (from its high-quality prediction) to update its memory bank, ensuring high-quality targets for the student to learn from.

The output includes:
- `teacher_feat`: Fused feature map after SAM3 memory attention.
- `mask_high_res`: The mask the teacher used to update its memory.
- `object_scores`: Objectness logits.

### Single GPU
```bash
bash stage2/scripts/save_teacher_embeddings.sh \
  TEACHER_CHECKPOINT=sam3_checkpoints/sam3.pt \
  FEATURES_DIR=output/stage2_features \
  OUTPUT_DIR=output/stage2_teacher \
  GPUS=1
```

### Multi-GPU
```bash
# 4 GPUs
bash stage2/scripts/save_teacher_embeddings.sh \
  TEACHER_CHECKPOINT=sam3_checkpoints/sam3.pt \
  FEATURES_DIR=output/stage2_features \
  OUTPUT_DIR=output/stage2_teacher \
  GPUS=4

# 8 GPUs
bash stage2/scripts/save_teacher_embeddings.sh \
  TEACHER_CHECKPOINT=sam3_checkpoints/sam3.pt \
  FEATURES_DIR=output/stage2_features \
  OUTPUT_DIR=output/stage2_teacher \
  GPUS=8
```

---

## Step 3 — Train the Hybrid Memory Module

We train the EfficientSAM3 student model. The student uses the **Hybrid Memory Module** (Global + Spatial Perceivers) to process the video history. We distill the teacher's memory-conditioned features into the student's output.

During training:
- The student receives the same images as the teacher.
- The student's memory is warmed up with the teacher's masks (Teacher Forcing).
- **Loss**: MSE + Cosine Similarity between the student's fused features and the cached `teacher_feat`.

### Single GPU
```bash
# 1x GPU training (override epochs for quick debug)
bash stage2/scripts/train_stage2.sh \
  CONFIG=stage2/configs/efficient_sam3_stage2.yaml \
  GPUS=1 \
  EPOCHS=12
```

### Multi-GPU (Single Node)
```bash
# 4 GPUs
bash stage2/scripts/train_stage2.sh \
  CONFIG=stage2/configs/efficient_sam3_stage2.yaml \
  GPUS=4 \
  EPOCHS=12

# 8 GPUs
bash stage2/scripts/train_stage2.sh \
  CONFIG=stage2/configs/efficient_sam3_stage2.yaml \
  GPUS=8 \
  EPOCHS=12
```

### Key Config Settings
Edit `stage2/configs/efficient_sam3_stage2.yaml`:
- `data.features_dir`: Path to `output/stage2_features`
- `data.teacher_dir`: Path to `output/stage2_teacher`
- `model.stage1_ckpt`: Path to merged Stage 1 checkpoint
- `train.batch_size`: Adjust based on GPU memory (default is often small due to video sequence length)

---

## Step 4 — Merge Stage‑1 Encoders + Stage‑2 Memory

Combine the Stage‑1 image/text students with the Stage‑2 checkpoint to produce the final EfficientSAM3 tracker.

```bash
python stage2/convert_stage2_weights.py \
  --image_student_checkpoint output/stage1/es_ev_s/ckpt_epoch_30.pth \
  --text_student_checkpoint output/stage1_text/mobileclip_s/ckpt_epoch_49.pth \
  --stage2_checkpoint output/stage2/stage2_epoch_12.pth \
  --output_path output/efficient_sam3_stage2.pt
```

The merged file keeps:
- Stage‑1 image/text encoders
- Stage‑2 Hybrid Memory Module + Efficient Attention
- Original SAM3 prompt encoder, mask decoder, and heads

---

## Quick Reference

| Step | Script | Output |
|------|--------|--------|
| Feature cache | `stage2/scripts/extract_features.sh` | `output/stage2_features` |
| Teacher embeddings | `stage2/scripts/save_teacher_embeddings.sh` | `output/stage2_teacher` |
| Stage‑2 training | `stage2/scripts/train_stage2.sh` | `output/stage2/stage2_epoch_*.pth` |
| Weight merge | `stage2/convert_stage2_weights.py` | `output/efficient_sam3_stage2.pt` |

After Stage‑2 conversion you can load the tracker exactly like SAM3’s video model; see the root `README.md` for API usage.
