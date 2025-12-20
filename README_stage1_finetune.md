## Stage 1 Fine-tuning — Geometry-Aware Distillation

Stage 1 Fine-tuning extends the base Stage 1 distillation by training on diverse SA-1B samples with **prompt-conditioned** processing. This stage ensures the student backbone produces embeddings that work well with SAM3's geometry encoder and downstream segmentation components.

### Key Insight

SAM3's architecture has **prompt-dependent processing**:
- The `GeometryEncoder` pools features FROM the backbone output to encode box/point prompts
- It uses `grid_sample` (for points) and `roi_align` (for boxes) on FPN features
- This means prompt embeddings differ based on backbone quality
- Fine-tuning on diverse prompts helps the student produce features compatible with downstream components

### Training Strategy: Dual-Path Distillation

We use **two complementary losses** that work together:

**Loss 1 — Embedding Distillation:**
```
Image → Student Trunk → student_embedding
                            ↓
            MSE Loss ← teacher_embedding (saved from Stage 1)
```

**Loss 2 — Mask Distillation:**
```
student_embedding → Frozen SAM3 (FPN→GeomEnc→Transformer→SegHead) → student_mask
                                    ↓
                              BCE + Dice Loss
                                    ↑
teacher_embedding → Frozen SAM3 (same frozen components) → teacher_mask
```

Both embeddings go through the **SAME frozen SAM3 components**, so the only difference in mask outputs is due to embedding quality.

### Prerequisites

1. **Environment** – follow the root [Installation](README.md#installation) guide to
   create/activate the `efficientsam3` Conda environment and run
   `pip install -e ".[stage1]"`.
2. **Complete Stage 1 Training** – you need pretrained student weights from Stage 1 (or use provided checkpoints).
3. **Teacher Embeddings** – **reuse the same embeddings from Stage 1**. If you haven't already, follow [README_stage1.md](README_stage1.md#step-1--save-teacher-embeddings) to save teacher embeddings using `stage1/scripts/save_image_embeddings.sh`. The embeddings are saved to `output/stage1_teacher/embeddings/`.
4. **SAM3 Checkpoint** – download `sam3.pt` from [Hugging Face](https://huggingface.co/facebook/sam3/tree/main) into `sam3_checkpoints/`.
5. **SA-1B Dataset** – point `DATA.DATA_PATH` to your SA-1B root (must contain `images/{train,val}` and `annotations/{train,val}`).

> **Note:** Geometry fine-tuning uses the **exact same teacher embeddings** as Stage 1. There is no need to run a separate embedding export - just point `TEACHER_EMBED_PATH` to your existing Stage 1 embeddings.

### Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINABLE COMPONENT                          │
│                                                                 │
│  Input Image (B, 3, 1008, 1008)                                │
│       │                                                         │
│       ▼                                                         │
│  StudentTrunk (RepViT/TinyViT/EfficientViT)                    │
│       │                                                         │
│       ▼                                                         │
│  Projection Head (Conv → BN → GELU → Conv)                     │
│       │                                                         │
│       ▼                                                         │
│  Student Embeddings (B, 1024, 72, 72) ─────────────────────────┼──► Loss 1: MSE
└───────┼─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FROZEN SAM3 COMPONENTS                        │
│                                                                 │
│  SimpleFPN Neck (frozen)                                       │
│       │                                                         │
│       ▼                                                         │
│  GeometryEncoder (frozen)  ←── Box/Point Prompts               │
│       │     (grid_sample/roi_align on FPN features)            │
│       ▼                                                         │
│  TransformerEncoder (frozen)                                   │
│       │                                                         │
│       ▼                                                         │
│  TransformerDecoder (frozen)                                   │
│       │                                                         │
│       ▼                                                         │
│  SegmentationHead (frozen)                                     │
│       │                                                         │
│       ▼                                                         │
│  Student Masks ────────────────────────────────────────────────┼──► Loss 2: BCE+Dice
└─────────────────────────────────────────────────────────────────┘
```

**Parameter Summary:**

| Component | Trainable | Parameters |
|-----------|-----------|------------|
| StudentTrunk (backbone + projection) | ✅ Yes | ~14-20M |
| SimpleFPN (4 conv layers) | ❌ Frozen | ~4M |
| GeometryEncoder | ❌ Frozen | ~10M |
| TransformerEncoder | ❌ Frozen | ~100M |
| SegmentationHead | ❌ Frozen | ~350M |

### Step 1 — Prepare Inputs

| Requirement | Notes |
|-------------|-------|
| **SA-1B dataset** | Point `DATA.DATA_PATH` to the folder that contains `images/{train,val}` and `annotations/{train,val}` (defaults to `data/sa-1b`). |
| **Teacher embeddings** | **Reuse Stage 1 embeddings.** Point `DISTILL.TEACHER_EMBED_PATH` to `output/stage1_teacher/embeddings/`. If not saved yet, run `bash stage1/scripts/save_image_embeddings.sh` first (see [README_stage1.md](README_stage1.md#step-1--save-teacher-embeddings)). |
| **SAM3 checkpoint** | Download `sam3.pt` and set `MODEL.SAM3_CHECKPOINT` in the config (or pass via `--sam3-checkpoint`). |
| **Stage 1 checkpoint** | Set `MODEL.PRETRAINED` to your Stage 1 student checkpoint (or pass via `--pretrained`). |
| **Output directory** | All outputs (logs, checkpoints) are saved under `output_geometry_finetune/`. |

### Step 2 — Train with Geometry Fine-tuning

Set `MODEL.BACKBONE` via the config file (see table below) and launch the fine-tuning run.

```bash
# Single GPU (override CFG/DATA_PATH/OUTPUT inline as needed)
bash stage1_geometry_finetune/scripts/train_geometry_finetune.sh \
  CFG=stage1_geometry_finetune/configs/es_rv_m.yaml \
  DATA_PATH=data/sa-1b \
  OUTPUT=output_geometry_finetune/es_rv_m \
  BATCH_SIZE=4 \
  GPUS=1
```

**Or run the Python entry point directly:**

```bash
python stage1_geometry_finetune/train_geometry_finetune.py \
  --cfg stage1_geometry_finetune/configs/es_rv_m.yaml \
  --data-path data/sa-1b \
  --pretrained output/efficient_sam3_repvit_m.pt \
  --sam3-checkpoint sam3_checkpoints/sam3.pt \
  --teacher-embed-path output/stage1_teacher/embeddings \
  --output output_geometry_finetune
```

**Output structure:**
```
output_geometry_finetune/es_rv_m/
├── config.json           # Training config
├── log_rank0.txt         # Training logs
├── ckpt_epoch_0.pth      # Checkpoints per epoch
├── ckpt_epoch_1.pth
└── ...
└── ckpt_epoch_29.pth     # Final checkpoint
```

### Configuration

Students are selected via `MODEL.BACKBONE` in the config. The table below maps
the model zoo to configuration files.

| Model | Backbone | Config file |
|-------|----------|-------------|
| ES-RV-S | `repvit_m0_9` | `stage1_geometry_finetune/configs/es_rv_s.yaml` |
| ES-RV-M | `repvit_m1_1` | `stage1_geometry_finetune/configs/es_rv_m.yaml` |
| ES-RV-L | `repvit_m2_3` | `stage1_geometry_finetune/configs/es_rv_l.yaml` |
| ES-TV-S | `tiny_vit_5m` | `stage1_geometry_finetune/configs/es_tv_s.yaml` |
| ES-TV-M | `tiny_vit_11m` | `stage1_geometry_finetune/configs/es_tv_m.yaml` |
| ES-TV-L | `tiny_vit_21m` | `stage1_geometry_finetune/configs/es_tv_l.yaml` |
| ES-EV-S | `efficientvit_b0` | `stage1_geometry_finetune/configs/es_ev_s.yaml` |
| ES-EV-M | `efficientvit_b1` | `stage1_geometry_finetune/configs/es_ev_m.yaml` |
| ES-EV-L | `efficientvit_b2` | `stage1_geometry_finetune/configs/es_ev_l.yaml` |


Loss formula:
```
total_loss = λ₁ × MSE(student_embed, teacher_embed)
           + λ₂ × BCE(student_mask, teacher_mask)
           + λ₃ × Dice(student_mask, teacher_mask)
```

### Step 3 — Package the Student with SAM3 Heads

After fine-tuning, replace the image encoder weights in the pretrained merged model.
This approach guarantees **100% structure match** with the original model.

```bash
python stage1_geometry_finetune/convert_geometry_finetune.py \
  --finetune-ckpt output_geometry_finetune/es_rv_m/ckpt_epoch_29.pth \
  --pretrained output/efficient_sam3_repvit_m.pt \
  --output output/efficient_sam3_repvit_m_geometry.pt
```

**Expected output:**
```
Student encoder keys extracted: 653
Keys matched in pretrained: 653
Replaced 653 weights in pretrained model

=== Verification ===
  ✓ c.weight
  ✓ bn.weight
  ✓ bn.bias

============================================================
Output saved to: output/efficient_sam3_repvit_m_geometry.pt
Total keys in output: 1698
Image encoder weights replaced: 653
✓ All weights successfully replaced!
============================================================
```

**What happens:**
- The script loads your pretrained merged model (e.g., `efficient_sam3_repvit_m.pt`)
- Extracts `student_trunk.*` keys from the finetune checkpoint (653 image encoder weights)
- Replaces only those weights in the pretrained model
- All other SAM3 components (FPN, geometry encoder, transformer, segmentation head) remain unchanged
- Result: 100% compatible model with geometry-finetuned image encoder

**Final output structure:**
```
output/
├── stage1_teacher/               # Teacher embedding export (from Stage 1)
│   └── embeddings/
├── stage1/                       # Stage 1 student training
│   └── es_rv_m/
├── output_geometry_finetune/     # Geometry fine-tuning
│   └── es_rv_m/
│       └── ckpt_epoch_29.pth     # Training checkpoint
└── efficient_sam3_repvit_m_geometry.pt  # Final merged model (1698 keys, 653 updated)
```


