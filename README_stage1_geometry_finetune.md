## Stage 1 Geometry Finetune — Geometry-in-the-Loop Knowledge Distillation

Stage 1 Geometry Finetune completes the distillation by training the decoder and
prompt encoder with **geometry-in-the-loop** (iterative refinement with
points/boxes/masks). The pipeline: 1) train on converted Stage 1 checkpoints,
2) merge text encoder back for deployment.

### Prerequisites

1. **Environment** – activate `efficientsam3` and install Stage 1 deps
   (`pip install -e ".[stage1]"`).
2. **Stage 1 checkpoints** – converted weights in `output/efficient_sam3_*.pt`
   (see [README_stage1.md](README_stage1.md) Step 3).
3. **Dataset** – SA-1B with `DATA.LOAD_GT_MASK: True` (ground truth masks).
4. **Teacher embeddings** – reuse from Stage 1 at
   `output/stage1_teacher/embeddings/`.
5. **SAM3 checkpoint** – `sam3_checkpoints/sam3.pt` for teacher predictions.

### How It Works

**Geometry-in-the-Loop:**  
Student and teacher both predict masks from the same geometric prompts
(points/boxes/masks). For N iterations, sample correction points from
disagreement regions, add them to the prompt, and refine. Losses: BCE (5.0),
Dice (5.0), IoU (1.0).

**Mask Prompt Preparation (Consistent with SAM3):**
- Ground truth masks downsampled to **256×256** (SAM's low-res mask input size)
- Uses `F.interpolate(..., mode='bilinear', align_corners=False)`
- Passed as `mask_input` parameter to `predict_inst()` (same API as SAM2/SAM3)
- During iterative refinement, previous prediction becomes next iteration's mask prompt

**What's Trained:**
- ✅ **Image Encoder** (efficient backbone) — Only trainable component
- ❄️ **Prompt Encoder** — **Frozen** (for Stage 2 compatibility)
- ❄️ **Mask Decoder** — **Frozen** (for Stage 2 compatibility)
- ❌ **Text Encoder** — Excluded (saves ~354M GPU memory), merged after training

**Why Freeze Prompt Encoder & Mask Decoder?**  
Stage 2 trains a memory bank using the teacher's frozen prompt encoder + mask decoder.
If we train these here, Stage 2's memory bank learns to work with SAM3's components,
but inference uses modified ones → performance mismatch. Only the image encoder is
trained to adapt its features to work with SAM3's frozen prompt encoder and decoder.

### Step 1 — Train Geometry Finetune

Train on converted Stage 1 checkpoints. Text encoder is excluded to save ~354M
GPU memory.

```bash
# RepViT-M (single GPU)
bash stage1_geometry_finetune/scripts/train_stage1_geometry_finetune.sh \
  CFG=stage1_geometry_finetune/configs/es_rv_m.yaml \
  DATA_PATH=data/sa-1b \
  OUTPUT=output/stage1_geometry_finetune/es_rv_m \
  BATCH_SIZE=4 \
  GPUS=1

# Eight GPUs
bash stage1_geometry_finetune/scripts/train_stage1_geometry_finetune.sh \
  CFG=stage1_geometry_finetune/configs/es_rv_m.yaml \
  GPUS=8 \
  BATCH_SIZE=4
```

**Available configs** (all M-size with mask prompts):

| Config | Backbone | Params | Pretrained |
|--------|----------|--------|------------|
| `es_rv_m.yaml` | RepViT-M1.1 | 7.77M | `output/efficient_sam3_repvit_m.pt` |
| `es_tv_m.yaml` | TinyViT-11M | 10.55M | `output/efficient_sam3_tinyvit_m.pt` |
| `es_ev_m.yaml` | EfficientViT-B1 | 4.64M | `output/efficient_sam3_efficientvit_m.pt` |

**Output:**
```
output/stage1_geometry_finetune/es_rv_m/
├── ckpt_epoch_9.pth  # Final checkpoint
└── log_rank0.txt
```

### Step 2 — Merge Text Encoder

Merge SAM3 text encoder (~354M params) with finetuned weights. This script also
automatically prepends the `detector.` prefix to the finetuned weights to match
the SAM3 architecture.

```bash
python stage1_geometry_finetune/convert_finetuned_weights.py \
  --finetuned-ckpt output/stage1_geometry_finetune/es_rv_m/ckpt_epoch_9.pth \
  --sam3-ckpt sam3_checkpoints/sam3.pt \
  --output output/efficient_sam3_repvit_m_finetuned.pt
```

**Result:** Complete EfficientSAM3 checkpoint ready for deployment. The script
ensures all keys are correctly prefixed (e.g., `detector.backbone...`) and
includes the text encoder from the teacher.

### Step 3 — Use the Model

```python
from sam3.model_builder import build_efficientsam3_image_model

model = build_efficientsam3_image_model(
    checkpoint_path='output/efficient_sam3_repvit_m_finetuned.pt',
    backbone_type='repvit',
    model_name='m1_1',
    enable_inst_interactivity=True,
)
```

See `sam3/efficientsam3_examples/efficientsam3_for_sam1_task_example.py` for usage.

### Key Settings

| Setting | Value | Notes |
|---------|-------|-------|
| **Epochs** | 10 | Shorter than Stage 1 (finetuning) |
| **Learning Rate** | 3.2e-3 | Higher for faster convergence |
| **Batch Size** | 4 per GPU | Reduce if OOM |
| **Prompt Types** | box/point/mask | All geometric prompts, random selection |
| **Refinement Iters** | 3 | Iterative correction with sampled points |
| **Mask Input Size** | 256×256 | Low-res mask prompt (SAM standard) |
| **Text Encoder** | Excluded | Saves ~354M GPU memory |
| **Losses** | BCE (5.0), Dice (5.0), IoU (1.0) | Decoder distillation |

**Freezing Config:**
- `FREEZE_IMAGE_ENCODER: False` → Encoder is **trained** (only trainable component)
- `FREEZE_PROMPT_ENCODER: True` → Prompt encoder is **frozen** (Stage 2 compatibility)
- `FREEZE_MASK_DECODER: True` → Decoder is **frozen** (Stage 2 compatibility)
- `ENABLE_TEXT_ENCODER: False` → Text encoder **excluded** (merged after training)

Only change `FREEZE_PROMPT_ENCODER` and `FREEZE_MASK_DECODER` to `False` if you don't plan to use Stage 2 memory bank.

### Multi-Stage Pipeline Design

**Why freeze prompt encoder and mask decoder for Stage 2 compatibility?**

| Stage | Image Encoder | Prompt Encoder | Mask Decoder | Memory Bank | Purpose |
|-------|---------------|----------------|--------------|-------------|---------|
| **Stage 1** | ✅ Train | ❌ None | ❌ None | ❌ None | Feature matching |
| **Geometry Finetune** | ✅ Train | ❄️ **Frozen** | ❄️ **Frozen** | ❌ None | Encoder adapts to prompts |
| **Stage 2** | ❄️ Frozen | ❄️ Frozen | ❄️ Frozen | ✅ Train | Memory learns with SAM3 |
| **Stage 3** | ✅ Train | ✅ Train | ✅ Train | ✅ Train | End-to-end optimization |

**The Problem:**  
If prompt encoder or mask decoder are trained in geometry finetune, then in Stage 2:
- Memory bank trains with **SAM3's original prompt encoder + decoder**
- But at inference, we use **modified prompt encoder + decoder** with the memory
- This component mismatch can degrade performance

**The Solution:**  
Keep both prompt encoder and decoder frozen, training only the image encoder. The encoder
learns to produce features that work well with SAM3's frozen components. The prompt encoder
and decoder are then trained jointly with the memory bank in Stages 2/3.

**Alternative approach:** Skip geometry finetune entirely and train encoder + prompt encoder +
decoder + memory bank jointly in Stage 2/3. This requires more compute but allows end-to-end
optimization.
