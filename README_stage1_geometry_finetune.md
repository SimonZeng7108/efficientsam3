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

### Step 1 — Save Teacher Embeddings (Optional)

**Note:** If you have already generated teacher embeddings during Stage 1 (Image Encoder Distillation), you can **skip this step** and reuse them.

This step runs a forward pass through the teacher model to save embeddings, which speeds up training by avoiding re-computation.

```bash
# Single GPU
bash stage1/scripts/save_stage1_embeddings.sh \
  CFG=stage1/configs/teacher/sam_vit_huge_sa1b.yaml \
  DATA_PATH=data/sa-1b \
  OUTPUT=output/stage1_teacher \
  GPUS=1

# Eight GPUs
bash stage1/scripts/save_stage1_embeddings.sh \
  CFG=stage1/configs/teacher/sam_vit_huge_sa1b.yaml \
  DATA_PATH=data/sa-1b \
  OUTPUT=output/stage1_teacher \
  GPUS=8 \
  BATCH_SIZE=32
```

### Step 2 — Train Geometry Finetune

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

### Step 3 — Merge Text Encoder

Merge SAM3 text encoder (~354M params) with finetuned weights. This script also
automatically prepends the `detector.` prefix to the finetuned weights to match
the SAM3 architecture.

```bash
python stage1_geometry_finetune/convert_finetuned_weights.py \
  --finetuned-ckpt output/stage1_geometry_finetune/es_rv_m/ckpt_epoch_9.pth \
  --sam3-ckpt sam3_checkpoints/sam3.pt \
  --output output/efficient_sam3_repvit_m_finetuned.pt
```

### Step 4 — Use the Model

```python
from sam3.model_builder import build_efficientsam3_image_model

model = build_efficientsam3_image_model(
    checkpoint_path='output/efficient_sam3_repvit_m_finetuned.pt',
    backbone_type='repvit',
    model_name='m1_1',
    enable_inst_interactivity=True,
)
```
