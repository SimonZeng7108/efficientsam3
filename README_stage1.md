## Stage 1 — SAM3 Encoder Distillation

Stage 1 compresses the SAM3 vision encoder into nine lightweight backbones
(`RepViT`, `TinyViT`, and `EfficientViT`) and the text encoder into lightweight
MobileCLIP variants (`MobileCLIP-S0`, `MobileCLIP-S1`, and `MobileCLIP2-L`). The pipeline has three discrete phases: 1) export SAM3
teacher embeddings (image on SA-1B, text on Recap-DataComp-1B), 2) train student encoders
to regress those embeddings, and 3) splice the student weights back into the
full SAM3 checkpoint for deployment.

### Prerequisites

1. **Environment** – follow the root [Installation](README.md#installation) guide to
   create/activate the `efficientsam3` Conda environment and run
   `pip install -e ".[stage1]"` (installs PyTorch, decord, and all Stage‑1 deps).
2. **Datasets**:
   - **SA-1B** (for Vision): make sure `DATA.DATA_PATH` points to your SA-1B root (must contain `images/{train,val}` and `annotations/{train,val}`).
   - **Recap-DataComp-1B** (for Text): We use Recap-DataComp-1B for text distillation.
     - Download the parquet files using `python data/download_datacomp.py`.
     - Ensure `DATA.DATA_PATH` points to the folder containing the parquet files (e.g., `data/recap_subset`).
     - Set `DATA.DATASET` to `recap_datacomp` in the config.

   **Note:** We currently distill vision from a 1% subset of SA-1B and text from a 1% subset of Recap-DataComp-1B.
   - Download links are provided in `data/sa-1b-1p.txt`.
   - After downloading and extracting the tar archives, run `python data/reorg_sa1b.py` to reorganize the files into the required train/val structure.

3. **Teacher weights** – download `sam3.pt` (or another SAM3 checkpoint) from [Hugging Face](https://huggingface.co/facebook/sam3/tree/main) into
   `sam3_checkpoints/` and set `MODEL.RESUME` inside
   `stage1/configs/teacher/sam_vit_huge_sa1b.yaml`.
4. **Parameter Analysis** – You can compare the parameter count of the SAM3 teacher model and the student models using the provided script:
   ```bash
   # View summary of all student models vs teacher
   PYTHONPATH=sam3 python stage1/compare_models.py --student all
   ```

   **Parameter Breakdown:**
   - **SAM3 Teacher (Total)**: 860.06M parameters.
     - **Vision Backbone**: 461.84M (Target for replacement).
     - **Language Backbone**: 353.72M (Target for replacement).
     - **Decoder/Heads**: ~45M (Retained).

   **Student Savings (Vision Encoder):**
   | Student Model | Backbone | Params | vs. Teacher (461M) |
   | :--- | :--- | :--- | :--- |
   | **ES-EV-S** | EfficientViT-B0 | **0.68M** | **99.85% smaller** |
   | **ES-EV-M** | EfficientViT-B1 | **4.64M** | **99.00% smaller** |
   | **ES-EV-L** | EfficientViT-B2 | **14.98M** | **96.76% smaller** |
   | **ES-RV-S** | RepViT-M0.9 | **4.72M** | **98.98% smaller** |
   | **ES-RV-M** | RepViT-M1.1 | **7.77M** | **98.32% smaller** |
   | **ES-RV-L** | RepViT-M2.3 | **22.40M** | **95.15% smaller** |
   | **ES-TV-S** | TinyViT-5M | **5.07M** | **98.90% smaller** |
   | **ES-TV-M** | TinyViT-11M | **10.55M** | **97.72% smaller** |
   | **ES-TV-L** | TinyViT-21M | **20.62M** | **95.53% smaller** |

   **Student Savings (Text Encoder):**
   
   **Teacher Breakdown**:
   - **Backbone**: 353.46M
   - **Resizer**: 0.26M (1024 $\to$ 256)
   - **Total**: 353.72M

   **Full Student Architecture**:
   The student model replaces the entire teacher text encoder, including the embedding layer. It uses the same tokenizer but learns its own embeddings (initialized randomly or from MobileCLIP).
   
   Structure: `Tokenizer -> Student Embed -> Student Transformer -> Projector`

   | Student Model | Backbone | Params | vs. Teacher (354M) |
   | :--- | :--- | :--- | :--- |
   | **ES-MC-S** | MobileCLIP-S0 | **42.57M** | **87.96% smaller** |
   | **ES-MC-M** | MobileCLIP-S1 | **63.56M** | **82.03% smaller** |
   | **ES-MC-L** | MobileCLIP2-L | **123.6M** | **65.06% smaller** |

7. **Shape Verification** – All student backbones have been verified to produce the correct embedding shapes to match the SAM3 teacher.

### 1. Prepare Inputs

| Requirement | Notes |
|-------------|-------|
| **SA-1B dataset** | Point `DATA.DATA_PATH` to the folder that contains `images/{train,val}` and `annotations/{train,val}` (defaults to `data/sa-1b`). |
| **Recap-DataComp dataset** | Point `DATA.DATA_PATH` to the folder that contains parquet files (defaults to `data/recap_subset`). |
| **SAM3 checkpoint** | Download `sam3.pt` (e.g. from HuggingFace `facebook/sam3`) and set `MODEL.RESUME` in `stage1/configs/teacher/sam_vit_huge_sa1b.yaml`. |
| **Output directory** | All outputs (logs, embeddings, checkpoints) are saved under `output/`. |

### Step 1 — Save Teacher Embeddings

**A. Image Embeddings (SA-1B)**

**This is a one-time forward pass** through the teacher model on the entire
SA-1B dataset. The embeddings are saved once to
`output/stage1_teacher/embeddings/`, then reused for all
student training epochs.

> **Note:** The text encoder (~354M params) is disabled during this step to reduce memory usage, as we only need image embeddings. Similarly, the vision encoder is disabled when saving text embeddings.

Use the provided launcher or run the Python entry point directly.

```bash
# Recommended helper (override CFG/DATA_PATH/OUTPUT inline as KEY=VALUE)
# Single GPU
bash stage1/scripts/save_image_embeddings.sh \
  CFG=stage1/configs/teacher/sam_vit_huge_sa1b.yaml \
  DATA_PATH=data/sa-1b \
  OUTPUT=output/stage1_teacher \
  BATCH_SIZE=64 \
  GPUS=1
```

**B. Text Embeddings (Recap-DataComp-1B)**

Similarly, save the teacher text embeddings on the Recap-DataComp-1B dataset.

```bash
bash stage1/scripts/save_text_embeddings.sh \
  CFG=stage1/configs/teacher/sam_text_teacher.yaml \
  DATA_PATH=data/recap_subset \
  OUTPUT=output/stage1_text_teacher \
  BATCH_SIZE=64 \
  GPUS=1 \
  --opts DATA.DATASET recap_datacomp
```

**Output structure**:
```
output/stage1_teacher/        # Image Embeddings
├── config.json               # Config from embedding export
├── log_rank0.txt             # Logs
└── embeddings/               # Actual teacher embeddings
    ├── rank0-keys.txt        # Image IDs
    ├── rank0-values.bin      # Embeddings (float16)
    ├── rank1-keys.txt        # (if using multiple GPUs)
    └── rank1-values.bin

output/stage1_text_teacher/   # Text Embeddings
├── config.json
├── log_rank0.txt
└── embeddings/
    ├── rank0-keys.txt        # Image IDs
    └── rank0-values.bin      # Embeddings (float16)
```

The scripts produce sharded binary files in `output/*/embeddings/`.



### Step 2 — Train Student Encoders

**A. Vision Encoders**

Set `MODEL.BACKBONE` via the config file (see table below) and launch the
student distillation run.

```bash
# Single GPU (override CFG/DATA_PATH/OUTPUT inline as needed)
bash stage1/scripts/train_image_student.sh \
  CFG=stage1/configs/es_rv_m.yaml \
  DATA_PATH=data/sa-1b \
  OUTPUT=output/stage1/repvit_m1 \
  BATCH_SIZE=4 \
  GPUS=1
```

**B. Text Encoders**

Set `MODEL.BACKBONE` in the config file to one of:
`MobileCLIP-S0`, `MobileCLIP-S1`, `MobileCLIP2-L`.

```bash
bash stage1/scripts/train_text_student.sh \
  CFG=stage1/configs/es_mc_s.yaml \
  DATA_PATH=data \
  OUTPUT=output/stage1_text/mobileclip_s \
  BATCH_SIZE=64 \
  GPUS=1
```

**Output structure**:
```
output/stage1/repvit_m1/
├── config.json           # Training config
├── log_rank0.txt         # Training logs
├── ckpt_epoch_0.pth      # Checkpoints per epoch
├── ckpt_epoch_1.pth
└── ...
└── ckpt_epoch_29.pth     # Final checkpoint
```

Students are selected via `MODEL.BACKBONE` in the config. The table below maps
the model zoo to configuration files.

| Model | Backbone | Config file |
|-------|----------|-------------|
| ES-RV-S | `repvit_m0_9` | `stage1/configs/es_rv_s.yaml` |
| ES-RV-M | `repvit_m1_1` | `stage1/configs/es_rv_m.yaml` |
| ES-RV-L | `repvit_m2_3` | `stage1/configs/es_rv_l.yaml` |
| ES-TV-S | `tiny_vit_5m` | `stage1/configs/es_tv_s.yaml` |
| ES-TV-M | `tiny_vit_11m` | `stage1/configs/es_tv_m.yaml` |
| ES-TV-L | `tiny_vit_21m` | `stage1/configs/es_tv_l.yaml` |
| ES-EV-S | `efficientvit_b0` | `stage1/configs/es_ev_s.yaml` |
| ES-EV-M | `efficientvit_b1` | `stage1/configs/es_ev_m.yaml` |
| ES-EV-L | `efficientvit_b2` | `stage1/configs/es_ev_l.yaml` |
| | | |
| ES-MC-S | `MobileCLIP-S0` | `stage1/configs/es_mc_s.yaml` |
| ES-MC-M | `MobileCLIP-S1` | `stage1/configs/es_mc_m.yaml` |
| ES-MC-L | `MobileCLIP2-L` | `stage1/configs/es_mc_l.yaml` |

Key config fields:

| Field | Description |
|-------|-------------|
| `DISTILL.TEACHER_EMBED_PATH` | Directory created during the teacher pass. |
| `DISTILL.EMBED_SIZE` / `EMBED_DIM` | Embedding grid size (default `64×64×256`). Must match the saved blobs. |
| `DATA.BATCH_SIZE`, `DATA.NUM_WORKERS` | Input pipeline throughput controls. |
| `OUTPUT`, `TAG` | Where checkpoints and TensorBoard logs are written. |

Stage‑1 loss = masked per-pixel MSE + 1.0 * Cosine Similarity computed on the resized embedding maps.
Padding pixels are filtered via `build_valid_mask`, so each student only learns
from valid pixels. The training script supports DDP + AMP by default.

### Step 3 — Package the Student with SAM3 Heads

After training, merge the distilled encoder with the full SAM3 checkpoint so it
can run end-to-end inference (prompt encoder + mask decoder).

**A. Vision Encoder Merge**

**Example for EfficientViT-B0 (ES-EV-S):**
```bash
python stage1/convert_image_encoder_weights_stage1.py \
  --student-ckpt output/stage1/es_ev_s/ckpt_epoch_49.pth \
  --sam3-ckpt sam3_checkpoints/sam3.pt \
  --output output/efficient_sam3_efficientvit_b0.pt
```

**B. Text Encoder Merge**

**Example for MobileCLIP-S0:**
```bash
python stage1/convert_text_encoder_weights_stage1.py \
  --student-ckpt output/stage1_text/mobileclip_s/ckpt_epoch_49.pth \
  --sam3-ckpt sam3_checkpoints/sam3.pt \
  --output output/efficient_sam3_text_s.pt
```

**C. Merge Both Encoders (Recommended)**

To create a fully efficient model with both student vision and text encoders:

> **Note:** If you perform this step, you do not need to run the individual merge steps (A and B) above.

```bash
python stage1/convert_both_encoders_weights_stage1.py \
  --image-student-ckpt output/stage1/es_ev_s/ckpt_epoch_30.pth \
  --text-student-ckpt output/stage1_text/mobileclip_s/ckpt_epoch_49.pth \
  --sam3-ckpt sam3_checkpoints/sam3.pt \
  --image-model-name efficientvit_b0 \
  --text-model-name mobileclip_s0
```

**Final output structure**:
```
output/
├── stage1_teacher/           # Teacher embedding export
│   ├── config.json
│   ├── log_rank0.txt
│   └── embeddings/           # Embeddings (reused by all students)
├── stage1/                   # Student training
│   └── repvit_m1/            # Checkpoints
└── efficient_sam3_efficientvit_b0_mobileclip_s0.pth  # Final merged model
```

The script copies student encoder weights into the SAM3 checkpoint under
`detector.backbone.vision_backbone.trunk.model.*` and preserves all other components (prompt encoder, mask
decoder). The `--replace-prefix` argument ensures the original teacher backbone is removed.

### Helper Scripts

The `stage1/scripts` folder contains ready-to-use launchers:

| Script | Purpose | Customisation knobs |
|--------|---------|---------------------|
| `save_image_embeddings.sh` | Runs `stage1/save_embedding_image_stage1.py` under `torchrun` for exporting teacher embeddings. | Override `CFG`, `DATA_PATH`, `OUTPUT`, `GPUS`, `MASTER_PORT`, and append additional CLI flags (e.g. `--check-saved-embed`). |
| `train_image_student.sh` | Launches `stage1/train_image_encoder_stage1.py` with a chosen student config. | Override `CFG`, `DATA_PATH`, `OUTPUT`, `BATCH_SIZE`, `GPUS`, `MASTER_PORT`, or pass through extra flags such as `--use-sync-bn`. |
| `save_text_embeddings.sh` | Runs `stage1/save_embedding_text_stage1.py` for exporting teacher text embeddings. | Same as above. |
| `train_text_student.sh` | Launches `stage1/train_text_encoder_stage1.py` for text student training. | Same as above. |

