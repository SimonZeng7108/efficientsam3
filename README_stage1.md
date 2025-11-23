## Stage 1 — SAM3 Encoder Distillation

Stage 1 compresses the SAM3 vision encoder into nine lightweight backbones
(`RepViT`, `TinyViT`, and `EfficientViT`). The pipeline has three discrete
phases: 1) export SAM3 teacher embeddings on SA-1B, 2) train student encoders
to regress those embeddings via masked MSE, and 3) splice the student weights
back into the full SAM3 checkpoint for deployment.

### Prerequisites

1. **Environment** – follow the root [Installation](README.md#installation) guide to
   create/activate the `efficientsam3` Conda environment and run
   `pip install -e ".[stage1]"` (installs PyTorch, decord, and all Stage‑1 deps).
2. **Dataset** – make sure `DATA.DATA_PATH` (defined in `stage1/configs/base_stage1.yaml`) points to your SA-1B root (the tree
   must contain `images/{train,val}` and `annotations/{train,val}`). By default
   configs/scripts read from `data/sa-1b`.

   **Note:** We currently distill from a 10% subset of SA-1B.
   - Download links are provided in `data/sa-1b-10p.txt`.
   - After downloading and extracting the tar archives, run `python data/reorg_sa1b.py` to reorganize the files into the required train/val structure.

3. **Teacher weights** – download `sam3.pt` (or another SAM3 checkpoint) from [Hugging Face](https://huggingface.co/facebook/sam3/tree/main) into
   `sam3_checkpoints/` and set `MODEL.RESUME` inside
   `stage1/configs/teacher/sam_vit_huge_sa1b.yaml`.
4. **Distributed launch** – both helper scripts use `torchrun`. Override
   `GPUS`, `MASTER_PORT`, `BATCH_SIZE`, etc. via environment variables or pass
   additional CLI flags directly to the underlying Python scripts.
5. **Image size** – SAM3’s ViT backbone is trained at `1008×1008`. Keep
   `DATA.IMG_SIZE` set to `1008` in both teacher and student configs (or via
   `--opts DATA.IMG_SIZE 1008`) so the rotary embeddings match; larger crops
   such as `1024` will trigger an assertion inside the teacher model.
6. **Parameter Analysis** – You can compare the parameter count of the SAM3 teacher model and the student models using the provided script:
   ```bash
   # View summary of all student models vs teacher
   PYTHONPATH=sam3 python stage1/compare_models.py --student all
   
   # View detailed breakdown of a specific student model (e.g. RepViT-M1.1)
   PYTHONPATH=sam3 python stage1/compare_models.py --student es_rv_m
   ```

   **Parameter Breakdown:**
   - **SAM3 Teacher (Total)**: 860.06M parameters.
     - **Vision Backbone**: 461.84M (Target for replacement).
     - **Language Backbone**: 353.72M (Retained).
     - **Decoder/Heads**: ~45M (Retained).

   **Student Savings (Vision Encoder Only):**
   | Student Model | Backbone | Params | vs. Teacher Encoder (461M) |
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

7. **Shape Verification** – All 9 student backbones have been verified to produce the correct embedding shape (72x72) to match the SAM3 teacher (stride 14 at 1008x1008 input).
   - **RepViT & EfficientViT**: Passed natively.
   - **TinyViT**: Required a fix in `stage1/model.py` to correctly calculate the output resolution (32x32) for the 1008px input, resolving a mismatch where the adapter expected 31x31.

### 1. Prepare Inputs

| Requirement | Notes |
|-------------|-------|
| **SA-1B dataset** | Point `DATA.DATA_PATH` to the folder that contains `images/{train,val}` and `annotations/{train,val}` (defaults to `data/sa-1b`). |
| **SAM3 checkpoint** | Download `sam3.pt` (e.g. from HuggingFace `facebook/sam3`) and set `MODEL.RESUME` in `stage1/configs/teacher/sam_vit_huge_sa1b.yaml`. |
| **Output directory** | All outputs (logs, embeddings, checkpoints) are saved under `output/`. Teacher embeddings go to `output/stage1_teacher/embeddings/`, student checkpoints to `output/stage1/<model_name>/`. |

### Step 1 — Save Teacher Embeddings

**This is a one-time forward pass** through the teacher model on the entire
SA-1B dataset. The embeddings are saved once to
`output/stage1_teacher/embeddings/`, then reused for all
student training epochs.

> **Note:** The text encoder (~354M params) is disabled during this step to reduce memory usage, as we only need image embeddings.

Use the provided launcher or run the Python entry point directly.

```bash
# Recommended helper (override CFG/DATA_PATH/OUTPUT inline as KEY=VALUE)
# Single GPU
bash stage1/scripts/save_stage1_embeddings.sh \
  CFG=stage1/configs/teacher/sam_vit_huge_sa1b.yaml \
  DATA_PATH=data/sa-1b \
  OUTPUT=output/stage1_teacher \
  GPUS=1

# Eight GPUs (default GPUS=8 so this shows it explicitly)
bash stage1/scripts/save_stage1_embeddings.sh \
  CFG=stage1/configs/teacher/sam_vit_huge_sa1b.yaml \
  DATA_PATH=data/sa-1b \
  OUTPUT=output/stage1_teacher \
  GPUS=8 \
  BATCH_SIZE=32

# Optional: verify saved blobs after the export
bash stage1/scripts/save_stage1_embeddings.sh \
  CFG=stage1/configs/teacher/sam_vit_huge_sa1b.yaml \
  DATA_PATH=data/sa-1b \
  OUTPUT=output/stage1_teacher \
  --check-saved-embed
```

**Output structure**:
```
output/stage1_teacher/
├── config.json               # Config from embedding export
├── log_rank0.txt             # Logs
└── embeddings/               # Actual teacher embeddings
    ├── rank0-keys.txt        # Image IDs
    ├── rank0-values.bin      # Embeddings (float16)
    ├── rank1-keys.txt        # (if using multiple GPUs)
    └── rank1-values.bin
```

Both commands wrap:

```
torchrun --nproc_per_node <GPUS> stage1/save_embedding_stage1.py \
  --cfg stage1/configs/teacher/sam_vit_huge_sa1b.yaml \
  --data-path data/sa-1b \
  --output output/stage1_teacher
```

The script produces sharded binary files in
`output/stage1_teacher/embeddings/`. Each record stores the
augmentation seed (first four bytes) followed by a float16 embedding grid of
shape `(EMBED_DIM, EMBED_SIZE, EMBED_SIZE)`.



### Step 2 — Train Student Encoders

Set `MODEL.BACKBONE` via the config file (see table below) and launch the
student distillation run. The helper script wires up `torchrun` with the desired
world size, batch size, and config file.

```bash
# Single GPU (override CFG/DATA_PATH/OUTPUT inline as needed)
bash stage1/scripts/train_stage1_student.sh \
  CFG=stage1/configs/es_rv_m.yaml \
  DATA_PATH=data/sa-1b \
  OUTPUT=output/stage1/repvit_m1 \
  BATCH_SIZE=4 \
  GPUS=1

# Eight GPUs (default GPUS=8, shown explicitly here)
bash stage1/scripts/train_stage1_student.sh \
  CFG=stage1/configs/es_rv_m.yaml \
  DATA_PATH=data/sa-1b \
  OUTPUT=output/stage1/repvit_m1 \
  BATCH_SIZE=32 \
  GPUS=8
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

**Example for EfficientViT-B0 (ES-EV-S):**
```bash
python stage1/convert_stage1_weights.py \
  --student-ckpt output/stage1/es_ev_s/ckpt_epoch_49.pth \
  --sam3-ckpt sam3_checkpoints/sam3.pt \
  --output output/efficient_sam3_efficientvit_b0.pt \
  --target-prefix detector.backbone.vision_backbone.trunk.model. \
  --replace-prefix detector.backbone.vision_backbone.trunk.
```

**Example for RepViT-M1.1 (ES-RV-M):**
```bash
python stage1/convert_stage1_weights.py \
  --student-ckpt output/stage1/repvit_m1/ckpt_epoch_29.pth \
  --sam3-ckpt sam3_checkpoints/sam3.pt \
  --output output/efficient_sam3_repvit_m1.pt \
  --target-prefix detector.backbone.vision_backbone.trunk.model. \
  --replace-prefix detector.backbone.vision_backbone.trunk.
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
└── efficient_sam3_efficientvit_b0.pt  # Final merged model
```

The script copies student encoder weights into the SAM3 checkpoint under
`detector.backbone.vision_backbone.trunk.model.*` and preserves all other components (prompt encoder, mask
decoder). The `--replace-prefix` argument ensures the original teacher backbone is removed.

### Helper Scripts

The `stage1/scripts` folder contains two ready-to-use launchers:

| Script | Purpose | Customisation knobs |
|--------|---------|---------------------|
| `save_stage1_embeddings.sh` | Runs `stage1/save_embedding_stage1.py` under `torchrun` for exporting teacher embeddings. | Override `CFG`, `DATA_PATH`, `OUTPUT`, `GPUS`, `MASTER_PORT`, and append additional CLI flags (e.g. `--check-saved-embed`). |
| `train_stage1_student.sh` | Launches `stage1/train_stage1.py` with a chosen student config. | Override `CFG`, `DATA_PATH`, `OUTPUT`, `BATCH_SIZE`, `GPUS`, `MASTER_PORT`, or pass through extra flags such as `--use-sync-bn`. |

Both scripts also honour `NNODES`, `NODE_RANK`, `RDZV_BACKEND`, and `RDZV_ENDPOINT`
for multi-node rendezvous. By default they run single-node training with
`torchrun --nnodes 1 --nproc_per_node ${GPUS}`.

### Files of Interest

* `stage1/model.py` — builders for student and teacher encoders (RepViT, TinyViT,
  EfficientViT wrappers + SAM3 teacher wrapper).
* `stage1/train_stage1.py` — distilled training loop (no EdgeSAM dependency).
* `stage1/save_embedding_stage1.py` — SAM3 embedding exporter and checker.
* `stage1/data/*` — SA-1B/COCO datasets plus deterministic augmentation manager.
* `stage1/convert_stage1_weights.py` — merges a Stage‑1 checkpoint with the full
  SAM3 weights (analogous to EdgeSAM Phase 1 conversion).

Customize the configs as needed (output directories, LR schedule, etc.) and run
the steps above for each backbone to reproduce the Stage‑1 model zoo.