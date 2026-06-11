## Stage 3 — EfficientSAM3 Joint Encoder Fine-Tuning

Stage 3 fine-tunes the **Stage 1 student image encoder** and **Stage 1 student
text encoder** jointly inside the full SAM3 image model, while keeping the
decoder, geometry encoder, and scoring head frozen. The pipeline has four
discrete phases: 1) merge Stage 1 student encoders with SAM3 teacher weights,
2) sanity-check the merged model on GPU, 3) fine-tune the encoders on a
mixed text-grounded detection dataset, and 4) merge the trained encoder
checkpoint back into a full model for evaluation.

### Prerequisites

1. **Environment** – follow the root [Installation](README.md#installation) guide to
   create/activate the `efficientsam3` Conda environment and run
   `pip install -e ".[stage1]"`.
2. **Stage 1 checkpoints**:
   - Image encoder checkpoints under `output/stage1_image_2p/` (e.g. `es_tv_m/ckpt_epoch_49.pth`).
   - Text encoder checkpoints under `output/stage1_text/` (e.g. `mobileclip_s1_5dataset_ctx16_fixed/ckpt_epoch_79.pth`).
3. **SAM3 teacher weights** – download `sam3.pt` from
   [Hugging Face](https://huggingface.co/facebook/sam3/tree/main) into
   `sam3_checkpoints/`.
4. **Datasets** – prepare the following data roots:

| Dataset | Path | Notes |
|---------|------|-------|
| **COCO** | `data/coco` | `annotations/instances_{train,val}2017.json` + `images/{train,val}2017/` |
| **LVIS** | `data/lvis` | `annotations/lvis_v1_{train,val}.json` + `images/` (symlinked from COCO) |
| **ODINW** | `data/odinw` | Roboflow-style COCO JSON per subset (`train/_annotations.coco.json`) |
| **RF100-VL** | `data/rf100-vl` | Roboflow-100 with COCO JSON per subset |
| **RefCOCO** (optional) | `data/refcoco` | Parquet shards (`train-*.parquet`, `val-*.parquet`) |

If your local paths differ, update `paths.*` in
`sam3/sam3/train/configs/stage3/mixed/stage3_mixed_local_train.yaml`.

### 1. Prepare Inputs

| Requirement | Notes |
|-------------|-------|
| **Stage 1 image encoder** | A trained student checkpoint (e.g. `output/stage1_image_2p/es_tv_m/ckpt_epoch_49.pth`). |
| **Stage 1 text encoder** | A trained student checkpoint (e.g. `output/stage1_text/mobileclip_s1_5dataset_ctx16_fixed/ckpt_epoch_79.pth`). |
| **SAM3 checkpoint** | `sam3_checkpoints/sam3.pt` — the full teacher model. |
| **BPE vocabulary** | `sam3/assets/bpe_simple_vocab_16e6.txt.gz` (shipped in-tree). |
| **Output directory** | All outputs (merged checkpoints, training logs, final models) go under `output/`. |

### Step 1 — Merge Stage 1 Encoders

Use the Stage 1 merge utility to splice both student encoders into a single
SAM3-format checkpoint. This replaces the teacher vision and text backbone
weights with the student weights while keeping all other SAM3 components
(geometry encoder, transformer decoder, scoring head) intact.

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

The merged checkpoint at
`output/efficient_sam3_stage3_es_tv_m_mobileclip_s1_ctx16.pth` contains the
full model (student encoders + frozen SAM3 heads) and is used as the starting
point for Stage 3 training.

**Supported encoder combinations:**

| Image Student | Text Student | `--image-model-name` | `--text-model-name` |
|---------------|--------------|----------------------|---------------------|
| ES-TV-S | MobileCLIP-S0 | `tinyvit_5m` | `mobileclip_s0` |
| ES-TV-M | MobileCLIP-S1 | `tinyvit_11m` | `mobileclip_s1` |
| ES-TV-L | MobileCLIP2-L | `tinyvit_21m` | `mobileclip2_l` |
| ES-RV-S | MobileCLIP-S0 | `repvit_m0_9` | `mobileclip_s0` |
| ES-RV-M | MobileCLIP-S1 | `repvit_m1_1` | `mobileclip_s1` |
| ES-RV-L | MobileCLIP2-L | `repvit_m2_3` | `mobileclip2_l` |
| ES-EV-S | MobileCLIP-S0 | `efficientvit_b0` | `mobileclip_s0` |
| ES-EV-M | MobileCLIP-S1 | `efficientvit_b1` | `mobileclip_s1` |
| ES-EV-L | MobileCLIP2-L | `efficientvit_b2` | `mobileclip2_l` |

Any image student can be paired with any text student; the table above shows
the default pairings.

### Step 2 — GPU Sanity Check (Recommended)

Before launching a long training run, verify that the merged checkpoint loads
correctly, parameters are frozen/unfrozen as expected, gradients flow only
through the student encoders, and encoder outputs are reasonable.

**Run locally (single GPU):**

```bash
PYTHONPATH=sam3:. python stage3/sanity_check_gpu.py \
  --merged-ckpt output/efficient_sam3_stage3_es_tv_m_mobileclip_s1_ctx16.pth \
  --sam3-ckpt   sam3_checkpoints/sam3.pt
```

**Run via Slurm:**

```bash
bash scripts/sanity_check_gpu.sh
```

The sanity check verifies:

1. **Model loading** — the merged checkpoint loads into the EfficientSAM3 model on GPU.
2. **Freeze state** — only `backbone.vision_backbone.trunk.*` and `backbone.language_backbone.*` are trainable; all other parameters are frozen.
3. **Vision forward** — the student image encoder produces 3 FPN levels with correct shapes and gradients flow to vision encoder parameters only.
4. **Text forward** — the student text encoder produces language features with `d_model=256` and gradients flow to text encoder parameters only.
5. **Teacher comparison** — encoder outputs are compared against the SAM3 teacher via cosine similarity and RMSE (non-trivial differences are expected since the student is distilled, not identical).
6. **Data pipeline** — basic check that COCO annotation files exist.

To skip the teacher comparison (saves ~30s and ~5GB VRAM):

```bash
SKIP_TEACHER=1 bash scripts/sanity_check_gpu.sh
```

### Step 3 — Smoke Test (Recommended)

Run a short 2-epoch training job to verify the full pipeline end-to-end before
launching a long run. This uses tiny dataset subsets (~500 samples per source).

```bash
bash scripts/train_stage3_smoketest.sh
```

Expected behavior: training loss should decrease over the 2 epochs (~140 → ~120
for `train_all_loss`), validation should run after each epoch, and the job
should complete in under 30 minutes on 4 GPUs.

### Step 4 — Train with the SAM3 Trainer

Stage 3 uses the original SAM3 Hydra trainer through the Stage 3 entry point.

**Local (4 GPUs, 30 epochs):**

```bash
PYTHONPATH=sam3:. python stage3/train_stage3.py \
  -c configs/stage3/mixed/stage3_mixed_h200_30ep_train.yaml \
  --use-cluster 0 \
  --num-gpus 4
```

**Slurm (recommended):**

```bash
sbatch scripts/train_stage3_h200.sh
```

The reusable local runner that the Slurm launcher invokes is:

```bash
stage3/scripts/train_stage3_mixed.sh \
  CFG=configs/stage3/mixed/stage3_mixed_h200_30ep_train.yaml \
  GPUS=4
```

**Resume an interrupted run:**

```bash
# Via the local runner:
stage3/scripts/train_stage3_mixed.sh \
  CFG=configs/stage3/mixed/stage3_mixed_h200_30ep_train.yaml \
  GPUS=4 \
  RESUME_FROM=output/stage3/mixed_h200_30ep_es_tv_m_mc_s1_ctx16/checkpoints/checkpoint.pt

# Or via Slurm:
RESUME_FROM=output/stage3/.../checkpoints/checkpoint.pt \
  sbatch scripts/train_stage3_h200.sh
```

**What happens inside the model:**

- Loads the merged Stage 1 checkpoint through `build_efficientsam3_image_model`.
- Freezes every parameter by default.
- Unfreezes only:
  - `backbone.vision_backbone.trunk.*` (student image encoder)
  - `backbone.language_backbone.*` (student text encoder)
- Keeps frozen SAM3 modules in eval mode during training (dropout/batchnorm
  stay frozen) while trainable encoder modules are in train mode.
- Trains on detection + grounding losses (bbox, GIoU, CE, presence) with
  box-conditioned text queries (GT boxes jittered by `RandomizeInputBbox`).
- Validates on held-out splits at configurable epoch intervals.
- Saves checkpoints periodically (encoder weights only — see
  [Checkpoint Format](#checkpoint-format) below).

**Key training hyperparameters** (from `stage3_mixed_local_train.yaml`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lr_vision_backbone` | `2.5e-5` | After 0.1× `lr_scale` |
| `lr_language_backbone` | `5e-6` | After 0.1× `lr_scale` |
| `lr_transformer` | `8e-5` | After 0.1× `lr_scale` (frozen, no effect) |
| `wd` | `0.1` | Weight decay (bias and LayerNorm excluded) |
| `resolution` | `1008` | Input image resolution |
| `train_batch_size` | `3` (local) / `4` (H200) | Per-GPU batch size |
| `max_data_epochs` | `30` | Total training epochs |
| Scheduler | `InverseSquareRootParamScheduler` | With warmup=20, cooldown=20, timescale=20 |
| AMP | `bfloat16` | Mixed precision training |

### Step 5 — Merge Checkpoint for Evaluation

Stage 3 training only saves the encoder weights to reduce checkpoint size.
To produce a full checkpoint for evaluation, graft the trained encoder weights
back onto the original merged checkpoint:

```bash
python stage3/merge_stage3_checkpoint_for_eval.py \
  --stage3-ckpt  output/stage3/mixed_h200_30ep_es_tv_m_mc_s1_ctx16/checkpoints/checkpoint.pt \
  --base-ckpt    output/efficient_sam3_stage3_es_tv_m_mobileclip_s1_ctx16.pth \
  --output       output/efficient_sam3_stage3_finetuned_es_tv_m_mc_s1.pth
```

The output file has the same key structure as the input merged checkpoint and
can be loaded directly by `build_efficientsam3_image_model`.

### Step 6 — Evaluation

Use the merged evaluation checkpoint for inference or benchmarking:

```python
from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_efficientsam3_image_model(
    checkpoint_path="output/efficient_sam3_stage3_finetuned_es_tv_m_mc_s1.pth",
    backbone_type="tinyvit",
    model_name="11m",
    text_encoder_type="MobileCLIP-S1",
    text_encoder_context_length=16,
    text_encoder_pos_embed_table_size=16,
    interpolate_pos_embed=False,
    enable_segmentation=True,
    enable_inst_interactivity=False,
)

processor = Sam3Processor(model)
inference_state = processor.set_image(image)
inference_state = processor.set_text_prompt(prompt="dog", state=inference_state)
masks = inference_state["masks"]
scores = inference_state["scores"]
```

For COCO evaluation:

```bash
python eval/eval_coco.py \
  --coco_root data/coco \
  --output_dir output \
  --checkpoint output/efficient_sam3_stage3_finetuned_es_tv_m_mc_s1.pth \
  --backbone_type tinyvit \
  --model_name 11m \
  --text_encoder_type MobileCLIP-S1
```

### Checkpoint Format

Stage 3 uses a **partial checkpoint** strategy to save disk space and avoid
redundantly storing frozen weights that never change.

**What gets saved** (trainable encoder weights):

- `backbone.vision_backbone.trunk.*`
- `backbone.language_backbone.*`

**What gets skipped** (frozen, configured via `skip_saving_parameters`):

- `backbone.vision_backbone.convs.*` (FPN neck convolutions)
- `geometry_encoder.*`
- `transformer.*` (decoder)
- `dot_prod_scoring.*`

| Checkpoint | Keys | Size | Purpose |
|------------|------|------|---------|
| Merged input (`.pth`) | All model keys | ~490MB | Full model for loading |
| Trainer output (`checkpoint.pt`) | Encoder keys + optimizer/scheduler state | ~340MB | Training resumption |
| Eval output (`.pth`) | All model keys (encoders updated) | ~490MB | Inference/benchmarking |

When resuming training from a partial checkpoint, the trainer loads the encoder
weights from `checkpoint.pt` and re-initializes the frozen weights from the
model definition (which loads them from the merged checkpoint at model build time).

### Step 7 — Geometry-Aware Fine-Tuning (Recommended)

Stage 3 supports geometry-aware training to keep the student image encoder's
FPN features compatible with the frozen SAM3 geometry encoder. This augments
text queries with geometry-only (box) queries so the geometry encoder's
cross-attention provides gradient signal back to the trainable image encoder.

**Smoke test:**

```bash
sbatch scripts/train_stage3_geo_smoketest.sh
```

**Production training (single node, 24h):**

```bash
sbatch scripts/train_stage3_geo_h200_1node.sh
```

**Production training (multi-node, 4 nodes × 4 GPUs):**

```bash
sbatch scripts/train_stage3_geo_h200_multinode.sh
```

The geometry training uses the `srun`-based DDP entry point
(`stage3/train_stage3_srun.py`) which maps Slurm environment variables to
PyTorch DDP variables. Auto-resume is built in — resubmitting the same script
will automatically resume from the latest checkpoint.

**How it works:**

1. `AddGeometricQueries` (in `stage3/transforms/geometry_sampling.py`) injects
   geometry-only queries with `query_text="geometric"` targeting random objects
   with valid masks (probability 50%, max 4 per image).
2. `RandomGeometricInputsAPI` (existing SAM3 transform) samples a noised
   bounding box from each object's GT mask for these queries.
3. The frozen geometry encoder processes box prompts via cross-attention with
   the trainable image encoder's FPN features, providing gradient signal.
4. No architecture changes needed — only the data pipeline is modified.

**Memory notes:** The decoder's attention scales quadratically with query count.
Geometry configs use `max_find_queries_per_img=40` (reduced from 64) and
`max_geo_queries=4` to keep peak memory under 50GB on H200.

### Design Decisions

- **Encoder-only fine-tuning**: Stage 3 keeps the SAM3 decoder, geometry
  encoder, and scoring head frozen. This is the cleanest starting point —
  the student encoders learn to produce features that work with the existing
  SAM3 heads, without risking catastrophic forgetting of the decoder's
  pre-trained capabilities.

- **Geometry-aware training**: The frozen geometry encoder's cross-attention
  over FPN features creates an implicit dependency on image feature quality.
  By training with geometry prompts alongside text prompts, the student image
  encoder learns features compatible with both pathways.

- **Detection losses only**: Segmentation is disabled (`enable_segmentation=False`)
  to keep per-sample GPU memory at ~18GB. The detection + grounding losses
  (bbox L1, GIoU, focal CE, presence) provide sufficient training signal for
  encoder adaptation. Segmentation can be enabled for future experiments by
  setting `enable_segmentation: true` in the config and adding mask losses.

- **Partial checkpointing**: Only trainable encoder weights are saved to avoid
  wasting disk space on frozen parameters that never change. The
  `merge_stage3_checkpoint_for_eval.py` script reconstructs the full model
  for evaluation.

- **Consistent with earlier stages**: `stage1/` owns encoder distillation,
  `stage1_geometry_finetune/` owns prompt-aware image refinement, and `stage3/`
  owns joint encoder fine-tuning inside the full SAM3 model.

### Config Reference

All configs live under `sam3/sam3/train/configs/stage3/mixed/`.

| Config | Epochs | Batch | Notes |
|--------|--------|-------|-------|
| `stage3_mixed_local_train.yaml` | 30 | 3 | Base config with all paths and hyperparameters |
| `stage3_mixed_smoketest_train.yaml` | 2 | 2 | Tiny subsets (~500 samples) for quick validation |
| `stage3_mixed_h200_30ep_train.yaml` | 30 | 4 | H200 production run (text-only) |
| `stage3_mixed_geo_smoketest_train.yaml` | 2 | 2 | Geometry-enabled smoke test |
| `stage3_mixed_geo_h200_train.yaml` | 10 | 2 | Geometry-enabled production (24h) |

### Helper Scripts

| Script | Purpose | Key overrides |
|--------|---------|---------------|
| `stage3/scripts/train_stage3_mixed.sh` | Local training launcher (invoked by Slurm jobs) | `CFG`, `GPUS`, `USE_CLUSTER`, `MASTER_PORT`, `RESUME_FROM` |
| `stage3/train_stage3_srun.py` | `srun`-based DDP entry point for multi-node | Maps `SLURM_PROCID`→`RANK`, etc. |
| `scripts/train_stage3_h200.sh` | Slurm launcher for text-only H200 training | `CONFIG_NAME`, `GPUS`, `TIME_LIMIT`, `RESUME_FROM` |
| `scripts/train_stage3_geo_smoketest.sh` | Slurm launcher for geometry smoketest | 1 node, 4 GPUs, 1h |
| `scripts/train_stage3_geo_h200_1node.sh` | Slurm launcher for geometry production (1 node) | 4 GPUs, 24h, auto-resume |
| `scripts/train_stage3_geo_h200_multinode.sh` | Slurm launcher for geometry production (4 nodes) | 16 GPUs, 24h, auto-resume |

### Repository Layout

```text
stage3/
├── __init__.py
├── model.py                          # build_stage3_model: freeze/unfreeze + train mode patching
├── train_stage3.py                   # Hydra entry point (calls sam3.train.train.main)
├── train_stage3_srun.py              # srun-based DDP entry point for multi-node training
├── sanity_check_gpu.py               # GPU verification: loading, freeze, forward, teacher comparison
├── sanity_check_merge.py             # CPU key-alignment verification
├── merge_stage3_checkpoint_for_eval.py  # Graft trained encoders back into full checkpoint
├── verify_checkpoint_integrity.py    # Full checkpoint integrity verification
├── data/
│   ├── __init__.py
│   ├── mixed_text_mask_dataset.py    # Multi-source dataset (COCO, LVIS, ODINW, RF100-VL, RefCOCO)
│   └── sa1b_pseudo_labels.py         # SA-1B pseudo-label dataset (experimental)
├── transforms/
│   ├── __init__.py
│   └── geometry_sampling.py          # AddGeometricQueries: inject geometry-only box queries
├── data_engine/
│   ├── __init__.py
│   ├── annotations.py                # SA-1B annotation parsing
│   ├── audit.py                      # Rendered audit of pseudo labels
│   ├── build_manifest.py             # Grouped JSONL manifest builder
│   ├── generate.py                   # VLM-based pseudo-label generation
│   └── visualize_sa1b.py             # SA-1B visualization utilities
└── scripts/
    └── train_stage3_mixed.sh         # Reusable local training runner

scripts/
├── train_stage3_h200.sh              # Slurm launcher: H200 training (text-only)
├── train_stage3_smoketest.sh         # Slurm launcher: text-only smoke test
├── train_stage3_geo_smoketest.sh     # Slurm launcher: geometry-enabled smoke test
├── train_stage3_geo_h200_1node.sh    # Slurm launcher: geometry production (1 node)
├── train_stage3_geo_h200_multinode.sh # Slurm launcher: geometry production (4 nodes)
└── sanity_check_gpu.sh               # Slurm launcher: GPU sanity check

sam3/sam3/train/configs/stage3/mixed/
├── stage3_mixed_local_train.yaml                 # Base config
├── stage3_mixed_smoketest_train.yaml             # 2-epoch smoke test (text-only)
├── stage3_mixed_geo_smoketest_train.yaml         # 2-epoch smoke test (geometry-enabled)
├── stage3_mixed_geo_h200_train.yaml              # Geometry production (10 epochs, 24h)
├── stage3_mixed_h200_30ep_train.yaml             # H200 production (text-only)
├── stage3_mixed_stable_30ep_train.yaml           # Stable 30-epoch
├── stage3_mixed_h200_30ep_epochckpt_train.yaml   # H200, COCO+LVIS only
└── stage3_mixed_h200_30ep_fullft_train.yaml      # Full-model fine-tuning
```

### Output Structure

```text
output/
├── efficient_sam3_stage3_es_tv_m_mobileclip_s1_ctx16.pth   # Step 1: merged checkpoint
├── efficient_sam3_stage3_finetuned_es_tv_m_mc_s1.pth       # Step 5: eval checkpoint (text-only)
├── efficient_sam3_stage3_geo_finetuned_es_tv_m_mc_s1.pth   # Step 7: eval checkpoint (geometry)
└── stage3/
    ├── mixed_h200_30ep_es_tv_m_mc_s1_ctx16/               # Text-only training
    │   ├── config.yaml
    │   ├── checkpoints/
    │   │   ├── checkpoint.pt           # Latest (rolling)
    │   │   └── checkpoint_N.pt         # Per-epoch
    │   ├── logs/
    │   └── tensorboard/
    └── geo_h200_es_tv_m_mc_s1_ctx16/                      # Geometry-enabled training
        ├── config.yaml
        ├── checkpoints/
        │   ├── checkpoint.pt
        │   └── checkpoint_N.pt
        ├── logs/
        └── tensorboard/
```

### Notes

- The first Stage 3 experiment uses **ES-TV-M** (TinyViT-11M) + **MobileCLIP-S1**
  as the default pair. Other combinations follow the same pipeline with
  different checkpoint paths and config `backbone_type`/`model_name` values.
- For multi-node training, use the `srun`-based scripts which handle
  `MASTER_ADDR`/`MASTER_PORT` automatically from Slurm environment variables.
- The `stage3/data_engine/` directory contains experimental tools for generating
  SA-1B pseudo text-mask labels via VLMs. These are not wired into the default
  training config.
