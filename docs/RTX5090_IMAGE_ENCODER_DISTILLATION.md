# RTX 5090 SAM3 Image Encoder Distillation Smoke Run

This runbook describes how to reproduce the Stage 1 SAM3 image encoder distillation pipeline on a single RTX 5090 workstation or one L40S Slurm GPU. The goal is to validate the full pipeline end to end with a small deterministic random subset before spending multi-day compute on the larger 1% SA-1B run.

## Summary

- Hardware target: 1x RTX 5090 with 32 GB VRAM.
- Teacher: official SAM3 image model checkpoint, using only the frozen image trunk.
- Students: ES-RV-S / RepViT-M0.9, ES-RV-M / RepViT-M1.1, and ES-RV-L / RepViT-M2.3.
- Dataset for smoke run: 1120 randomly sampled SA-1B training images, approximately 0.01% of full SA-1B. The subset is materialized once with seed `5090`; teacher export and all student runs read that same subset without a second random sampling pass.
- Loss: masked feature MSE plus masked channel-wise cosine loss.
- Output target: a merged EfficientSAM3 checkpoint with the distilled image encoder and original SAM3 heads.
- Scratch run root: `/storage/scratch1/9/eliu354/efficientsam3_distill_smoke`.

The subset manifest at `data/SA-1B-0.01P/subset_manifest.json` records the selected keys. The smoke configs use `DATA.RANDOM_SAMPLE=False` when pointed at the materialized subset, so teacher embedding export and student training use exactly the same image keys.

## 1. One-Command L40S Run

The repo includes a scratch-based runner that creates the environment, downloads data/checkpoints, builds the deterministic 0.01% subset, exports teacher embeddings, trains ES-RV-S, ES-RV-M, and ES-RV-L for 3 smoke epochs each, and merges one checkpoint per student size.

Submit to one L40S GPU:

```bash
cd /storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation
sbatch scripts/slurm_l40s_image_distill_smoke.sbatch
```

Or run interactively on a workstation/GPU shell:

```bash
cd /storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation
bash scripts/run_image_encoder_distill_smoke.sh
```

All heavy artifacts are written under:

```text
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/
├── conda_env/
├── conda_pkgs/
├── cache/
├── data/
│   └── SA-1B-0.01P/
├── sam3_checkpoints/
└── output/
```

The runner sets `CLEAN_INTERMEDIATE=1` by default, so after creating `SA-1B-0.01P`, it removes the downloaded 1% tar directory and the temporary 1% reorganized dataset inside the run root. The retained dataset is the deterministic 1120-image subset plus teacher embeddings and checkpoints.

## 2. Scratch Environment Preflight

Before a GPU allocation starts, the same scratch conda environment can be created and checked without downloading SA-1B or running CUDA:

```bash
cd /storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation
bash scripts/preflight_image_encoder_distill.sh
```

After dependencies are already installed, rerun only the checks with:

```bash
PREFLIGHT_INSTALL_DEPS=0 bash scripts/preflight_image_encoder_distill.sh
```

The preflight writes logs to:

```text
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/preflight_*.log
```

It validates dependency installation, PyTorch/core package imports, YAML parsing, config resolution for the teacher plus ES-RV-S/M/L smoke configs, and CPU construction of the three RepViT student image encoders.

## 3. Manual Environment

Create the environment from scratch:

```bash
conda create -n efficientsam3 python=3.12 -y
conda activate efficientsam3

cd /storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation
pip install -U pip
pip install -e ".[stage1]"
```

Verify CUDA and PyTorch see the RTX 5090:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 1024**3, "GiB")
PY
```

## 4. Checkpoint

Download the official SAM3 checkpoint into scratch:

```bash
mkdir -p /storage/scratch1/9/eliu354/efficientsam3_distill_smoke/sam3_checkpoints
huggingface-cli download facebook/sam3 sam3.pt \
  --local-dir /storage/scratch1/9/eliu354/efficientsam3_distill_smoke/sam3_checkpoints \
  --local-dir-use-symlinks False
```

If the checkpoint is downloaded manually, place it at:

```text
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/sam3_checkpoints/sam3.pt
```

## 5. Data

For the first RTX 5090 reproduction, use the official SA-1B data source but only download the repo-provided 1% shard list. The helper then materializes 1120 randomly selected images from that available subset, matching approximately 0.01% of full SA-1B.

```bash
RUN_ROOT=/storage/scratch1/9/eliu354/efficientsam3_distill_smoke
mkdir -p "${RUN_ROOT}/data"
bash data/download_sa1b.sh data/sa-1b-1p.txt "${RUN_ROOT}/data/sa-1b-1p" 4
```

Reorganize the downloaded tar files into the layout expected by the Stage 1 dataloader:

```bash
cd "${RUN_ROOT}/data"
python /storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation/data/reorg_sa1b.py
```

The training data path must contain:

```text
images/train/*.jpg
annotations/train/*.json
images/val/*.jpg
annotations/val/*.json
```

Create the deterministic 0.01% subset:

```bash
cd /storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation
python data/create_sa1b_subset.py \
  --source "${RUN_ROOT}/data/SA-1B-1P" \
  --output "${RUN_ROOT}/data/SA-1B-0.01P" \
  --num-samples 1120 \
  --seed 5090 \
  --mode hardlink
```

Expected storage for the smoke run:

- SA-1B 1% raw/extracted files: plan for roughly 250-300 GB if keeping both tar files and extracted data.
- Teacher embeddings for 1120 images: about 11.3 GiB, because each embedding is `1024 x 72 x 72` fp16.
- Checkpoints and logs: usually a few GB for the smoke run.

## 6. Export Teacher Image Embeddings

Start conservatively on the RTX 5090 with `BATCH_SIZE=1`. Increase to `2` only after confirming memory headroom.

```bash
bash stage1/scripts/save_image_embeddings.sh \
  CFG=stage1/configs/teacher/sam_vit_huge_sa1b_5090_smoke.yaml \
  DATA_PATH=/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/data/SA-1B-0.01P \
  OUTPUT=/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/stage1_teacher \
  BATCH_SIZE=1 \
  GPUS=1 \
  --opts \
    MODEL.RESUME /storage/scratch1/9/eliu354/efficientsam3_distill_smoke/sam3_checkpoints/sam3.pt \
    DATA.RANDOM_SAMPLE False \
    DISTILL.TEACHER_EMBED_PATH /storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/stage1_teacher/embeddings
```

The log now reports:

- sample count
- embedding shape
- estimated embedding storage
- image throughput
- total ETA
- peak GPU memory

Expected output:

```text
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/stage1_teacher/
├── config.json
├── log_rank0.txt
└── embeddings/
    ├── rank0-keys.txt
    └── rank0-values.bin
```

Optional integrity check:

```bash
bash stage1/scripts/save_image_embeddings.sh \
  CFG=stage1/configs/teacher/sam_vit_huge_sa1b_5090_smoke.yaml \
  DATA_PATH=data/sa-1b \
  OUTPUT=output/rtx5090_smoke/stage1_teacher_check \
  BATCH_SIZE=1 \
  GPUS=1 \
  --check-saved-embed
```

## 7. Train Student Image Encoders

The one-command runner trains all three RepViT sizes by default. To run one student manually, use ES-RV-M / RepViT-M1.1 as the balanced first check:

```bash
bash stage1/scripts/train_image_student.sh \
  CFG=stage1/configs/es_rv_m_5090_smoke.yaml \
  DATA_PATH=/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/data/SA-1B-0.01P \
  OUTPUT=/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/stage1/es_rv_m \
  BATCH_SIZE=4 \
  GPUS=1 \
  --opts \
    DATA.RANDOM_SAMPLE False \
    DISTILL.TEACHER_EMBED_PATH /storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/stage1_teacher/embeddings
```

The smoke config trains for 3 epochs. The log now reports:

- total samples
- steps per epoch
- effective batch size
- image throughput
- current epoch ETA
- total training ETA across all remaining epochs
- peak GPU memory

If the workstation has memory headroom, try `BATCH_SIZE=8` for ES-RV-S/M. If it OOMs, use `BATCH_SIZE=2` and keep the run moving. ES-RV-L defaults to `BATCH_SIZE=2`.

To manually test the other two sizes, change `CFG`, `OUTPUT`, and the batch size:

```text
ES-RV-S: CFG=stage1/configs/es_rv_s_5090_smoke.yaml, OUTPUT=.../stage1/es_rv_s, BATCH_SIZE=4
ES-RV-L: CFG=stage1/configs/es_rv_l_5090_smoke.yaml, OUTPUT=.../stage1/es_rv_l, BATCH_SIZE=2
```

## 8. Merge Student Encoder with SAM3 Heads

After a smoke training run finishes:

```bash
python stage1/convert_image_encoder_weights_stage1.py \
  --student-ckpt /storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/stage1/es_rv_m/ckpt_epoch_2.pth \
  --sam3-ckpt /storage/scratch1/9/eliu354/efficientsam3_distill_smoke/sam3_checkpoints/sam3.pt \
  --output /storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/efficient_sam3_repvit_m_smoke.pt
```

This checkpoint keeps the original SAM3 prompt and mask heads, and replaces only the image encoder trunk with the distilled ES-RV-M student.

The one-command runner writes:

```text
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/efficient_sam3_repvit_s_smoke.pt
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/efficient_sam3_repvit_m_smoke.pt
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/efficient_sam3_repvit_l_smoke.pt
```

## 9. Move from Smoke Run to Larger Run

Once the smoke run succeeds, keep the same architecture and increase training scale:

```bash
bash stage1/scripts/train_image_student.sh \
  CFG=stage1/configs/es_rv_m_5090_smoke.yaml \
  DATA_PATH=/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/data/SA-1B-0.01P \
  OUTPUT=/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/stage1/es_rv_m_1120_50ep \
  BATCH_SIZE=4 \
  GPUS=1 \
  --opts TRAIN.EPOCHS 50 TRAIN.WARMUP_EPOCHS 5
```

For a larger random subset, override both teacher export and student training with the same values:

```bash
--opts DATA.NUM_SAMPLES 10000 DATA.RANDOM_SAMPLE True DATA.SAMPLE_SEED 5090
```

The teacher embedding export must be rerun whenever `DATA.NUM_SAMPLES`, `DATA.RANDOM_SAMPLE`, `DATA.SAMPLE_SEED`, image size, or embedding shape changes.

## 10. Reporting Expected Time

Use the first 50-100 logged steps as the reliable estimate. The code reports `throughput` and `total_eta` directly in:

```text
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/stage1_teacher/log_rank0.txt
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/stage1/es_rv_s/log_rank0.txt
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/stage1/es_rv_m/log_rank0.txt
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/stage1/es_rv_l/log_rank0.txt
```

For manager reporting:

- RTX 5090 can validate the pipeline end to end on a deterministic 0.01% SA-1B smoke run across small, medium, and large RepViT image encoders.
- Full 1% SA-1B training is feasible but likely slow on one workstation because it requires large teacher embedding storage and many student training steps.
- H100/H200 access would mainly reduce teacher embedding export time, allow larger batch sizes, and shorten the multi-epoch student distillation wall-clock time.
