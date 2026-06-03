# RTX 5090 SAM3 Image Encoder Distillation Smoke Run

This runbook describes how to reproduce the Stage 1 SAM3 image encoder distillation pipeline on a single RTX 5090 workstation or one L40S Slurm GPU. The goal is to validate the full pipeline end to end with a small deterministic random subset before spending multi-day compute on the larger 1% SA-1B run.

## Summary

- Hardware target: 1x RTX 5090 with 32 GB VRAM.
- Verified cluster smoke baseline: one PACE Phoenix L40S job, Slurm job `9402755`, `gpu-l40s`, QOS `embers`, completed on 2026-06-03.
- Teacher: official SAM3 image model checkpoint, using only the frozen image trunk.
- Students: ES-RV-S / RepViT-M0.9, ES-RV-M / RepViT-M1.1, and ES-RV-L / RepViT-M2.3.
- Dataset for smoke run: 1120 randomly sampled SA-1B training images, approximately 0.01% of full SA-1B. The subset is materialized once with seed `5090`; teacher export and all student runs read that same subset without a second random sampling pass.
- Loss: masked feature MSE plus masked channel-wise cosine loss.
- Output target: a merged EfficientSAM3 checkpoint with the distilled image encoder and original SAM3 heads.
- PACE scratch baseline: `/storage/scratch1/9/eliu354/efficientsam3_distill_smoke`.
- Workstation run root: choose a large local SSD path, for example `/data/efficientsam3_distill_smoke` or `$HOME/efficientsam3_distill_smoke`.

The subset manifest at `data/SA-1B-0.01P/subset_manifest.json` records the selected keys. The smoke configs use `DATA.RANDOM_SAMPLE=False` when pointed at the materialized subset, so teacher embedding export and student training use exactly the same image keys.

## 1. PACE Baseline

The PACE Phoenix validation used Slurm only for the cluster run. Submit cluster jobs with `embers`, never `inferno`:

```bash
cd /storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation
sbatch scripts/slurm_l40s_image_distill_smoke.sbatch
```

The verified PACE smoke run produced:

```text
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/efficient_sam3_repvit_s_smoke.pt
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/efficient_sam3_repvit_m_smoke.pt
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/efficient_sam3_repvit_l_smoke.pt
```

## 2. Local RTX 5090 One-Command Run

On the workstation, do not use `sbatch`, PACE partitions, or `/storage/project` paths. Start in a normal local shell where the NVIDIA driver is installed and CUDA is visible.

Choose local paths first:

```bash
export REPO_DIR="$HOME/src/EfficientSam3-Distillation"
export RUN_ROOT="/data/efficientsam3_distill_smoke"
export ENV_DIR="${RUN_ROOT}/conda_env"
mkdir -p "$(dirname "${REPO_DIR}")" "${RUN_ROOT}"
```

Clone or copy the repo to the workstation. If cloning from the remote, check out the branch that contains the smoke pipeline:

```bash
cd "$(dirname "${REPO_DIR}")"
git clone <REPO_URL> "$(basename "${REPO_DIR}")"
cd "${REPO_DIR}"
git checkout image-encoder-distill-pipeline
```

If the repo was copied manually from PACE, set `REPO_DIR` to that local copy and `cd` there.

Run the full local smoke pipeline:

```bash
cd "${REPO_DIR}"
REPO_DIR="${REPO_DIR}" \
RUN_ROOT="${RUN_ROOT}" \
ENV_DIR="${ENV_DIR}" \
bash scripts/run_image_encoder_distill_smoke.sh
```

The runner creates the environment, downloads data/checkpoints, builds the deterministic 0.01% subset, exports teacher embeddings, trains ES-RV-S, ES-RV-M, and ES-RV-L for 3 smoke epochs each, and merges one checkpoint per student size.

All heavy artifacts are written under the local `RUN_ROOT`:

```text
${RUN_ROOT}/
├── conda_env/
├── conda_pkgs/
├── cache/
├── data/
│   └── SA-1B-0.01P/
├── sam3_checkpoints/
└── output/
```

The runner sets `CLEAN_INTERMEDIATE=1` by default, so after creating `SA-1B-0.01P`, it removes the downloaded 1% tar directory and the temporary 1% reorganized dataset inside the run root. The retained dataset is the deterministic 1120-image subset plus teacher embeddings and checkpoints.

## 3. Local Environment Preflight

Before the full workstation run, the same local conda environment can be created and checked without downloading SA-1B:

```bash
cd "${REPO_DIR}"
REPO_DIR="${REPO_DIR}" RUN_ROOT="${RUN_ROOT}" ENV_DIR="${ENV_DIR}" \
  bash scripts/preflight_image_encoder_distill.sh
```

After dependencies are already installed, rerun only the checks with:

```bash
REPO_DIR="${REPO_DIR}" RUN_ROOT="${RUN_ROOT}" ENV_DIR="${ENV_DIR}" \
  PREFLIGHT_INSTALL_DEPS=0 bash scripts/preflight_image_encoder_distill.sh
```

The preflight writes logs to:

```text
${RUN_ROOT}/preflight_*.log
```

It validates dependency installation, PyTorch/core package imports, YAML parsing, config resolution for the teacher plus ES-RV-S/M/L smoke configs, and CPU construction of the three RepViT student image encoders.

## 4. Local CPU Asset Preparation

On PACE this step was submitted as a CPU Slurm job while the GPU job was pending. On the workstation, run it directly from a local shell:

```bash
cd "${REPO_DIR}"
REPO_DIR="${REPO_DIR}" RUN_ROOT="${RUN_ROOT}" ENV_DIR="${ENV_DIR}" \
  bash scripts/prepare_image_encoder_distill_assets.sh
```

This uses the same local run root, downloads `sam3.pt`, downloads the repo-provided SA-1B shard list, reorganizes it with a bounded worker count, materializes the deterministic 1120-image subset, and removes intermediate tar/reorganized data when `CLEAN_INTERMEDIATE=1`.

Logs are written to:

```text
${RUN_ROOT}/prepare_assets_*.log
```

The GPU runner reuses these assets and skips checkpoint/data preparation if they already exist.

## 5. Manual Local Environment

The one-command runner creates and repairs the local environment automatically. If creating a workstation environment manually, start with the normal Stage 1 extra:

```bash
conda create -n efficientsam3 python=3.12 -y
conda activate efficientsam3

cd "${REPO_DIR}"
pip install -U pip
pip install -e ".[stage1]"
```

If that install fails on `mmcv` or resolves a PyTorch wheel that cannot initialize CUDA on the local driver, use the same fallback dependency set that passed on PACE L40S:

```bash
pip install -e . --no-deps
pip install --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  torch==2.11.0+cu128 torchvision==0.26.0+cu128
pip install \
  "timm>=1.0.17" "numpy>=1.26.4" tqdm "ftfy==6.1.1" regex \
  "iopath>=0.1.10" typing_extensions huggingface_hub psutil \
  "decord>=0.6.0" "mmengine>=0.10.4" "pycocotools>=2.0.7" \
  "yacs>=0.1.8" "Pillow>=10.0.0" "opencv-python>=4.9.0.80" \
  "scipy>=1.10.0" "scikit-image>=0.21.0" "scikit-learn>=1.3.0" \
  "tensorboard>=2.12.0" "einops>=0.7.0" "hydra-core>=1.3.2" \
  "submitit>=1.5.1" "fvcore>=0.1.5.post20221221" \
  "fairscale>=0.4.13" pandas pyyaml segment-anything
```

Verify CUDA and PyTorch see the RTX 5090 before exporting teacher embeddings:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 1024**3, "GiB")
PY
```

Do not continue if `torch.cuda.is_available()` is `False`; fix the NVIDIA driver, CUDA-compatible PyTorch wheel, or shell environment first.

## 6. Checkpoint

Download the official SAM3 checkpoint into the local run root:

```bash
mkdir -p "${RUN_ROOT}/sam3_checkpoints"
hf download facebook/sam3 sam3.pt \
  --local-dir "${RUN_ROOT}/sam3_checkpoints"
```

`facebook/sam3` is a gated Hugging Face repository. If `HF_HOME` is pointed at a scratch cache that has not been logged in, either run `hf auth login` for that cache or export `HF_TOKEN_PATH` to an already approved token file before downloading.

If the checkpoint is downloaded manually, place it at:

```text
${RUN_ROOT}/sam3_checkpoints/sam3.pt
```

## 7. Data

For the first RTX 5090 reproduction, use the official SA-1B data source but only download the repo-provided 1% shard list. The helper then materializes 1120 randomly selected images from that available subset, matching approximately 0.01% of full SA-1B.

```bash
cd "${REPO_DIR}"
mkdir -p "${RUN_ROOT}/data"
bash data/download_sa1b.sh data/sa-1b-1p.txt "${RUN_ROOT}/data/sa-1b-1p" 4
```

As of 2026-06-02 on PACE Phoenix, the checked-in official CDN links returned HTTP `403 Forbidden`. The scratch runners therefore default to downloading the same shard names from the Hugging Face dataset mirror `ssbai/sa1b`:

```bash
cd "${REPO_DIR}"
bash data/download_sa1b_hf.sh \
  data/sa-1b-1p.txt \
  "${RUN_ROOT}/data/sa-1b-1p" \
  ssbai/sa1b
```

To force the original TSV/CDN downloader after refreshing `data/sa-1b-1p.txt`, run the one-command scripts with:

```bash
cd "${REPO_DIR}"
REPO_DIR="${REPO_DIR}" RUN_ROOT="${RUN_ROOT}" ENV_DIR="${ENV_DIR}" \
  SA1B_DOWNLOAD_BACKEND=tsv bash scripts/run_image_encoder_distill_smoke.sh
```

Reorganize the downloaded tar files into the layout expected by the Stage 1 dataloader:

```bash
cd "${RUN_ROOT}/data"
python "${REPO_DIR}/data/reorg_sa1b.py"
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
cd "${REPO_DIR}"
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

## 8. Export Teacher Image Embeddings

Start conservatively on the RTX 5090 with `BATCH_SIZE=1`. Increase to `2` only after confirming memory headroom.

```bash
cd "${REPO_DIR}"
bash stage1/scripts/save_image_embeddings.sh \
  CFG=stage1/configs/teacher/sam_vit_huge_sa1b_5090_smoke.yaml \
  DATA_PATH="${RUN_ROOT}/data/SA-1B-0.01P" \
  OUTPUT="${RUN_ROOT}/output/stage1_teacher" \
  BATCH_SIZE=1 \
  GPUS=1 \
  --opts \
    MODEL.RESUME "${RUN_ROOT}/sam3_checkpoints/sam3.pt" \
    DATA.RANDOM_SAMPLE False \
    DISTILL.TEACHER_EMBED_PATH "${RUN_ROOT}/output/stage1_teacher/embeddings"
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
${RUN_ROOT}/output/stage1_teacher/
├── config.json
├── log_rank0.txt
└── embeddings/
    ├── rank0-keys.txt
    └── rank0-values.bin
```

Optional integrity check:

```bash
cd "${REPO_DIR}"
bash stage1/scripts/save_image_embeddings.sh \
  CFG=stage1/configs/teacher/sam_vit_huge_sa1b_5090_smoke.yaml \
  DATA_PATH="${RUN_ROOT}/data/SA-1B-0.01P" \
  OUTPUT="${RUN_ROOT}/output/stage1_teacher_check" \
  BATCH_SIZE=1 \
  GPUS=1 \
  --check-saved-embed
```

## 9. Train Student Image Encoders

The one-command runner trains all three RepViT sizes by default. To run one student manually, use ES-RV-M / RepViT-M1.1 as the balanced first check:

```bash
cd "${REPO_DIR}"
bash stage1/scripts/train_image_student.sh \
  CFG=stage1/configs/es_rv_m_5090_smoke.yaml \
  DATA_PATH="${RUN_ROOT}/data/SA-1B-0.01P" \
  OUTPUT="${RUN_ROOT}/output/stage1/es_rv_m" \
  BATCH_SIZE=4 \
  GPUS=1 \
  --opts \
    DATA.RANDOM_SAMPLE False \
    DISTILL.TEACHER_EMBED_PATH "${RUN_ROOT}/output/stage1_teacher/embeddings"
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

## 10. Merge Student Encoder with SAM3 Heads

After a smoke training run finishes:

```bash
cd "${REPO_DIR}"
python stage1/convert_image_encoder_weights_stage1.py \
  --student-ckpt "${RUN_ROOT}/output/stage1/es_rv_m/ckpt_epoch_2.pth" \
  --sam3-ckpt "${RUN_ROOT}/sam3_checkpoints/sam3.pt" \
  --output "${RUN_ROOT}/output/efficient_sam3_repvit_m_smoke.pt"
```

This checkpoint keeps the original SAM3 prompt and mask heads, and replaces only the image encoder trunk with the distilled ES-RV-M student.

The one-command runner writes:

```text
${RUN_ROOT}/output/efficient_sam3_repvit_s_smoke.pt
${RUN_ROOT}/output/efficient_sam3_repvit_m_smoke.pt
${RUN_ROOT}/output/efficient_sam3_repvit_l_smoke.pt
```

## 11. Move from Smoke Run to Larger Run

Once the smoke run succeeds, keep the same architecture and increase training scale:

```bash
cd "${REPO_DIR}"
bash stage1/scripts/train_image_student.sh \
  CFG=stage1/configs/es_rv_m_5090_smoke.yaml \
  DATA_PATH="${RUN_ROOT}/data/SA-1B-0.01P" \
  OUTPUT="${RUN_ROOT}/output/stage1/es_rv_m_1120_50ep" \
  BATCH_SIZE=4 \
  GPUS=1 \
  --opts TRAIN.EPOCHS 50 TRAIN.WARMUP_EPOCHS 5
```

For a larger random subset, override both teacher export and student training with the same values:

```bash
--opts DATA.NUM_SAMPLES 10000 DATA.RANDOM_SAMPLE True DATA.SAMPLE_SEED 5090
```

The teacher embedding export must be rerun whenever `DATA.NUM_SAMPLES`, `DATA.RANDOM_SAMPLE`, `DATA.SAMPLE_SEED`, image size, or embedding shape changes.

## 12. Completion Checks

After the one-command RTX 5090 run, verify the same artifacts that the L40S baseline produced:

```bash
wc -l "${RUN_ROOT}/output/stage1_teacher/embeddings/rank0-keys.txt"
ls -lh \
  "${RUN_ROOT}/output/stage1_teacher/embeddings/rank0-values.bin" \
  "${RUN_ROOT}/output/efficient_sam3_repvit_s_smoke.pt" \
  "${RUN_ROOT}/output/efficient_sam3_repvit_m_smoke.pt" \
  "${RUN_ROOT}/output/efficient_sam3_repvit_l_smoke.pt"
tail -40 "${RUN_ROOT}"/run_*.log
```

Expected baseline from the successful L40S run:

```text
Teacher key count: 1120
Teacher values size: 11890856320 bytes
ES-RV-S merged size: 1714333907 bytes
ES-RV-M merged size: 1727107813 bytes
ES-RV-L merged size: 1786850967 bytes
```

The exact checkpoint byte sizes may differ if package versions, PyTorch serialization, or training settings change. The required completion signal is that all three `efficient_sam3_repvit_*_smoke.pt` files exist, each corresponding `stage1/es_rv_*/log_rank0.txt` reaches the final configured epoch, and the latest run log ends with all three merged checkpoint paths and `Done.`

## 13. Reporting Expected Time

Use the first 50-100 logged steps as the reliable estimate. The code reports `throughput` and `total_eta` directly in:

```text
${RUN_ROOT}/output/stage1_teacher/log_rank0.txt
${RUN_ROOT}/output/stage1/es_rv_s/log_rank0.txt
${RUN_ROOT}/output/stage1/es_rv_m/log_rank0.txt
${RUN_ROOT}/output/stage1/es_rv_l/log_rank0.txt
```

For manager reporting:

- RTX 5090 can validate the pipeline end to end on a deterministic 0.01% SA-1B smoke run across small, medium, and large RepViT image encoders.
- Full 1% SA-1B training is feasible but likely slow on one workstation because it requires large teacher embedding storage and many student training steps.
- H100/H200 access would mainly reduce teacher embedding export time, allow larger batch sizes, and shorten the multi-epoch student distillation wall-clock time.
