# L40S Image Encoder Distillation Smoke Run Record

Date: 2026-06-02
Repo: `/storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation`
Branch: `image-encoder-distill-pipeline`
Pipeline commit: `8abc6e3` (`Add scratch image encoder distillation smoke pipeline`)
Preflight commit: `703862c` (`Add scratch preflight for image distillation`)
Scratch root: `/storage/scratch1/9/eliu354/efficientsam3_distill_smoke`

## Objective

Run the SAM3 Stage 1 image encoder distillation pipeline end to end on one GPU, using a reproducible 0.01% SA-1B subset and three RepViT student image encoder sizes: ES-RV-S, ES-RV-M, and ES-RV-L.

## Configuration

- Slurm partition: `gpu-l40s`
- GPU request: `gres/gpu:l40s:1`
- Account: `gts-agarg35`
- QOS: `embers`
- Time limit: `08:00:00`
- CPU request: `4`
- Memory request: `96G`
- Dataset target: `1120` SA-1B train image/annotation pairs
- Subset seed: `5090`
- Teacher batch size: `1`
- Student batch sizes: ES-RV-S `4`, ES-RV-M `4`, ES-RV-L `2`
- Student epochs: `3`
- Student specs: `es_rv_s`, `es_rv_m`, `es_rv_l`
- Runner: `scripts/run_image_encoder_distill_smoke.sh`
- Preflight: `scripts/preflight_image_encoder_distill.sh`
- CPU asset prep: `scripts/prepare_image_encoder_distill_assets.sh`
- Slurm script: `scripts/slurm_l40s_image_distill_smoke.sbatch`

## Scratch Layout

The runner writes heavy artifacts only under scratch:

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

`CLEAN_INTERMEDIATE=1` is enabled, so the temporary 1% SA-1B tar/reorganized data is removed after `SA-1B-0.01P` is created.

## Submitted Job

Submission command:

```bash
sbatch scripts/slurm_l40s_image_distill_smoke.sbatch
```

Submitted job:

```text
Job ID: 9400333
Initial state: PENDING
Reason: Priority
```

The job was accepted by Slurm after adding the required account/QOS and reducing the CPU request to satisfy the `gpu-l40s` CPU:GPU policy. The job was still pending when the runner was extended from one ES-RV-M student to the ES-RV-S/M/L smoke matrix; because the Slurm script calls the repo runner at execution time, the pending job will use the current branch worktree if it starts before further edits.

## Monitoring

Check scheduler state:

```bash
squeue -j 9400333 -o '%i %.12P %.20j %.8T %.10M %.20R'
```

Check Slurm stdout from the repo directory:

```bash
tail -f sam3_img_smoke-9400333.out
```

Check run logs after the job starts:

```bash
ls -lt /storage/scratch1/9/eliu354/efficientsam3_distill_smoke/run_*.log
tail -f /storage/scratch1/9/eliu354/efficientsam3_distill_smoke/run_*.log
```

Scratch environment preflight:

```bash
bash scripts/preflight_image_encoder_distill.sh
PREFLIGHT_INSTALL_DEPS=0 bash scripts/preflight_image_encoder_distill.sh
ls -lt /storage/scratch1/9/eliu354/efficientsam3_distill_smoke/preflight_*.log
```

## Preflight Result

Completed on 2026-06-02 before the L40S allocation started.

Logs:

```text
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/preflight_20260602_185803.log
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/preflight_20260602_190650.log
```

Evidence:

- Scratch conda env created at `/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/conda_env`.
- Full `.[stage1]` install attempted first and failed at `mmcv` metadata build; fallback image-distillation dependency set completed successfully.
- PyTorch import works in the scratch env: `torch 2.12.0+cu130`; login shell reports `cuda_available False`, expected outside a GPU allocation.
- Core imports passed: `torchvision 0.27.0+cu130`, `timm 1.0.27`, `cv2 4.13.0`, `mmengine 0.10.7`.
- Teacher smoke config parsed with `samples=1120`, `random_sample=False`, `batch=1`.
- ES-RV-S config parsed and model construction passed: `repvit_m0_9`, `batch=4`, `epochs=3`, `params=14551264`.
- ES-RV-M config parsed and model construction passed: `repvit_m1_1`, `batch=4`, `epochs=3`, `params=17739408`.
- ES-RV-L config parsed and model construction passed: `repvit_m2_3`, `batch=2`, `epochs=3`, `params=32499848`.
- Scratch usage after preflight: about `8.7G`.
- Slurm job `9400333` was still `PENDING (Priority)` after preflight; no teacher embeddings, training logs, or merged checkpoints exist yet.

## CPU Asset Preparation

The L40S job remained `PENDING (Priority)`, so checkpoint/data preparation was split into a CPU-safe path that can run while waiting for a GPU:

```bash
sbatch scripts/slurm_prepare_image_encoder_assets.sbatch
```

The asset prep writes:

```text
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/prepare_assets_*.log
```

It prepares:

```text
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/sam3_checkpoints/sam3.pt
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/data/SA-1B-0.01P/subset_manifest.json
```

`data/reorg_sa1b.py` now accepts `--num-workers` and defaults to `SLURM_CPUS_PER_TASK`, so the CPU and GPU jobs do not oversubscribe the node during tar extraction/reorganization.

First CPU asset job attempt:

```text
Job ID: 9400793
State: FAILED
ExitCode: 1:0
Log: sam3_img_assets-9400793.out
Scratch log: /storage/scratch1/9/eliu354/efficientsam3_distill_smoke/prepare_assets_20260602_191337.log
```

Failure reason: the installed Hugging Face CLI reports `huggingface-cli` as deprecated and no longer working. The download paths now use `hf download` and omit the removed `--local-dir-use-symlinks` option.

Follow-up auth check:

- Scratch-only `HF_HOME=/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/cache/huggingface` was not logged in and `hf download --dry-run` returned `Access denied. This repository requires approval.`
- The default submitted environment is logged in to Hugging Face as `danny010324`; `hf download --dry-run facebook/sam3 sam3.pt` passed.
- With scratch `HF_HOME` plus `HF_TOKEN_PATH` pointing at the approved default token file, `hf download --dry-run` also passed.
- The scratch scripts now default `HF_HOME` to `${RUN_ROOT}/cache/huggingface` while reusing an ambient `${HF_HOME}/token` via `HF_TOKEN_PATH` when available.

## 2026-06-02 GPU/CPU Data Download Failure

The L40S job eventually started, but failed before teacher embedding export:

```text
Job ID: 9400333
State: FAILED
Log: sam3_img_smoke-9400333.out
Scratch log: /storage/scratch1/9/eliu354/efficientsam3_distill_smoke/run_20260602_204457.log
Node: atl1-1-01-002-2-0
Failure point: SA-1B shard download/reorganization
```

The second CPU asset prep attempt failed at the same point:

```text
Job ID: 9400866
State: FAILED
Log: sam3_img_assets-9400866.out
Scratch log: /storage/scratch1/9/eliu354/efficientsam3_distill_smoke/prepare_assets_20260602_192234.log
```

Observed failure:

- Every official CDN URL in `data/sa-1b-1p.txt` returned HTTP `403 Forbidden` on PACE.
- The old downloader left empty tar files and returned success, so the reorganizer attempted to extract ten empty tar files.
- No `SA-1B-0.01P` subset, teacher embeddings, student logs, or merged checkpoints were produced.
- The SAM3 checkpoint does exist at `/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/sam3_checkpoints/sam3.pt`.

Fix in current worktree:

- `data/download_sa1b.sh` removes failed/empty partial files and exits nonzero if any expected shard is missing.
- `data/download_sa1b_hf.sh` adds a Hugging Face dataset mirror backend for the same shard filenames.
- `scripts/prepare_image_encoder_distill_assets.sh` and `scripts/run_image_encoder_distill_smoke.sh` default `SA1B_DOWNLOAD_BACKEND=hf` with `SA1B_HF_REPO=ssbai/sa1b`.
- Set `SA1B_DOWNLOAD_BACKEND=tsv` only after refreshing `data/sa-1b-1p.txt` with working official links.

Immediate retry submission initially failed while the Slurm controller was unavailable:

```bash
cd /storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation
sbatch scripts/slurm_prepare_image_encoder_assets.sbatch
```

Error:

```text
Batch job submission failed: Unable to contact slurm controller (connect failure)
```

## 2026-06-02 Successful Asset Prep Retry

After Slurm recovered, CPU asset prep was resubmitted:

```text
Job ID: 9402240
Partition: cpu-medium
QOS: embers
State: COMPLETED
ExitCode: 0:0
Elapsed: 00:15:45
Node: atl1-1-03-004-2-1
Scratch log: /storage/scratch1/9/eliu354/efficientsam3_distill_smoke/prepare_assets_20260602_224506.log
```

Evidence:

- `SA1B_DOWNLOAD_BACKEND=hf` and `SA1B_HF_REPO=ssbai/sa1b` were used.
- All ten SA-1B shards downloaded successfully from the Hugging Face mirror.
- `data/reorg_sa1b.py` found `111860` image/annotation pairs, split them into `100674` train and `11186` val pairs, and moved them with `0` failures.
- The deterministic subset was created at `/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/data/SA-1B-0.01P`.
- `subset_manifest.json` records `seed=5090`, `requested_num_samples=1120`, and `actual_num_samples=1120`.
- Verified file counts: `1120` train images and `1120` train annotations.
- Scratch usage after cleanup is about `13G`; the intermediate raw tar and reorganized 1% data were removed.

The dependent L40S smoke job was submitted after the asset job:

```text
Job ID: 9402252
Partition: gpu-l40s
QOS: embers
Dependency: afterok:9402240
State after asset prep: PENDING (Priority)
```

## 2026-06-02 First GPU Retry Failure

Job `9402252` started on `atl1-1-03-007-31-0` and failed before any teacher embeddings were written:

```text
State: FAILED
ExitCode: 1:0
Elapsed: 00:02:53
Log: sam3_img_smoke-9402252.out
Scratch log: /storage/scratch1/9/eliu354/efficientsam3_distill_smoke/run_20260602_230405.log
```

Root causes:

- The fallback dependency install had allowed `torch>=2.7.0` to resolve to `torch 2.12.0+cu130`.
- The L40S node driver reported version `12090`, so CUDA initialization failed for the CUDA 13 PyTorch wheel and `torch.cuda.is_available()` was `False`.
- The fallback dependency list also missed `pandas`, which is imported by the Stage 1 data package.

Fix:

- The fallback dependency path now pins `torch==2.11.0+cu128` and `torchvision==0.26.0+cu128` from `https://download.pytorch.org/whl/cu128`.
- The fallback dependency path now installs `pandas`.
- `scripts/run_image_encoder_distill_smoke.sh` exits before teacher export if CUDA is unavailable in a GPU allocation.
- The existing scratch env was repaired in place and verified on the login node:

```text
torch 2.11.0+cu128
torchvision 0.26.0+cu128
pandas 3.0.3
cuda_available_login False
```

Replacement GPU job:

```text
Job ID: 9402450
Partition: gpu-l40s
QOS: embers
State at submission: PENDING (Priority)
```

## 2026-06-02 Second GPU Retry: Teacher Pass, Student Failure

Job `9402450` started on `atl1-1-03-007-31-0` and verified the repaired CUDA stack:

```text
torch 2.11.0+cu128
cuda_available True
cuda_device NVIDIA L40S
cuda_mem_gib 44.39215087890625
```

Teacher embedding export completed successfully:

```text
Scratch log: /storage/scratch1/9/eliu354/efficientsam3_distill_smoke/run_20260602_231856.log
Teacher log: /storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/stage1_teacher/log_rank0.txt
Samples: 1120
Embedding shape: 1024 x 72 x 72 fp16
Elapsed: 0:02:08
Final throughput: about 8.68 img/s
Peak GPU memory: 2144 MB
Embedding values: /storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/stage1_teacher/embeddings/rank0-values.bin
Embedding keys: /storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/stage1_teacher/embeddings/rank0-keys.txt
```

The run then failed at the first ES-RV-S training step:

```text
Job ID: 9402450
State: FAILED
ExitCode: 1:0
Elapsed: 00:04:24
Failure point: ES-RV-S DataLoader reading saved teacher embeddings
```

Root cause:

- `stage1/data/augmentation/dataset_wrapper.py` used `int(np.frombuffer(...))`.
- With the current NumPy stack this returns a one-element array, and converting that array directly to `int` raises `TypeError: only 0-dimensional arrays can be converted to Python scalars`.

Fix:

- Seed parsing now indexes the scalar explicitly: `np.frombuffer(...)[0]`.
- Verified against the generated teacher embedding package: key `sa_307767`, integer seed `1861567313`, fp16 embedding vector shape `(5308416,)`.

Expected final artifacts:

```text
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/data/SA-1B-0.01P/subset_manifest.json
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/stage1_teacher/log_rank0.txt
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/stage1/es_rv_s/log_rank0.txt
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/stage1/es_rv_m/log_rank0.txt
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/stage1/es_rv_l/log_rank0.txt
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/efficient_sam3_repvit_s_smoke.pt
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/efficient_sam3_repvit_m_smoke.pt
/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/output/efficient_sam3_repvit_l_smoke.pt
```

## Current Limitation

The current Codex tool shell is not itself on a GPU node (`nvidia-smi` is unavailable). Scratch assets are prepared, but teacher embedding export and ES-RV-S/M/L student training still require the pending L40S Slurm allocation.
