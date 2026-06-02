# L40S Image Encoder Distillation Smoke Run Record

Date: 2026-06-02
Repo: `/storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation`
Branch: `image-encoder-distill-pipeline`
Pipeline commit: `8abc6e3` (`Add scratch image encoder distillation smoke pipeline`)
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
- Scratch usage after preflight: about `7.8G`.
- Slurm job `9400333` was still `PENDING (Priority)` after preflight; no teacher embeddings, training logs, or merged checkpoints exist yet.

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

The current Codex tool shell is not itself on a GPU node (`nvidia-smi` is unavailable). Scratch is writable from the login shell, but the actual CUDA run still requires the pending Slurm GPU allocation.
