#!/usr/bin/env bash
#SBATCH --job-name=sam3_img_smoke
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=sam3_img_smoke-%j.out

set -euo pipefail

export REPO_DIR=/storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation
export RUN_ROOT=/storage/scratch1/9/eliu354/efficientsam3_distill_smoke
export NUM_SAMPLES=1120
export SAMPLE_SEED=5090
export TEACHER_BATCH_SIZE=1
export STUDENT_BATCH_SIZE=4
export STUDENT_EPOCHS=3
export STUDENT_SPECS="es_rv_s:stage1/configs/es_rv_s_5090_smoke.yaml:stage1/es_rv_s:efficient_sam3_repvit_s_smoke.pt:4 es_rv_m:stage1/configs/es_rv_m_5090_smoke.yaml:stage1/es_rv_m:efficient_sam3_repvit_m_smoke.pt:4 es_rv_l:stage1/configs/es_rv_l_5090_smoke.yaml:stage1/es_rv_l:efficient_sam3_repvit_l_smoke.pt:2"
export NUM_WORKERS=4
export DOWNLOAD_CONCURRENCY=4
export CLEAN_INTERMEDIATE=1

bash "${REPO_DIR}/scripts/run_image_encoder_distill_smoke.sh"
