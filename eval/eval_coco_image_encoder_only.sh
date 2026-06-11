#!/bin/bash
#SBATCH --job-name=es3_coco_img_only
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --account=brics.b5cz
#SBATCH --output=/home/b5cz/simonz.b5cz/program/stage2/efficientsam3/output/eval_coco_img_only_%j.out
#SBATCH --error=/home/b5cz/simonz.b5cz/program/stage2/efficientsam3/output/eval_coco_img_only_%j.err

set -euo pipefail

# Usage (via environment variables):
# sbatch eval/eval_coco_image_encoder_only.sh \
#   --export=ALL,MODEL_CKPT=/abs/path/to/merged_image_only.pt,BACKBONE=tinyvit,SIZE=m,NUM_SAMPLES=-1
#
# Required:
#   MODEL_CKPT : path to merged checkpoint (image encoder student, original text path)
#   BACKBONE   : one of tinyvit | repvit | efficientvit
#   SIZE       : one of s | m | l
#
# Optional:
#   COCO_ROOT    : default ${BASE_DIR}/data/coco
#   NUM_SAMPLES  : default -1 (full val set)
#   PROMPT_MODE  : default interactive
#   USE_TRINECK  : default 0 (set to 1 to pass --use_trineck)

module load cuda/12.6
source /home/b5cz/yuxuanjj.b5cz/miniforge3/etc/profile.d/conda.sh
conda activate efficientsam3

BASE_DIR="/home/b5cz/simonz.b5cz/program/stage2/efficientsam3"
cd "${BASE_DIR}"

COCO_ROOT="${COCO_ROOT:-${BASE_DIR}/data/coco}"
MODEL_CKPT="${MODEL_CKPT:-}"
BACKBONE="${BACKBONE:-}"
SIZE="${SIZE:-}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
PROMPT_MODE="${PROMPT_MODE:-interactive}"
USE_TRINECK="${USE_TRINECK:-0}"

if [[ -z "${MODEL_CKPT}" || -z "${BACKBONE}" || -z "${SIZE}" ]]; then
  echo "ERROR: MODEL_CKPT, BACKBONE, and SIZE are required."
  exit 1
fi

if [[ ! -f "${MODEL_CKPT}" ]]; then
  echo "Missing checkpoint: ${MODEL_CKPT}"
  exit 1
fi
if [[ ! -f "${COCO_ROOT}/annotations/instances_val2017.json" ]]; then
  echo "Missing COCO annotation file: ${COCO_ROOT}/annotations/instances_val2017.json"
  exit 1
fi

RUN_DIR="${BASE_DIR}/output/coco_eval_image_only_${BACKBONE}_${SIZE}_${SLURM_JOB_ID}"
MODEL_DIR="${RUN_DIR}/models"
mkdir -p "${MODEL_DIR}"

# eval/eval_coco.py expects filename pattern efficient_sam3_{backbone}_{size}.pt
TARGET_NAME="efficient_sam3_${BACKBONE}_${SIZE}.pt"
ln -sf "${MODEL_CKPT}" "${MODEL_DIR}/${TARGET_NAME}"

echo "=== GPU Info ==="
nvidia-smi

echo "=== COCO Eval (Image Encoder Only) ==="
echo "MODEL_CKPT=${MODEL_CKPT}"
echo "BACKBONE=${BACKBONE} SIZE=${SIZE}"
echo "COCO_ROOT=${COCO_ROOT}"
echo "NUM_SAMPLES=${NUM_SAMPLES}"
echo "PROMPT_MODE=${PROMPT_MODE}"
echo "USE_TRINECK=${USE_TRINECK}"
echo "RUN_DIR=${RUN_DIR}"

TRINECK_ARG=()
if [[ "${USE_TRINECK}" == "1" ]]; then
  TRINECK_ARG=(--use_trineck)
fi

PYTHONPATH="${BASE_DIR}:${BASE_DIR}/sam3" \
python eval/eval_coco.py \
  --coco_root "${COCO_ROOT}" \
  --output_dir "${MODEL_DIR}" \
  --num_samples "${NUM_SAMPLES}" \
  --prompt_mode "${PROMPT_MODE}" \
  --device cuda:0 \
  "${TRINECK_ARG[@]}" | tee "${RUN_DIR}/eval.log"

echo "=== Done: ${RUN_DIR}/eval.log ==="
