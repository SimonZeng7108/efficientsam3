#!/bin/bash
#SBATCH --job-name=es3_coco_both
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --account=brics.b5cz
#SBATCH --output=/home/b5cz/simonz.b5cz/program/stage2/efficientsam3/output/eval_coco_both_%j.out
#SBATCH --error=/home/b5cz/simonz.b5cz/program/stage2/efficientsam3/output/eval_coco_both_%j.err

set -euo pipefail

# Usage (via environment variables):
# sbatch eval/eval_coco_both_student_encoders.sh \
#   --export=ALL,MODEL_CKPT=/abs/path/to/efficient_sam3_tinyvit_m_mobileclip_s0_ctx16_5dataset.pt,NUM_SAMPLES=-1
#
# Required:
#   MODEL_CKPT : path to merged checkpoint (student image encoder + student text encoder)
#
# Optional:
#   BACKBONE    : tinyvit | repvit | efficientvit (auto-detected from filename when possible)
#   SIZE        : s | m | l (auto-detected from filename when possible)
#   COCO_ROOT   : default ${BASE_DIR}/data/coco
#   NUM_SAMPLES : default -1
#   PROMPT_MODE : default interactive
#   USE_TRINECK : default 0 (set to 1 to pass --use_trineck)

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

if [[ -z "${MODEL_CKPT}" ]]; then
  echo "ERROR: MODEL_CKPT is required."
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

# Auto-detect backbone and size from checkpoint name, e.g.
# efficient_sam3_tinyvit_m_mobileclip_s0_ctx16_5dataset.pt
if [[ -z "${BACKBONE}" || -z "${SIZE}" ]]; then
  base="$(basename "${MODEL_CKPT}" .pt)"
  if [[ "${base}" =~ ^efficient_sam3_(tinyvit|repvit|efficientvit)_([sml])_ ]]; then
    BACKBONE="${BASH_REMATCH[1]}"
    SIZE="${BASH_REMATCH[2]}"
  fi
fi

if [[ -z "${BACKBONE}" || -z "${SIZE}" ]]; then
  echo "ERROR: Could not infer BACKBONE/SIZE from MODEL_CKPT. Set BACKBONE and SIZE explicitly."
  exit 1
fi

RUN_DIR="${BASE_DIR}/output/coco_eval_both_${BACKBONE}_${SIZE}_${SLURM_JOB_ID}"
MODEL_DIR="${RUN_DIR}/models"
mkdir -p "${MODEL_DIR}"

# eval/eval_coco.py expects filename pattern efficient_sam3_{backbone}_{size}.pt
TARGET_NAME="efficient_sam3_${BACKBONE}_${SIZE}.pt"
ln -sf "${MODEL_CKPT}" "${MODEL_DIR}/${TARGET_NAME}"

echo "=== GPU Info ==="
nvidia-smi

echo "=== COCO Eval (Both Student Encoders) ==="
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
