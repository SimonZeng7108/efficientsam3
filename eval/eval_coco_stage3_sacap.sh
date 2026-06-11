#!/bin/bash
#SBATCH --job-name=es3_coco_s3sacap
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --account=brics.b5cz
#SBATCH --output=/home/b5cz/simonz.b5cz/program/stage2/efficientsam3/output/eval_coco_s3sacap_%j.out
#SBATCH --error=/home/b5cz/simonz.b5cz/program/stage2/efficientsam3/output/eval_coco_s3sacap_%j.err

set -euo pipefail

LINE="${1:-img_fpn_tvm_99_1_20ep}"
MODEL_CKPT="${2:-}"

module load cuda/12.6
source /home/b5cz/yuxuanjj.b5cz/miniforge3/etc/profile.d/conda.sh
conda activate efficientsam3
deactivate 2>/dev/null || true
unset VIRTUAL_ENV

PYTHON="/home/b5cz/yuxuanjj.b5cz/miniforge3/envs/efficientsam3/bin/python"
BASE_DIR="/home/b5cz/simonz.b5cz/program/stage2/efficientsam3"
cd "${BASE_DIR}"

if [[ -z "${MODEL_CKPT}" ]]; then
    # Default CKPT if not passed
    case "${LINE}" in
        img_fpn_tvm_99_1_20ep) MODEL_CKPT="${BASE_DIR}/output/stage3/sacap_sa1b_tvm_mcs0_ctx16_seg_img_fpn_99_1_20ep/checkpoints/checkpoint.pt" ;;
        img_fpn_rvm_99_1_20ep) MODEL_CKPT="${BASE_DIR}/output/stage3/sacap_sa1b_rvm_mcs0_ctx16_seg_img_fpn_99_1_20ep/checkpoints/checkpoint.pt" ;;
        img_fpn_evm_99_1_20ep) MODEL_CKPT="${BASE_DIR}/output/stage3/sacap_sa1b_evm_mcs0_ctx16_seg_img_fpn_99_1_20ep/checkpoints/checkpoint.pt" ;;
        all_sources_tvm_20ep_negfix) MODEL_CKPT="${BASE_DIR}/output/stage3/sacap_sa1b_refcoco_gold_tvm_mcs0_ctx16_seg_img_fpn_20ep_negfix/checkpoints/checkpoint.pt" ;;
        fullft_tvm_30ep) MODEL_CKPT="${BASE_DIR}/output/stage3/sacap_sa1b_tvm_mcs0_ctx16_seg_fullft_30ep/checkpoints/checkpoint.pt" ;;
        *) echo "Unknown LINE=${LINE}"; exit 1 ;;
    esac
fi

VIS_BACKBONE="tinyvit"
SYMLINK_NAME="efficient_sam3_tinyvit_m.pt"
BASE_CKPT="${BASE_DIR}/output/sam3_stage1_image_5p/efficient_sam3_tv_m_mobileclip_s0_ctx16_5p_highmse.pt"
if [[ "${LINE}" == *"rvm"* ]]; then
    VIS_BACKBONE="repvit"
    SYMLINK_NAME="efficient_sam3_repvit_m.pt"
    BASE_CKPT="${BASE_DIR}/output/sam3_stage1_image_5p/efficient_sam3_rv_m_mobileclip_s0_ctx16_5p_highmse.pt"
elif [[ "${LINE}" == *"evm"* ]]; then
    VIS_BACKBONE="efficientvit"
    SYMLINK_NAME="efficient_sam3_efficientvit_m.pt"
    BASE_CKPT="${BASE_DIR}/output/sam3_stage1_image_5p/efficient_sam3_ev_m_mobileclip_s0_ctx16_5p_highmse.pt"
fi

COCO_ROOT="${COCO_ROOT:-${BASE_DIR}/data/coco}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"

if [[ ! -f "${MODEL_CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${MODEL_CKPT}"
  exit 1
fi

RUN_DIR="${BASE_DIR}/output/coco_eval_s3sacap_${SLURM_JOB_ID}"
MODEL_DIR="${RUN_DIR}/models"
mkdir -p "${MODEL_DIR}"
ln -sf "${MODEL_CKPT}" "${MODEL_DIR}/${SYMLINK_NAME}"

echo "=== Stage 3 SACap COCO Eval ==="
echo "LINE       : ${LINE}"
echo "BASE_CKPT  : ${BASE_CKPT}"
echo "MODEL_CKPT : ${MODEL_CKPT}"
echo "SYMLINK    : ${SYMLINK_NAME}"
echo "OUTPUT_DIR : ${RUN_DIR}"

PYTHONPATH="${BASE_DIR}:${BASE_DIR}/sam3" \
"${PYTHON}" eval/eval_coco.py \
  --coco_root "${COCO_ROOT}" \
  --output_dir "${MODEL_DIR}" \
  --num_samples "${NUM_SAMPLES}" \
  --base-checkpoint "${BASE_CKPT}" \
  --text-encoder-type MobileCLIP-S0 \
  --context-length 16 \
  --device cuda:0 \
  | tee "${RUN_DIR}/eval.log"

echo "=== Done: ${RUN_DIR}/eval.log ==="
