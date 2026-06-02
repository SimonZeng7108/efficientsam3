#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation}"
RUN_ROOT="${RUN_ROOT:-/storage/scratch1/9/eliu354/efficientsam3_distill_smoke}"
ENV_DIR="${ENV_DIR:-${RUN_ROOT}/conda_env}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${RUN_ROOT}/conda_pkgs}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${RUN_ROOT}/cache/pip}"
HF_HOME="${HF_HOME:-${RUN_ROOT}/cache/huggingface}"
DATA_ROOT="${DATA_ROOT:-${RUN_ROOT}/data}"
RAW_TAR_DIR="${RAW_TAR_DIR:-${DATA_ROOT}/sa-1b-1p}"
REORG_ROOT="${REORG_ROOT:-${DATA_ROOT}/SA-1B-1P}"
SUBSET_ROOT="${SUBSET_ROOT:-${DATA_ROOT}/SA-1B-0.01P}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/output}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/sam3_checkpoints}"
SAM3_CKPT="${SAM3_CKPT:-${CHECKPOINT_DIR}/sam3.pt}"
NUM_SAMPLES="${NUM_SAMPLES:-1120}"
SAMPLE_SEED="${SAMPLE_SEED:-5090}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-1}"
STUDENT_BATCH_SIZE="${STUDENT_BATCH_SIZE:-4}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-3}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DOWNLOAD_CONCURRENCY="${DOWNLOAD_CONCURRENCY:-4}"
CLEAN_INTERMEDIATE="${CLEAN_INTERMEDIATE:-1}"
STUDENT_SPECS="${STUDENT_SPECS:-es_rv_s:stage1/configs/es_rv_s_5090_smoke.yaml:stage1/es_rv_s:efficient_sam3_repvit_s_smoke.pt:4 es_rv_m:stage1/configs/es_rv_m_5090_smoke.yaml:stage1/es_rv_m:efficient_sam3_repvit_m_smoke.pt:${STUDENT_BATCH_SIZE} es_rv_l:stage1/configs/es_rv_l_5090_smoke.yaml:stage1/es_rv_l:efficient_sam3_repvit_l_smoke.pt:2}"

export CONDA_PKGS_DIRS PIP_CACHE_DIR HF_HOME

mkdir -p "${RUN_ROOT}" "${DATA_ROOT}" "${OUTPUT_ROOT}" "${CHECKPOINT_DIR}" \
  "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "${HF_HOME}"

LOG_FILE="${RUN_ROOT}/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Run log: ${LOG_FILE}"
echo "Repo: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Subset root: ${SUBSET_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Student specs: ${STUDENT_SPECS}"

if ! command -v conda >/dev/null 2>&1 && [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # Slurm non-interactive shells often do not load the conda shell function.
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is required." >&2
  exit 1
fi

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  echo "Creating conda environment at ${ENV_DIR}"
  conda create -y -p "${ENV_DIR}" python=3.12
else
  echo "Using existing conda environment at ${ENV_DIR}"
fi

PYTHON="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"
export PATH="${ENV_DIR}/bin:${PATH}"

"${PYTHON}" -m pip install -U pip setuptools wheel

echo "Installing EfficientSAM3 Stage 1 dependencies"
if ! "${PIP}" install -e "${REPO_DIR}[stage1]"; then
  echo "Full stage1 extra install failed; retrying with image-distillation dependency set."
  "${PIP}" install -e "${REPO_DIR}" --no-deps
  "${PIP}" install \
    "torch>=2.7.0" "torchvision>=0.18.0" \
    "timm>=1.0.17" "numpy>=1.26.4" tqdm "ftfy==6.1.1" regex \
    "iopath>=0.1.10" typing_extensions huggingface_hub psutil \
    "decord>=0.6.0" "mmengine>=0.10.4" "pycocotools>=2.0.7" \
    "yacs>=0.1.8" "Pillow>=10.0.0" "opencv-python>=4.9.0.80" \
    "scipy>=1.10.0" "scikit-image>=0.21.0" "scikit-learn>=1.3.0" \
    "tensorboard>=2.12.0" "einops>=0.7.0" "hydra-core>=1.3.2" \
    "submitit>=1.5.1" "fvcore>=0.1.5.post20221221" \
    "fairscale>=0.4.13" pyyaml segment-anything
fi

"${PYTHON}" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
    print("cuda_mem_gib", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY

if [ ! -s "${SAM3_CKPT}" ]; then
  echo "Downloading SAM3 checkpoint to ${SAM3_CKPT}"
  "${ENV_DIR}/bin/huggingface-cli" download facebook/sam3 sam3.pt \
    --local-dir "${CHECKPOINT_DIR}" \
    --local-dir-use-symlinks False
else
  echo "Using existing SAM3 checkpoint at ${SAM3_CKPT}"
fi

if [ ! -d "${SUBSET_ROOT}/images/train" ] || \
   [ "$(find "${SUBSET_ROOT}/images/train" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)" -lt "${NUM_SAMPLES}" ]; then
  if [ ! -d "${REORG_ROOT}/images/train" ]; then
    echo "Downloading SA-1B 1% shards to ${RAW_TAR_DIR}"
    bash "${REPO_DIR}/data/download_sa1b.sh" "${REPO_DIR}/data/sa-1b-1p.txt" "${RAW_TAR_DIR}" "${DOWNLOAD_CONCURRENCY}"

    echo "Reorganizing SA-1B shards under ${DATA_ROOT}"
    (
      cd "${DATA_ROOT}"
      "${PYTHON}" "${REPO_DIR}/data/reorg_sa1b.py"
    )
  else
    echo "Using existing reorganized SA-1B data at ${REORG_ROOT}"
  fi

  echo "Creating deterministic ${NUM_SAMPLES}-sample SA-1B subset"
  "${PYTHON}" "${REPO_DIR}/data/create_sa1b_subset.py" \
    --source "${REORG_ROOT}" \
    --output "${SUBSET_ROOT}" \
    --num-samples "${NUM_SAMPLES}" \
    --seed "${SAMPLE_SEED}" \
    --mode hardlink

  if [ "${CLEAN_INTERMEDIATE}" = "1" ]; then
    echo "Cleaning intermediate SA-1B 1% raw/reorganized data inside ${DATA_ROOT}"
    rm -rf "${RAW_TAR_DIR}" "${REORG_ROOT}"
  fi
else
  echo "Using existing subset at ${SUBSET_ROOT}"
fi

TEACHER_OUTPUT="${OUTPUT_ROOT}/stage1_teacher"
TEACHER_EMB="${TEACHER_OUTPUT}/embeddings"

echo "Exporting teacher image embeddings"
bash "${REPO_DIR}/stage1/scripts/save_image_embeddings.sh" \
  CFG="${REPO_DIR}/stage1/configs/teacher/sam_vit_huge_sa1b_5090_smoke.yaml" \
  DATA_PATH="${SUBSET_ROOT}" \
  OUTPUT="${TEACHER_OUTPUT}" \
  BATCH_SIZE="${TEACHER_BATCH_SIZE}" \
  GPUS=1 \
  --opts \
    MODEL.RESUME "${SAM3_CKPT}" \
    DATA.NUM_SAMPLES "${NUM_SAMPLES}" \
    DATA.RANDOM_SAMPLE False \
    DATA.NUM_WORKERS "${NUM_WORKERS}" \
    DISTILL.TEACHER_EMBED_PATH "${TEACHER_EMB}"

FINAL_EPOCH=$((STUDENT_EPOCHS - 1))
MERGED_CHECKPOINTS=()
for spec in ${STUDENT_SPECS}; do
  IFS=':' read -r STUDENT_NAME STUDENT_CFG STUDENT_OUTPUT_SUBDIR MERGED_NAME SPEC_BATCH_SIZE <<EOF
${spec}
EOF
  if [ -z "${STUDENT_NAME}" ] || [ -z "${STUDENT_CFG}" ] || \
     [ -z "${STUDENT_OUTPUT_SUBDIR}" ] || [ -z "${MERGED_NAME}" ]; then
    echo "ERROR: invalid STUDENT_SPECS entry: ${spec}" >&2
    exit 1
  fi
  SPEC_BATCH_SIZE="${SPEC_BATCH_SIZE:-${STUDENT_BATCH_SIZE}}"
  STUDENT_OUTPUT="${OUTPUT_ROOT}/${STUDENT_OUTPUT_SUBDIR}"
  MERGED_CKPT="${OUTPUT_ROOT}/${MERGED_NAME}"

  echo "Training ${STUDENT_NAME} student image encoder"
  bash "${REPO_DIR}/stage1/scripts/train_image_student.sh" \
    CFG="${REPO_DIR}/${STUDENT_CFG}" \
    DATA_PATH="${SUBSET_ROOT}" \
    OUTPUT="${STUDENT_OUTPUT}" \
    BATCH_SIZE="${SPEC_BATCH_SIZE}" \
    GPUS=1 \
    --opts \
      TRAIN.EPOCHS "${STUDENT_EPOCHS}" \
      DATA.NUM_SAMPLES "${NUM_SAMPLES}" \
      DATA.RANDOM_SAMPLE False \
      DATA.NUM_WORKERS "${NUM_WORKERS}" \
      DISTILL.TEACHER_EMBED_PATH "${TEACHER_EMB}"

  echo "Merging ${STUDENT_NAME} image encoder into SAM3 checkpoint"
  "${PYTHON}" "${REPO_DIR}/stage1/convert_image_encoder_weights_stage1.py" \
    --student-ckpt "${STUDENT_OUTPUT}/ckpt_epoch_${FINAL_EPOCH}.pth" \
    --sam3-ckpt "${SAM3_CKPT}" \
    --output "${MERGED_CKPT}"
  MERGED_CHECKPOINTS+=("${MERGED_CKPT}")
done

echo "Done."
echo "Subset: ${SUBSET_ROOT}"
echo "Teacher log: ${TEACHER_OUTPUT}/log_rank0.txt"
echo "Student logs:"
for spec in ${STUDENT_SPECS}; do
  IFS=':' read -r STUDENT_NAME _ STUDENT_OUTPUT_SUBDIR _ _ <<EOF
${spec}
EOF
  echo "  ${STUDENT_NAME}: ${OUTPUT_ROOT}/${STUDENT_OUTPUT_SUBDIR}/log_rank0.txt"
done
echo "Merged checkpoints:"
for ckpt in "${MERGED_CHECKPOINTS[@]}"; do
  echo "  ${ckpt}"
done
