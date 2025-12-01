#!/usr/bin/env bash
#
set -euo pipefail

# Allow KEY=VALUE overrides on the command line
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    TEACHER_CHECKPOINT=*|DATASET_PATH=*|OUTPUT_DIR=*|BATCH_SIZE=*|GPUS=*|MASTER_PORT=*|NNODES=*|NODE_RANK=*|RDZV_BACKEND=*|RDZV_ENDPOINT=*)
      key=${arg%%=*}
      value=${arg#*=}
      printf -v "$key" '%s' "$value"
      ;;
    *)
      EXTRA_ARGS+=("$arg")
      ;;
  esac
done
set -- "${EXTRA_ARGS[@]}"

TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-sam3_checkpoints/sam3.pt}"
DATASET_PATH="${DATASET_PATH:-data/sa-v/extracted_frames}"
OUTPUT_DIR="${OUTPUT_DIR:-output/stage2_features}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GPUS="${GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29503}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
RDZV_BACKEND="${RDZV_BACKEND:-c10d}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:${MASTER_PORT}}"

TORCHRUN_ARGS=(--nproc_per_node "${GPUS}")
if [ "${NNODES}" -gt 1 ]; then
  TORCHRUN_ARGS+=(--nnodes "${NNODES}" --node_rank "${NODE_RANK}" --rdzv_backend "${RDZV_BACKEND}" --rdzv_endpoint "${RDZV_ENDPOINT}")
else
  TORCHRUN_ARGS+=(--nnodes 1 --master_port "${MASTER_PORT}")
fi

PY_ARGS=(
  --teacher_checkpoint "${TEACHER_CHECKPOINT}"
  --dataset_path "${DATASET_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --batch_size "${BATCH_SIZE}"
)

PYTHONPATH=. python -m torch.distributed.run "${TORCHRUN_ARGS[@]}" \
  stage2/extract_features_stage2.py \
  "${PY_ARGS[@]}" \
  "$@"
