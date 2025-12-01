#!/usr/bin/env bash

set -euo pipefail

EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    TEACHER_CHECKPOINT=*|FEATURES_DIR=*|OUTPUT_DIR=*|GPUS=*|MASTER_PORT=*|NNODES=*|NODE_RANK=*|RDZV_BACKEND=*|RDZV_ENDPOINT=*|IMAGE_SIZE=*|INIT_BOX=*)
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
FEATURES_DIR="${FEATURES_DIR:-output/stage2_features}"
OUTPUT_DIR="${OUTPUT_DIR:-output/stage2_teacher}"
IMAGE_SIZE="${IMAGE_SIZE:-1008}"
INIT_BOX="${INIT_BOX:-128}"
GPUS="${GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29605}"
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
  --features_dir "${FEATURES_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --image_size "${IMAGE_SIZE}"
  --init_box "${INIT_BOX}"
)

PYTHONPATH=. python -m torch.distributed.run "${TORCHRUN_ARGS[@]}" \
  stage2/save_teacher_embeddings_stage2.py \
  "${PY_ARGS[@]}" \
  "$@"

