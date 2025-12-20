#!/usr/bin/env bash
#
set -euo pipefail

# Allow KEY=VALUE overrides passed after the script name.
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    CFG=*|DATA_PATH=*|OUTPUT=*|BATCH_SIZE=*|GPUS=*|MASTER_PORT=*|NNODES=*|NODE_RANK=*|RDZV_BACKEND=*|RDZV_ENDPOINT=*|PRETRAINED=*|SAM3_CHECKPOINT=*|TEACHER_EMBED_PATH=*)
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

CFG="${CFG:-stage1_geometry_finetune/configs/repvit_m1_1_geometry.yaml}"
DATA_PATH="${DATA_PATH:-data/sa-1b}"
OUTPUT="${OUTPUT:-output_geometry_finetune}"
BATCH_SIZE="${BATCH_SIZE:-}"
GPUS="${GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29503}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
RDZV_BACKEND="${RDZV_BACKEND:-c10d}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:${MASTER_PORT}}"
PRETRAINED="${PRETRAINED:-}"
SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-sam3_checkpoints/sam3.pt}"
TEACHER_EMBED_PATH="${TEACHER_EMBED_PATH:-output/stage1_teacher/embeddings}"

TORCHRUN_ARGS=(--nproc_per_node "${GPUS}")
if [ "${NNODES}" -gt 1 ]; then
  TORCHRUN_ARGS+=(--nnodes "${NNODES}" --node_rank "${NODE_RANK}" --rdzv_backend "${RDZV_BACKEND}" --rdzv_endpoint "${RDZV_ENDPOINT}")
else
  TORCHRUN_ARGS+=(--nnodes 1 --master_port "${MASTER_PORT}")
fi

PY_ARGS=(
  --cfg "${CFG}"
  --data-path "${DATA_PATH}"
  --output "${OUTPUT}"
)

if [ -n "${BATCH_SIZE}" ]; then
  PY_ARGS+=(--batch-size "${BATCH_SIZE}")
fi

if [ -n "${PRETRAINED}" ]; then
  PY_ARGS+=(--pretrained "${PRETRAINED}")
fi

PY_ARGS+=(--sam3-checkpoint "${SAM3_CHECKPOINT}")
PY_ARGS+=(--teacher-embed-path "${TEACHER_EMBED_PATH}")

PYTHONPATH=. torchrun "${TORCHRUN_ARGS[@]}" \
  stage1_geometry_finetune/train_geometry_finetune.py \
  "${PY_ARGS[@]}" \
  "$@"
