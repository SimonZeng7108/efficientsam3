#!/usr/bin/env bash
#
set -euo pipefail

# Allow KEY=VALUE overrides on the command line (e.g. CFG=..., GPUS=8, etc.)
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    CFG=*|DATA_PATH=*|OUTPUT=*|GPUS=*|BATCH_SIZE=*|MASTER_PORT=*|NNODES=*|NODE_RANK=*|RDZV_BACKEND=*|RDZV_ENDPOINT=*)
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

CFG="${CFG:-stage1/configs/teacher/sam_vit_huge_sa1b.yaml}"
DATA_PATH="${DATA_PATH:-data/sa-1b}"
OUTPUT="${OUTPUT:-output/stage1_teacher}"
GPUS="${GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
RDZV_BACKEND="${RDZV_BACKEND:-c10d}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:${MASTER_PORT}}"
BATCH_SIZE="${BATCH_SIZE:-}"

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

PYTHONPATH=. python -m torch.distributed.run "${TORCHRUN_ARGS[@]}" \
  stage1/save_embedding_stage1.py \
  "${PY_ARGS[@]}" \
  "$@"

