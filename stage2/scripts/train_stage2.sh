#!/usr/bin/env bash
#
set -euo pipefail

# Allow KEY=VALUE overrides on the command line
EXTRA_ARGS=()
EPOCHS=""
for arg in "$@"; do
  case "$arg" in
    CONFIG=*|GPUS=*|MASTER_PORT=*|NNODES=*|NODE_RANK=*|RDZV_BACKEND=*|RDZV_ENDPOINT=*|EPOCHS=*)
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

CONFIG="${CONFIG:-stage2/configs/efficient_sam3_stage2.yaml}"
GPUS="${GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29504}"
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

PY_ARGS=(--config "${CONFIG}")
if [ -n "${EPOCHS}" ]; then
  # Update the config dict in the python script if passed as arg
  # Actually train_stage2.py doesn't have --epochs arg in argparse directly if it loads from config.
  # Let's check train_stage2.py again.
  # It seems I need to modify train_stage2.py to accept --epochs override or rely on config.
  # Checking previous read_file of train_stage2.py...
  # It DOES NOT have --epochs in argparse. It reads from config.
  # I will update train_stage2.py to accept --epochs argument.
  PY_ARGS+=(--epochs "${EPOCHS}")
fi

PYTHONPATH=. python -m torch.distributed.run "${TORCHRUN_ARGS[@]}" \
  stage2/train_stage2.py \
  "${PY_ARGS[@]}" \
  "$@"
