#!/bin/bash

# Stage 1 Geometry Finetune Training Script
# Prompt-in-the-Loop Knowledge Distillation

# Initialize an array for arguments to pass to the python script
PY_ARGS=()

# Parse arguments
for arg in "$@"
do
    if [[ "$arg" == *=* ]]; then
        # It's a variable assignment like CFG=...
        export "$arg"
    else
        # It's a flag or other argument for the python script
        PY_ARGS+=("$arg")
    fi
done

# Override these via environment variables or command line
CFG="${CFG:-stage1_geometry_finetune/configs/es_rv_m.yaml}"
DATA_PATH="${DATA_PATH:-data/sa-1b}"
OUTPUT="${OUTPUT:-output/stage1_geometry_finetune/es_rv_m}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GPUS="${GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29502}"

# Distributed training settings
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
RDZV_BACKEND="${RDZV_BACKEND:-c10d}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:${MASTER_PORT}}"

echo "============================================"
echo "Stage 1 Geometry Finetune Training"
echo "============================================"
echo "Config: $CFG"
echo "Data Path: $DATA_PATH"
echo "Output: $OUTPUT"
echo "Batch Size: $BATCH_SIZE"
echo "GPUs: $GPUS"
echo "Master Port: $MASTER_PORT"
echo "============================================"

# Set PYTHONPATH to include sam3
export PYTHONPATH="${PYTHONPATH}:$(pwd)/sam3"

torchrun \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  --rdzv_backend=${RDZV_BACKEND} \
  --rdzv_endpoint=${RDZV_ENDPOINT} \
  stage1_geometry_finetune/train_stage1_geometry_finetune.py \
    --cfg ${CFG} \
    --data-path ${DATA_PATH} \
    --output ${OUTPUT} \
    --batch-size ${BATCH_SIZE} \
    "${PY_ARGS[@]}"
