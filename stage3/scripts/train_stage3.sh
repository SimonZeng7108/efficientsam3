#!/bin/bash
# Stage 3 Training Launch Script
#
# Usage:
#   # Single GPU
#   bash stage3/scripts/train_stage3.sh es_rv_m trunk_only
#
#   # Multi-GPU (4x)
#   NGPU=4 bash stage3/scripts/train_stage3.sh es_rv_m trunk_only
#
# Arguments:
#   $1 - backbone config name: es_rv_m | es_tv_m | es_ev_m
#   $2 - trainable scope: trunk_only | trunk_fpn | trunk_seghead
#   $3 - (optional) extra args

set -e

CONFIG=${1:-es_rv_m}
SCOPE=${2:-trunk_only}
EXTRA_ARGS=${@:3}

NGPU=${NGPU:-1}

CFG_FILE="stage3/configs/${CONFIG}.yaml"
if [ ! -f "$CFG_FILE" ]; then
    echo "Config not found: $CFG_FILE"
    exit 1
fi

COMMON_ARGS="--cfg $CFG_FILE \
    --data-path data/sa-1b-1p_reorg \
    --sam3-checkpoint sam3_checkpoints/sam3.pt \
    --teacher-embed-dir output/stage3_teacher_embeddings \
    --trainable-scope $SCOPE \
    --tag ${SCOPE} \
    $EXTRA_ARGS"

if [ "$NGPU" -gt 1 ]; then
    torchrun --nproc_per_node=$NGPU stage3/train_stage3.py $COMMON_ARGS
else
    python stage3/train_stage3.py $COMMON_ARGS
fi
