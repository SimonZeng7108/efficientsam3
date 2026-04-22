#!/bin/bash
# Stage 3 Phase-1 Ablation: 3 scopes x 3 backbones = 9 runs
#
# Usage:
#   bash stage3/scripts/run_ablation.sh [num_samples] [epochs]
#
# Default: 10000 samples, 5 epochs per run

set -e

NUM_SAMPLES=${1:-10000}
EPOCHS=${2:-5}

BACKBONES=("es_rv_m" "es_tv_m" "es_ev_m")
SCOPES=("trunk_only" "trunk_fpn" "trunk_seghead")

echo "=== Stage 3 Phase-1 Ablation ==="
echo "Samples: $NUM_SAMPLES, Epochs: $EPOCHS"
echo "Backbones: ${BACKBONES[@]}"
echo "Scopes: ${SCOPES[@]}"
echo ""

for BB in "${BACKBONES[@]}"; do
    for SCOPE in "${SCOPES[@]}"; do
        TAG="ablation_${SCOPE}"
        echo "--- Running: $BB / $SCOPE ---"

        python stage3/train_stage3.py \
            --cfg "stage3/configs/${BB}.yaml" \
            --data-path data/sa-1b-1p_reorg \
            --sam3-checkpoint sam3_checkpoints/sam3.pt \
            --teacher-embed-dir output/stage3_teacher_embeddings \
            --trainable-scope "$SCOPE" \
            --tag "$TAG" \
            --output "output_stage3_ablation/${BB}" \
            --opts \
                DATA.NUM_SAMPLES "$NUM_SAMPLES" \
                TRAIN.EPOCHS "$EPOCHS" \
                SAVE_FREQ 5

        echo "--- Finished: $BB / $SCOPE ---"
        echo ""
    done
done

echo "=== Ablation complete. Evaluate with: ==="
echo "  bash stage3/scripts/eval_stage3.sh output_stage3_ablation"
