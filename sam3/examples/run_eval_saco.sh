#!/bin/bash
#SBATCH --job-name=eval_saco
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Example usage: 
# sbatch sam3/examples/run_eval_saco.sh

PYTHON_SCRIPT="sam3/examples/eval_saco_veval_with_sam3.py"
CHECKPOINT="/home/b5cz/simonz.b5cz/program/stage2/efficientsam3/sam3_checkpoints/sam3.pt"
DATA_DIR="/home/b5cz/simonz.b5cz/program/stage2/efficientsam3/data/sa-co/all"
OUTPUT_DIR="/home/b5cz/simonz.b5cz/program/stage2/efficientsam3/output/eval_saco_sam3_pt"
DATASET="saco_veval_yt1b_val"

# Ensure output dir exists
mkdir -p $OUTPUT_DIR
mkdir -p slurm_logs

echo "Running evaluation on $DATASET with checkpoint $CHECKPOINT"

python $PYTHON_SCRIPT \
    --checkpoint $CHECKPOINT \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --gpus 4
