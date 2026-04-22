#!/bin/bash
# Stage 3 Evaluation Script
#
# Evaluates Stage 3 fine-tuned checkpoints on COCO using eval/eval_coco.py.
# Walks the output directory tree looking for checkpoint files.
#
# Usage:
#   bash stage3/scripts/eval_stage3.sh <output_dir> [coco_root] [num_samples]
#
# Examples:
#   bash stage3/scripts/eval_stage3.sh output_stage3
#   bash stage3/scripts/eval_stage3.sh output_stage3_ablation data/coco 500

set -e

OUTPUT_DIR=${1:?Usage: eval_stage3.sh <output_dir> [coco_root] [num_samples]}
COCO_ROOT=${2:-data/coco}
NUM_SAMPLES=${3:--1}

python -c "
import os, sys, glob
sys.path.insert(0, '.')
from eval.eval_coco import evaluate_model

output_dir = '$OUTPUT_DIR'
coco_root = '$COCO_ROOT'
num_samples = int('$NUM_SAMPLES')

backbone_map = {
    'es_rv_m': ('repvit', 'm1.1'),
    'es_tv_m': ('tinyvit', '11m'),
    'es_ev_m': ('efficientvit', 'b1'),
}

def find_checkpoints(root):
    \"\"\"Walk output tree and yield (backbone_key, scope_tag, ckpt_path) tuples.\"\"\"
    for dirpath, dirnames, filenames in os.walk(root):
        pth_files = sorted([f for f in filenames if f.endswith('.pth')])
        if not pth_files:
            continue
        rel = os.path.relpath(dirpath, root)
        parts = rel.replace(os.sep, '/').split('/')
        bb_key = None
        for p in parts:
            if p in backbone_map:
                bb_key = p
                break
        if bb_key is None:
            continue
        tag = '/'.join(p for p in parts if p != bb_key) or 'default'
        yield bb_key, tag, os.path.join(dirpath, pth_files[-1])

results = {}
for bb_key, tag, ckpt_path in find_checkpoints(output_dir):
    backbone_type, model_name = backbone_map[bb_key]
    label = f'{bb_key}/{tag}'
    print(f'\\nEvaluating {label} => {os.path.basename(ckpt_path)}...')
    try:
        result = evaluate_model(
            ckpt_path, backbone_type, model_name, coco_root,
            num_samples=num_samples,
        )
        if result is not None:
            miou, elapsed = result
            results[label] = {'miou': miou, 'time': elapsed}
    except Exception as e:
        print(f'  ERROR: {e}')

print('\\n=== Stage 3 Evaluation Results ===')
for name in sorted(results):
    d = results[name]
    print(f'  {name}: mIoU={d[\"miou\"]:.4f}, Time={d[\"time\"]:.2f}s')
"
