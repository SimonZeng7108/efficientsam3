#!/usr/bin/env python3
"""
Dedicated COCO val2017 eval for EVm/RVm checkpoints.
Bypasses the filename-parsing logic in eval_coco.py by calling evaluate_model directly.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_coco import evaluate_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Stage3 checkpoint (.pt)')
    parser.add_argument('--base-checkpoint', required=True, help='Stage1 base checkpoint (.pt)')
    parser.add_argument('--backbone', required=True, choices=['efficientvit', 'repvit', 'tinyvit'],
                        help='Vision backbone type')
    parser.add_argument('--model-name', required=True,
                        help='Model name string (e.g. b1, m1.1, tv_m)')
    parser.add_argument('--coco-root', default='data/coco')
    parser.add_argument('--output-dir', default='output/coco_eval_manual')
    parser.add_argument('--num-samples', type=int, default=-1)
    parser.add_argument('--device', default=None)
    parser.add_argument('--text-encoder-context-length', type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    miou, elapsed = evaluate_model(
        model_path=args.checkpoint,
        backbone=args.backbone,
        model_name=args.model_name,
        coco_root=args.coco_root,
        split='val2017',
        num_samples=args.num_samples,
        device=args.device,
        base_checkpoint=args.base_checkpoint,
        text_encoder_type='MobileCLIP-S0',
        text_encoder_context_length=args.text_encoder_context_length,
    )
    print(f"=== Final mIoU: {miou:.4f} (elapsed: {elapsed:.1f}s) ===")
    results_path = os.path.join(args.output_dir, 'results.txt')
    with open(results_path, 'w') as f:
        f.write(f"mIoU: {miou:.4f}\n")
        f.write(f"Elapsed: {elapsed:.1f}s\n")

if __name__ == '__main__':
    main()
