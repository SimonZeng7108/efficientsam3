import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from tqdm import tqdm
import argparse

# Add the parent directory to sys.path to allow importing sam3
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Assuming running from root where 'sam3' package is located
from sam3 import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.device import get_device

def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

import time

def evaluate_model(model_path, backbone, model_name, coco_root, split='val2017', num_samples=-1, device=None):
    print(f"Evaluating model: {model_path}")
    start_time = time.time()

    if device is None:
        device = get_device()
    
    # Load Model
    try:
        model = build_efficientsam3_image_model(
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz", 
            enable_inst_interactivity=True,
            checkpoint_path=model_path,
            load_from_HF=False,
            backbone_type=backbone,
            model_name=model_name,
        )
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        return None

    processor = Sam3Processor(model)

    # Load COCO
    ann_file = os.path.join(coco_root, f'annotations/instances_{split}.json')
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return None
        
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    
    if num_samples > 0:
        img_ids = img_ids[:num_samples]

    ious = []
    
    for img_id in tqdm(img_ids, desc=f"Eval {os.path.basename(model_path)}"):
        img_info = coco.loadImgs(img_id)[0]
        # Handle potential double folder structure
        img_path = os.path.join(coco_root, 'images', split, split, img_info['file_name'])
        if not os.path.exists(img_path):
             img_path = os.path.join(coco_root, 'images', split, img_info['file_name'])
             if not os.path.exists(img_path):
                # Try without split folder
                img_path = os.path.join(coco_root, 'images', img_info['file_name'])
                if not os.path.exists(img_path):
                    # print(f"Image not found: {img_path}")
                    continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

        # Get annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        if not anns:
            continue

        # Process image
        try:
            inference_state = processor.set_image(image)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue

        for ann in anns:
            if ann['iscrowd']:
                continue
            
            bbox = ann['bbox'] # x, y, w, h
            # Convert to x1, y1, x2, y2
            box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            
            # Run inference
            with torch.no_grad():
                masks, scores, _ = model.predict_inst(
                    inference_state,
                    point_coords=None,
                    point_labels=None,
                    box=box[None, :],
                    multimask_output=False,
                )
            
            if isinstance(masks, torch.Tensor):
                pred_mask = masks[0].cpu().numpy() > 0
            else:
                pred_mask = masks[0] > 0
            
            # Get GT mask
            gt_mask = coco.annToMask(ann)
            
            iou = calculate_iou(pred_mask, gt_mask)
            ious.append(iou)

    if not ious:
        print("No valid evaluations.")
        return 0.0

    miou = np.mean(ious)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"mIoU for {os.path.basename(model_path)}: {miou:.4f}")
    print(f"Time taken: {elapsed_time:.2f}s")
    return miou, elapsed_time

def main():
    parser = argparse.ArgumentParser(description='Evaluate EfficientSAM3 on COCO')
    parser.add_argument('--coco_root', type=str, default='data/coco', help='Path to COCO dataset')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory containing models')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to evaluate')
    from sam3.device import get_device
    parser.add_argument('--device', type=str, default=str(get_device()))
    args = parser.parse_args()

    models_dir = args.output_dir
    if not os.path.exists(models_dir):
        print(f"Output directory not found: {models_dir}")
        return

    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt') and 'efficient_sam3' in f]
    
    # Mapping from size char to model_name
    size_mapping = {
        'efficientvit': {'s': 'b0', 'm': 'b1', 'l': 'b2'},
        'repvit': {'s': 'm0.9', 'm': 'm1.1', 'l': 'm2.3'},
        'tinyvit': {'s': '5m', 'm': '11m', 'l': '21m'}
    }

    results = {}

    for model_file in sorted(model_files):
        # Format: efficient_sam3_{backbone}_{size}.pt
        parts = model_file.replace('.pt', '').split('_')
        if len(parts) < 4:
            print(f"Skipping {model_file}, cannot parse name.")
            continue
        
        backbone = parts[2]
        size = parts[3]
        
        if backbone not in size_mapping or size not in size_mapping[backbone]:
            print(f"Skipping {model_file}, unknown backbone or size.")
            continue
            
        model_name = size_mapping[backbone][size]
        model_path = os.path.join(models_dir, model_file)
        
        result = evaluate_model(model_path, backbone, model_name, args.coco_root, num_samples=args.num_samples, device=args.device)
        if result is not None:
            miou, elapsed_time = result
            results[model_file] = {'miou': miou, 'time': elapsed_time}

    print("\n=== Final Results ===")
    for model, data in results.items():
        print(f"{model}: mIoU={data['miou']:.4f}, Time={data['time']:.2f}s")

if __name__ == "__main__":
    main()
