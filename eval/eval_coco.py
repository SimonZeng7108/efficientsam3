import os
import sys
import json
from contextlib import nullcontext
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
from sam3.device import get_autocast_device_type, get_autocast_dtype, get_device

def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def box_iou_xyxy(box_a, box_b):
    """Compute IoU between two boxes in xyxy format (numpy arrays)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

import time


def inference_autocast(device):
    torch_device = torch.device(device)
    if torch_device.type not in ("cuda", "mps"):
        return nullcontext()
    return torch.autocast(
        device_type=get_autocast_device_type(torch_device),
        dtype=get_autocast_dtype(torch_device),
    )

def evaluate_model(model_path, backbone, model_name, coco_root, split='val2017',
                    num_samples=-1, device=None, prompt_mode='interactive',
                    use_trineck=False):
    print(f"Evaluating model: {model_path}  (prompt_mode={prompt_mode}, use_trineck={use_trineck})")
    start_time = time.time()

    if device is None:
        device = get_device()

    if prompt_mode not in {"interactive", "processor"}:
        raise ValueError(f"Unsupported prompt_mode={prompt_mode!r}. Use 'interactive' or 'processor'.")
    use_interactive = (prompt_mode == "interactive")

    # Load Model — always enable inst_interactivity so both modes work
    try:
        model = build_efficientsam3_image_model(
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz", 
            enable_inst_interactivity=True,
            checkpoint_path=model_path,
            load_from_HF=False,
            backbone_type=backbone,
            model_name=model_name,
            use_trineck=use_trineck,
        )
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        return None

    processor = Sam3Processor(model, device=device)

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
            with torch.no_grad(), inference_autocast(device):
                inference_state = processor.set_image(image)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue

        for ann in anns:
            if ann['iscrowd']:
                continue
            
            bbox = ann['bbox'] # x, y, w, h
            x, y, w, h = bbox

            if use_interactive:
                # SAM1-style interactive prediction with absolute-pixel xyxy box
                try:
                    box = np.array(
                        [x, y, x + w, y + h], dtype=np.float32
                    )
                    with torch.no_grad(), inference_autocast(device):
                        masks, scores, _ = model.predict_inst(
                            inference_state,
                            point_coords=None,
                            point_labels=None,
                            box=box[None, :],
                            multimask_output=False,
                        )
                    if masks is None or len(scores) == 0:
                        continue
                    if isinstance(masks, torch.Tensor):
                        pred_mask = masks[0].cpu().numpy() > 0
                    else:
                        pred_mask = np.asarray(masks[0]) > 0
                except Exception as e:
                    continue
            else:
                # SAM3 detection-style geometric prompt (normalized cxcywh)
                cx = (x + w / 2.0) / image.width
                cy = (y + h / 2.0) / image.height
                nw = w / image.width
                nh = h / image.height

                with torch.no_grad(), inference_autocast(device):
                    inference_state = processor.add_geometric_prompt(
                        [cx, cy, nw, nh],
                        True,
                        inference_state,
                    )

                masks = inference_state.get("masks")
                scores = inference_state.get("scores")
                boxes_pred = inference_state.get("boxes")
                if masks is None or scores is None or len(scores) == 0:
                    processor.reset_all_prompts(inference_state)
                    continue

                # Match predicted detection to GT box by IoU
                gt_box_xyxy = np.array([x, y, x + w, y + h])
                if boxes_pred is not None and len(boxes_pred) > 0:
                    best_iou = -1
                    best_idx = 0
                    for i, pred_box in enumerate(boxes_pred):
                        pred_box_np = pred_box.cpu().numpy()
                        biou = box_iou_xyxy(gt_box_xyxy, pred_box_np)
                        if biou > best_iou:
                            best_iou = biou
                            best_idx = i
                else:
                    best_idx = 0

                if isinstance(masks, torch.Tensor):
                    pred_mask = masks[best_idx, 0].cpu().numpy() > 0
                else:
                    pred_mask = masks[best_idx][0] > 0

                processor.reset_all_prompts(inference_state)
            
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
    parser.add_argument('--prompt_mode', type=str, default='interactive', choices=['interactive', 'processor'],
                        help='Prompting mode: interactive (SAM-style box prompt) or processor (detection mode).')
    parser.add_argument('--use_trineck', action='store_true',
                        help='Use SAM3.1 TriNeck vision backbone when building the model.')
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
        
        result = evaluate_model(
            model_path,
            backbone,
            model_name,
            args.coco_root,
            num_samples=args.num_samples,
            device=args.device,
            prompt_mode=args.prompt_mode,
            use_trineck=args.use_trineck,
        )
        if result is not None:
            miou, elapsed_time = result
            results[model_file] = {'miou': miou, 'time': elapsed_time}

    print("\n=== Final Results ===")
    for model, data in results.items():
        print(f"{model}: mIoU={data['miou']:.4f}, Time={data['time']:.2f}s")

if __name__ == "__main__":
    main()
