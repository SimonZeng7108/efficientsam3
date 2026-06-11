import argparse
import json
import os
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

from sam3 import build_efficientsam3_image_model
from sam3.device import get_autocast_device_type, get_autocast_dtype, get_device
from sam3.model.sam3_image_processor import Sam3Processor


def inference_autocast(device):
    torch_device = torch.device(device)
    if torch_device.type not in ("cuda", "mps"):
        return nullcontext()
    return torch.autocast(
        device_type=get_autocast_device_type(torch_device),
        dtype=get_autocast_dtype(torch_device),
    )


def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def box_iou_xyxy(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, (box_a[2] - box_a[0])) * max(0.0, (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0])) * max(0.0, (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def overlay_mask(image_rgb, mask, color, alpha=0.45):
    out = image_rgb.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * color_arr
    return out.clip(0, 255).astype(np.uint8)


def resolve_coco_image_path(coco_root, split, file_name):
    candidates = [
        os.path.join(coco_root, "images", split, split, file_name),
        os.path.join(coco_root, "images", split, file_name),
        os.path.join(coco_root, "images", file_name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def choose_annotation(anns):
    valid = [a for a in anns if not a.get("iscrowd", 0)]
    if not valid:
        return None
    return max(valid, key=lambda a: float(a.get("area", 0.0)))


def load_targets_from_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("records", [])
    # Keep stable order by example index when available.
    records = sorted(records, key=lambda r: int(r.get("example_index", 0)))
    targets = []
    for r in records:
        if "img_id" not in r or "ann_id" not in r:
            continue
        targets.append((int(r["img_id"]), int(r["ann_id"])))
    return targets


def build_model(checkpoint_path, backbone, model_name, use_trineck, device):
    model = build_efficientsam3_image_model(
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        enable_inst_interactivity=True,
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
        backbone_type=backbone,
        model_name=model_name,
        use_trineck=use_trineck,
    )
    model.to(device)
    model.eval()
    return model


def run_examples(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    ann_file = os.path.join(args.coco_root, "annotations", f"instances_{args.split}.json")
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"Missing annotation file: {ann_file}")

    if args.device is None:
        args.device = str(get_device())

    model = build_model(
        checkpoint_path=args.checkpoint,
        backbone=args.backbone,
        model_name=args.model_name,
        use_trineck=args.use_trineck,
        device=args.device,
    )
    processor = Sam3Processor(model, device=args.device)

    coco = COCO(ann_file)
    img_ids = coco.getImgIds()

    targets = None
    if args.manifest_in:
        targets = load_targets_from_manifest(args.manifest_in)
        if len(targets) == 0:
            raise ValueError(f"No valid (img_id, ann_id) targets found in {args.manifest_in}")

    records = []
    saved = 0

    iterator = targets if targets is not None else img_ids
    for item in iterator:
        if saved >= args.num_examples:
            break

        if targets is not None:
            img_id, ann_id_target = item
        else:
            img_id = item
            ann_id_target = None

        img_info = coco.loadImgs(img_id)[0]
        img_path = resolve_coco_image_path(args.coco_root, args.split, img_info["file_name"])
        if img_path is None:
            continue

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        if ann_id_target is not None:
            ann = next((a for a in anns if int(a.get("id", -1)) == int(ann_id_target)), None)
            if ann is None:
                continue
            if ann.get("iscrowd", 0):
                continue
        else:
            ann = choose_annotation(anns)
        if ann is None:
            continue

        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        with torch.no_grad(), inference_autocast(args.device):
            state = processor.set_image(image)

        x, y, w, h = [float(v) for v in ann["bbox"]]

        if args.prompt_mode == "interactive":
            box = np.array([x, y, x + w, y + h], dtype=np.float32)
            with torch.no_grad(), inference_autocast(args.device):
                masks, scores, _ = model.predict_inst(
                    state,
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
        else:
            cx = (x + w / 2.0) / image.width
            cy = (y + h / 2.0) / image.height
            nw = w / image.width
            nh = h / image.height
            with torch.no_grad(), inference_autocast(args.device):
                state = processor.add_geometric_prompt([cx, cy, nw, nh], True, state)

            masks = state.get("masks")
            boxes_pred = state.get("boxes")
            if masks is None or len(masks) == 0:
                processor.reset_all_prompts(state)
                continue

            gt_box = np.array([x, y, x + w, y + h], dtype=np.float32)
            best_idx = 0
            if boxes_pred is not None and len(boxes_pred) > 0:
                best_iou = -1.0
                for i, pb in enumerate(boxes_pred):
                    pb_np = pb.detach().cpu().numpy()
                    biou = box_iou_xyxy(gt_box, pb_np)
                    if biou > best_iou:
                        best_iou = biou
                        best_idx = i

            if isinstance(masks, torch.Tensor):
                pred_mask = masks[best_idx, 0].detach().cpu().numpy() > 0
            else:
                pred_mask = np.asarray(masks[best_idx][0]) > 0
            processor.reset_all_prompts(state)

        gt_mask = coco.annToMask(ann).astype(bool)
        iou = calculate_iou(pred_mask, gt_mask)

        gt_overlay = overlay_mask(image_np, gt_mask, color=(60, 170, 75))
        pred_overlay = overlay_mask(image_np, pred_mask, color=(220, 55, 75))

        # Draw prompt bbox on all panes
        bbox_xyxy = (x, y, x + w, y + h)
        panes = []
        labels = [
            f"Image + Prompt Box | img_id={img_id}",
            "GT Mask Overlay",
            f"Pred Mask Overlay | IoU={iou:.3f}",
        ]
        for pane_np, label in zip([image_np, gt_overlay, pred_overlay], labels):
            pane = Image.fromarray(pane_np)
            draw = ImageDraw.Draw(pane)
            draw.rectangle(bbox_xyxy, outline=(255, 214, 10), width=3)
            draw.rectangle((8, 8, min(8 + len(label) * 7 + 12, pane.width - 8), 30), fill=(0, 0, 0))
            draw.text((14, 13), label, fill=(255, 255, 255))
            panes.append(np.array(pane))

        combined = np.concatenate(panes, axis=1)
        out_name = f"{saved:03d}_img{img_id}_ann{ann['id']}_iou{iou:.3f}.png"
        out_path = os.path.join(args.output_dir, out_name)
        Image.fromarray(combined).save(out_path)

        records.append(
            {
                "example_index": saved,
                "img_id": int(img_id),
                "ann_id": int(ann["id"]),
                "file_name": img_info["file_name"],
                "bbox_xywh": [x, y, w, h],
                "iou": iou,
                "prompt_mode": args.prompt_mode,
                "output_image": out_path,
            }
        )
        saved += 1

    manifest_path = os.path.join(args.output_dir, "examples_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"num_examples": saved, "records": records}, f, indent=2)

    print(f"Saved {saved} examples to {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a few COCO qualitative examples with EfficientSAM3")
    parser.add_argument("--coco_root", type=str, default="data/coco")
    parser.add_argument("--split", type=str, default="val2017")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_examples", type=int, default=8)
    parser.add_argument("--prompt_mode", type=str, default="interactive", choices=["interactive", "processor"])
    parser.add_argument("--backbone", type=str, default="tinyvit")
    parser.add_argument("--model_name", type=str, default="11m")
    parser.add_argument("--use_trineck", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--manifest_in", type=str, default=None, help="Optional path to a previous examples_manifest.json to reproduce the same (img_id, ann_id) set.")
    args = parser.parse_args()

    run_examples(args)
