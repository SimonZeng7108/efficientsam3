#!/usr/bin/env python3
"""
Visualize COCO and LVIS image-text-mask triplets.
Produces an HTML page with sample images showing overlaid segmentation masks
and their corresponding text labels.
"""

import json
import os
import random
import base64
from io import BytesIO
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

BASE = "/home/b5cz/simonz.b5cz/program/stage2/efficientsam3"
OUTPUT_DIR = os.path.join(BASE, "output", "data_viz")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COCO_ANN = os.path.join(BASE, "data/coco/annotations/instances_train2017.json")
COCO_IMG = os.path.join(BASE, "data/coco/images/train2017")
LVIS_ANN = os.path.join(BASE, "data/lvis/annotations/lvis_v1_train.json")
LVIS_IMG = os.path.join(BASE, "data/lvis/images/train2017")

N_SAMPLES = 8
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
]


def decode_segmentation(ann, h, w):
    """Decode COCO/LVIS segmentation to binary mask."""
    seg = ann.get("segmentation")
    if seg is None:
        return None
    if isinstance(seg, dict):
        if isinstance(seg["counts"], list):
            rle = mask_utils.frPyObjects(seg, h, w)
        else:
            rle = seg
        return mask_utils.decode(rle)
    elif isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, h, w)
        return mask_utils.decode(mask_utils.merge(rles))
    return None


def overlay_masks(image, anns, cat_id_to_name, max_anns=6):
    """Overlay colored masks with text labels on image."""
    img = image.copy()
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_text = ImageDraw.Draw(img)

    w, h = img.size
    labels = []

    for i, ann in enumerate(anns[:max_anns]):
        mask = decode_segmentation(ann, h, w)
        if mask is None:
            continue

        color = COLORS[i % len(COLORS)]
        cat_name = cat_id_to_name.get(ann["category_id"], f"cat_{ann['category_id']}")
        labels.append(cat_name)

        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_rgba[mask > 0] = (*color, 100)
        overlay = Image.alpha_composite(overlay, Image.fromarray(mask_rgba, "RGBA"))

        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            except (OSError, IOError):
                font = ImageFont.load_default()
            draw_text.text((cx, cy), cat_name, fill=color, font=font)

    img = img.convert("RGBA")
    result = Image.alpha_composite(img, overlay).convert("RGB")
    return result, labels


def img_to_b64(img, max_w=600):
    if img.width > max_w:
        ratio = max_w / img.width
        img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def sample_coco(n=N_SAMPLES):
    """Sample n images from COCO with their annotations."""
    print(f"Loading COCO annotations from {COCO_ANN}...")
    coco = COCO(COCO_ANN)
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    img_ids_with_anns = []
    for img_id in coco.getImgIds():
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        if 2 <= len(ann_ids) <= 10:
            img_ids_with_anns.append(img_id)
    random.seed(42)
    selected = random.sample(img_ids_with_anns, min(n, len(img_ids_with_anns)))

    results = []
    for img_id in selected:
        info = coco.loadImgs(img_id)[0]
        path = os.path.join(COCO_IMG, info["file_name"])
        if not os.path.exists(path):
            continue
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
        img = Image.open(path).convert("RGB")
        viz, labels = overlay_masks(img, anns, cats)
        results.append({
            "dataset": "COCO",
            "image_id": img_id,
            "file_name": info["file_name"],
            "size": f"{info['width']}x{info['height']}",
            "n_objects": len(anns),
            "labels": labels,
            "has_masks": all(decode_segmentation(a, info["height"], info["width"]) is not None for a in anns[:6]),
            "viz_b64": img_to_b64(viz),
            "orig_b64": img_to_b64(img),
        })
    return results


def sample_lvis(n=N_SAMPLES):
    """Sample n images from LVIS with their annotations."""
    print(f"Loading LVIS annotations from {LVIS_ANN}...")
    with open(LVIS_ANN) as f:
        lvis = json.load(f)

    cats = {c["id"]: c["name"].replace("_", " ") for c in lvis["categories"]}
    img_map = {i["id"]: i for i in lvis["images"]}

    anns_by_img = defaultdict(list)
    for ann in lvis["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    candidates = [iid for iid, anns in anns_by_img.items() if 2 <= len(anns) <= 10]
    random.seed(123)
    selected = random.sample(candidates, min(n, len(candidates)))

    results = []
    for img_id in selected:
        info = img_map[img_id]
        fname = info.get("file_name", info.get("coco_url", "").split("/")[-1])
        path = os.path.join(LVIS_IMG, fname)
        if not os.path.exists(path):
            path = os.path.join(LVIS_IMG, os.path.basename(fname))
        if not os.path.exists(path):
            continue
        anns = anns_by_img[img_id]
        img = Image.open(path).convert("RGB")
        viz, labels = overlay_masks(img, anns, cats)
        results.append({
            "dataset": "LVIS",
            "image_id": img_id,
            "file_name": fname,
            "size": f"{info['width']}x{info['height']}",
            "n_objects": len(anns),
            "labels": labels,
            "has_masks": all(decode_segmentation(a, info["height"], info["width"]) is not None for a in anns[:6]),
            "viz_b64": img_to_b64(viz),
            "orig_b64": img_to_b64(img),
        })
    return results


def build_html(coco_samples, lvis_samples):
    rows = ""
    for s in coco_samples + lvis_samples:
        label_tags = " ".join(f'<span class="tag">{l}</span>' for l in s["labels"])
        rows += f"""
        <div class="card">
          <div class="header">
            <span class="ds {s['dataset'].lower()}">{s['dataset']}</span>
            <span class="meta">ID: {s['image_id']} | {s['size']} | {s['n_objects']} objects | masks: {"yes" if s['has_masks'] else "NO"}</span>
          </div>
          <div class="images">
            <div class="img-col">
              <div class="label">Original</div>
              <img src="data:image/jpeg;base64,{s['orig_b64']}" />
            </div>
            <div class="img-col">
              <div class="label">Masks + Text Labels</div>
              <img src="data:image/jpeg;base64,{s['viz_b64']}" />
            </div>
          </div>
          <div class="tags">Text queries: {label_tags}</div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>COCO & LVIS Data Pairs</title>
<style>
body {{ font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; max-width: 1200px; margin: 0 auto; }}
h1 {{ color: #e94560; margin-bottom: 5px; }}
h2 {{ color: #aaa; font-weight: normal; margin-top: 0; }}
.card {{ background: #16213e; border-radius: 12px; padding: 20px; margin: 20px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
.header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }}
.ds {{ padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 13px; }}
.ds.coco {{ background: #e94560; }}
.ds.lvis {{ background: #0f3460; }}
.meta {{ color: #888; font-size: 13px; }}
.images {{ display: flex; gap: 16px; }}
.img-col {{ flex: 1; }}
.img-col img {{ width: 100%; border-radius: 8px; }}
.label {{ font-size: 12px; color: #888; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 1px; }}
.tags {{ margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; }}
.tag {{ background: #0f3460; padding: 4px 10px; border-radius: 20px; font-size: 13px; color: #fff; }}
.summary {{ background: #0f3460; padding: 16px 20px; border-radius: 10px; margin: 20px 0; }}
.summary td {{ padding: 4px 16px; }}
</style></head><body>
<h1>COCO & LVIS — Image / Text / Mask Triplets</h1>
<h2>Training data samples for Stage 3 segmentation-enabled finetuning</h2>
<div class="summary">
<table>
<tr><td><b>COCO samples</b></td><td>{len(coco_samples)} images shown, each with category name (text) + instance masks</td></tr>
<tr><td><b>LVIS samples</b></td><td>{len(lvis_samples)} images shown, each with category name (text) + instance masks</td></tr>
<tr><td><b>Mask format</b></td><td>Polygon / RLE segmentation annotations (decoded to binary masks)</td></tr>
<tr><td><b>Text format</b></td><td>Category names used as text queries during training</td></tr>
</table>
</div>
{rows}
</body></html>"""
    return html


if __name__ == "__main__":
    coco_samples = sample_coco(N_SAMPLES)
    print(f"Sampled {len(coco_samples)} COCO images")
    lvis_samples = sample_lvis(N_SAMPLES)
    print(f"Sampled {len(lvis_samples)} LVIS images")

    html = build_html(coco_samples, lvis_samples)
    out_path = os.path.join(OUTPUT_DIR, "data_pairs.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Saved visualization to {out_path}")
