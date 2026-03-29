#!/usr/bin/env python3
"""
Save COCO and LVIS image-text-mask triplets as PNG files for review.
Each image shows: original on left, mask overlay on right, text labels listed.
"""

import json
import os
import random
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

N_SAMPLES = 6
COLORS = [
    (255, 50, 50), (50, 255, 50), (50, 100, 255), (255, 255, 50),
    (255, 50, 255), (50, 255, 255), (200, 100, 255), (255, 180, 50),
    (50, 200, 255), (180, 255, 50), (255, 50, 180), (50, 255, 180),
]


def get_font(size=16):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def decode_segmentation(ann, h, w):
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


def create_viz_panel(image, anns, cat_id_to_name, dataset_name, img_id, max_anns=8):
    """Create a side-by-side panel: original | mask overlay, with text labels below."""
    w, h = image.size
    target_w = 500
    ratio = target_w / w
    target_h = int(h * ratio)
    img_resized = image.resize((target_w, target_h), Image.LANCZOS)

    overlay_img = img_resized.copy().convert("RGBA")
    labels_with_colors = []

    for i, ann in enumerate(anns[:max_anns]):
        mask = decode_segmentation(ann, h, w)
        if mask is None:
            continue
        mask_resized = np.array(
            Image.fromarray(mask).resize((target_w, target_h), Image.NEAREST)
        )
        color = COLORS[i % len(COLORS)]
        cat_name = cat_id_to_name.get(ann["category_id"], f"id_{ann['category_id']}")
        labels_with_colors.append((cat_name, color))

        mask_rgba = np.zeros((target_h, target_w, 4), dtype=np.uint8)
        mask_rgba[mask_resized > 0] = (*color, 120)
        overlay_layer = Image.fromarray(mask_rgba, "RGBA")
        overlay_img = Image.alpha_composite(overlay_img, overlay_layer)

        draw = ImageDraw.Draw(overlay_img)
        ys, xs = np.where(mask_resized > 0)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            font = get_font(13)
            bbox = draw.textbbox((cx, cy), cat_name, font=font)
            draw.rectangle(
                [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
                fill=(0, 0, 0, 180),
            )
            draw.text((cx, cy), cat_name, fill=(*color, 255), font=font)

    overlay_rgb = overlay_img.convert("RGB")

    label_h = 30 + 22 * ((len(labels_with_colors) + 3) // 4)
    panel_w = target_w * 2 + 30
    panel_h = target_h + 60 + label_h
    panel = Image.new("RGB", (panel_w, panel_h), (25, 25, 45))
    draw = ImageDraw.Draw(panel)

    title_font = get_font(18)
    small_font = get_font(13)
    label_font = get_font(14)

    draw.text(
        (15, 8),
        f"{dataset_name}  |  ID: {img_id}  |  {w}x{h}  |  {len(anns)} objects  |  masks: {'yes' if labels_with_colors else 'no'}",
        fill=(200, 200, 200),
        font=title_font,
    )

    y_off = 38
    draw.text((15, y_off), "Original", fill=(150, 150, 150), font=small_font)
    panel.paste(img_resized, (15, y_off + 18))
    draw.text((target_w + 20, y_off), "Segmentation Masks + Text Labels", fill=(150, 150, 150), font=small_font)
    panel.paste(overlay_rgb, (target_w + 20, y_off + 18))

    label_y = y_off + 18 + target_h + 8
    draw.text((15, label_y), "Text queries:", fill=(180, 180, 180), font=small_font)
    x_cursor = 120
    for name, color in labels_with_colors:
        tag_w = label_font.getlength(name) + 16
        if x_cursor + tag_w > panel_w - 15:
            label_y += 22
            x_cursor = 15
        draw.rounded_rectangle(
            [x_cursor, label_y, x_cursor + tag_w, label_y + 20],
            radius=10,
            fill=(*color, 40),
            outline=color,
        )
        draw.text((x_cursor + 8, label_y + 2), name, fill=color, font=label_font)
        x_cursor += tag_w + 8

    return panel


def sample_and_viz_coco():
    print(f"Loading COCO annotations...")
    coco = COCO(COCO_ANN)
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    candidates = [
        iid for iid in coco.getImgIds()
        if 3 <= len(coco.getAnnIds(imgIds=iid, iscrowd=False)) <= 8
    ]
    random.seed(42)
    selected = random.sample(candidates, min(N_SAMPLES, len(candidates)))

    panels = []
    for img_id in selected:
        info = coco.loadImgs(img_id)[0]
        path = os.path.join(COCO_IMG, info["file_name"])
        if not os.path.exists(path):
            continue
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
        img = Image.open(path).convert("RGB")
        panel = create_viz_panel(img, anns, cats, "COCO", img_id)
        panels.append(panel)
    return panels


def sample_and_viz_lvis():
    print(f"Loading LVIS annotations...")
    with open(LVIS_ANN) as f:
        lvis = json.load(f)

    cats = {c["id"]: c["name"].replace("_", " ") for c in lvis["categories"]}
    img_map = {i["id"]: i for i in lvis["images"]}

    anns_by_img = defaultdict(list)
    for ann in lvis["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    candidates = [iid for iid, a in anns_by_img.items() if 3 <= len(a) <= 8]
    random.seed(123)
    selected = random.sample(candidates, min(N_SAMPLES, len(candidates)))

    panels = []
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
        panel = create_viz_panel(img, anns, cats, "LVIS", img_id)
        panels.append(panel)
    return panels


if __name__ == "__main__":
    coco_panels = sample_and_viz_coco()
    lvis_panels = sample_and_viz_lvis()

    all_panels = coco_panels + lvis_panels
    for i, panel in enumerate(all_panels):
        ds = "coco" if i < len(coco_panels) else "lvis"
        idx = i if i < len(coco_panels) else i - len(coco_panels)
        path = os.path.join(OUTPUT_DIR, f"{ds}_sample_{idx}.png")
        panel.save(path)
        print(f"Saved {path}")

    print(f"\nTotal: {len(coco_panels)} COCO + {len(lvis_panels)} LVIS samples saved to {OUTPUT_DIR}/")
