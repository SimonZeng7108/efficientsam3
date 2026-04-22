"""SA-1B Dataset for Stage 3 End-to-End Fine-Tuning.

Each ``__getitem__`` returns **one** (image, caption, box, points, mask) tuple
drawn from a random annotation inside the image. This "one prompt per sample"
policy keeps text / geometry / mask prompts mutually consistent and matches
how SAM3 is queried at inference time.

Fields returned per sample:
    image                (3, S, S) float, normalized, padded to IMG_SIZE
    img_size_before_pad  (2,) int
    box_cxcywh_norm      (4,) float in [0, 1]
    points_norm          (K, 2) float in [0, 1]
    gt_mask              (1, S, S) float 0/1 in padded coords
    text                 str
    teacher_embedding    (C, He, We) or None
    teacher_valid        bool
    key                  str
"""

from __future__ import annotations

import glob
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageFile
from pycocotools import mask as mask_utils
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

from stage1.data.transforms import ResizeLongestSide

ImageFile.LOAD_TRUNCATED_IMAGES = True

TEXT_LABEL_KEYS = ("label_2", "label_5", "label_10")
PLACEHOLDER_PREFIX = "sa1b crop instance"
DUMMY_TEXT = "visual"


def _is_valid_text(value: Optional[str]) -> bool:
    return bool(value) and not value.startswith(PLACEHOLDER_PREFIX)


class SA1BStage3Dataset(torch.utils.data.Dataset):
    """SA-1B single-prompt dataset for Stage 3 text + geometry supervised FT."""

    def __init__(
        self,
        data_root: str,
        img_size: int = 1008,
        split: str = "train",
        num_samples: int = -1,
        pixel_mean: List[float] = (123.675, 116.28, 103.53),
        pixel_std: List[float] = (58.395, 57.12, 57.375),
        num_sample_points: int = 3,
        box_jitter: bool = True,
        box_jitter_frac: float = 0.1,
        sort_by_area: bool = True,
        teacher_embed_dir: Optional[str] = None,
        teacher_embed_dtype: str = "float32",
        text_label_mode: str = "random",
        deterministic: bool = False,
    ):
        super().__init__()
        self.data_root = data_root
        self.img_size = img_size
        self.split = split
        self.pixel_mean = torch.tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.tensor(pixel_std).view(-1, 1, 1)
        self.transform = ResizeLongestSide(img_size)

        self.num_sample_points = num_sample_points
        self.box_jitter = box_jitter
        self.box_jitter_frac = box_jitter_frac
        self.sort_by_area = sort_by_area

        self.teacher_embed_dir = teacher_embed_dir
        dtype_str = str(teacher_embed_dtype).lower()
        self.teacher_embed_dtype = (
            torch.float16 if dtype_str in ("float16", "fp16", "half") else torch.float32
        )

        self.text_label_mode = text_label_mode
        self.deterministic = deterministic
        self.num_samples = num_samples
        self._epoch = 0

        self._prepare_data()

    def _prepare_data(self):
        self.data: List[tuple] = []
        self.keys: List[str] = []

        anno_dir = os.path.join(self.data_root, "annotations", self.split)
        img_dir = os.path.join(self.data_root, "images", self.split)

        enhanced_files = sorted(glob.glob(os.path.join(anno_dir, "*_enhanced.json")))
        for anno_path in enhanced_files:
            stem = Path(anno_path).stem.replace("_enhanced", "")
            img_path = os.path.join(img_dir, f"{stem}.jpg")
            if not os.path.exists(img_path):
                continue
            self.data.append((img_path, anno_path))
            self.keys.append(stem)
            if 0 < self.num_samples <= len(self.data):
                break

        print(
            f"[SA1BStage3Dataset][{self.split}]: {len(self.data)} images "
            f"with enhanced annotations"
        )

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.data)

    def _rng(self, idx: int) -> np.random.Generator:
        if self.deterministic:
            seed = (idx * 1_000_003 + self._epoch * 97) & 0xFFFFFFFF
            return np.random.default_rng(seed)
        return np.random.default_rng()

    def __getitem__(self, idx: int, _retry: int = 0) -> Optional[Dict[str, torch.Tensor]]:
        if _retry >= 32:
            raise RuntimeError(f"Too many retries at index {idx}")

        img_path, anno_path = self.data[idx]
        key = self.keys[idx]
        rng = self._rng(idx)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self), _retry + 1)

        img_t = pil_to_tensor(img).float()
        orig_h, orig_w = img_t.shape[1:]

        with open(anno_path, "r") as f:
            anno_json = json.load(f)

        height = anno_json["image"]["height"]
        width = anno_json["image"]["width"]
        annotations = anno_json["annotations"]
        if len(annotations) == 0:
            return self.__getitem__((idx + 1) % len(self), _retry + 1)

        valid_indices = [
            i
            for i, r in enumerate(annotations)
            if any(_is_valid_text(r.get(k)) for k in TEXT_LABEL_KEYS)
        ]
        if not valid_indices:
            valid_indices = list(range(len(annotations)))

        if self.sort_by_area:
            areas = np.array(
                [
                    annotations[i].get("area", annotations[i]["bbox"][2] * annotations[i]["bbox"][3])
                    for i in valid_indices
                ],
                dtype=np.float64,
            )
            probs = areas / (areas.sum() + 1e-6)
            picked = valid_indices[int(rng.choice(len(valid_indices), p=probs))]
        else:
            picked = valid_indices[int(rng.integers(0, len(valid_indices)))]

        record = annotations[picked]

        text = self._pick_text_label(record, rng)

        bbox_xywh = np.asarray(record["bbox"], dtype=np.float32)
        if self.box_jitter:
            bbox_xywh = self._jitter_xywh(bbox_xywh, width, height, rng)
        x, y, w, h = bbox_xywh.tolist()
        box_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)

        sample_pts = record.get("mask_sample_points_xy", [])
        if sample_pts and self.num_sample_points > 0:
            pts = np.asarray(sample_pts, dtype=np.float32)
            k = min(self.num_sample_points, len(pts))
            sel = rng.choice(len(pts), size=k, replace=False)
            points_xy = pts[sel]
        else:
            pc = record.get("point_coords", [[x + w / 2, y + h / 2]])
            points_xy = np.asarray(pc, dtype=np.float32).reshape(-1, 2)[:1]

        seg = record["segmentation"]
        if isinstance(seg, list):
            rles = mask_utils.frPyObjects(seg, height, width)
            rle = mask_utils.merge(rles)
        elif isinstance(seg["counts"], list):
            rle = mask_utils.frPyObjects(seg, height, width)
        else:
            rle = seg
        gt_mask_np = mask_utils.decode(rle)
        if gt_mask_np.ndim == 3:
            gt_mask_np = gt_mask_np.max(axis=-1)

        img_t = self.transform.apply_image_torch(img_t[None]).squeeze(0)
        img_t = self._normalize(img_t)
        box_xyxy_t = torch.from_numpy(box_xyxy).unsqueeze(0)
        box_xyxy_t = self.transform.apply_boxes_torch(box_xyxy_t, (orig_h, orig_w))[0]
        pts_t = torch.from_numpy(points_xy)
        pts_t = self.transform.apply_coords_torch(pts_t, (orig_h, orig_w))
        gt_mask_t = torch.from_numpy(gt_mask_np).float().unsqueeze(0)
        gt_mask_t = self.transform.apply_masks_torch(gt_mask_t, (orig_h, orig_w))

        pre_pad_size = torch.tensor(img_t.shape[1:], dtype=torch.int32)
        img_t = self._pad(img_t)
        gt_mask_t = self._pad(gt_mask_t)

        box_xyxy_t = box_xyxy_t.clamp(0, self.img_size - 1)
        box_norm = box_xyxy_t / self.img_size
        cx = (box_norm[0] + box_norm[2]) / 2
        cy = (box_norm[1] + box_norm[3]) / 2
        bw = (box_norm[2] - box_norm[0]).clamp(min=1e-4)
        bh = (box_norm[3] - box_norm[1]).clamp(min=1e-4)
        box_cxcywh_norm = torch.stack([cx, cy, bw, bh])

        pts_t = pts_t.clamp(0, self.img_size - 1) / self.img_size

        teacher_embedding = None
        teacher_valid = False
        if self.teacher_embed_dir:
            emb_path = os.path.join(self.teacher_embed_dir, f"{key}.pt")
            if os.path.exists(emb_path):
                try:
                    emb = torch.load(emb_path, map_location="cpu", weights_only=True)
                    teacher_embedding = emb.to(self.teacher_embed_dtype)
                    teacher_valid = True
                except Exception:
                    teacher_embedding = None

        return {
            "image": img_t,
            "img_size_before_pad": pre_pad_size,
            "box_cxcywh_norm": box_cxcywh_norm,
            "points_norm": pts_t,
            "gt_mask": gt_mask_t,
            "text": text,
            "teacher_embedding": teacher_embedding,
            "teacher_valid": teacher_valid,
            "key": key,
        }

    def _pick_text_label(self, record: dict, rng: np.random.Generator) -> str:
        available: Dict[str, str] = {}
        for k in TEXT_LABEL_KEYS:
            v = record.get(k)
            if _is_valid_text(v):
                available[k] = v

        if not available:
            return DUMMY_TEXT

        mode = self.text_label_mode
        if mode in available:
            return available[mode]
        if mode == "random":
            keys = list(available.keys())
            return available[keys[int(rng.integers(0, len(keys)))]]
        return next(iter(available.values()))

    def _jitter_xywh(
        self,
        bbox_xywh: np.ndarray,
        img_w: float,
        img_h: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        x, y, w, h = bbox_xywh.tolist()
        jitter_x = self.box_jitter_frac * w
        jitter_y = self.box_jitter_frac * h
        cx = x + w / 2 + rng.normal(0, jitter_x)
        cy = y + h / 2 + rng.normal(0, jitter_y)
        w = max(1.0, w * (1 + rng.normal(0, self.box_jitter_frac)))
        h = max(1.0, h * (1 + rng.normal(0, self.box_jitter_frac)))
        x = float(np.clip(cx - w / 2, 0, img_w - 1))
        y = float(np.clip(cy - h / 2, 0, img_h - 1))
        w = float(np.clip(w, 1.0, img_w - x))
        h = float(np.clip(h, 1.0, img_h - y))
        return np.array([x, y, w, h], dtype=np.float32)

    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.pixel_mean) / self.pixel_std

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return F.pad(x, (0, self.img_size - w, 0, self.img_size - h))


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Stack the per-sample tensors. Points are padded to the max K in the batch."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}

    images = torch.stack([b["image"] for b in batch])
    img_sizes = torch.stack([b["img_size_before_pad"] for b in batch])
    boxes = torch.stack([b["box_cxcywh_norm"] for b in batch])
    gt_masks = torch.stack([b["gt_mask"] for b in batch])

    max_k = max(b["points_norm"].shape[0] for b in batch)
    B = len(batch)
    points = torch.zeros(B, max_k, 2)
    point_mask = torch.ones(B, max_k, dtype=torch.bool)
    for i, b in enumerate(batch):
        k = b["points_norm"].shape[0]
        if k > 0:
            points[i, :k] = b["points_norm"]
            point_mask[i, :k] = False

    texts = [b["text"] for b in batch]
    keys = [b["key"] for b in batch]

    teacher_valid = torch.tensor([b["teacher_valid"] for b in batch], dtype=torch.bool)
    teacher_embeddings = None
    if teacher_valid.any():
        shapes = {
            tuple(b["teacher_embedding"].shape)
            for b in batch
            if b["teacher_embedding"] is not None
        }
        if len(shapes) == 1:
            ref_shape = shapes.pop()
            teacher_embeddings = torch.zeros(B, *ref_shape)
            for i, b in enumerate(batch):
                if b["teacher_embedding"] is not None:
                    teacher_embeddings[i] = b["teacher_embedding"].float()
        else:
            teacher_valid = torch.zeros(B, dtype=torch.bool)

    return {
        "images": images,
        "img_sizes": img_sizes,
        "boxes_cxcywh_norm": boxes,
        "points_norm": points,
        "point_mask": point_mask,
        "gt_mask": gt_masks,
        "texts": texts,
        "keys": keys,
        "teacher_embeddings": teacher_embeddings,
        "teacher_valid": teacher_valid,
    }
