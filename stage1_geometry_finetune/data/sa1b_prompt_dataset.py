"""
SA1B Dataset with Prompt Loading for Stage 2 Geometry Fine-tuning.

This dataset loads:
1. Images from SA-1B
2. Box and point prompts from annotations
3. Saved teacher trunk embeddings (from Stage 1 embedding saving)
"""

import os
import copy
import json
import glob
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image, ImageFile
from pycocotools import mask as mask_utils

from stage1.data.transforms import ResizeLongestSide


ImageFile.LOAD_TRUNCATED_IMAGES = True


class SA1BPromptDataset(torch.utils.data.Dataset):
    """
    SA-1B dataset with prompt loading for Stage 2 geometry fine-tuning.
    
    Returns:
        - image: Preprocessed image tensor
        - boxes: Box prompts (N, 4) in xyxy format
        - points: Point prompts (N, 2)
        - point_labels: Point labels (N,) - 1 for foreground
        - img_size_before_pad: Original size before padding (for valid masking)
        - teacher_embedding: Saved teacher trunk embedding (optional)
    """
    
    def __init__(
        self,
        data_root: str,
        img_size: int = 1024,
        split: str = 'train',
        num_samples: int = -1,
        sort_by_area: bool = True,
        filter_by_area: Optional[Tuple[float, float]] = None,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        load_gt_mask: bool = False,
        max_prompts_per_image: int = 16,
        min_prompts_per_image: int = 1,
        fix_seed: bool = False,
        mask_nms_thresh: float = 0.8,
        box_jitter: bool = True,
        use_box_prompts: bool = True,
        use_point_prompts: bool = True,
        # Teacher embedding loading
        teacher_embed_path: Optional[str] = None,
        embed_dim: int = 1024,
        embed_size: int = 72,  # Match Stage 1 teacher (1008/14=72)
    ):
        super().__init__()
        
        self.data_root = data_root
        self.img_size = img_size
        self.split = split
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        self.transform = ResizeLongestSide(img_size)
        
        # Prompt settings
        self.sort_by_area = sort_by_area
        self.filter_by_area = filter_by_area
        self.load_gt_mask = load_gt_mask
        self.max_prompts_per_image = max_prompts_per_image
        self.min_prompts_per_image = min_prompts_per_image
        self.fix_seed = fix_seed
        self.mask_nms_thresh = mask_nms_thresh
        self.box_jitter = box_jitter
        self.use_box_prompts = use_box_prompts
        self.use_point_prompts = use_point_prompts
        
        # Teacher embedding settings
        self.teacher_embed_path = teacher_embed_path
        self.embed_dim = embed_dim
        self.embed_size = embed_size
        self.embed_shape = (embed_dim, embed_size, embed_size)
        
        # Load embedding manager if path provided
        # Embedding format: 4 bytes seed (int32) + embedding (float16)
        # Embedding shape: (embed_dim, embed_size, embed_size)
        # Item size = 4 + embed_dim * embed_size * embed_size * 2 (float16 = 2 bytes)
        self.embed_manager = None
        if teacher_embed_path and os.path.exists(teacher_embed_path):
            from stage1.data.augmentation.manager import TxtManager
            item_size = 4 + embed_dim * embed_size * embed_size * 2  # seed + embedding
            self.embed_manager = TxtManager(teacher_embed_path, item_size, rank=0)
        
        self.num_samples = num_samples
        self.prepare_data()
        
        # For reproducibility with distributed training
        self._epoch = 0
    
    def prepare_data(self):
        """Load image paths and annotation paths."""
        self.data = []
        self.keys = []
        
        counter = 0
        img_pattern = f'{self.data_root}/images/{self.split}/*.jpg'
        
        for img_path in sorted(glob.glob(img_pattern)):
            name = Path(img_path).stem
            anno_path = f'{self.data_root}/annotations/{self.split}/{name}.json'
            
            if not os.path.exists(anno_path):
                continue
            
            self.data.append((img_path, anno_path))
            self.keys.append(name)
            
            counter += 1
            if self.num_samples > 0 and counter >= self.num_samples:
                break
        
        print(f"SA1BPromptDataset: Loaded {len(self.data)} samples from {self.split}")
    
    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling."""
        self._epoch = epoch
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, anno_path = copy.deepcopy(self.data[idx])
        key = self.keys[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img = pil_to_tensor(img)
        original_size = img.shape[1:]  # (H, W)
        
        # Load annotations
        with open(anno_path, 'r') as f:
            anno_json = json.load(f)
        
        height, width = anno_json['image']['height'], anno_json['image']['width']
        anno_raw = anno_json['annotations']
        
        # Extract prompts
        boxes_list = []
        points_list = []
        masks_list = []
        areas = []
        
        for record in anno_raw:
            # Box prompt (xywh format)
            box = np.asarray(record['bbox'])
            boxes_list.append(box)
            
            # Point prompt
            point = np.asarray(record['point_coords'])
            points_list.append(point)
            
            # Optional mask
            if self.load_gt_mask:
                segm = record['segmentation']
                if type(segm) == list:
                    rles = mask_utils.frPyObjects(segm, height, width)
                    rle = mask_utils.merge(rles)
                elif type(segm['counts']) == list:
                    rle = mask_utils.frPyObjects(segm, height, width)
                else:
                    rle = segm
                mask = mask_utils.decode(rle)
                masks_list.append(mask)
            
            areas.append(record['area'])
        
        areas = np.array(areas)
        
        # Apply mask NMS if requested
        if self.mask_nms_thresh > 0 and self.load_gt_mask and len(masks_list) > 0:
            boxes_list, points_list, masks_list = self._apply_mask_nms(
                boxes_list, points_list, masks_list, areas
            )
        
        # Convert to tensors
        boxes = torch.from_numpy(np.stack(boxes_list, axis=0))  # (N, 4) xywh
        points = torch.from_numpy(np.stack(points_list, axis=0))  # (N, 1, 2) or (N, 2)
        
        if len(points.shape) == 3:
            points = points.squeeze(1)  # (N, 2)
        
        if self.load_gt_mask and len(masks_list) > 0:
            gt_masks = torch.from_numpy(np.stack(masks_list, axis=0))
        else:
            gt_masks = None
        
        # Apply box jittering for data augmentation
        if self.box_jitter:
            boxes = self._jitter_boxes(boxes, height, width)
        
        # Sort by area if requested (larger objects first)
        if self.sort_by_area:
            area = boxes[:, 2] * boxes[:, 3]
            indices = torch.argsort(area, descending=True)
            boxes = boxes[indices]
            points = points[indices]
            if gt_masks is not None:
                gt_masks = gt_masks[indices]
        
        # Apply transforms (resize)
        # NOTE: ResizeLongestSide.apply_boxes_torch expects boxes in xyxy.
        img = self.transform.apply_image_torch(img[None].float()).squeeze(0)
        boxes_xyxy = self._xywh_to_xyxy(boxes)
        boxes_xyxy = self.transform.apply_boxes_torch(boxes_xyxy, original_size)
        points = self.transform.apply_coords_torch(points, original_size)
        if gt_masks is not None:
            gt_masks = self.transform.apply_masks_torch(gt_masks, original_size)
        
        # Record size before padding
        img_size_before_pad = torch.tensor(img.shape[1:], dtype=torch.int32)  # (H, W)
        
        # Normalize and pad
        img = self._pad(self._norm(img))
        if gt_masks is not None:
            gt_masks = self._pad(gt_masks)
        
        # Filter by area if requested
        if self.filter_by_area is not None:
            boxes_xyxy, points, gt_masks = self._filter_by_area(
                boxes_xyxy, points, gt_masks
            )
        
        # Limit number of prompts
        num_prompts = boxes_xyxy.shape[0]
        if num_prompts > self.max_prompts_per_image:
            if self.fix_seed:
                torch.manual_seed(idx + self._epoch * 10000)
            selected = torch.randperm(num_prompts)[:self.max_prompts_per_image]
            boxes_xyxy = boxes_xyxy[selected]
            points = points[selected]
            if gt_masks is not None:
                gt_masks = gt_masks[selected]
        
        # Ensure minimum prompts
        if boxes_xyxy.shape[0] < self.min_prompts_per_image:
            return self.__getitem__((idx + 1) % len(self))
        
        # Point labels (all foreground for now)
        point_labels = torch.ones(points.shape[0], dtype=torch.long)
        
        # Normalize boxes to [0, 1]
        boxes_normalized = boxes_xyxy / self.img_size
        points_normalized = points / self.img_size
        
        result = {
            'image': img,
            'boxes': boxes_normalized if self.use_box_prompts else None,
            'points': points_normalized if self.use_point_prompts else None,
            'point_labels': point_labels if self.use_point_prompts else None,
            'img_size_before_pad': img_size_before_pad,
            'key': key,
        }
        
        if gt_masks is not None:
            result['gt_masks'] = gt_masks
        
        # Load teacher embedding if available
        if self.embed_manager is not None:
            try:
                raw_data = self.embed_manager.read(key)
                seed_size = 4  # 1 int32 value for seed (4 bytes)
                embedding_data = np.frombuffer(raw_data[seed_size:], dtype=np.float16)
                teacher_embedding = torch.from_numpy(
                    embedding_data.reshape(self.embed_shape).copy()
                ).float()
                result['teacher_embedding'] = teacher_embedding
            except Exception as e:
                # If embedding not found, skip this sample
                print(f"Warning: Could not load embedding for {key}: {e}")
                return self.__getitem__((idx + 1) % len(self))
        
        return result
    
    def _apply_mask_nms(self, boxes_list, points_list, masks_list, areas):
        """Apply mask NMS to remove highly overlapping masks."""
        tot_mask = np.zeros_like(masks_list[0])
        keep_indices = []
        sort_idx = np.argsort(-areas)
        
        for i in sort_idx:
            gt_mask = masks_list[i]
            if gt_mask.sum() == 0:
                continue
            ratio = (gt_mask * tot_mask).sum() / gt_mask.sum()
            if ratio <= self.mask_nms_thresh:
                keep_indices.append(i)
                tot_mask = (tot_mask + gt_mask).clip(max=1)
        
        # Reorder to keep original relative order
        keep_indices = sorted(keep_indices)
        
        boxes_list = [boxes_list[i] for i in keep_indices]
        points_list = [points_list[i] for i in keep_indices]
        masks_list = [masks_list[i] for i in keep_indices]
        
        return boxes_list, points_list, masks_list
    
    def _jitter_boxes(self, boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Apply random jittering to box coordinates."""
        # boxes: (N, 4) in xywh format
        boxes_xyxy = self._xywh_to_xyxy(boxes)
        N = boxes_xyxy.shape[0]
        
        # Random jitter proportional to box size
        w = boxes[:, 2]
        h = boxes[:, 3]
        
        delta_w = (w[:, None] * 0.1 * torch.randn(N, 2)).clamp(-20, 20)
        delta_h = (h[:, None] * 0.1 * torch.randn(N, 2)).clamp(-20, 20)
        
        boxes_xyxy[:, 0::2] = (boxes_xyxy[:, 0::2] + delta_w).clamp(0, width)
        boxes_xyxy[:, 1::2] = (boxes_xyxy[:, 1::2] + delta_h).clamp(0, height)
        
        # Convert back to xywh
        return self._xyxy_to_xywh(boxes_xyxy)
    
    def _xywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from xywh to xyxy format."""
        x, y, w, h = boxes.unbind(-1)
        return torch.stack([x, y, x + w, y + h], dim=-1)
    
    def _xyxy_to_xywh(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from xyxy to xywh format."""
        x1, y1, x2, y2 = boxes.unbind(-1)
        return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)
    
    def _filter_by_area(self, boxes, points, masks):
        """Filter prompts by box area."""
        area_min, area_max = self.filter_by_area
        area_min = float('-inf') if area_min is None else eval(str(area_min))
        area_max = float('inf') if area_max is None else eval(str(area_max))
        
        # boxes in xyxy format
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        area = w * h
        
        selected = (area >= area_min) & (area <= area_max)
        if selected.sum() > 0:
            boxes = boxes[selected]
            points = points[selected]
            if masks is not None:
                masks = masks[selected]
        
        return boxes, points, masks
    
    def _norm(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize image."""
        return (img - self.pixel_mean) / self.pixel_std
    
    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad to square."""
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        return F.pad(x, (0, padw, 0, padh))


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-length prompts."""
    # Stack images
    images = torch.stack([item['image'] for item in batch])
    img_sizes = torch.stack([item['img_size_before_pad'] for item in batch])
    keys = [item['key'] for item in batch]
    
    # Pad prompts to max length in batch
    max_prompts = max(item['boxes'].shape[0] if item['boxes'] is not None else 0 
                      for item in batch)
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    boxes = torch.zeros(batch_size, max_prompts, 4)
    points = torch.zeros(batch_size, max_prompts, 2)
    point_labels = torch.zeros(batch_size, max_prompts, dtype=torch.long)
    prompt_mask = torch.ones(batch_size, max_prompts, dtype=torch.bool)  # True = masked
    
    for i, item in enumerate(batch):
        if item['boxes'] is not None:
            n = item['boxes'].shape[0]
            boxes[i, :n] = item['boxes']
            prompt_mask[i, :n] = False
        if item['points'] is not None:
            n = item['points'].shape[0]
            points[i, :n] = item['points']
            point_labels[i, :n] = item['point_labels']
    
    result = {
        'images': images,
        'boxes': boxes,
        'points': points,
        'point_labels': point_labels,
        'prompt_mask': prompt_mask,
        'img_sizes': img_sizes,
        'keys': keys,
    }
    
    # Handle teacher embeddings
    if 'teacher_embedding' in batch[0]:
        embeddings = torch.stack([item['teacher_embedding'] for item in batch])
        result['teacher_embeddings'] = embeddings
    
    # Handle GT masks if present
    if 'gt_masks' in batch[0] and batch[0]['gt_masks'] is not None:
        # Pad masks to max prompts
        max_prompts = max(item['gt_masks'].shape[0] for item in batch 
                         if 'gt_masks' in item and item['gt_masks'] is not None)
        mask_h, mask_w = batch[0]['gt_masks'].shape[-2:]
        gt_masks = torch.zeros(batch_size, max_prompts, mask_h, mask_w)
        for i, item in enumerate(batch):
            if 'gt_masks' in item and item['gt_masks'] is not None:
                n = item['gt_masks'].shape[0]
                gt_masks[i, :n] = item['gt_masks']
        result['gt_masks'] = gt_masks
    
    return result
