# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml
from tqdm import tqdm
import glob
from PIL import Image
import numpy as np
import logging
import sys
from termcolor import colored
from timm.scheduler.cosine_lr import CosineLRScheduler

from sam3.model_builder import build_sam3_video_model
from sam3.model.efficient_sam3 import EfficientSam3Tracker
from sam3.model.hybrid_memory import HybridMemoryModule
from sam3.model.vl_combiner import SAM3VLBackbone

# Simple logger utility
def create_logger(output_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = os.path.join(output_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Simple LR scheduler builder
def build_scheduler_from_args(args, optimizer, num_batches):
    num_steps = int(args.epochs * num_batches)
    warmup_steps = int(args.warmup_epochs * num_batches)
    
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=args.min_lr,
        warmup_lr_init=args.warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )
    return lr_scheduler

# Placeholder for dataset
class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.video_dirs = sorted(glob.glob(os.path.join(root_dir, "*")))
        self.transform = transform

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        video_dir = self.video_dirs[idx]
        frame_paths = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
        # Load frames
        frames = []
        for p in frame_paths[:8]: # Limit frames for memory training
            img = Image.open(p).convert("RGB")
            img = img.resize((1024, 1024)) # Resize to SAM input
            frames.append(np.array(img))
        
        frames = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
        return frames

def train(args):
    logger = create_logger(args.output_dir)
    logger.info(f"Starting Stage 2 Training with args: {args}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Teacher Model (SAM3)
    logger.info("Loading Teacher Model...")
    teacher_model = build_sam3_video_model(
        checkpoint_path=args.teacher_checkpoint,
        apply_temporal_disambiguation=True
    )
    teacher_model.to(device)
    teacher_model.eval()
    
    # 2. Load Student Model (EfficientSAM3)
    logger.info("Loading Student Model...")
    
    # To ensure the student uses the SAM3 original weights for encoders and other parts,
    # we will initialize the EfficientSam3Tracker using the Teacher's components where possible,
    # or load the SAM3 checkpoint into a new instance.
    
    # Since EfficientSam3Tracker needs a backbone, and we want the SAM3 backbone (ViT-Huge),
    # we can reuse the teacher's backbone to save memory and ensure identical weights.
    # The teacher_model is a Sam3VideoInference wrapper.
    # teacher_model.detector is Sam3ImageOnVideoMultiGPU
    # teacher_model.detector.backbone is SAM3VLBackbone
    
    teacher_backbone = teacher_model.detector.backbone
    
    # Create EfficientSam3Tracker with the Teacher's backbone
    student_tracker = EfficientSam3Tracker(
        backbone=teacher_backbone, # Shared backbone (Frozen)
        hybrid_memory_dim=args.memory_dim,
        num_global_latents=args.num_global_latents,
        num_spatial_latents=args.num_spatial_latents,
        window_size=args.window_size
    )
    
    # Load SAM3 weights into the rest of the student tracker (Mask Decoder, Prompt Encoder, etc.)
    # We can load from the teacher_model.tracker
    # teacher_model.tracker is Sam3TrackerPredictor
    
    # Copy weights from teacher tracker to student tracker
    # We need to be careful not to overwrite the new Hybrid Memory and Efficient Attention
    # if they have same names (they shouldn't, or we filter).
    
    teacher_tracker_state = teacher_model.tracker.state_dict()
    student_tracker_state = student_tracker.state_dict()
    
    # Filter out keys that don't match or are part of the new modules
    # New modules: maskmem_backbone (HybridMemory), transformer (EfficientAttention)
    # Old modules: maskmem_backbone (SimpleMaskEncoder), transformer (RoPEAttention)
    # We should NOT load maskmem_backbone and transformer keys.
    
    keys_to_load = {}
    for k, v in teacher_tracker_state.items():
        if k in student_tracker_state:
            # Skip memory backbone and transformer as they are architecturally different
            if "maskmem_backbone" in k or "transformer" in k:
                continue
            if v.shape == student_tracker_state[k].shape:
                keys_to_load[k] = v
            else:
                logger.warning(f"Skipping {k} due to shape mismatch: {v.shape} vs {student_tracker_state[k].shape}")
        else:
            # print(f"Skipping {k} (not in student)")
            pass
            
    missing, unexpected = student_tracker.load_state_dict(keys_to_load, strict=False)
    logger.info(f"Loaded SAM3 weights into Student Tracker. Missing: {len(missing)} (Expected for new modules)")
    
    student_tracker.to(device)
    student_tracker.train()
    
    # Freeze components that should be fixed (Backbone, Mask Decoder, Prompt Encoder)
    # Backbone is already shared from teacher (which is eval/frozen?)
    # teacher_model.eval() was called, so backbone is in eval mode.
    # But we should ensure requires_grad is False for safety.
    for param in student_tracker.backbone.parameters():
        param.requires_grad = False
        
    # Freeze SAM heads
    for param in student_tracker.sam_prompt_encoder.parameters():
        param.requires_grad = False
    for param in student_tracker.sam_mask_decoder.parameters():
        param.requires_grad = False
        
    # Only Hybrid Memory and Efficient Transformer should be trainable
    # Verify
    trainable_params = [p for p in student_tracker.parameters() if p.requires_grad]
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # Data
    # dataset = VideoDataset(args.dataset_path)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # Mock dataloader length
    num_batches = 100 
    
    lr_scheduler = build_scheduler_from_args(args, optimizer, num_batches)
    
    # Loss
    mse_loss = nn.MSELoss()
    
    logger.info("Starting Training...")
    # Mock training loop
    global_step = 0
    for epoch in range(args.epochs):
        # for batch in dataloader:
            # frames = batch.to(device)
            
            # 1. Extract Image Features (using Teacher or Stage 1 Encoder)
            # with torch.no_grad():
            #     image_features = teacher_model.image_encoder(frames)
            
            # 2. Teacher Forward
            # with torch.no_grad():
            #     teacher_mem_out = teacher_model.tracker(image_features)
            
            # 3. Student Forward
            # student_mem_out = student_tracker(image_features)
            
            # 4. Loss
            # loss = mse_loss(student_mem_out, teacher_mem_out)
            
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            # lr_scheduler.step_update(global_step)
            # global_step += 1
        pass
            
        logger.info(f"Epoch {epoch} complete.")
        
        # Save checkpoint
        torch.save(student_tracker.state_dict(), os.path.join(args.output_dir, f"epoch_{epoch}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="stage2/configs/efficient_sam3_stage2.yaml")
    parser.add_argument("--teacher_checkpoint", type=str, default="sam3_checkpoints/sam3.pt")
    parser.add_argument("--dataset_path", type=str, required=False)
    parser.add_argument("--output_dir", type=str, default="output/stage2")
    
    # Overrides
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4) # EdgeTAM uses 1e-4
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.05) # Common ViT setting
    
    parser.add_argument("--memory_dim", type=int, default=256)
    parser.add_argument("--num_global_latents", type=int, default=64) # EdgeTAM default
    parser.add_argument("--num_spatial_latents", type=int, default=192) # EdgeTAM default
    parser.add_argument("--window_size", type=int, default=8)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    train(args)
