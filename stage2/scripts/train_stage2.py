# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import glob
import numpy as np
import logging
import sys
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.append(os.getcwd())

from sam3.model_builder import build_sam3_video_model
from sam3.model.efficient_sam3 import EfficientSam3Tracker

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

# Logger
def create_logger(output_dir, rank=0):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    # Only log to console/file on rank 0
    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        if output_dir:
            file_handler = logging.FileHandler(os.path.join(output_dir, 'training.log'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger

class CachedFeatureDataset(Dataset):
    def __init__(self, root_dir, seq_len=8):
        self.root_dir = root_dir
        self.video_dirs = sorted(glob.glob(os.path.join(root_dir, "*")))
        # Filter directories only
        self.video_dirs = [d for d in self.video_dirs if os.path.isdir(d)]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        video_dir = self.video_dirs[idx]
        feat_path = os.path.join(video_dir, "features.pt")
        
        try:
            data = torch.load(feat_path, map_location="cpu")
            feats = data["feat"] # [N, C, H, W]
            pos = data["pos"]    # [N, C, H, W]
            
            N = feats.shape[0]
            if N > self.seq_len:
                # Random crop or start
                start = np.random.randint(0, N - self.seq_len)
                feats = feats[start:start+self.seq_len]
                pos = pos[start:start+self.seq_len]
            else:
                # Pad if needed (simple repeat)
                while feats.shape[0] < self.seq_len:
                    feats = torch.cat([feats, feats], dim=0)
                    pos = torch.cat([pos, pos], dim=0)
                feats = feats[:self.seq_len]
                pos = pos[:self.seq_len]
                
            return feats, pos
        except Exception as e:
            # print(f"Error loading {feat_path}: {e}")
            # Return dummy
            return torch.zeros(self.seq_len, 256, 64, 64), torch.zeros(self.seq_len, 256, 64, 64)

def _to_sequence(feat: torch.Tensor) -> torch.Tensor:
    # input [B, C, H, W] -> [HW, B, C]
    return feat.flatten(2).permute(2, 0, 1).contiguous()

def train(args):
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logger = create_logger(args.output_dir, rank)
        logger.info(f"Starting Stage 2 Training with args: {args}")
        logger.info(f"Distributed Init: Rank {rank}/{world_size}, Local Rank {local_rank}")
    else:
        logger = create_logger(None, rank)
    
    # 1. Load Teacher Model (SAM3)
    if rank == 0:
        logger.info("Loading Teacher Model (SAM3)...")
    
    try:
        teacher_model = build_sam3_video_model(
            checkpoint_path=args.teacher_checkpoint,
            apply_temporal_disambiguation=True,
            device=device
        )
    except Exception as e:
        if rank == 0:
            logger.error(f"Error loading teacher: {e}")
        return

    teacher_model.eval()
    teacher_tracker = teacher_model.tracker
    
    # 2. Initialize Student Tracker (EfficientSAM3)
    if rank == 0:
        logger.info("Initializing Student Tracker...")
    student_tracker = EfficientSam3Tracker(
        hybrid_memory_dim=256,
        num_global_latents=64,
        num_spatial_latents=192,
        window_size=8,
        # Pass necessary components to match SAM3 structure if needed
        backbone=teacher_tracker.backbone, # Shared backbone (frozen)
        transformer=None, # Will be created by EfficientSam3Tracker init
        maskmem_backbone=None, # Will be created by EfficientSam3Tracker init
    )
    # Copy necessary components from teacher
    student_tracker.hidden_dim = teacher_tracker.hidden_dim
    student_tracker.sam_prompt_embed_dim = teacher_tracker.sam_prompt_embed_dim
    student_tracker.sam_image_embedding_size = teacher_tracker.sam_image_embedding_size
    # Student shares the prompt encoder and mask decoder from teacher (frozen)
    student_tracker.sam_prompt_encoder = teacher_tracker.sam_prompt_encoder
    student_tracker.sam_mask_decoder = teacher_tracker.sam_mask_decoder
    
    student_tracker.to(device)
    
    # Wrap in DDP
    student_tracker = DDP(student_tracker, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    student_tracker.train()
    
    # 3. Data Loader
    dataset = CachedFeatureDataset(args.dataset_path, seq_len=args.seq_len)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    
    # 4. Optimizer
    optimizer = optim.AdamW(student_tracker.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    
    # 5. Training Loop
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        if rank == 0:
            pbar = tqdm(dataloader)
        else:
            pbar = dataloader
        
        for batch_idx, (feats, pos) in enumerate(pbar):
            feats = feats.to(device) # [B, T, C, H, W]
            pos = pos.to(device)
            
            B, T, C, H, W = feats.shape
            
            # Initialize Memory for this batch
            teacher_out_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
            student_out_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
            
            # Frame 0: Initialization (Dummy Mask)
            dummy_mask = torch.zeros(B, 1, 256, 256).to(device)
            dummy_mask[:, :, 100:150, 100:150] = 1.0 # Center box
            
            f0 = feats[:, 0] # [B, C, H, W]
            p0 = pos[:, 0]
            
            # Convert to sequence: [HW, B, C]
            f0_seq = _to_sequence(f0)
            p0_seq = _to_sequence(p0)
            
            # Teacher Encode Frame 0
            with torch.no_grad():
                teacher_tracker.track_step(
                    frame_idx=0,
                    is_init_cond_frame=True,
                    current_vision_feats=[f0_seq],
                    current_vision_pos_embeds=[p0_seq],
                    feat_sizes=[(H, W)],
                    image=None,
                    point_inputs=None,
                    mask_inputs=dummy_mask,
                    output_dict=teacher_out_dict,
                    num_frames=T,
                    run_mem_encoder=True
                )
                
            # Student Encode Frame 0
            student_tracker.module.track_step(
                frame_idx=0,
                is_init_cond_frame=True,
                current_vision_feats=[f0_seq],
                current_vision_pos_embeds=[p0_seq],
                feat_sizes=[(H, W)],
                image=None,
                point_inputs=None,
                mask_inputs=dummy_mask,
                output_dict=student_out_dict,
                num_frames=T,
                run_mem_encoder=True
            )
            
            batch_loss = 0
            optimizer.zero_grad()
            
            # Loop over subsequent frames
            for t in range(1, T):
                ft = feats[:, t]
                pt = pos[:, t]
                
                ft_seq = _to_sequence(ft)
                pt_seq = _to_sequence(pt)
                
                # Teacher Step
                with torch.no_grad():
                    # 1. Get Target Features
                    teacher_feat_mem = teacher_tracker._prepare_memory_conditioned_features(
                        frame_idx=t,
                        is_init_cond_frame=False,
                        current_vision_feats=[ft_seq],
                        current_vision_pos_embeds=[pt_seq],
                        feat_sizes=[(H, W)],
                        output_dict=teacher_out_dict,
                        num_frames=T
                    )
                    
                    # 2. Advance Teacher State
                    teacher_out = teacher_tracker.track_step(
                        frame_idx=t,
                        is_init_cond_frame=False,
                        current_vision_feats=[ft_seq],
                        current_vision_pos_embeds=[pt_seq],
                        feat_sizes=[(H, W)],
                        image=None,
                        point_inputs=None,
                        mask_inputs=None,
                        output_dict=teacher_out_dict,
                        num_frames=T,
                        run_mem_encoder=True
                    )
                    
                    teacher_pred_mask = teacher_out["pred_masks_high_res"]
                    obj_score = teacher_out["object_score_logits"]
                    
                # Student Step
                # 1. Get Student Features
                student_feat_mem = student_tracker.module._prepare_memory_conditioned_features(
                    frame_idx=t,
                    is_init_cond_frame=False,
                    current_vision_feats=[ft_seq],
                    current_vision_pos_embeds=[pt_seq],
                    feat_sizes=[(H, W)],
                    output_dict=student_out_dict,
                    num_frames=T
                )
                
                # 2. Distillation Loss
                loss = loss_fn(student_feat_mem, teacher_feat_mem)
                batch_loss += loss
                
                # 3. Update Student Memory (Teacher Forcing)
                # We reuse track_step which calls _prepare_memory again, but this is necessary 
                # to trigger the mask decoder and memory encoder with the forced mask.
                student_tracker.module.track_step(
                    frame_idx=t,
                    is_init_cond_frame=False,
                    current_vision_feats=[ft_seq],
                    current_vision_pos_embeds=[pt_seq],
                    feat_sizes=[(H, W)],
                    image=None,
                    point_inputs=None,
                    mask_inputs=None,
                    output_dict=student_out_dict,
                    num_frames=T,
                    run_mem_encoder=True,
                    teacher_forced_mask=teacher_pred_mask,
                    teacher_forced_object_score=obj_score
                )
            
            batch_loss.backward()
            optimizer.step()
            
            if rank == 0:
                pbar.set_description(f"Loss: {batch_loss.item()/T:.4f}")
        
        # Save Checkpoint
        if rank == 0:
            ckpt_path = os.path.join(args.output_dir, f"stage2_epoch_{epoch+1}.pth")
            torch.save(student_tracker.module.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

    cleanup_distributed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="stage2/configs/efficient_sam3_stage2.yaml")
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/stage2")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    
    args = parser.parse_args()

    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        # Update args with config values if not explicitly provided via CLI
        if args.batch_size is None:
            args.batch_size = config.get("train", {}).get("batch_size", 2)
        if args.lr is None:
            args.lr = float(config.get("train", {}).get("lr", 1e-4))
        if args.epochs is None:
            args.epochs = config.get("train", {}).get("epochs", 10)
        if args.seq_len is None:
            args.seq_len = config.get("data", {}).get("seq_len", 8)
        
        # Also allow overriding dataset paths from config if not provided (though currently required)
        # But args.dataset_path is required in parser, so we can't easily override it here unless we make it optional.
        # For now, we'll leave required args as is.
        
    # Set defaults if still None
    if args.batch_size is None: args.batch_size = 2
    if args.lr is None: args.lr = 1e-4
    if args.epochs is None: args.epochs = 10
    if args.seq_len is None: args.seq_len = 8

    train(args)
