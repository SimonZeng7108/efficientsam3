# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
import logging
import os
import random
import sys
import time
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import yaml

# repo root for relative imports
sys.path.append(os.getcwd())

from sam3.model_builder import build_sam3_video_model
from sam3.model.hybrid_memory import HybridMemoryModule
from sam3.model.efficient_attention import EfficientRoPEAttention, Attention
from sam3.model.decoder import TransformerDecoderLayerv2, TransformerEncoderCrossAttention
from sam3.model.model_misc import TransformerWrapper
from stage2.data import Stage2SequenceDataset


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=rank
        )
        dist.barrier()
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def create_logger(output_dir: str, rank: int) -> logging.Logger:
    logger = logging.getLogger("stage2_train")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    if rank == 0:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_efficient_transformer(hidden_dim: int, mem_dim: int):
    self_attention = EfficientRoPEAttention(
        embedding_dim=hidden_dim,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_fa3=False,
        use_rope_real=False,
    )
    cross_attention = Attention(
        embedding_dim=hidden_dim,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=mem_dim,
    )
    encoder_layer = TransformerDecoderLayerv2(
        cross_attention_first=False,
        activation="relu",
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=self_attention,
        d_model=hidden_dim,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attention,
    )
    encoder = TransformerEncoderCrossAttention(
        remove_cross_attention_layers=[],
        batch_first=True,
        d_model=hidden_dim,
        frozen=False,
        pos_enc_at_input=True,
        layer=encoder_layer,
        num_layers=4,
        use_act_checkpoint=False,
    )
    return TransformerWrapper(encoder=encoder, decoder=None, d_model=hidden_dim)


def build_student_tracker(cfg: Dict[str, Any], device: torch.device):
    base_model = build_sam3_video_model(
        checkpoint_path=cfg["model"]["stage1_ckpt"],
        apply_temporal_disambiguation=True,
        device=device,
    )
    tracker = base_model.tracker
    mem_dim = cfg["model"]["hybrid_dim"]
    tracker.maskmem_backbone = HybridMemoryModule(
        dim=mem_dim,
        in_dim=tracker.hidden_dim,
        num_global_latents=cfg["model"]["num_global_latents"],
        num_spatial_latents=cfg["model"]["num_spatial_latents"],
        window_size=cfg["model"]["window_size"],
    )
    tracker.mem_dim = mem_dim
    # Re-init projection layer for object pointers as it might have been initialized with a different dimension
    tracker.obj_ptr_tpos_proj = torch.nn.Linear(tracker.hidden_dim, tracker.mem_dim).to(device)
    tracker.transformer = build_efficient_transformer(tracker.hidden_dim, mem_dim)
    # re-init spatial embeddings
    tracker.maskmem_tpos_enc = torch.nn.Parameter(
        torch.zeros(tracker.num_maskmem, 1, 1, tracker.mem_dim, device=device)
    )
    torch.nn.init.trunc_normal_(tracker.maskmem_tpos_enc, std=0.02)
    tracker.no_obj_embed_spatial = torch.nn.Parameter(
        torch.zeros(1, tracker.mem_dim, device=device)
    )
    torch.nn.init.trunc_normal_(tracker.no_obj_embed_spatial, std=0.02)
    tracker = tracker.to(device)
    for param in tracker.parameters():
        param.requires_grad = False
    for param in tracker.maskmem_backbone.parameters():
        param.requires_grad = True
    for param in tracker.transformer.parameters():
        param.requires_grad = True
    return tracker


def flatten_feat(feat: torch.Tensor) -> torch.Tensor:
    # [B, C, H, W] -> [HW, B, C]
    return feat.flatten(2).permute(2, 0, 1).contiguous()


def run_video(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    device: torch.device,
):
    tracker = model
    # Batch comes already sliced by dataset to seq_len
    # batch shape: [B, seq_len, C, H, W] -> wait, Dataset returns stack [seq_len, C, H, W] (no B yet)
    # DataLoader adds B dimension: [B, seq_len, C, H, W]
    # Since batch_size=1, we have [1, seq_len, C, H, W]
    
    # Let's squeeze batch dim if present, but actually we want to iterate over time 't'
    # The dataset returns fields with shape [T, ...] 
    # DataLoader collates them to [B, T, ...]
    
    b, t_steps = batch["f0"].shape[:2]
    
    # Assuming B=1 for now as per logic. 
    # If B > 1, we need to handle multiple videos in parallel, but tracker.track_step usually handles 1 video?
    # SAM2/3 tracker usually maintains state per object, but can handle batches if designed.
    # But here, let's iterate over batch dimension if needed or assert B=1.
    assert b == 1, "Batch size > 1 not fully supported in this simple loop"
    
    # Remove batch dimension
    f0 = batch["f0"][0].to(device)
    f1 = batch["f1"][0].to(device)
    f2 = batch["f2"][0].to(device)
    p0 = batch["p0"][0].to(device)
    p1 = batch["p1"][0].to(device)
    p2 = batch["p2"][0].to(device)
    teacher_feat = batch["teacher_feat"][0].to(device)
    teacher_mask = batch["mask_high_res"][0].to(device)
    teacher_scores = batch["object_scores"][0].to(device)
    is_init_flags = batch["is_init"][0].to(device)

    student_out = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    losses = []

    # Iterate over the sequence provided by dataset
    for t in range(t_steps):
        f0_t = f0[t].unsqueeze(0)
        f1_t = f1[t].unsqueeze(0)
        f2_t = f2[t].unsqueeze(0)
        p0_t = p0[t].unsqueeze(0)
        p1_t = p1[t].unsqueeze(0)
        p2_t = p2[t].unsqueeze(0)

        feats = [flatten_feat(f0_t), flatten_feat(f1_t), flatten_feat(f2_t)]
        poses = [flatten_feat(p0_t), flatten_feat(p1_t), flatten_feat(p2_t)]
        feat_sizes = [(f0_t.shape[-2], f0_t.shape[-1]),
                      (f1_t.shape[-2], f1_t.shape[-1]),
                      (f2_t.shape[-2], f2_t.shape[-1])]

        is_init = is_init_flags[t].item()
        
        # Force initialization for the first frame of the chunk
        if t == 0:
            is_init = True
        
        if is_init:
            # Reset memory for this sequence if we hit an initialization frame
            student_out = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        
        # Prepare features (student memory attention)
        fused = tracker._prepare_memory_conditioned_features(
            frame_idx=t, # Frame index is relative to start of this batch/sequence for tracker state? 
                         # Tracker maintains state based on frame_idx. 
                         # If we feed frame_idx=0, 1, 2... it's fine.
            is_init_cond_frame=is_init,
            current_vision_feats=feats[-1:],
            current_vision_pos_embeds=poses[-1:],
            feat_sizes=feat_sizes[-1:],
            output_dict=student_out,
            num_frames=t_steps,
        )

        if not is_init:
            target = teacher_feat[t].unsqueeze(0)
            mse = F.mse_loss(fused, target)
            cos = 1 - F.cosine_similarity(
                fused.flatten(1), target.flatten(1), dim=-1
            ).mean()
            losses.append(mse + cfg["train"]["cosine_weight"] * cos)

        # Update student memory with Teacher Forcing
        current_out = tracker.track_step(
            frame_idx=t,
            is_init_cond_frame=is_init,
            current_vision_feats=feats,
            current_vision_pos_embeds=poses,
            feat_sizes=feat_sizes,
            image=None,
            point_inputs=None,
            mask_inputs=None,
            output_dict=student_out,
            num_frames=t_steps,
            run_mem_encoder=True,
            teacher_forced_mask=teacher_mask[t],
            teacher_forced_object_score=teacher_scores[t],
        )
        
        # Store output in student_out so it can be used as memory for future frames
        # Frame 0 (init) is conditioning. Subsequent frames are usually non-conditioning
        # unless we explicitly want to add them as "prompts".
        # In standard VOS, we usually just append to non_cond_frame_outputs.
        # The _prepare... function looks at non_cond_frame_outputs for memory.
        # But wait, if is_init is True, Sam3TrackerBase usually treats it as "cond_frame".
        # Let's stick to: is_init -> cond, else -> non_cond.
        
        if is_init:
            student_out["cond_frame_outputs"][t] = current_out
        else:
            student_out["non_cond_frame_outputs"][t] = current_out

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
        
    return torch.stack(losses).mean()


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 training for EfficientSAM3"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="stage2/configs/efficient_sam3_stage2.yaml",
        help="Path to Stage2 config YAML",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    logger = create_logger(cfg["output"]["dir"], rank)
    if rank == 0:
        logger.info(f"Loaded config from {args.config}")
        logger.info(f"Starting training on {world_size} GPUs for {cfg['train']['epochs']} epochs")

    dataset = Stage2SequenceDataset(
        cfg["data"]["features_dir"],
        cfg["data"]["teacher_dir"],
        seq_len=cfg["data"]["seq_len"],
        max_frames=cfg["data"].get("max_frames", -1),
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        sampler=sampler,
        num_workers=cfg["train"]["workers"],
        pin_memory=True,
        drop_last=False,
    )

    tracker = build_student_tracker(cfg, device)
    if world_size > 1:
        tracker = DDP(tracker, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, tracker.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    for epoch in range(cfg["train"]["epochs"]):
        sampler.set_epoch(epoch)
        tracker.train()
        epoch_loss = 0.0
        step = 0
        iterator = tqdm(dataloader, disable=rank != 0)
        start_time = time.time()
        for batch in iterator:
            optimizer.zero_grad()
            loss = run_video(
                tracker.module if isinstance(tracker, DDP) else tracker,
                batch,
                cfg,
                device,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, tracker.parameters()),
                cfg["train"]["grad_clip"],
            )
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            if rank == 0:
                iterator.set_description(f"Epoch {epoch+1} Loss {loss.item():.4f}")

        avg_loss = epoch_loss / max(1, step)
        if rank == 0:
            logger.info(
                f"Epoch {epoch+1}/{cfg['train']['epochs']} "
                f"- loss: {avg_loss:.4f} - time: {time.time() - start_time:.1f}s"
            )
            ckpt_path = os.path.join(
                cfg["output"]["dir"], f"stage2_epoch_{epoch+1}.pth"
            )
            state = (
                tracker.module.state_dict()
                if isinstance(tracker, DDP)
                else tracker.state_dict()
            )
            torch.save(state, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
