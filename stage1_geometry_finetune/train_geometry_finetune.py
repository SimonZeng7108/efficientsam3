"""Stage 1 Geometry Fine-tuning Training Script.

This script trains the student backbone using:
1. Embedding distillation (MSE loss on trunk outputs)
2. Mask distillation (BCE + Dice loss on predicted masks)

Usage:
    python train_geometry_finetune.py --cfg configs/repvit_m1_1.yaml \
        --data-path /path/to/sa1b \
        --pretrained /path/to/stage1_checkpoint.pth \
        --sam3-checkpoint /path/to/sam3.pt \
        --teacher-embed-path /path/to/teacher_embeddings
"""

import os
import sys
import time
import random
import argparse
import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

# Add parent directory to path for imports
# Add stage1 first so its relative imports (from utils import ...) work correctly
sys.path.insert(0, str(Path(__file__).parent.parent / 'stage1'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from stage1_geometry_finetune.config import get_config
from stage1_geometry_finetune.data import build_loader
from stage1_geometry_finetune.model import GeometryFinetuneModel, load_stage1_weights
from stage1_geometry_finetune.losses import GeometryFinetuningLoss, create_valid_mask

from stage1.logger import create_logger
from stage1.lr_scheduler import build_scheduler
from stage1.optimizer import build_optimizer
from stage1.my_meter import AverageMeter
from stage1.utils import save_checkpoint, load_checkpoint, NativeScalerWithGradNormCount


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from xyxy format to cxcywh format.
    
    Args:
        boxes: (B, N, 4) in xyxy format, normalized to [0, 1]
        
    Returns:
        boxes: (B, N, 4) in cxcywh format
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def parse_option():
    parser = argparse.ArgumentParser(
        "EfficientSAM3 Stage 2 Geometry Fine-tuning", add_help=False
    )
    
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE",
                        help='path to config file')
    parser.add_argument('--opts', nargs='+', default=None,
                        help="Modify config options by adding 'KEY VALUE' pairs")
    
    # Data
    parser.add_argument('--batch-size', type=int, help="batch size per GPU")
    parser.add_argument('--data-path', type=str, help='path to SA-1B dataset')
    
    # Model
    parser.add_argument('--pretrained', type=str, 
                        help='path to Stage 1 pretrained checkpoint')
    parser.add_argument('--resume', type=str, 
                        help='path to checkpoint to resume from')
    parser.add_argument('--sam3-checkpoint', type=str,
                        help='path to SAM3 checkpoint for frozen components')
    parser.add_argument('--teacher-embed-path', type=str,
                        help='path to saved teacher embeddings')
    
    # Training
    parser.add_argument('--accumulation-steps', type=int,
                        help='gradient accumulation steps')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help='use gradient checkpointing')
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable automatic mixed precision')
    parser.add_argument('--only-cpu', action='store_true',
                        help='use CPU only')
    
    # Output
    parser.add_argument('--output', type=str, default='output_geometry_finetune',
                        help='output directory')
    parser.add_argument('--tag', type=str, default='default',
                        help='tag for experiment')
    
    # Evaluation
    parser.add_argument('--eval', action='store_true',
                        help='evaluate only')
    parser.add_argument('--throughput', action='store_true',
                        help='test throughput only')
    
    # Distributed
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')
    
    args = parser.parse_args()
    
    config = get_config(args)
    return args, config


def main(config):
    # Build data loaders
    logger.info("Building data loaders...")
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config)
    logger.info(f"Training samples: {len(dataset_train)}")
    
    # Build model with SAM3 for DUAL-PATH distillation
    logger.info("Building model (DUAL-PATH: embedding + mask distillation)...")
    model = GeometryFinetuneModel(
        student_backbone_name=config.MODEL.BACKBONE,
        sam3_checkpoint_path=config.MODEL.SAM3_CHECKPOINT,
        embed_dim=config.DISTILL.EMBED_DIM,
        embed_size=config.DISTILL.EMBED_SIZE,
        img_size=config.DATA.IMG_SIZE,
        freeze_fpn=True,
    )
    
    # Load Stage 1 pretrained weights
    if config.MODEL.PRETRAINED:
        load_stage1_weights(model, config.MODEL.PRETRAINED, logger)
    
    model.cuda()
    
    # Distributed training
    model_without_ddp = model
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[config.LOCAL_RANK],
            find_unused_parameters=config.TRAIN.FIND_UNUSED_PARAMETERS,
        )
        model_without_ddp = model.module
    
    # Count parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_parameters / 1e6:.2f}M")
    
    # Build loss function
    criterion = GeometryFinetuningLoss(
        embedding_weight=config.DISTILL.EMBEDDING_LOSS_WEIGHT,
        mask_bce_weight=config.DISTILL.MASK_BCE_WEIGHT,
        mask_dice_weight=config.DISTILL.MASK_DICE_WEIGHT,
        mask_focal_weight=config.DISTILL.MASK_FOCAL_WEIGHT,
        iou_weight=config.DISTILL.IOU_LOSS_WEIGHT,
        temperature=config.DISTILL.TEMPERATURE,
    )
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(config, model)
    n_iter_per_epoch = len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS
    lr_scheduler = build_scheduler(config, optimizer, n_iter_per_epoch)
    loss_scaler = NativeScalerWithGradNormCount()
    
    # Resume from checkpoint
    max_accuracy = 0.0
    if config.TRAIN.AUTO_RESUME:
        resume_file = os.path.join(config.OUTPUT, 'ckpt_epoch_latest.pth')
        if os.path.exists(resume_file):
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"Auto-resuming from {resume_file}")
    
    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger
        )
    
    # Training loop
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if dist.is_initialized():
            data_loader_train.sampler.set_epoch(epoch)
        dataset_train.set_epoch(epoch)
        
        train_one_epoch(
            config, model, criterion, data_loader_train,
            optimizer, lr_scheduler, loss_scaler, epoch, logger
        )
        
        # Save checkpoint
        if (epoch + 1) % config.SAVE_FREQ == 0 or epoch == config.TRAIN.EPOCHS - 1:
            save_checkpoint(
                config, epoch, model_without_ddp, max_accuracy,
                optimizer, lr_scheduler, loss_scaler, logger
            )
        
        # Validation
        if data_loader_val is not None and (epoch + 1) % config.SAVE_FREQ == 0:
            val_loss = validate(config, model, criterion, data_loader_val, logger)
            logger.info(f"Validation loss: {val_loss:.4f}")
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {datetime.timedelta(seconds=int(total_time))}")


def train_one_epoch(config, model, criterion, data_loader, optimizer, 
                    lr_scheduler, loss_scaler, epoch, logger):
    """
    Train for one epoch with DUAL-PATH distillation:
    
    Loss 1 (Embedding): MSE(student_embedding, teacher_embedding)
    Loss 2 (Mask): BCE+Dice(student_mask, teacher_mask)
    
    Both masks are generated by running embeddings through frozen SAM3 components.
    """
    model.train()
    
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    meters = defaultdict(AverageMeter)
    
    # Check if mask distillation is enabled
    use_mask_loss = (
        config.DISTILL.MASK_BCE_WEIGHT > 0
        or config.DISTILL.MASK_DICE_WEIGHT > 0
        or config.DISTILL.MASK_FOCAL_WEIGHT > 0
        or config.DISTILL.IOU_LOSS_WEIGHT > 0
    )
    
    start = time.time()
    end = time.time()
    
    for idx, batch in enumerate(data_loader):
        # Move data to GPU
        images = batch['images'].cuda(non_blocking=True)
        teacher_embeddings = batch['teacher_embeddings'].cuda(non_blocking=True)
        img_sizes = batch['img_sizes']
        
        # Prompts for mask prediction
        boxes = batch['boxes'].cuda(non_blocking=True) if 'boxes' in batch else None
        points = batch['points'].cuda(non_blocking=True) if 'points' in batch else None
        point_labels = batch['point_labels'].cuda(non_blocking=True) if 'point_labels' in batch else None
        prompt_mask = batch['prompt_mask'].cuda(non_blocking=True) if 'prompt_mask' in batch else None
        
        batch_size = images.shape[0]
        
        # Create valid mask for embeddings (handles padding)
        valid_mask = create_valid_mask(
            batch_size=batch_size,
            embed_size=config.DISTILL.EMBED_SIZE,
            img_sizes=img_sizes,
            img_size=config.DATA.IMG_SIZE,
            device=images.device,
        )
        
        with autocast(enabled=config.AMP_ENABLE):
            # Get model reference (handle DDP)
            model_ref = model.module if dist.is_initialized() else model
            
            # Forward student trunk → student embeddings
            student_embeddings = model_ref.forward_student(images)

            # Optional: Mask distillation
            student_masks = None
            teacher_masks = None
            student_iou = None
            teacher_iou = None

            if use_mask_loss and (boxes is not None or points is not None):
                # Convert boxes from xyxy normalized to cxcywh normalized
                # Dataset returns xyxy format, SAM3 expects cxcywh
                boxes_cxcywh = xyxy_to_cxcywh(boxes) if boxes is not None else None
                
                # Get teacher masks (teacher embedding → frozen SAM3 → mask)
                with torch.no_grad():
                    teacher_out = model_ref.forward_mask_prediction(
                        trunk_output=teacher_embeddings,
                        boxes=boxes_cxcywh,
                        points=points,
                        point_labels=point_labels,
                        prompt_mask=prompt_mask,
                    )
                    teacher_masks = teacher_out.get('pred_masks')
                
                # Get student masks (student embedding → frozen SAM3 → mask)
                student_out = model_ref.forward_mask_prediction(
                    trunk_output=student_embeddings,
                    boxes=boxes_cxcywh,
                    points=points,
                    point_labels=point_labels,
                    prompt_mask=prompt_mask,
                )
                student_masks = student_out.get('pred_masks')

                # Try to find IoU-like predictions if they exist in the model outputs.
                # IMPORTANT: `pred_logits` are class logits, not IoU.
                for key in ('pred_ious', 'pred_iou', 'iou_pred', 'iou_preds'):
                    if teacher_iou is None and key in teacher_out:
                        teacher_iou = teacher_out[key]
                    if student_iou is None and key in student_out:
                        student_iou = student_out[key]

            # Compute combined loss ONCE (prevents double-counting embedding loss)
            losses = criterion(
                student_embedding=student_embeddings,
                teacher_embedding=teacher_embeddings,
                student_masks=student_masks,
                teacher_masks=teacher_masks,
                student_iou=student_iou,
                teacher_iou=teacher_iou,
                valid_mask=valid_mask,
            )
        
        # Scale loss for gradient accumulation
        loss = losses['total_loss'] / config.TRAIN.ACCUMULATION_STEPS
        
        # Backward
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(
            loss, optimizer,
            clip_grad=config.TRAIN.CLIP_GRAD,
            parameters=model.parameters(),
            create_graph=is_second_order,
            update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0
        )
        
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS
            )
        
        torch.cuda.synchronize()
        
        # Update meters
        loss_meter.update(losses['total_loss'].item(), batch_size)
        if grad_norm is not None:
            norm_meter.update(grad_norm)
        
        for k, v in losses.items():
            if k != 'total_loss' and isinstance(v, torch.Tensor):
                meters[k].update(v.item(), batch_size)
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Logging
        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            
            loss_str = ' '.join([f'{k} {v.val:.4f} ({v.avg:.4f})' 
                                for k, v in meters.items()])
            
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]  '
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}  '
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})  '
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})  '
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})  '
                f'{loss_str}  mem {memory_used:.0f}MB'
            )
    
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, model, criterion, data_loader, logger):
    """Validate the model."""
    model.eval()
    
    loss_meter = AverageMeter()
    
    for batch in data_loader:
        images = batch['images'].cuda(non_blocking=True)
        teacher_embeddings = batch['teacher_embeddings'].cuda(non_blocking=True)
        img_sizes = batch['img_sizes']
        
        batch_size = images.shape[0]
        
        valid_mask = create_valid_mask(
            batch_size=batch_size,
            embed_size=config.DISTILL.EMBED_SIZE,
            img_sizes=img_sizes,
            img_size=config.DATA.IMG_SIZE,
            device=images.device,
        )
        
        with autocast(enabled=config.AMP_ENABLE):
            student_embeddings = model.module.forward_student(images) if dist.is_initialized() \
                else model.forward_student(images)
            
            losses = criterion(
                student_embedding=student_embeddings,
                teacher_embedding=teacher_embeddings,
                valid_mask=valid_mask,
            )
        
        loss_meter.update(losses['total_loss'].item(), batch_size)
    
    return loss_meter.avg


if __name__ == '__main__':
    args, config = parse_option()
    
    # Setup distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(config.LOCAL_RANK)
    else:
        rank = 0
        world_size = 1
    
    # Set random seed
    seed = config.SEED + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    # Create output directory
    os.makedirs(config.OUTPUT, exist_ok=True)
    
    # Setup logger
    logger = create_logger(
        output_dir=config.OUTPUT,
        dist_rank=rank,
        name=f'geometry_finetune_{config.TAG}'
    )
    
    if rank == 0:
        logger.info(f"Config:\n{config}")
    
    main(config)
