"""Stage 3 End-to-End Fine-Tuning Training Script.

Trains the student EfficientSAM3 model (student vision trunk + frozen student
MobileCLIP-S0 text encoder + frozen SAM3 transformer / seghead) with hybrid
supervision on SA-1B-1% enhanced annotations:

    L = lambda_emb  * MSE(student_embed, teacher_embed)        # distillation
      + lambda_bce  * BCE(best_mask_logits,   gt_mask)         # pixel level
      + lambda_dice * Dice(best_mask_logits,  gt_mask)         # region level
      + lambda_cls  * BCE(pred_logits, one_hot(best_query))    # score head

Each training sample is ONE prompt: (image, caption, box, K points, GT mask).
The decoder still emits Q mask hypotheses; we supervise the best-IoU query,
and teach ``pred_logits`` to concentrate on that slot so inference-time
``argmax`` selection is well-calibrated.

Usage:
    # Single GPU
    python stage3/train_stage3.py --cfg stage3/configs/es_tv_m.yaml \
        --data-path data/sa-1b-1p_reorg \
        --teacher-embed-dir output/stage3_teacher_embeddings \
        --trainable-scope trunk_only

    # 4-GPU DDP
    torchrun --nproc_per_node=4 stage3/train_stage3.py \
        --cfg stage3/configs/es_tv_m.yaml ...
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import autocast

sys.path.insert(0, str(Path(__file__).parent.parent / "stage1"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from stage3.config import get_config
from stage3.data import build_loader, set_loader_epoch
from stage3.losses import Stage3HybridLoss, compute_miou
from stage3.model import Stage3FinetuneModel
from stage1_geometry_finetune.losses import create_valid_mask

from stage1.logger import create_logger
from stage1.lr_scheduler import build_scheduler
from stage1.my_meter import AverageMeter
from stage1.optimizer import build_optimizer
from stage1.utils import load_checkpoint, save_checkpoint, NativeScalerWithGradNormCount


BACKBONE_MAP = {
    "repvit_m0_9": ("repvit", "m0.9"),
    "repvit_m1_1": ("repvit", "m1.1"),
    "repvit_m2_3": ("repvit", "m2.3"),
    "tiny_vit_5m": ("tinyvit", "5m"),
    "tiny_vit_11m": ("tinyvit", "11m"),
    "tiny_vit_21m": ("tinyvit", "21m"),
    "tinyvit_5m": ("tinyvit", "5m"),
    "tinyvit_11m": ("tinyvit", "11m"),
    "tinyvit_21m": ("tinyvit", "21m"),
    "efficientvit_b0": ("efficientvit", "b0"),
    "efficientvit_b1": ("efficientvit", "b1"),
    "efficientvit_b2": ("efficientvit", "b2"),
}


def resolve_backbone(name: str):
    key = name.lower()
    if key not in BACKBONE_MAP:
        raise ValueError(
            f"Unknown BACKBONE '{name}'. "
            f"Supported: {sorted(BACKBONE_MAP.keys())}"
        )
    return BACKBONE_MAP[key]


def parse_option():
    parser = argparse.ArgumentParser("EfficientSAM3 Stage 3 Fine-Tuning", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE")
    parser.add_argument("--opts", nargs="+", default=None)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--sam3-checkpoint", type=str, default=None,
                        help="Ignored (kept for CLI compatibility)")
    parser.add_argument("--teacher-embed-dir", type=str, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=None)
    parser.add_argument("--trainable-scope", type=str, default=None,
                        choices=["trunk_only", "trunk_fpn", "trunk_seghead"])
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--only-cpu", action="store_true")

    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    config = get_config(args)
    return args, config


def _mask_valid_region(img_sizes, mask_h, mask_w, img_size, device) -> torch.Tensor:
    """Build a (B, 1, mask_h, mask_w) 0/1 mask of valid (non-padded) pixels."""
    B = img_sizes.shape[0] if isinstance(img_sizes, torch.Tensor) else len(img_sizes)
    full = torch.zeros(B, 1, img_size, img_size, device=device)
    for bi in range(B):
        h, w = img_sizes[bi] if isinstance(img_sizes, torch.Tensor) else img_sizes[bi]
        full[bi, :, : int(h), : int(w)] = 1.0
    mv = F.interpolate(full, size=(mask_h, mask_w), mode="bilinear", align_corners=False)
    return (mv > 0.5).float()


def main(config):
    logger.info("Building data loaders...")
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(
        config, build_val=True
    )
    logger.info(f"Training samples: {len(dataset_train)}")
    if dataset_val is not None:
        logger.info(f"Validation samples: {len(dataset_val)}")

    backbone_type, model_name = resolve_backbone(config.MODEL.BACKBONE)
    logger.info(
        f"Building model (backbone={backbone_type}/{model_name}, "
        f"scope={config.MODEL.TRAINABLE_SCOPE})..."
    )

    model = Stage3FinetuneModel(
        backbone_type=backbone_type,
        model_name=model_name,
        stage1_checkpoint_path=config.MODEL.PRETRAINED or None,
        text_encoder_type="MobileCLIP-S0",
        trainable_scope=config.MODEL.TRAINABLE_SCOPE,
        img_size=config.DATA.IMG_SIZE,
    )
    model.cuda()

    model_without_ddp = model
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.LOCAL_RANK],
            find_unused_parameters=config.TRAIN.FIND_UNUSED_PARAMETERS,
        )
        model_without_ddp = model.module

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params / 1e6:.2f}M")

    criterion = Stage3HybridLoss(
        embedding_weight=config.DISTILL.EMBEDDING_LOSS_WEIGHT,
        mask_bce_weight=config.DISTILL.MASK_BCE_WEIGHT,
        mask_dice_weight=config.DISTILL.MASK_DICE_WEIGHT,
        mask_focal_weight=config.DISTILL.MASK_FOCAL_WEIGHT,
        classification_weight=config.DISTILL.CLASSIFICATION_WEIGHT,
    )

    optimizer = build_optimizer(config, model)
    n_iter_per_epoch = max(1, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    lr_scheduler = build_scheduler(config, optimizer, n_iter_per_epoch)
    loss_scaler = NativeScalerWithGradNormCount()

    max_accuracy = 0.0
    if config.TRAIN.AUTO_RESUME:
        resume_file = os.path.join(config.OUTPUT, "ckpt_epoch_latest.pth")
        if os.path.exists(resume_file):
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"Auto-resuming from {resume_file}")

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger
        )

    logger.info("Starting training...")
    start_time = time.time()

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        set_loader_epoch(data_loader_train, epoch)

        train_one_epoch(
            config, model, criterion, data_loader_train,
            optimizer, lr_scheduler, loss_scaler, epoch, logger,
        )

        if data_loader_val is not None:
            set_loader_epoch(data_loader_val, epoch)
            val_miou = validate(config, model, criterion, data_loader_val, epoch, logger)
            if val_miou > max_accuracy:
                max_accuracy = val_miou
                logger.info(f"New best mIoU: {max_accuracy:.4f}")

        if (epoch + 1) % config.SAVE_FREQ == 0 or epoch == config.TRAIN.EPOCHS - 1:
            save_checkpoint(
                config, epoch, model_without_ddp, max_accuracy,
                optimizer, lr_scheduler, loss_scaler, logger,
            )

    total_time = time.time() - start_time
    logger.info(f"Training completed in {datetime.timedelta(seconds=int(total_time))}")


def _forward_and_loss(config, model_ref, criterion, batch, device):
    """Shared forward / loss computation for train and val."""
    images = batch["images"].to(device, non_blocking=True)
    boxes = batch["boxes_cxcywh_norm"].to(device, non_blocking=True)
    points = batch["points_norm"].to(device, non_blocking=True)
    gt_mask = batch["gt_mask"].to(device, non_blocking=True)
    img_sizes = batch["img_sizes"]
    texts = batch["texts"]

    teacher_embeddings = batch.get("teacher_embeddings")
    teacher_valid = batch.get("teacher_valid")
    if teacher_embeddings is not None:
        teacher_embeddings = teacher_embeddings.to(device, non_blocking=True)
    if teacher_valid is not None:
        teacher_valid = teacher_valid.to(device, non_blocking=True)

    model_out = model_ref.forward_grounding(
        images=images,
        captions=texts,
        boxes_cxcywh_norm=boxes,
        points_norm=points,
    )

    pred_masks = model_out.get("pred_masks")
    pred_logits = model_out.get("pred_logits")
    trunk_features = model_out.get("trunk_features")

    B = images.shape[0]
    if pred_masks is not None:
        mask_h, mask_w = pred_masks.shape[-2:]
        gt_for_loss = gt_mask
        if gt_for_loss.shape[-2:] != (mask_h, mask_w):
            gt_for_loss = F.interpolate(
                gt_for_loss.float(), size=(mask_h, mask_w), mode="nearest"
            )
        else:
            gt_for_loss = gt_for_loss.float()
        mask_valid = _mask_valid_region(
            img_sizes, mask_h, mask_w, config.DATA.IMG_SIZE, device
        )
    else:
        gt_for_loss = gt_mask.float()
        mask_valid = None

    embed_valid = None
    if trunk_features is not None:
        embed_valid = create_valid_mask(
            batch_size=B,
            embed_size=config.DISTILL.EMBED_SIZE,
            img_sizes=img_sizes,
            img_size=config.DATA.IMG_SIZE,
            device=device,
        )

    losses, best_idx = criterion(
        student_embedding=trunk_features,
        teacher_embedding=teacher_embeddings,
        embed_valid_mask=embed_valid,
        embed_sample_mask=teacher_valid,
        pred_masks=pred_masks,
        pred_logits=pred_logits,
        gt_mask=gt_for_loss,
        mask_valid_mask=mask_valid,
    )

    with torch.no_grad():
        miou_val = 0.0
        if pred_masks is not None and best_idx is not None:
            batch_idx = torch.arange(B, device=device)
            best_mask = pred_masks[batch_idx, best_idx].unsqueeze(1).detach()
            miou_val = compute_miou(best_mask, gt_for_loss, mask_valid)

    return losses, miou_val, B


@torch.no_grad()
def validate(config, model, criterion, data_loader, epoch, logger):
    model.eval()
    miou_meter = AverageMeter()
    loss_meter = AverageMeter()
    device = torch.device("cuda")

    model_ref = model.module if dist.is_initialized() else model
    for batch in data_loader:
        if not batch:
            continue
        with autocast(enabled=config.AMP_ENABLE):
            losses, miou_val, B = _forward_and_loss(
                config, model_ref, criterion, batch, device
            )
        loss_meter.update(losses["total_loss"].item(), B)
        miou_meter.update(miou_val, B)

    logger.info(f"VAL [{epoch}] loss={loss_meter.avg:.4f} mIoU={miou_meter.avg:.4f}")
    model.train()
    return miou_meter.avg


def train_one_epoch(config, model, criterion, data_loader, optimizer,
                    lr_scheduler, loss_scaler, epoch, logger):
    model.train()
    device = torch.device("cuda")

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    miou_meter = AverageMeter()
    meters = defaultdict(AverageMeter)

    start = time.time()
    end = time.time()

    optimizer.zero_grad()

    for idx, batch in enumerate(data_loader):
        if not batch:
            continue

        model_ref = model.module if dist.is_initialized() else model

        with autocast(enabled=config.AMP_ENABLE):
            losses, miou_val, B = _forward_and_loss(
                config, model_ref, criterion, batch, device
            )
            loss = losses["total_loss"] / config.TRAIN.ACCUMULATION_STEPS

        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        update_grad = (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0
        grad_norm = loss_scaler(
            loss, optimizer,
            clip_grad=config.TRAIN.CLIP_GRAD,
            parameters=[p for p in model.parameters() if p.requires_grad],
            create_graph=is_second_order,
            update_grad=update_grad,
        )

        if update_grad:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS
            )

        loss_meter.update(losses["total_loss"].item(), B)
        if grad_norm is not None:
            norm_meter.update(grad_norm)
        miou_meter.update(miou_val, B)

        for k, v in losses.items():
            if k != "total_loss" and isinstance(v, torch.Tensor):
                meters[k].update(v.item(), B)

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            torch.cuda.synchronize()
            lr = optimizer.param_groups[0]["lr"]
            mem = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            loss_str = " ".join(f"{k} {v.val:.4f} ({v.avg:.4f})" for k, v in meters.items())
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]  "
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}  "
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})  "
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})  "
                f"mIoU {miou_meter.val:.4f} ({miou_meter.avg:.4f})  "
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})  "
                f"{loss_str}  mem {mem:.0f}MB"
            )

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))} "
        f"avg_loss={loss_meter.avg:.4f} avg_mIoU={miou_meter.avg:.4f}"
    )


if __name__ == "__main__":
    args, config = parse_option()

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(config.LOCAL_RANK)
        dist.init_process_group("nccl")
    else:
        rank = 0
        world_size = 1

    seed = config.SEED + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)

    logger = create_logger(
        output_dir=config.OUTPUT,
        dist_rank=rank,
        name=f"stage3_{config.TAG}",
    )

    if rank == 0:
        logger.info(f"Config:\n{config}")

    main(config)
