import os
import time
import argparse
import datetime
import random
from collections import defaultdict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from data import build_loader
from logger import create_logger
from lr_scheduler import build_scheduler
from model import build_student_model
from my_meter import AverageMeter
from optimizer import build_optimizer
from utils import (
    NativeScalerWithGradNormCount,
    add_common_args,
    auto_resume_helper,
    get_git_info,
    is_main_process,
    load_checkpoint,
    save_checkpoint,
)

try:
    import wandb
except ImportError:
    wandb = None


def parse_option():
    parser = argparse.ArgumentParser(
        "EfficientSAM3 Stage-1 training", add_help=False
    )
    add_common_args(parser)
    args = parser.parse_args()
    config = get_config(args)
    return args, config


# Training configuration notes:
# - Learning Rate: Increased to 1e-3 for faster convergence.
# - Loss Functions: Uses both Pixel-wise MSE and Cosine Similarity for feature alignment.
# - Weight Decay: Set to 0.01.
# - Epochs: Default set to 50.

def main(args, config):
    dataset_train, _, data_loader_train, _ = build_loader(config, build_val=False)

    logger.info(f"Creating model: {config.MODEL.BACKBONE}")
    model = build_student_model(config)
    if not args.only_cpu:
        model.cuda()

    if args.use_sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = build_optimizer(config, model)
    if not args.only_cpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.LOCAL_RANK],
            broadcast_buffers=False,
            find_unused_parameters=config.TRAIN.FIND_UNUSED_PARAMETERS,
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    loss_scaler = NativeScalerWithGradNormCount(
        grad_scaler_enabled=config.AMP_ENABLE
    )
    lr_scheduler = build_scheduler(
        config,
        optimizer,
        len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS,
    )

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}"
                )
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"auto resuming from {resume_file}")

    if config.MODEL.RESUME:
        load_checkpoint(
            config,
            model_without_ddp,
            optimizer,
            lr_scheduler,
            loss_scaler,
            logger,
        )
        if config.EVAL_MODE:
            return

    loss_writer = None
    if dist.get_rank() == 0:
        log_path = datetime.datetime.now().strftime("%Y-%m-%d/%H:%M:%S")
        loss_writer = SummaryWriter(f"{config.OUTPUT}/{log_path}")

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if hasattr(dataset_train, "set_epoch"):
            dataset_train.set_epoch(epoch)
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            args,
            config,
            model,
            data_loader_train,
            optimizer,
            epoch,
            lr_scheduler,
            loss_scaler,
            loss_writer,
        )

        if dist.get_rank() == 0 and (
            epoch % config.SAVE_FREQ == 0
            or epoch == (config.TRAIN.EPOCHS - 1)
        ):
            save_checkpoint(
                config,
                epoch,
                model_without_ddp,
                0.0,
                optimizer,
                lr_scheduler,
                loss_scaler,
                logger,
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")


def train_one_epoch(
    args,
    config,
    model,
    data_loader,
    optimizer,
    epoch,
    lr_scheduler,
    loss_scaler,
    loss_writer,
):
    model.train()
    set_bn_state(config, model)
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    meters = defaultdict(AverageMeter)

    start = time.time()
    end = time.time()
    data_tic = time.time()

    embed_shape = (
        config.DISTILL.EMBED_DIM,
        config.DISTILL.EMBED_SIZE,
        config.DISTILL.EMBED_SIZE,
    )

    for idx, ((samples, annos), (saved_embeddings, seeds)) in enumerate(
        data_loader
    ):
        samples = torch.stack(samples, dim=0).cuda(non_blocking=True)
        saved_embeddings = torch.from_numpy(
            np.stack(saved_embeddings, axis=0)
        ).float()
        saved_embeddings = saved_embeddings.view(
            samples.size(0), *embed_shape
        ).cuda(non_blocking=True)

        meters["data_time"].update(time.time() - data_tic)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            preds = model(samples)

        valid_mask = build_valid_mask(
            config, annos["img_size_before_pad"], preds.shape, preds.device
        )
        loss = masked_mse(preds, saved_embeddings, valid_mask)

        if config.DISTILL.COSINE > 0.0:
            loss += config.DISTILL.COSINE * masked_cosine_loss(
                preds, saved_embeddings, valid_mask
            )

        loss = loss / config.TRAIN.ACCUMULATION_STEPS
        loss_meter.update(loss.detach().item(), samples.size(0))

        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=config.TRAIN.CLIP_GRAD,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0,
        )
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS
            )

        loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)
        if grad_norm is not None and not torch.isnan(grad_norm):
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)

        torch.cuda.synchronize()

        batch_time.update(time.time() - end)
        end = time.time()
        data_tic = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            memory_used = (
                torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            )
            eta = batch_time.avg * (num_steps - idx)
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]  "
                f"eta {datetime.timedelta(seconds=int(eta))}  "
                f"lr {lr:.6f}  time {batch_time.val:.4f} ({batch_time.avg:.4f})  "
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})  "
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})  "
                f"loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})  "
                f"mem {memory_used:.0f}MB"
            )

        if loss_writer is not None:
            step = epoch * num_steps + idx
            loss_writer.add_scalar("loss/total", loss.item(), step)

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}"
    )


def build_valid_mask(config, img_size_before_pad, target_shape, device):
    batch_size = len(img_size_before_pad)
    img_size = config.DATA.IMG_SIZE
    valid = torch.zeros(batch_size, 1, img_size, img_size, device=device)
    for i in range(batch_size):
        h, w = img_size_before_pad[i][1:]
        valid[i, :, :h, :w] = 1
    valid = F.interpolate(
        valid, size=target_shape[-2:], mode="bilinear", align_corners=False
    )
    return (valid > 0.5).float()


def masked_mse(preds, teacher, mask):
    diff = (preds - teacher) * mask
    denom = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    loss = diff.square().sum(dim=(1, 2, 3)) / denom
    return loss.mean()


def masked_cosine_loss(preds, teacher, mask):
    # preds: (B, C, H, W)
    # teacher: (B, C, H, W)
    # mask: (B, 1, H, W)

    # Cosine similarity along channel dimension (dim=1)
    sim = F.cosine_similarity(preds, teacher, dim=1)  # (B, H, W)
    loss = 1.0 - sim

    # Apply mask
    # mask is (B, 1, H, W), squeeze to (B, H, W)
    mask = mask.squeeze(1)

    loss = loss * mask
    denom = mask.sum(dim=(1, 2)).clamp(min=1.0)
    loss = loss.sum(dim=(1, 2)) / denom
    return loss.mean()


def set_bn_state(config, model):
    if config.TRAIN.EVAL_BN_WHEN_TRAINING:
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()


if __name__ == "__main__":
    args, config = parse_option()
    config.defrost()
    config.freeze()

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = -1
        world_size = -1

    if args.only_cpu:
        ddp_backend = "gloo"
    else:
        torch.cuda.set_device(config.LOCAL_RANK)
        ddp_backend = "nccl"

    torch.distributed.init_process_group(
        backend=ddp_backend, init_method="env://", world_size=world_size, rank=rank
    )
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    linear_scaled_lr = (
        config.TRAIN.BASE_LR
        * config.DATA.BATCH_SIZE
        * dist.get_world_size()
        / 512.0
    )
    linear_scaled_warmup_lr = (
        config.TRAIN.WARMUP_LR
        * config.DATA.BATCH_SIZE
        * dist.get_world_size()
        / 512.0
    )
    linear_scaled_min_lr = (
        config.TRAIN.MIN_LR
        * config.DATA.BATCH_SIZE
        * dist.get_world_size()
        / 512.0
    )
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr *= config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr *= config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr *= config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(
        output_dir=config.OUTPUT,
        dist_rank=dist.get_rank(),
        name=f"{config.MODEL.NAME}",
    )

    if is_main_process():
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

        config_dict = dict(config)
        config_dict["git"] = get_git_info()
        if args.use_wandb and wandb is not None:
            wandb_output_path = config.OUTPUT
            wandb.init(
                project="EfficientSAM3-Stage1",
                config=config_dict,
                dir=wandb_output_path,
            )

    logger.info("===== git =====")
    logger.info(str(get_git_info()))
    logger.info(config.dump())

    main(args, config)


