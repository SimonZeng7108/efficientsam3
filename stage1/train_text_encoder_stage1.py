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
from model import build_text_student_model
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
        "EfficientSAM3 Stage-1 text encoder training", add_help=False
    )
    add_common_args(parser)
    args = parser.parse_args()
    config = get_config(args)
    return args, config


def main(args, config):
    dataset_train, _, data_loader_train, _ = build_loader(config, build_val=False)

    logger.info(f"Creating text student model")
    model = build_text_student_model(config, logger=logger)
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
    # set_bn_state(config, model) # Text encoder might not have BN, or we want to train it.
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
        config.DISTILL.NUM_EMBED,
        config.DISTILL.EMBED_DIM,
    )

    for idx, batch in enumerate(data_loader):
        # Handle batch structure
        # DatasetWrapper returns (item, (embeddings, seed))
        # item is the caption string from RecapCOCODataset
        
        # Case 1: Transposed batch [list_of_captions, [list_of_embeddings, list_of_seeds]]
        if isinstance(batch, list) and len(batch) == 2 and isinstance(batch[0], (list, tuple)) and len(batch[0]) > 0 and isinstance(batch[0][0], str):
             samples = batch[0] # list of strings
             saved_embeddings = batch[1][0]
             seeds = batch[1][1]
        # Case 2: List of tuples [(caption, (embeddings, seed)), ...]
        elif isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], tuple):
             samples = [item[0] for item in batch]
             saved_embeddings = [item[1][0] for item in batch]
             seeds = [item[1][1] for item in batch]
        else:
             # Fallback or error
             raise ValueError(f"Unexpected batch structure: {type(batch)}")

        # samples is list of strings
        saved_embeddings = torch.from_numpy(
            np.stack(saved_embeddings, axis=0)
        ).float()
        saved_embeddings = saved_embeddings.view(
            len(samples), *embed_shape
        ).cuda(non_blocking=True)

        meters["data_time"].update(time.time() - data_tic)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            # model returns (mask, memory, embeds)
            # memory is [Seq, Batch, Dim]
            _, preds, _ = model(samples, device="cuda")
            preds = preds.transpose(0, 1)

            # preds: [Batch, Seq, 256]
            # saved_embeddings: [Batch, Seq, 256]
            
            loss = text_mse(preds, saved_embeddings)

            if config.DISTILL.COSINE > 0.0:
                loss += config.DISTILL.COSINE * text_cosine_loss(
                    preds, saved_embeddings
                )

            loss = loss / config.TRAIN.ACCUMULATION_STEPS
        
        loss_meter.update(loss.detach().item(), len(samples))

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


def text_mse(preds, teacher):
    # preds: (B, Seq, C)
    # teacher: (B, Seq, C)
    return F.mse_loss(preds, teacher)


def text_cosine_loss(preds, teacher):
    # preds: (B, Seq, C)
    # teacher: (B, Seq, C)
    # Cosine similarity along channel dimension (dim=2)
    sim = F.cosine_similarity(preds, teacher, dim=2) # (B, Seq)
    loss = 1.0 - sim
    return loss.mean()


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

    # LR scaling: scale by batch_size/512 for large-scale training
    # Disable with TRAIN.SCALE_LR=False for tiny test datasets
    if config.TRAIN.SCALE_LR:
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
                project="EfficientSAM3-Stage1-Text",
                config=config_dict,
                dir=wandb_output_path,
            )

    logger.info("===== git =====")
    logger.info(str(get_git_info()))
    logger.info(config.dump())

    main(args, config)
