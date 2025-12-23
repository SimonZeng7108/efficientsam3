import os
import time
import random
import argparse
import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from config import get_config
from data import build_loader
from logger import create_logger
from model import build_image_teacher_model
from my_meter import AverageMeter
from utils import add_common_args


def parse_option():
    parser = argparse.ArgumentParser(
        "EfficientSAM3 save teacher embeddings", add_help=False
    )
    add_common_args(parser)
    parser.add_argument(
        "--check-saved-embed",
        action="store_true",
        help="Validate that stored embeddings match the teacher outputs",
    )
    args = parser.parse_args()
    config = get_config(args)
    return args, config


def main(config, args):
    dataset_train, _, data_loader_train, _ = build_loader(config, build_val=False)

    logger.info("Building SAM3 teacher encoder")
    model = build_image_teacher_model(config)
    model.cuda()

    os.makedirs(config.DISTILL.TEACHER_EMBED_PATH, exist_ok=True)

    # model = torch.nn.parallel.DistributedDataParallel(
    #     model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False
    # )

    if args.check_saved_embed:
        logger.info("Start checking embeddings")
    else:
        logger.info("Start saving embeddings")

    start_time = time.time()
    # Teacher embeddings are saved once (single forward pass), not per epoch
    dataset_train.set_epoch(0)
    data_loader_train.sampler.set_epoch(0)

    if args.check_saved_embed:
        check_embeddings_one_epoch(config, model, data_loader_train, epoch=0)
    else:
        save_embeddings_one_epoch(config, model, data_loader_train, epoch=0)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Embedding pipeline finished in {total_time_str}")


@torch.no_grad()
def save_embeddings_one_epoch(config, model, data_loader, epoch):
    model.eval()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    meters = defaultdict(AverageMeter)

    start = time.time()
    end = time.time()

    file_manager = data_loader.dataset.get_manager()

    for idx, ((samples, _), (keys, seeds)) in enumerate(data_loader):
        samples = torch.stack(samples, dim=0).cuda(non_blocking=True)
        seeds = np.stack(seeds, axis=0).astype(np.int32)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)

        torch.cuda.synchronize()

        write_tic = time.time()
        outputs = outputs.detach().to(dtype=torch.float16, device="cpu").numpy()

        for key, seed, output in zip(keys, seeds, outputs):
            payload = seed.tobytes() + output.tobytes()
            file_manager.write(key, payload)
        meters["write_time"].update(time.time() - write_tic)

        batch_time.update(time.time() - end)
        end = time.time()
        
        # Optional debug short-circuit
        if getattr(config.DATA, 'DEBUG', False) and idx >= 0:
            logger.info("DATA.DEBUG=True: breaking after 1 batch.")
            break

        if idx % config.PRINT_FREQ == 0:
            memory_used = (
                torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            )
            eta = batch_time.avg * (num_steps - idx)
            extra = "  ".join(
                f"{k} {v.val:.4f} ({v.avg:.4f})" for k, v in meters.items()
            )
            logger.info(
                f"Save: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]  "
                f"eta {datetime.timedelta(seconds=int(eta))}  "
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})  "
                f"{extra}  mem {memory_used:.0f}MB"
            )

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} save image embeddings takes "
        f"{datetime.timedelta(seconds=int(epoch_time))}"
    )


@torch.no_grad()
def check_embeddings_one_epoch(config, model, data_loader, epoch):
    model.eval()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    meters = defaultdict(AverageMeter)

    start = time.time()
    end = time.time()
    embed_shape = (
        config.DISTILL.EMBED_DIM,
        config.DISTILL.EMBED_SIZE,
        config.DISTILL.EMBED_SIZE,
    )

    for idx, ((samples, _), (saved_embeddings, seeds)) in enumerate(
        data_loader
    ):
        samples = torch.stack(samples, dim=0).cuda(non_blocking=True)
        saved_embeddings = torch.from_numpy(
            np.stack(saved_embeddings, axis=0)
        ).float()
        saved_embeddings = saved_embeddings.view(
            samples.size(0), *embed_shape
        ).cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)

        torch.cuda.synchronize()
        meters["error"].update(
            (outputs - saved_embeddings).abs().mean().item()
        )

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = (
                torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            )
            eta = batch_time.avg * (num_steps - idx)
            extra = "  ".join(
                f"{k} {v.val:.4f} ({v.avg:.4f})" for k, v in meters.items()
            )
            logger.info(
                f"Check: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]  "
                f"eta {datetime.timedelta(seconds=int(eta))}  "
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})  "
                f"{extra}  mem {memory_used:.0f}MB"
            )

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} check image embeddings takes "
        f"{datetime.timedelta(seconds=int(epoch_time))}"
    )


if __name__ == "__main__":
    args, config = parse_option()
    config.defrost()
    assert (
        len(config.DISTILL.TEACHER_EMBED_PATH) > 0
    ), "Please set DISTILL.TEACHER_EMBED_PATH"
    if not args.check_saved_embed:
        config.DISTILL.SAVE_TEACHER_EMBED = True
    config.freeze()

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )
    torch.distributed.barrier()

    seed = (
        config.SEED
        + dist.get_rank()
        + config.TRAIN.START_EPOCH * dist.get_world_size()
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(
        output_dir=config.OUTPUT,
        dist_rank=dist.get_rank(),
        name=f"{config.MODEL.NAME}",
    )

    if dist.get_rank() == 0:
        os.makedirs(config.DISTILL.TEACHER_EMBED_PATH, exist_ok=True)
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    logger.info(config.dump())

    main(config, args)
    
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


