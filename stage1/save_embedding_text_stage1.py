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
from model import build_text_teacher_model
from my_meter import AverageMeter
from utils import add_common_args


def parse_option():
    parser = argparse.ArgumentParser(
        "EfficientSAM3 save text teacher embeddings", add_help=False
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

    logger.info("Building SAM3 text teacher encoder")
    model = build_text_teacher_model(config)
    model.cuda()

    os.makedirs(config.DISTILL.TEACHER_EMBED_PATH, exist_ok=True)

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
        
        # Explicitly close the writer to ensure all data is flushed
        try:
            manager = dataset_train.get_manager()
            if manager.writer is not None:
                logger.info("Explicitly closing writer to flush data...")
                # Manually invoke cleanup
                if manager.writer.worker is not None:
                    manager.writer.msg_queue.put(manager.writer._WORKER_MSG.KILL)
                    manager.writer.worker.join()
                    manager.writer.worker = None
        except Exception as e:
            logger.warning(f"Error closing writer: {e}")

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

    for idx, batch in enumerate(data_loader):
        # Handle different batch structures
        # Case 1: batch is [list_of_captions, list_of_metadata] (Column-oriented)
        # Relaxed check: batch[0] can be list or tuple
        if isinstance(batch, list) and len(batch) == 2 and isinstance(batch[0], (list, tuple)) and len(batch[0]) > 0 and isinstance(batch[0][0], str):
            samples = batch[0]
            # batch[1] is [list_of_keys, list_of_seeds]
            # We found that batch[1] has length 2, and batch[1][0] is the list of keys
            keys = batch[1][0]
            seeds = batch[1][1]
        # Case 2: batch is list of (caption, (key, seed)) tuples (Row-oriented)
        elif isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], tuple):
            samples = [item[0] for item in batch]
            keys = [item[1][0] for item in batch]
            seeds = [item[1][1] for item in batch]
        else:
            # Fallback or error
            raise ValueError(f"Unknown batch structure. Type: {type(batch)}")
        
        if idx == 0:
            logger.info(f"Batch 0: len(batch)={len(batch)}")
            logger.info(f"Batch 0: type(batch[0])={type(batch[0])}, len(batch[0])={len(batch[0])}")
            logger.info(f"Batch 0: type(batch[1])={type(batch[1])}, len(batch[1])={len(batch[1])}")
            logger.info(f"Batch 0: len(keys)={len(keys)}, keys[0]={keys[0]}")

        seeds = np.array(seeds).astype(np.int32)

        # model expects list of strings
        # outputs: [Seq, Batch, 256]
        outputs = model(samples, device="cuda")

        torch.cuda.synchronize()

        write_tic = time.time()
        # Transpose to [Batch, Seq, 256] for saving
        outputs = outputs.transpose(0, 1).detach().to(dtype=torch.float16, device="cpu").numpy()

        for key, seed, output in zip(keys, seeds, outputs):
            key = str(key) # Ensure key is string
            payload = seed.tobytes() + output.tobytes()
            file_manager.write(key, payload)
        meters["write_time"].update(time.time() - write_tic)

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
                f"Save: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]  "
                f"eta {datetime.timedelta(seconds=int(eta))}  "
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})  "
                f"{extra}  mem {memory_used:.0f}MB"
            )
        
        # Break after 1 batch for testing
        if idx >= 0:
            logger.info("Breaking after 1 batch for testing.")
            break

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} save text embeddings takes "
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
    
    # For text, embed_shape is (Seq, Embed_Dim)
    # We assume NUM_EMBED is set to Seq length (e.g. 32)
    embed_shape = (
        config.DISTILL.NUM_EMBED,
        config.DISTILL.EMBED_DIM,
    )

    for idx, batch in enumerate(data_loader):
        # batch is list of (caption, (embeddings, seed))
        samples = [item[0] for item in batch]
        saved_embeddings = [item[1][0] for item in batch]
        seeds = [item[1][1] for item in batch]

        # samples: list of strings
        saved_embeddings = torch.from_numpy(
            np.stack(saved_embeddings, axis=0)
        ).float()
        
        # saved_embeddings: [Batch, Seq * Embed_Dim] -> [Batch, Seq, Embed_Dim]
        saved_embeddings = saved_embeddings.view(
            len(samples), *embed_shape
        ).cuda(non_blocking=True)

        # outputs: [Seq, Batch, 256]
        outputs = model(samples, device="cuda")
        
        # Transpose outputs to [Batch, Seq, 256] to match saved_embeddings
        outputs = outputs.transpose(0, 1)

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
        f"EPOCH {epoch} check text embeddings takes "
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
