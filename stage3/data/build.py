"""Data loader builder for Stage 3."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.distributed as dist

from .sa1b_stage3_dataset import SA1BStage3Dataset, collate_fn


def _worker_init_fn(worker_id: int):
    """Seed worker RNGs deterministically from torch.initial_seed().

    We also propagate the current ``_epoch`` from the main process through the
    dataset attribute so that ``deterministic=True`` samplers stay in sync
    across workers when ``persistent_workers=True``.
    """
    info = torch.utils.data.get_worker_info()
    base_seed = torch.initial_seed() % (2**31)
    random.seed(base_seed + worker_id)
    np.random.seed(base_seed + worker_id)
    ds = info.dataset
    ep = int(os.environ.get("STAGE3_EPOCH", "0"))
    if hasattr(ds, "set_epoch"):
        ds.set_epoch(ep)


def set_loader_epoch(loader: torch.utils.data.DataLoader, epoch: int):
    """Propagate the current epoch to persistent workers via env var.

    Workers read ``STAGE3_EPOCH`` on __init__ *and* we also call
    ``dataset.set_epoch`` on the main process copy so the ``fix_seed`` logic
    works for both stateful-sampling and non-persistent-worker setups.
    """
    os.environ["STAGE3_EPOCH"] = str(epoch)
    ds = loader.dataset
    if hasattr(ds, "set_epoch"):
        ds.set_epoch(epoch)
    if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)


def build_loader(config, build_val: bool = False):
    """Build data loaders for Stage 3 training."""
    persistent_workers = (
        bool(getattr(config.DATA, "PERSISTENT_WORKERS", False))
        and config.DATA.NUM_WORKERS > 0
    )
    prefetch_factor = int(getattr(config.DATA, "PREFETCH_FACTOR", 2))
    dl_kwargs = {}
    if config.DATA.NUM_WORKERS > 0:
        dl_kwargs["persistent_workers"] = persistent_workers
        dl_kwargs["prefetch_factor"] = prefetch_factor
        dl_kwargs["worker_init_fn"] = _worker_init_fn

    dataset_train = SA1BStage3Dataset(
        data_root=config.DATA.DATA_PATH,
        img_size=config.DATA.IMG_SIZE,
        split="train",
        num_samples=config.DATA.NUM_SAMPLES,
        num_sample_points=config.DATA.NUM_SAMPLE_POINTS,
        box_jitter=config.DATA.BOX_JITTER,
        sort_by_area=config.DATA.SORT_BY_AREA,
        teacher_embed_dir=(
            config.DISTILL.TEACHER_EMBED_DIR if config.DISTILL.USE_SAVED_EMBEDDINGS else None
        ),
        teacher_embed_dtype=config.DISTILL.TEACHER_EMBED_DTYPE,
        text_label_mode=config.DATA.TEXT_LABEL_MODE,
        deterministic=config.DISTILL.NO_RAND,
    )

    if dist.is_initialized():
        sampler_train = torch.utils.data.distributed.DistributedSampler(
            dataset_train, shuffle=True, drop_last=True
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        collate_fn=collate_fn,
        **dl_kwargs,
    )

    dataset_val, data_loader_val = None, None
    if build_val:
        val_num_samples = (
            min(500, config.DATA.NUM_SAMPLES)
            if config.DATA.NUM_SAMPLES > 0
            else 500
        )
        dataset_val = SA1BStage3Dataset(
            data_root=config.DATA.DATA_PATH,
            img_size=config.DATA.IMG_SIZE,
            split="val",
            num_samples=val_num_samples,
            num_sample_points=config.DATA.NUM_SAMPLE_POINTS,
            box_jitter=False,
            sort_by_area=False,
            teacher_embed_dir=(
                config.DISTILL.TEACHER_EMBED_DIR if config.DISTILL.USE_SAVED_EMBEDDINGS else None
            ),
            teacher_embed_dtype=config.DISTILL.TEACHER_EMBED_DTYPE,
            text_label_mode="label_10",
            deterministic=True,
        )

        if dist.is_initialized():
            sampler_val = torch.utils.data.distributed.DistributedSampler(
                dataset_val, shuffle=False
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False,
            collate_fn=collate_fn,
            **dl_kwargs,
        )

    return dataset_train, dataset_val, data_loader_train, data_loader_val
