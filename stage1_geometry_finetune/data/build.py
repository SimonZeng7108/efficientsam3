"""
Data loader builder for Stage 2.
"""

import torch
import torch.distributed as dist

from .sa1b_prompt_dataset import SA1BPromptDataset, collate_fn


def build_loader(config, build_val: bool = False):
    """Build data loaders for Stage 2 training."""
    
    # Training dataset
    dataset_train = SA1BPromptDataset(
        data_root=config.DATA.DATA_PATH,
        img_size=config.DATA.IMG_SIZE,
        split='train',
        num_samples=config.DATA.NUM_SAMPLES,
        sort_by_area=config.DATA.SORT_BY_AREA,
        filter_by_area=config.DATA.FILTER_BY_AREA,
        pixel_mean=config.DATA.MEAN,
        pixel_std=config.DATA.STD,
        load_gt_mask=config.DATA.LOAD_GT_MASK,
        max_prompts_per_image=config.DISTILL.MAX_PROMPTS,
        fix_seed=config.DISTILL.NO_RAND,
        mask_nms_thresh=config.DATA.MASK_NMS,
        box_jitter=config.DATA.BOX_JITTER,
        use_box_prompts=config.DISTILL.USE_BOX_PROMPTS,
        use_point_prompts=config.DISTILL.USE_POINT_PROMPTS,
        teacher_embed_path=config.DISTILL.TEACHER_EMBED_PATH if config.DISTILL.USE_SAVED_EMBEDDINGS else None,
        embed_dim=config.DISTILL.EMBED_DIM,
        embed_size=config.DISTILL.EMBED_SIZE,
    )
    
    # Distributed sampler
    if dist.is_initialized():
        sampler_train = torch.utils.data.distributed.DistributedSampler(
            dataset_train,
            shuffle=True,
            drop_last=True,
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    
    # Data loader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    # Validation dataset (optional)
    dataset_val = None
    data_loader_val = None
    
    if build_val:
        dataset_val = SA1BPromptDataset(
            data_root=config.DATA.DATA_PATH,
            img_size=config.DATA.IMG_SIZE,
            split='val',
            num_samples=min(1000, config.DATA.NUM_SAMPLES) if config.DATA.NUM_SAMPLES > 0 else 1000,
            sort_by_area=config.DATA.SORT_BY_AREA,
            filter_by_area=config.DATA.FILTER_BY_AREA,
            pixel_mean=config.DATA.MEAN,
            pixel_std=config.DATA.STD,
            load_gt_mask=config.DATA.LOAD_GT_MASK,
            max_prompts_per_image=config.DISTILL.MAX_PROMPTS,
            fix_seed=True,  # Always fix seed for validation
            mask_nms_thresh=config.DATA.MASK_NMS,
            box_jitter=False,  # No augmentation for validation
            use_box_prompts=config.DISTILL.USE_BOX_PROMPTS,
            use_point_prompts=config.DISTILL.USE_POINT_PROMPTS,
            teacher_embed_path=config.DISTILL.TEACHER_EMBED_PATH if config.DISTILL.USE_SAVED_EMBEDDINGS else None,
            embed_dim=config.DISTILL.EMBED_DIM,
            embed_size=config.DISTILL.EMBED_SIZE,
        )
        
        if dist.is_initialized():
            sampler_val = torch.utils.data.distributed.DistributedSampler(
                dataset_val,
                shuffle=False,
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
        )
    
    return dataset_train, dataset_val, data_loader_train, data_loader_val
