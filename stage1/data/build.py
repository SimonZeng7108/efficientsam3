import torch
import torch.distributed as dist
from mmengine.dataset import pseudo_collate

from .augmentation.dataset_wrapper import DatasetWrapper
from .sa1b_dataset import SA1BDataset
from .coco_dataset import COCODataset
from .coco_caption_dataset import COCOCaptionDataset
from .recap_coco_dataset import RecapCOCODataset
from .recap_datacomp_dataset import RecapDataCompDataset
from .text_annotations_dataset import TextAnnotationsDataset
from .sampler import MyDistributedSampler


def build_loader(config, build_val=True):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(
        is_train=True, config=config)
    config.freeze()

    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    
    if build_val:
        dataset_val, _ = build_dataset(is_train=False, config=config)
        print(
            f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")
    else:
        dataset_val = None

    sampler_train = MyDistributedSampler(
        dataset_train, shuffle=True,
        drop_last=False, padding=True, pair=False,
    )

    if build_val:
        sampler_val = MyDistributedSampler(
            dataset_val, shuffle=False,
            drop_last=False, padding=False, pair=False,
        )
    else:
        sampler_val = None

    # EdgeSAM Dataset Wrapper
    dataset_train = DatasetWrapper(
        dataset_train,
        logits_path=config.DISTILL.TEACHER_EMBED_PATH,
        topk=config.DISTILL.EMBED_DIM,
        write=config.DISTILL.SAVE_TEACHER_EMBED,
        num_embedding=config.DISTILL.NUM_EMBED,
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        # modified for EdgeSAM, we save image embeddings of all samples
        drop_last=not config.DISTILL.SAVE_TEACHER_EMBED,
        collate_fn=pseudo_collate
    )

    if build_val:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False,
            collate_fn=pseudo_collate
        )
    else:
        data_loader_val = None

    return dataset_train, dataset_val, data_loader_train, data_loader_val


def build_dataset(is_train, config):
    if config.DATA.DATASET == 'sa1b':
        num_samples = 100 if config.DATA.DEBUG else config.DATA.NUM_SAMPLES
        dataset = SA1BDataset(
            data_root=config.DATA.DATA_PATH,
            split='train' if is_train else 'val',
            img_size=config.DATA.IMG_SIZE,
            num_samples=num_samples,
            sort_by_area=config.DATA.SORT_BY_AREA,
            filter_by_area=config.DATA.FILTER_BY_AREA,
            pixel_mean=config.DATA.MEAN,
            pixel_std=config.DATA.STD,
            load_gt_mask=config.DATA.LOAD_GT_MASK,
            max_allowed_prompts=config.DISTILL.MAX_ALLOWED_PROMPTS,
            fix_seed=False,
            mask_nms_thresh=config.DATA.MASK_NMS,
            box_jitter=config.DATA.BOX_JITTER,
        )
        nb_classes = 0
    elif config.DATA.DATASET == 'coco_caption':
        num_samples = 100 if config.DATA.DEBUG else -1
        dataset = COCOCaptionDataset(
            data_root=config.DATA.DATA_PATH,
            split='train' if is_train else 'val',
            num_samples=num_samples,
        )
        nb_classes = 0
    elif config.DATA.DATASET == 'recap_coco':
        num_samples = 100 if config.DATA.DEBUG else -1
        dataset = RecapCOCODataset(
            data_root=config.DATA.DATA_PATH,
            split='train' if is_train else 'val',
            num_samples=num_samples,
        )
        nb_classes = 0
    elif config.DATA.DATASET == 'recap_datacomp':
        num_samples = 100 if config.DATA.DEBUG else -1
        dataset = RecapDataCompDataset(
            data_root=config.DATA.DATA_PATH,
            split='train' if is_train else 'val',
            num_samples=num_samples,
        )
        nb_classes = 0
    elif config.DATA.DATASET == 'text_annotations':
        num_samples = 100 if config.DATA.DEBUG else -1
        dataset = TextAnnotationsDataset(
            data_root=config.DATA.DATA_PATH,
            split='train' if is_train else 'val',
            num_samples=num_samples,
        )
        nb_classes = 0
    elif config.DATA.DATASET in ['coco', 'cocofied_lvis', 'lvis']:
        num_samples = 100 if config.DATA.DEBUG else -1
        dataset = COCODataset(
            data_root=config.DATA.DATA_PATH,
            split='train' if is_train else 'val',
            img_size=config.DATA.IMG_SIZE,
            num_samples=num_samples,
            sort_by_area=config.DATA.SORT_BY_AREA,
            filter_by_area=config.DATA.FILTER_BY_AREA,
            pixel_mean=config.DATA.MEAN,
            pixel_std=config.DATA.STD,
            load_gt_mask=config.DATA.LOAD_GT_MASK,
            max_allowed_prompts=config.DISTILL.MAX_ALLOWED_PROMPTS,
            fix_seed=False,
            annotation=config.DATA.DATASET,
        )
        nb_classes = 0
    else:
        raise NotImplementedError("We only support ImageNet, SA-1B, and COCO Now.")

    return dataset, nb_classes