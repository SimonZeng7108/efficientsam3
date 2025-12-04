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
    dice_loss,
    sigmoid_focal_loss,
    sigmoid_ce_loss,
    calculate_uncertainty,
)

try:
    import wandb
except ImportError:
    wandb = None


def parse_option():
    parser = argparse.ArgumentParser(
        "EfficientSAM3 Stage-1 Geometry Finetune", add_help=False
    )
    add_common_args(parser)
    args = parser.parse_args()
    config = get_config(args)
    return args, config


def main(args, config):
    dataset_train, _, data_loader_train, _ = build_loader(config, build_val=False)

    logger.info(f"Creating model: {config.MODEL.BACKBONE}")
    
    # Load the converted stage1 model (efficient encoder + SAM3 decoder)
    from sam3.model_builder import build_efficientsam3_image_model
    
    model = build_efficientsam3_image_model(
        bpe_path=None,
        checkpoint_path=config.MODEL.PRETRAINED,
        load_from_HF=False,
        enable_inst_interactivity=True,
        enable_text_encoder=config.DISTILL.ENABLE_TEXT_ENCODER,  # Keep text encoder if needed
        backbone_type=config.MODEL.BACKBONE.split('_')[0],  # repvit, tinyvit, or efficientvit
        model_name='_'.join(config.MODEL.BACKBONE.split('_')[1:]),
    )
    
    if not args.only_cpu:
        model.cuda()

    if args.use_sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Load teacher SAM3 model for prompt encoder and mask decoder
    logger.info("Loading teacher SAM3 model...")
    from sam3.model_builder import build_sam3_image_model
    
    teacher = build_sam3_image_model(
        checkpoint_path=config.MODEL.SAM3_CHECKPOINT,
        load_from_HF=False if config.MODEL.SAM3_CHECKPOINT else True,
        eval_mode=True,
        enable_segmentation=True,
        enable_inst_interactivity=True,
        enable_text_encoder=False,  # Disable text encoder
        compile=False,
    )
    
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()
    
    if not args.only_cpu:
        teacher.cuda()
    
    logger.info("Teacher model loaded successfully!")

    # Apply freezing if specified
    if config.DISTILL.FREEZE_IMAGE_ENCODER:
        logger.info("Freezing image encoder...")
        for param in model.backbone.parameters():
            param.requires_grad = False

    if config.DISTILL.FREEZE_PROMPT_ENCODER:
        logger.info("Freezing prompt encoder (geometry_encoder and sam_prompt_encoder)...")
        # Freeze SAM3's geometry encoder
        for param in model.geometry_encoder.parameters():
            param.requires_grad = False
        # Freeze SAM-style prompt encoder inside interactive predictor
        if hasattr(model, 'inst_interactive_predictor') and model.inst_interactive_predictor is not None:
            for param in model.inst_interactive_predictor.model.sam_prompt_encoder.parameters():
                param.requires_grad = False

    if config.DISTILL.FREEZE_MASK_DECODER:
        logger.info("Freezing mask decoder (sam_mask_decoder)...")
        # Freeze SAM-style mask decoder inside interactive predictor
        if hasattr(model, 'inst_interactive_predictor') and model.inst_interactive_predictor is not None:
            for param in model.inst_interactive_predictor.model.sam_mask_decoder.parameters():
                param.requires_grad = False

    if config.DISTILL.FREEZE_TEXT_ENCODER and hasattr(model, 'backbone') and hasattr(model.backbone, 'text'):
        if model.backbone.text is not None:
            logger.info("Freezing text encoder...")
            for param in model.backbone.text.parameters():
                param.requires_grad = False

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

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of trainable params: {n_parameters}")

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
            teacher,
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


def set_bn_state(config, model):
    if config.TRAIN.EVAL_BN_WHEN_TRAINING:
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()


def sample_point_in_mask(mask_s, mask_t, num_samples=1):
    """Sample points from FP/FN regions between student and teacher masks."""
    if len(mask_s.shape) == 4:
        mask_s = mask_s[:, 0]
        mask_t = mask_t[:, 0]

    device = mask_s.device
    sample_list = []
    label_list = []
    
    # False positives and false negatives
    fp = (mask_s != mask_t) * (mask_t == 0)
    fn = (mask_s != mask_t) * (mask_t == 1)
    fp_fn = fp | fn

    label_map = -2 * torch.ones_like(mask_s, dtype=torch.int32)
    label_map[fp] = 0  # Negative point
    label_map[fn] = 1  # Positive point

    _, h, w = mask_s.shape
    y_axis = torch.arange(h, device=device)[:, None].expand(h, w)
    x_axis = torch.arange(w, device=device)[None, :].expand(h, w)
    grid_points = torch.stack([x_axis, y_axis], dim=-1)

    for cur_fp_fn, cur_label_map in zip(fp_fn, label_map):
        if cur_fp_fn.sum() < num_samples * 10:
            sample_list.append(torch.zeros(num_samples, 2, device=device))
            label_list.append(-2 * torch.ones(num_samples, device=device))
        else:
            candidate_points = grid_points[cur_fp_fn]
            candidate_labels = cur_label_map[cur_fp_fn]
            selected = torch.randint(candidate_points.shape[0], (num_samples,))
            sample_list.append(candidate_points[selected])
            label_list.append(candidate_labels[selected])
    
    return torch.stack(sample_list, dim=0), torch.stack(label_list, dim=0)


def forward_sam3_batch(
    model,
    backbone_out,
    points=None,
    labels=None,
    boxes=None,
    masks=None,
    multimask_output=True,
    num_prompts=None,
):
    # model is Sam3Image (or wrapped in DDP)
    # We need to access the internal components
    
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
        
    predictor = model.inst_interactive_predictor
    tracker_model = predictor.model
    prompt_encoder = tracker_model.sam_prompt_encoder
    mask_decoder = tracker_model.sam_mask_decoder
    
    # Prepare features
    # We already have backbone_out. We need to extract features.
    # Use _prepare_backbone_features from tracker_model
    _, vision_feats, _, _ = tracker_model._prepare_backbone_features(backbone_out["sam2_backbone_out"])
    
    # vision_feats is a list of (HW, B, C) tensors.
    # We need (B, C, H, W).
    
    # Add no_mem_embed
    vision_feats[-1] = vision_feats[-1] + tracker_model.no_mem_embed
    
    # Reshape features
    feats = []
    for feat, feat_size in zip(vision_feats[::-1], predictor._bb_feat_sizes[::-1]):
        # feat: (HW, B, C)
        # permute(1, 2, 0) -> (B, C, HW)
        # view(B, C, H, W)
        B = feat.shape[1]
        C = feat.shape[2]
        H, W = feat_size
        feats.append(feat.permute(1, 2, 0).view(B, C, H, W))
        
    feats = feats[::-1]
    image_embed = feats[-1] # (B, 256, H, W)
    high_res_features = feats[:-1] # List of (B, 256, H, W)
    
    if num_prompts is not None and len(num_prompts) > 0:
        # Repeat features
        new_image_embed = []
        new_high_res = [[] for _ in range(len(high_res_features))]
        
        for i, n in enumerate(num_prompts):
            new_image_embed.append(image_embed[i].unsqueeze(0).repeat(n, 1, 1, 1))
            for j, feat in enumerate(high_res_features):
                new_high_res[j].append(feat[i].unsqueeze(0).repeat(n, 1, 1, 1))
                
        image_embed = torch.cat(new_image_embed, dim=0)
        high_res_features = [torch.cat(f, dim=0) for f in new_high_res]
    
    # Prepare prompts
    # points: (B, N, 2)
    # labels: (B, N)
    # boxes: (B, 4)
    # masks: (B, 1, 256, 256)
    
    # Embed prompts
    if boxes is not None:
        # boxes is (B, 4) -> (B, 1, 4)
        if len(boxes.shape) == 2:
            boxes = boxes.unsqueeze(1)
            
    sparse_embeddings, dense_embeddings = prompt_encoder(
        points=(points, labels) if points is not None else None,
        boxes=boxes,
        masks=masks,
    )

    # Resize dense_embeddings if needed (e.g. when using mask prompts with different resolution)
    if dense_embeddings.shape[-2:] != image_embed.shape[-2:]:
        dense_embeddings = F.interpolate(
            dense_embeddings,
            size=image_embed.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
    
    # Predict masks
    low_res_masks, iou_predictions, _, _ = mask_decoder(
        image_embeddings=image_embed,
        image_pe=prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,
        repeat_image=False, # We have batched image embeddings
        high_res_features=high_res_features,
    )
    
    return low_res_masks, iou_predictions, None


def train_one_epoch(
    args,
    config,
    model,
    teacher,
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

    for idx, ((samples, annos), (saved_embeddings, seeds)) in enumerate(
        data_loader
    ):
        samples = torch.stack(samples, dim=0).cuda(non_blocking=True)
        saved_embeddings = torch.from_numpy(
            np.stack(saved_embeddings, axis=0)
        ).float()
        embed_shape = (
            config.DISTILL.EMBED_DIM,
            config.DISTILL.EMBED_SIZE,
            config.DISTILL.EMBED_SIZE,
        )
        saved_embeddings = saved_embeddings.view(
            samples.size(0), *embed_shape
        ).cuda(non_blocking=True)

        meters["data_time"].update(time.time() - data_tic)

        img_bs = samples.shape[0]
        img_size_before_pad = annos['img_size_before_pad']
        img_size_pad = (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)
        
        # Get mask threshold from model's interactive predictor
        if not args.only_cpu:
            mask_threshold = model.module.inst_interactive_predictor.mask_threshold
        else:
            mask_threshold = model.inst_interactive_predictor.mask_threshold

        # Prepare image batch for SAM3 model (encode once, use for all iterations)
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            # Create a simple image batch dict that SAM3 expects
            img_batch = {
                'img_batch_all_stages': samples,
                'img_size_before_pad': img_size_before_pad,
            }
            # Get actual model
            actual_model = model.module if not args.only_cpu else model
            # Prepare backbone features using SAM3's method
            backbone_out = actual_model.backbone.forward_image(samples)
            
            # Project features for SAM3 mask decoder (required for high-res features)
            if "sam2_backbone_out" in backbone_out and backbone_out["sam2_backbone_out"] is not None:
                mask_decoder = actual_model.inst_interactive_predictor.model.sam_mask_decoder
                fpn_feats = backbone_out["sam2_backbone_out"]["backbone_fpn"]
                # Project level 0 and level 1
                fpn_feats[0] = mask_decoder.conv_s0(fpn_feats[0])
                fpn_feats[1] = mask_decoder.conv_s1(fpn_feats[1])

            # Teacher backbone
            with torch.no_grad():
                teacher_backbone_out = teacher.backbone.forward_image(samples)
                if "sam2_backbone_out" in teacher_backbone_out and teacher_backbone_out["sam2_backbone_out"] is not None:
                    t_mask_decoder = teacher.inst_interactive_predictor.model.sam_mask_decoder
                    t_fpn_feats = teacher_backbone_out["sam2_backbone_out"]["backbone_fpn"]
                    t_fpn_feats[0] = t_mask_decoder.conv_s0(t_fpn_feats[0])
                    t_fpn_feats[1] = t_mask_decoder.conv_s1(t_fpn_feats[1])

            # Store inference state for predict_inst calls
            inference_state = {
                'backbone_out': backbone_out,
                'img_size_before_pad': img_size_before_pad,
                'original_height': config.DATA.IMG_SIZE,
                'original_width': config.DATA.IMG_SIZE,
            }

        # Skip encoder embedding extraction for geometry finetune
        # Encoder distillation losses are disabled (all weights are 0.0)
        loss = dict()

        # Prepare prompts from annotations
        # Sample points from ground truth masks
        if 'gt_mask' in annos:
            gt_masks = annos['gt_mask']
            gt_masks = torch.cat(gt_masks, dim=0).cuda(non_blocking=True).squeeze(1)
        else:
            gt_masks = None

        # Get boxes
        boxes = annos.get('prompt_box', None)
        num_prompts = []
        if boxes is not None:
            for box in boxes:
                num_prompts.append(box.size(0))
            boxes = torch.cat(boxes, dim=0).cuda(non_blocking=True)
        
        # Sample points from masks
        points = None
        if config.DISTILL.PROMPT_MASK_TO_POINT and gt_masks is not None:
            point_list = []
            label_list = []
            for g in gt_masks:
                candidate_indices = g.nonzero()
                if len(candidate_indices) > 0:
                    selected_index = random.randint(0, len(candidate_indices) - 1)
                    p = candidate_indices[selected_index].flip(0)
                    l = torch.tensor(1, device=samples.device)
                else:
                    p = torch.zeros(2, device=samples.device)
                    l = torch.tensor(-2, device=samples.device)
                point_list.append(p)
                label_list.append(l)
            point_coords = torch.stack(point_list, dim=0)[:, None]
            point_labels = torch.stack(label_list, dim=0)[:, None]
            points = (point_coords, point_labels)

        # Convert box to point if specified
        if config.DISTILL.PROMPT_BOX_TO_POINT and boxes is not None:
            center_x = (boxes[:, 0] + boxes[:, 2]) / 2
            center_y = (boxes[:, 1] + boxes[:, 3]) / 2
            point_coords = torch.stack([center_x, center_y], dim=1)[:, None]
            point_labels = torch.ones(point_coords.shape[:2], device=samples.device)
            points = (point_coords, point_labels)

        # Prepare mask prompts (low-resolution mask input)
        masks = None
        if config.DISTILL.USE_MASK_PROMPT and gt_masks is not None:
            # Downsample ground truth masks to 256x256 (SAM's low-res mask input size)
            mask_input_size = 256
            masks = gt_masks.unsqueeze(1).float()  # Add channel dimension
            masks = F.interpolate(
                masks,
                size=(mask_input_size, mask_input_size),
                mode='bilinear',
                align_corners=False
            )
            # Optionally add some noise to make it more robust
            if config.DISTILL.get('MASK_PROMPT_NOISE', 0.0) > 0:
                noise = torch.randn_like(masks) * config.DISTILL.MASK_PROMPT_NOISE
                masks = masks + noise

        # Decide prompt type for this iteration
        cur_prompt_type = config.DISTILL.PROMPT_TYPE
        cur_decoder_iters = config.DISTILL.DECODE_ITERS
        cur_multimask_output = config.DISTILL.MULTIMASK_OUTPUT
        
        # Random prompt type selection for diversity
        num_prompt_types = len([p for p in cur_prompt_type if p in ['point', 'box', 'mask']])
        if num_prompt_types > 1:
            rand_val = torch.rand(1).item()
            if 'mask' in cur_prompt_type and rand_val < 1.0 / num_prompt_types:
                points = None
                boxes = None
                cur_prompt_type = ['mask']
                # Mask prompts typically use single mask output initially
                if not config.DISTILL.get('MULTIMASK_ON_MASK', False):
                    cur_multimask_output = 1
            elif 'point' in cur_prompt_type and 'box' in cur_prompt_type:
                if rand_val < 2.0 / num_prompt_types:
                    points = None
                    masks = None
                    cur_prompt_type = ['box']
                    if not config.DISTILL.ITER_ON_BOX:
                        cur_decoder_iters = 1
                    if not config.DISTILL.MULTIMASK_ON_BOX:
                        cur_multimask_output = 1
                else:
                    boxes = None
                    masks = None
                    cur_prompt_type = ['point']

        if 'point' not in cur_prompt_type:
            points = None
        if 'box' not in cur_prompt_type:
            boxes = None
        if 'mask' not in cur_prompt_type:
            masks = None

        # Build valid mask for loss computation
        valid = torch.zeros(img_bs, 1, *img_size_pad, device=samples.device)
        valid_list = []
        for img_i in range(img_bs):
            h, w = img_size_before_pad[img_i][1:]
            valid[img_i, :, :h, :w] = 1
            valid_list.append(valid[img_i:img_i + 1].expand(num_prompts[img_i], *valid.shape[1:]))
        valid = torch.cat(valid_list, dim=0)

        prev_point = points
        prev_mask = None
        
        # Iterative refinement loop
        for iter_i in range(cur_decoder_iters):
            if iter_i > 0:
                # Sample refinement points from disagreement regions
                with torch.no_grad():
                    valid_down = F.interpolate(valid, mask_s.shape[2:], mode="bilinear", align_corners=False)
                    mask_s_bin = (mask_s.detach() > mask_threshold) * valid_down
                    mask_t_bin = (mask_t.detach() > mask_threshold) * valid_down

                    if mask_t.shape[1] > 1:
                        max_iou_idx = iou_t.argmax(dim=1)
                        batch_range = torch.arange(mask_s.shape[0], device=mask_s.device)
                        mask_s_bin = mask_s_bin[batch_range, max_iou_idx].unsqueeze(1)
                        mask_t_bin = mask_t_bin[batch_range, max_iou_idx].unsqueeze(1)
                    
                    # Option 1: Use previous iteration's mask as mask prompt
                    if config.DISTILL.MASK_PROMPT_FROM_PREV_ITER and masks is not None:
                        # Use student's prediction as low-res mask input
                        mask_input_size = 256
                        prev_mask = F.interpolate(
                            mask_s.detach(),
                            size=(mask_input_size, mask_input_size),
                            mode='bilinear',
                            align_corners=False
                        )
                        if prev_mask.shape[1] > 1:
                            # Use best mask based on IoU
                            prev_mask = prev_mask[batch_range, max_iou_idx].unsqueeze(1)
                        masks = prev_mask
                    
                    # Option 2: Sample correction points from disagreement
                    point, label = sample_point_in_mask(mask_s_bin, mask_t_bin, config.DISTILL.POINTS_PER_REFINE_ITER)

                    point[:, :, 0] = point[:, :, 0] / mask_s_bin.shape[3] * img_size_pad[1]
                    point[:, :, 1] = point[:, :, 1] / mask_s_bin.shape[2] * img_size_pad[0]

                    del mask_s_bin, mask_t_bin
                    if prev_point is not None:
                        point = torch.cat([prev_point[0], point], dim=1)
                        label = torch.cat([prev_point[1], label], dim=1)
                    points = (point, label)
                    prev_point = points

            # Student forward pass
            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                mask_s, iou_s, _ = forward_sam3_batch(
                    model,
                    backbone_out,
                    points[0] if points else None,
                    points[1] if points else None,
                    boxes,
                    masks,
                    (cur_multimask_output > 1),
                    num_prompts=num_prompts
                )

            # Teacher forward pass
            with torch.no_grad():
                mask_t, iou_t, _ = forward_sam3_batch(
                    teacher,
                    teacher_backbone_out,
                    points[0] if points else None,
                    points[1] if points else None,
                    boxes,
                    masks,
                    (cur_multimask_output > 1),
                    num_prompts=num_prompts
                )

            # Compute decoder losses
            if config.DISTILL.DECODER_BCE > 0 or config.DISTILL.DECODER_FOCAL > 0 or config.DISTILL.DECODER_DICE > 0:
                valid_down = F.interpolate(valid, mask_s.shape[2:], mode='bilinear', align_corners=False)
                _mask_s = mask_s.float()
                _mask_t = mask_t

                temperature = config.DISTILL.TEMPERATURE
                _mask_s /= temperature
                _mask_t /= temperature

                target_logit = config.DISTILL.USE_TEACHER_LOGITS
                if not target_logit:
                    _mask_t = (_mask_t > mask_threshold).float()

                if config.DISTILL.DECODER_BCE > 0:
                    _tmp = sigmoid_ce_loss(_mask_s, _mask_t, valid_down,
                                          target_logit) * config.DISTILL.DECODER_BCE / cur_decoder_iters
                    key = f'dec_bce_{iter_i}'
                    loss[key] = (loss[key] + _tmp) if key in loss else _tmp

                if config.DISTILL.DECODER_FOCAL > 0:
                    _tmp = sigmoid_focal_loss(_mask_s, _mask_t, valid_down,
                                             target_logit) * config.DISTILL.DECODER_FOCAL / cur_decoder_iters
                    key = f'dec_focal_{iter_i}'
                    loss[key] = (loss[key] + _tmp) if key in loss else _tmp

                if config.DISTILL.DECODER_DICE > 0:
                    _tmp = dice_loss(_mask_s, _mask_t, valid_down,
                                    target_logit) * config.DISTILL.DECODER_DICE / cur_decoder_iters
                    key = f'dec_dice_{iter_i}'
                    loss[key] = (loss[key] + _tmp) if key in loss else _tmp

            if config.DISTILL.DECODER_IOU > 0:
                _tmp = F.mse_loss(iou_s, iou_t) * config.DISTILL.DECODER_IOU / cur_decoder_iters
                key = f'dec_iou_{iter_i}'
                loss[key] = (loss[key] + _tmp) if key in loss else _tmp

        # Aggregate losses
        for key in loss:
            loss[key] = loss[key] / config.TRAIN.ACCUMULATION_STEPS
            meters[key].update(loss[key].item(), len(samples))

        total_loss = sum(loss.values())

        if loss_writer is not None and dist.get_rank() == 0:
            display_dict = {'total': total_loss.item()}
            for key in loss:
                display_dict[key] = loss[key].item()
            loss_writer.add_scalars('loss', display_dict, epoch * num_steps + idx)

        loss_meter.update(total_loss.item(), samples.size(0))

        grad_norm = loss_scaler(
            total_loss,
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
            
            extra_meters_str = ''
            for k, v in meters.items():
                if k != 'data_time':
                    extra_meters_str += f'{k} {v.val:.4f} ({v.avg:.4f})  '
            
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]  "
                f"eta {datetime.timedelta(seconds=int(eta))}  "
                f"lr {lr:.6f}  time {batch_time.val:.4f} ({batch_time.avg:.4f})  "
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})  "
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})  "
                f"{extra_meters_str}"
                f"mem {memory_used:.0f}MB"
            )

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}"
    )


def build_valid_mask(config, img_size_before_pad, embed_shape, device):
    """Build a mask to filter out padding pixels."""
    batch_size = embed_shape[0]
    embed_h, embed_w = embed_shape[-2:]
    valid_mask = torch.zeros(batch_size, 1, embed_h, embed_w, device=device)
    
    for i in range(batch_size):
        h, w = img_size_before_pad[i][1:]
        valid_h = int(h / config.DATA.IMG_SIZE * embed_h)
        valid_w = int(w / config.DATA.IMG_SIZE * embed_w)
        valid_mask[i, :, :valid_h, :valid_w] = 1.0
    
    return valid_mask


def masked_cosine_loss(pred, target, mask):
    """Compute cosine similarity loss with masking."""
    pred = F.normalize(pred, p=2, dim=1)
    target = F.normalize(target, p=2, dim=1)
    cosine_sim = (pred * target).sum(dim=1, keepdim=True)
    loss = (1 - cosine_sim) * mask
    return loss.sum() / mask.sum()


if __name__ == "__main__":
    args, config = parse_option()
    config.defrost()
    config.freeze()

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
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

    # Linear scale the learning rate
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
        config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    )
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = (
            linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        )
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(
        output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}"
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
                project="EfficientSAM3-GeometryFinetune",
                config=config_dict,
                dir=wandb_output_path,
            )

    logger.info("===== git =====")
    logger.info(str(get_git_info()))

    logger.info(config.dump())

    main(args, config)
