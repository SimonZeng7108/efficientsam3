# --------------------------------------------------------
# Stage 2 Geometry Fine-tuning Configuration
# --------------------------------------------------------

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 4  # Smaller batch size due to mask computation
_C.DATA.DATA_PATH = ''
_C.DATA.DATASET = 'sa1b'
_C.DATA.MEAN = [123.675, 116.28, 103.53]
_C.DATA.STD = [58.395, 57.12, 57.375]
_C.DATA.IMG_SIZE = 1008  # Match Stage 1 teacher (1008/14=72 spatial size)
_C.DATA.INTERPOLATION = 'bicubic'
_C.DATA.PIN_MEMORY = True
_C.DATA.NUM_WORKERS = 8
_C.DATA.DEBUG = False
_C.DATA.NUM_SAMPLES = -1
_C.DATA.FILTER_BY_AREA = [None, None]
_C.DATA.SORT_BY_AREA = True  # Sort by area for consistent prompt selection
_C.DATA.LOAD_GT_MASK = False  # We use teacher masks, not GT
_C.DATA.BOX_JITTER = True  # Data augmentation for robustness
_C.DATA.MASK_NMS = 0.8  # Remove highly overlapping masks
_C.DATA.MAX_PROMPTS_PER_IMAGE = 16  # Limit prompts per image

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = 'efficient_sam3'
_C.MODEL.NAME = 'efficient_sam3'
_C.MODEL.BACKBONE = 'repvit_m1_1'  # Student backbone
_C.MODEL.PRETRAINED = ''  # Stage 1 pretrained weights
_C.MODEL.RESUME = ''  # Resume training checkpoint
_C.MODEL.SAM3_CHECKPOINT = ''  # Path to SAM3 checkpoint for frozen components

# -----------------------------------------------------------------------------
# Distillation settings
# -----------------------------------------------------------------------------
_C.DISTILL = CN()
_C.DISTILL.ENABLED = True
_C.DISTILL.EMBED_DIM = 1024  # Trunk output dimension
_C.DISTILL.EMBED_SIZE = 72  # Spatial size after trunk (1008/14=72 for SAM3)
_C.DISTILL.NUM_EMBED = _C.DISTILL.EMBED_SIZE * _C.DISTILL.EMBED_SIZE

# Teacher embeddings (saved from Stage 1 or recomputed)
_C.DISTILL.TEACHER_EMBED_PATH = ''
_C.DISTILL.USE_SAVED_EMBEDDINGS = True  # Use saved trunk embeddings for efficiency

# Loss weights
# NOTE: Measured raw loss scales (before training converges):
#   embed_mse ~ 1000-1500 (large initially, decreases as student learns)
#   mask_bce ~ 0.3-1.5, mask_dice ~ 0.97-1.0, total ~ 1.5-2.5
# Weight = mask_total / embed_mse ≈ 2.0 / 1330 ≈ 0.0015
_C.DISTILL.EMBEDDING_LOSS_WEIGHT = 0.0015  # MSE on trunk embeddings (empirically tuned)
_C.DISTILL.MASK_BCE_WEIGHT = 1.0  # BCE loss on masks
_C.DISTILL.MASK_DICE_WEIGHT = 1.0  # Dice loss on masks
_C.DISTILL.MASK_FOCAL_WEIGHT = 0.0  # Focal loss (optional)
# NOTE: SAM3 doesn't output IoU predictions, so this is disabled
_C.DISTILL.IOU_LOSS_WEIGHT = 0.0  # IoU prediction matching (disabled - SAM3 has no IoU output)

# Temperature for mask distillation
_C.DISTILL.TEMPERATURE = 1.0

# Prompt settings
_C.DISTILL.USE_BOX_PROMPTS = True
_C.DISTILL.USE_POINT_PROMPTS = True
_C.DISTILL.MAX_PROMPTS = 16
_C.DISTILL.NO_RAND = True

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 50  # Shorter than Stage 1, fine-tuning
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.WEIGHT_DECAY = 0.01  # Smaller weight decay for fine-tuning
_C.TRAIN.BASE_LR = 1e-4  # Smaller LR for fine-tuning
_C.TRAIN.WARMUP_LR = 1e-7
_C.TRAIN.MIN_LR = 1e-6
_C.TRAIN.CLIP_GRAD = 1.0
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.ACCUMULATION_STEPS = 4  # Effective batch size = 4 * 4 = 16
_C.TRAIN.USE_CHECKPOINT = False
_C.TRAIN.LAYER_LR_DECAY = 1.0
_C.TRAIN.EVAL_BN_WHEN_TRAINING = False
_C.TRAIN.FIND_UNUSED_PARAMETERS = False

_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.AMP_ENABLE = True
_C.OUTPUT = ''
_C.TAG = 'default'
_C.SAVE_FREQ = 1
_C.PRINT_FREQ = 10
_C.SEED = 0
_C.EVAL_MODE = False
_C.THROUGHPUT_MODE = False
_C.LOCAL_RANK = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
            config.defrost()
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.sam3_checkpoint:
        config.MODEL.SAM3_CHECKPOINT = args.sam3_checkpoint
    if args.teacher_embed_path:
        config.DISTILL.TEACHER_EMBED_PATH = args.teacher_embed_path
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.disable_amp or args.only_cpu:
        config.AMP_ENABLE = False
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True

    # Set local rank for distributed training
    if 'LOCAL_RANK' in os.environ:
        config.LOCAL_RANK = int(os.environ['LOCAL_RANK'])

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    update_config(config, args)
    return config
