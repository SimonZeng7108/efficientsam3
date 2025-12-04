# --------------------------------------------------------
# Stage 1 Geometry Finetune Configuration
# Prompt-in-the-Loop Knowledge Distillation
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
_C.DATA.BATCH_SIZE = 4
_C.DATA.DATA_PATH = ''
_C.DATA.DATASET = 'sa1b'
_C.DATA.MEAN = [123.675, 116.28, 103.53]
_C.DATA.STD = [58.395, 57.12, 57.375]
_C.DATA.IMG_SIZE = 1008
_C.DATA.INTERPOLATION = 'bicubic'
_C.DATA.PIN_MEMORY = True
_C.DATA.NUM_WORKERS = 8
_C.DATA.DEBUG = False
_C.DATA.NUM_SAMPLES = -1
_C.DATA.FILTER_BY_AREA = [None, None]
_C.DATA.SORT_BY_AREA = False
_C.DATA.LOAD_GT_MASK = True  # Required for geometry finetune
_C.DATA.BOX_JITTER = False
_C.DATA.MASK_NMS = -1.0

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = 'efficient_sam3'
_C.MODEL.NAME = 'efficient_sam3'
_C.MODEL.BACKBONE = 'repvit_m0_9'
_C.MODEL.PRETRAINED = ''  # Path to converted stage1 checkpoint
_C.MODEL.RESUME = ''
_C.MODEL.SAM3_CHECKPOINT = 'sam3_checkpoints/sam3.pt'  # Teacher SAM3 checkpoint

# -----------------------------------------------------------------------------
# Distillation settings
# -----------------------------------------------------------------------------
_C.DISTILL = CN()
_C.DISTILL.ENABLED = True
_C.DISTILL.ENCODER_ONLY = False  # Now finetune full model with decoder
_C.DISTILL.EMBED_DIM = 1024
_C.DISTILL.EMBED_SIZE = 72
_C.DISTILL.NUM_EMBED = _C.DISTILL.EMBED_SIZE * _C.DISTILL.EMBED_SIZE
_C.DISTILL.TEACHER_EMBED_PATH = 'output/stage1_teacher/embeddings'
_C.DISTILL.SAVE_TEACHER_EMBED = False
_C.DISTILL.NO_RAND = True
_C.DISTILL.MAX_ALLOWED_PROMPTS = 16

# Encoder distillation (optional, can be disabled)
_C.DISTILL.PIXEL_WISE = 0.0  # Encoder already trained
_C.DISTILL.CHANNEL_WISE = 0.0
_C.DISTILL.CORRELATION = 0.0
_C.DISTILL.COSINE = 0.0

# Decoder distillation (main focus)
_C.DISTILL.DECODER_BCE = 5.0
_C.DISTILL.DECODER_FOCAL = 0.0
_C.DISTILL.DECODER_DICE = 5.0
_C.DISTILL.DECODER_IOU = 1.0
_C.DISTILL.DECODER_ATTN = 0.0
_C.DISTILL.USE_TEACHER_LOGITS = True
_C.DISTILL.TEMPERATURE = 1.0
_C.DISTILL.POINT_REND_SAMPLING = False

# Prompt settings
_C.DISTILL.PROMPT_TYPE = ['box', 'point']  # Types of prompts to use: 'box', 'point', 'mask'
_C.DISTILL.PROMPT_BOX_TO_POINT = False  # Convert box to center point
_C.DISTILL.PROMPT_MASK_TO_POINT = True  # Sample point from GT mask
_C.DISTILL.USE_MASK_PROMPT = False  # Use ground truth mask as prompt (low-res mask input)
_C.DISTILL.MASK_PROMPT_FROM_PREV_ITER = True  # Use previous iteration's mask as prompt
_C.DISTILL.DECODE_ITERS = 2  # Number of refinement iterations
_C.DISTILL.POINTS_PER_REFINE_ITER = 1  # Points to sample per iteration
_C.DISTILL.ITER_ON_BOX = True  # Apply iterative refinement on box prompts
_C.DISTILL.MULTIMASK_OUTPUT = 4  # Number of mask outputs (1 or 3)
_C.DISTILL.MULTIMASK_ON_BOX = True  # Use multimask for box prompts

# Freezing options
_C.DISTILL.FREEZE_IMAGE_ENCODER = False  # Train encoder to adapt to geometry prompts
_C.DISTILL.FREEZE_PROMPT_ENCODER = True  # Freeze for Stage 2 memory bank compatibility
_C.DISTILL.FREEZE_MASK_DECODER = True    # Freeze for Stage 2 memory bank compatibility
_C.DISTILL.FREEZE_TEXT_ENCODER = True    # Keep text encoder frozen if enabled
_C.DISTILL.ENABLE_TEXT_ENCODER = False  # Exclude text encoder to save memory (merge after training)
_C.DISTILL.INIT_FROM_TEACHER = False   # Already initialized from stage1 conversion

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 10  # Shorter than stage1 since we're finetuning
_C.TRAIN.WARMUP_EPOCHS = 0
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 3.2e-3  # Higher LR for finetuning
_C.TRAIN.WARMUP_LR = 3.2e-5
_C.TRAIN.MIN_LR = 3.2e-4
_C.TRAIN.CLIP_GRAD = 0.01
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.ACCUMULATION_STEPS = 1
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
_C.AMP_ENABLE = False  # Disable AMP for stability during geometry finetune
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
    config.DISTILL.NUM_EMBED = (
        config.DISTILL.EMBED_SIZE * config.DISTILL.EMBED_SIZE
    )
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

    if args.local_rank is None and 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    config.LOCAL_RANK = args.local_rank

    config.DISTILL.NUM_EMBED = (
        config.DISTILL.EMBED_SIZE * config.DISTILL.EMBED_SIZE
    )

    config.freeze()


def get_config(args=None):
    config = _C.clone()
    if args is not None:
        update_config(config, args)
    return config
