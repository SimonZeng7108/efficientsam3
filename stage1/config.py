# --------------------------------------------------------
# Stage 1 Distillation Configuration
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
_C.DATA.BATCH_SIZE = 64
_C.DATA.DATA_PATH = ''
_C.DATA.DATASET = 'sa1b'
_C.DATA.MEAN = [123.675, 116.28, 103.53]
_C.DATA.STD = [58.395, 57.12, 57.375]
_C.DATA.IMG_SIZE = 1024
_C.DATA.INTERPOLATION = 'bicubic'
_C.DATA.PIN_MEMORY = True
_C.DATA.NUM_WORKERS = 8
_C.DATA.DEBUG = False
_C.DATA.NUM_SAMPLES = -1
_C.DATA.FILTER_BY_AREA = [None, None]
_C.DATA.SORT_BY_AREA = False
_C.DATA.LOAD_GT_MASK = False
_C.DATA.BOX_JITTER = False
_C.DATA.MASK_NMS = -1.0

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = 'efficient_sam3'
_C.MODEL.NAME = 'efficient_sam3'
_C.MODEL.BACKBONE = 'repvit_m0_9'
_C.MODEL.PRETRAINED = ''
_C.MODEL.RESUME = ''

# -----------------------------------------------------------------------------
# Distillation settings
# -----------------------------------------------------------------------------
_C.DISTILL = CN()
_C.DISTILL.ENABLED = True
_C.DISTILL.ENCODER_ONLY = True
_C.DISTILL.EMBED_DIM = 1024
_C.DISTILL.EMBED_SIZE = 64
_C.DISTILL.NUM_EMBED = _C.DISTILL.EMBED_SIZE * _C.DISTILL.EMBED_SIZE
_C.DISTILL.PIXEL_WISE = 1.0
_C.DISTILL.CHANNEL_WISE = 0.0
_C.DISTILL.CORRELATION = 0.0
_C.DISTILL.COSINE = 1.0
_C.DISTILL.TEACHER_EMBED_PATH = 'data/teacher_embeddings'
_C.DISTILL.SAVE_TEACHER_EMBED = False
_C.DISTILL.NO_RAND = True
_C.DISTILL.MAX_ALLOWED_PROMPTS = -1

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
_C.TRAIN.CLIP_GRAD = 5.0
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

    # config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    config.DISTILL.NUM_EMBED = (
        config.DISTILL.EMBED_SIZE * config.DISTILL.EMBED_SIZE
    )

    config.freeze()


def get_config(args=None):
    config = _C.clone()
    if args is not None:
        update_config(config, args)
    return config

