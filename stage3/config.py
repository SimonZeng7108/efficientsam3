"""Stage 3 End-to-End Fine-Tuning Configuration."""

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()
_C.BASE = ['']

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 2
_C.DATA.DATA_PATH = ''
_C.DATA.DATASET = 'sa1b_enhanced'
_C.DATA.MEAN = [123.675, 116.28, 103.53]
_C.DATA.STD = [58.395, 57.12, 57.375]
_C.DATA.IMG_SIZE = 1008
_C.DATA.INTERPOLATION = 'bicubic'
_C.DATA.PIN_MEMORY = True
_C.DATA.NUM_WORKERS = 4
_C.DATA.PERSISTENT_WORKERS = True
_C.DATA.PREFETCH_FACTOR = 2
_C.DATA.NUM_SAMPLES = -1
_C.DATA.SORT_BY_AREA = True
_C.DATA.BOX_JITTER = True
_C.DATA.MASK_NMS = 0.8
_C.DATA.MAX_PROMPTS_PER_IMAGE = 8
_C.DATA.NUM_SAMPLE_POINTS = 3
_C.DATA.TEXT_LABEL_MODE = 'random'

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = 'efficient_sam3'
_C.MODEL.NAME = 'efficient_sam3_stage3'
_C.MODEL.BACKBONE = 'repvit_m1_1'
_C.MODEL.PRETRAINED = ''
_C.MODEL.RESUME = ''
_C.MODEL.SAM3_CHECKPOINT = ''
_C.MODEL.TRAINABLE_SCOPE = 'trunk_only'

# ---------------------------------------------------------------------------
# Distillation / Supervision
# ---------------------------------------------------------------------------
_C.DISTILL = CN()
_C.DISTILL.ENABLED = True
_C.DISTILL.EMBED_DIM = 1024
_C.DISTILL.EMBED_SIZE = 72
_C.DISTILL.NUM_EMBED = _C.DISTILL.EMBED_SIZE * _C.DISTILL.EMBED_SIZE

_C.DISTILL.TEACHER_EMBED_DIR = ''
_C.DISTILL.USE_SAVED_EMBEDDINGS = True
_C.DISTILL.TEACHER_EMBED_DTYPE = 'float32'

_C.DISTILL.EMBEDDING_LOSS_WEIGHT = 0.0015
_C.DISTILL.MASK_BCE_WEIGHT = 1.0
_C.DISTILL.MASK_DICE_WEIGHT = 1.0
_C.DISTILL.MASK_FOCAL_WEIGHT = 0.0
_C.DISTILL.CLASSIFICATION_WEIGHT = 0.1

_C.DISTILL.USE_BOX_PROMPTS = True
_C.DISTILL.USE_POINT_PROMPTS = True
_C.DISTILL.MAX_PROMPTS = 8
_C.DISTILL.NO_RAND = False

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 30
_C.TRAIN.WARMUP_EPOCHS = 2
_C.TRAIN.WEIGHT_DECAY = 0.01
_C.TRAIN.BASE_LR = 5e-5
_C.TRAIN.WARMUP_LR = 1e-7
_C.TRAIN.MIN_LR = 1e-6
_C.TRAIN.CLIP_GRAD = 1.0
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.ACCUMULATION_STEPS = 8
_C.TRAIN.USE_CHECKPOINT = False
_C.TRAIN.FIND_UNUSED_PARAMETERS = True

_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 20
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
_C.AMP_ENABLE = True
_C.OUTPUT = ''
_C.TAG = 'default'
_C.SAVE_FREQ = 5
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
    if os.environ.get('RANK', '0') == '0':
        print(f'=> merge config from {cfg_file}')
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
    if args.teacher_embed_dir:
        config.DISTILL.TEACHER_EMBED_DIR = args.teacher_embed_dir
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.trainable_scope:
        config.MODEL.TRAINABLE_SCOPE = args.trainable_scope
    if args.disable_amp or args.only_cpu:
        config.AMP_ENABLE = False
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True

    if 'LOCAL_RANK' in os.environ:
        config.LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    config.freeze()


def get_config(args):
    config = _C.clone()
    update_config(config, args)
    return config
