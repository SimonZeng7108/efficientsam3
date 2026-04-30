# Fewshot YAML 配置入口设计

## 目标

为 `fewshot_adapter` 增加轻量 YAML 配置入口，让数据转换、模型加载、adapter 微调、loss、评估阈值都能通过一个配置文件调整，同时保留命令行参数覆盖能力，方便 GPU 实验快速改参和复现实验。

## 官方参数来源

本设计不照搬 SAM3 官方 Hydra 全量配置，只抽取和当前少样本目标检测闭环直接相关的默认值：

- `resolution=1008`、输入归一化均值方差 `0.5` 来自 SAM3 官方 eval / train 配置。
- EfficientSAM3-S 使用 `backbone_type=efficientvit`、`model_name=b0`，对应 `efficient_sam3_efficientvit_s.pt`。
- detection fine-tune 默认不训练 mask，`enable_segmentation=false`、`use_masks=false` 对齐 Roboflow detection 配置。
- matcher / box loss 使用官方值：`cost_class=2.0`、`cost_bbox=5.0`、`cost_giou=2.0`、`loss_bbox=5.0`、`loss_giou=2.0`、`loss_ce=20.0`、`presence_loss=20.0`、`pos_weight=10.0`、`alpha=0.25`、`gamma=2.0`。
- 推理阈值默认 `score_threshold=0.3`，贴近官方 thresholded postprocessor；评估匹配仍用常见 `iou_threshold=0.5`。

## YAML 结构

配置采用大写分组，靠近 EfficientSAM3 / Stage1 现有配置风格：

```yaml
DATA:
  DATATRAIN: dataset/DataTrain.txt
  IMAGE_DIR: dataset/images
  OUTPUT_DIR: dataset_json
  FULL_GROUND_TRUTH: dataset_json/full_gt.json
  IMAGE_MAP: dataset_json/image_map.json
  IMG_SIZE: 1008

MODEL:
  CHECKPOINT: sam3_checkpoints/efficient_sam3_efficientvit_s.pt
  BACKBONE_TYPE: efficientvit
  MODEL_NAME: b0
  DEVICE: cuda
  ENABLE_SEGMENTATION: false

ADAPTER:
  NUM_PROMPT_TOKENS: 8
  PROMPT_DIM: 256
  PROMPT_ADAPTER_DIM: 64
  TRAIN_DOT_PROD_SCORING: true
  TRAIN_BBOX_EMBED: false
  TRAIN_DECODER_CROSS_ATTENTION: false

TRAIN:
  OUTPUT_ROOT: runs/native_fewshot
  SEED: 0
  MAX_ROUNDS: 10
  STEPS_PER_ROUND: 80
  LEARNING_RATE: 0.00008
  WEIGHT_DECAY: 0.1

EVAL:
  LABEL: obj
  SCORE_THRESHOLD: 0.3
  IOU_THRESHOLD: 0.5
  LOCALIZATION_ERROR_THRESHOLD: 0.1
  IOU_MODE: hbb

LOSS:
  COST_CLASS: 2.0
  COST_BBOX: 5.0
  COST_GIOU: 2.0
  LOSS_CE: 20.0
  LOSS_BBOX: 5.0
  LOSS_GIOU: 2.0
  PRESENCE_LOSS: 20.0
  POS_WEIGHT: 10.0
  ALPHA: 0.25
  GAMMA: 2.0
  USE_PRESENCE: true
  USE_MASKS: false
```

## 行为

- `convert_datatrain` 支持 `--config`，从 `DATA.DATATRAIN`、`DATA.IMAGE_DIR`、`DATA.OUTPUT_DIR` 读取路径。
- `train_native_efficientsam3_fewshot` 支持 `--config`，把 YAML 映射到 `NativeFewShotLoopConfig`、`NativeAdapterConfig`、`NativeLossConfig`。
- 命令行参数优先级高于 YAML，适合临时覆盖 `--seed`、`--score-threshold`、`--steps-per-round` 等实验参数。
- 训练启动后保存最终生效配置到 `OUTPUT_ROOT/resolved_config.yaml`，用于复现实验。
- 配置加载模块必须是轻量纯 Python，不能要求 GPU 或 torch；没有 PyYAML 时给出清晰错误。

## 测试

- 配置加载测试：默认值、YAML 覆盖、未知 section / key 报错。
- dataclass 映射测试：YAML 能正确构造 loop / adapter / loss 配置。
- CLI 测试：`--config` 可运行，命令行覆盖优先生效。
- 轻量环境测试：不需要 torch 即可测试配置解析和 CLI 参数处理。
