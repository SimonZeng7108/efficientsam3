"""少样本 LoRA 交互式同类目标查找的配置定义。

这个文件是调参入口，尽量把每个参数都写清楚：
- `ModelConfig`：控制 EfficientSAM3 原生模型如何构建。
- `LoRAConfig`：控制 LoRA 注入位置和低秩参数规模。
- `TrainingConfig`：控制每一轮少样本微调的时间、步数和 loss。
- `EvaluationConfig`：控制每轮全量评估的阈值和后处理。
- `FewShotLoRAConfig`：把上面几组配置组合成一次完整实验。

每个 dataclass 字段都带 `metadata["help_zh"]`，方便后续做中文配置导出、
CLI help 或实验记录；这也能避免“参数很多但不知道干什么”的情况。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


def _help(text: str) -> dict[str, str]:
    """给 dataclass 字段附加中文说明，测试会检查每个配置项都必须填写。"""

    return {"help_zh": text}


@dataclass(frozen=True)
class ModelConfig:
    # 主路线固定使用 EfficientSAM3 image model；这里保留参数是为了后续复现实验可切换。
    backbone_type: str = field(
        default="efficientvit",
        metadata=_help("视觉学生主干类型；本任务主路线使用 efficientvit。"),
    )
    # EfficientViT 默认使用 b0，代表轻量模型，适合后续边缘设备方向。
    model_name: str = field(
        default="b0",
        metadata=_help("视觉主干具体型号；EfficientViT 推荐从 b0 开始。"),
    )
    # 开启 segmentation head 后才能用 pred_masks 拟合 OBB，也能训练 polygon mask loss。
    enable_segmentation: bool = field(
        default=True,
        metadata=_help("是否启用分割头；开启后可训练 mask loss 并从预测 mask 拟合 OBB。"),
    )
    # 服务器实验一般需要显式传入 checkpoint；不传则只构建模型结构，不加载权重。
    checkpoint_path: Path | None = field(
        default=None,
        metadata=_help("EfficientSAM3 checkpoint 路径；服务器正式实验建议显式指定。"),
    )
    # 默认使用 SAM3 原生文本编码器；如需轻量文本编码器可在这里配置。
    text_encoder_type: str | None = field(
        default=None,
        metadata=_help("文本编码器类型；None 表示使用 SAM3 默认文本编码器。"),
    )


@dataclass(frozen=True)
class LoRAConfig:
    # rank 越大可训练容量越强，但 adapter 更大、每轮训练更慢。
    rank: int = field(default=8, metadata=_help("LoRA 低秩维度；越大容量越强但训练更慢。"))
    # alpha 控制 LoRA 分支缩放，实际缩放系数为 alpha / rank。
    alpha: float = field(
        default=16.0,
        metadata=_help("LoRA 缩放超参；实际缩放系数为 alpha/rank。"),
    )
    # 少样本很容易过拟合，dropout 可用于抑制 adapter 记死单张图。
    dropout: float = field(
        default=0.0,
        metadata=_help("LoRA 分支 dropout；默认关闭，必要时用于缓解少样本过拟合。"),
    )
    # 只命中 LiteMLA 的 qkv/proj 内部 Conv2d，保持与 fewshot_adapter 完全隔离。
    target_suffixes: tuple[str, ...] = field(
        default=("qkv.conv", "proj.conv"),
        metadata=_help("LoRA 注入目标后缀；默认只注入 LiteMLA.qkv.conv 和 LiteMLA.proj.conv。"),
    )


@dataclass(frozen=True)
class TrainingConfig:
    # 每轮同时受 step 和 wall-clock 限制，满足“单轮目标耗时 < 60 秒”。
    max_steps_per_round: int = field(
        default=80,
        metadata=_help("每轮最多训练步数；会与时间上限共同限制一轮训练。"),
    )
    max_seconds_per_round: float = field(
        default=60.0,
        metadata=_help("每轮最多训练秒数；达到时间上限即停止当前轮。"),
    )
    learning_rate: float = field(
        default=1e-3,
        metadata=_help("LoRA 参数学习率；只作用于可训练 adapter 参数。"),
    )
    weight_decay: float = field(
        default=0.0,
        metadata=_help("优化器权重衰减；默认 0，避免少样本 adapter 被过度正则。"),
    )
    use_amp: bool = field(
        default=True,
        metadata=_help("是否在 CUDA 上启用 AMP 混合精度以降低显存和加速训练。"),
    )
    loss_normalization: str = field(
        default="local",
        metadata=_help("SAM3 loss 归一化方式；单机少样本默认 local，避免触发分布式 all-reduce。"),
    )
    enable_mask_loss: bool = field(
        default=True,
        metadata=_help("是否启用原生 mask loss；开启后 polygon mask 会参与监督。"),
    )


@dataclass(frozen=True)
class EvaluationConfig:
    score_threshold: float = field(
        default=0.5,
        metadata=_help("预测分数阈值；低于该分数的预测会被过滤。"),
    )
    mask_threshold: float = field(
        default=0.5,
        metadata=_help("预测 mask 二值化阈值；用于从 pred_masks 拟合 OBB。"),
    )
    iou_threshold: float = field(
        default=0.5,
        metadata=_help("OBB IoU 成功匹配阈值；达到该阈值才算 TP。"),
    )
    localization_iou_threshold: float = field(
        default=0.1,
        metadata=_help("定位错误阈值；有重叠但未达匹配阈值时记为 localization_error。"),
    )
    nms_iou_threshold: float = field(
        default=0.5,
        metadata=_help("单类别 OBB NMS 阈值；用于移除重复预测。"),
    )
    text_prompt: str = field(
        default="object",
        metadata=_help("无实例 label 可用时的兜底文本 prompt；评估时不使用 GT 框。"),
    )


@dataclass(frozen=True)
class FewShotLoRAConfig:
    output_dir: Path = field(metadata=_help("实验输出目录；每个子数据集会在这里生成 summary.json 和 adapter。"))
    max_rounds: int = field(
        default=5,
        metadata=_help("每个子数据集最多交互轮数；成功或达到该值后停止。"),
    )
    annotation_filename: str = field(
        default="DetectTrainData.txt",
        metadata=_help("标注文件名；可改为 DetectTrainData_sample5.txt 做 smoke test。"),
    )
    image_size: int = field(
        default=1008,
        metadata=_help("模型输入方图尺寸；与 EfficientSAM3 训练/推理默认尺寸保持一致。"),
    )
    device: str = field(
        default="cuda",
        metadata=_help("运行设备，例如 cuda 或 cpu；正式训练建议在 Linux CUDA 服务器上运行。"),
    )
    model: ModelConfig = field(
        default_factory=ModelConfig,
        metadata=_help("模型构建相关配置，包含 backbone、checkpoint 和分割头开关。"),
    )
    lora: LoRAConfig = field(
        default_factory=LoRAConfig,
        metadata=_help("LoRA 注入和 adapter 参数规模配置。"),
    )
    training: TrainingConfig = field(
        default_factory=TrainingConfig,
        metadata=_help("每轮 LoRA 微调相关配置。"),
    )
    evaluation: EvaluationConfig = field(
        default_factory=EvaluationConfig,
        metadata=_help("全量评估、阈值和后处理相关配置。"),
    )

    def dataset_output_dir(self, dataset_dir: Path) -> Path:
        name = dataset_dir.name or str(dataset_dir).strip().replace(":", "")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "dataset"
        return self.output_dir / safe_name
