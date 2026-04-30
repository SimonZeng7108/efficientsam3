"""少样本闭环 YAML 配置。

这里使用项目自己的轻量配置结构，而不是直接接入 SAM3 官方 Hydra 配置。
默认值尽量对齐官方 detection fine-tune / eval 配置，字段名保持大写风格，
方便和 EfficientSAM3 现有 `stage1/configs/*.yaml` 一起阅读。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

from ..native.adapter import NativeAdapterConfig
from ..native.loss import NativeLossConfig
from ..native.trainer import NativeFewShotLoopConfig


@dataclass(frozen=True)
class FewShotDataSection:
    datatrain: str | None = None
    image_dir: str | None = None
    output_dir: str = "dataset_json"
    full_ground_truth: str = "dataset_json/full_gt.json"
    image_map: str = "dataset_json/image_map.json"
    img_size: int = 1008


@dataclass(frozen=True)
class FewShotModelSection:
    checkpoint: str = "sam3_checkpoints/efficient_sam3_efficientvit_s.pt"
    backbone_type: str = "efficientvit"
    model_name: str = "b0"
    device: str = "cuda"
    enable_segmentation: bool = False


@dataclass(frozen=True)
class FewShotAdapterSection:
    num_prompt_tokens: int = 8
    prompt_dim: int = 256
    prompt_adapter_dim: int = 64
    prompt_init_std: float = 0.02
    train_dot_prod_scoring: bool = True
    train_bbox_embed: bool = False
    train_decoder_cross_attention: bool = False


@dataclass(frozen=True)
class FewShotTrainSection:
    output_root: str = "runs/native_fewshot"
    seed: int = 0
    max_rounds: int = 10
    steps_per_round: int = 80
    learning_rate: float = 8e-5
    weight_decay: float = 0.1


@dataclass(frozen=True)
class FewShotEvalSection:
    label: str | None = None
    score_threshold: float = 0.3
    iou_threshold: float = 0.5
    localization_error_threshold: float = 0.1
    iou_mode: str = "hbb"


@dataclass(frozen=True)
class FewShotLossSection:
    cost_class: float = 2.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0
    loss_ce: float = 20.0
    loss_bbox: float = 5.0
    loss_giou: float = 2.0
    presence_loss: float = 20.0
    pos_weight: float = 10.0
    alpha: float = 0.25
    gamma: float = 2.0
    use_presence: bool = True
    o2m_weight: float = 2.0
    o2m_matcher_alpha: float = 0.3
    o2m_matcher_threshold: float = 0.4
    o2m_matcher_topk: int = 4
    use_o2m_matcher_on_o2m_aux: bool = False
    use_masks: bool = False
    loss_mask: float = 200.0
    loss_dice: float = 10.0


@dataclass(frozen=True)
class FewShotExperimentConfig:
    data: FewShotDataSection = FewShotDataSection()
    model: FewShotModelSection = FewShotModelSection()
    adapter: FewShotAdapterSection = FewShotAdapterSection()
    train: FewShotTrainSection = FewShotTrainSection()
    eval: FewShotEvalSection = FewShotEvalSection()
    loss: FewShotLossSection = FewShotLossSection()


_SECTION_TYPES = {
    "DATA": FewShotDataSection,
    "MODEL": FewShotModelSection,
    "ADAPTER": FewShotAdapterSection,
    "TRAIN": FewShotTrainSection,
    "EVAL": FewShotEvalSection,
    "LOSS": FewShotLossSection,
}

_SECTION_ATTRS = {
    "DATA": "data",
    "MODEL": "model",
    "ADAPTER": "adapter",
    "TRAIN": "train",
    "EVAL": "eval",
    "LOSS": "loss",
}


def load_fewshot_config(path: str | Path | None = None) -> FewShotExperimentConfig:
    """读取 YAML 配置；`path=None` 时返回官方口径默认值。"""
    config = FewShotExperimentConfig()
    if path is None:
        return config
    payload = _read_yaml(path)
    if payload is None:
        return config
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a YAML mapping")
    return apply_config_overrides(config, payload)


def apply_config_overrides(
    config: FewShotExperimentConfig,
    overrides: Mapping[str, Any],
) -> FewShotExperimentConfig:
    """把 YAML 或 CLI 覆盖项合并进配置对象。

    `None` 表示未覆盖，会被跳过；未知 section 或 key 会直接报错，避免拼写错误
    悄悄生效失败。
    """
    next_config = config
    for section_name, section_payload in overrides.items():
        normalized_section = _normalize_key(section_name)
        if normalized_section not in _SECTION_TYPES:
            raise ValueError(f"unknown config section: {section_name}")
        if section_payload is None:
            continue
        if not isinstance(section_payload, Mapping):
            raise ValueError(f"{section_name} section must be a mapping")
        attr_name = _SECTION_ATTRS[normalized_section]
        current_section = getattr(next_config, attr_name)
        updated_section = _apply_section_overrides(
            current_section,
            section_payload,
            section_name=normalized_section,
        )
        next_config = replace(next_config, **{attr_name: updated_section})
    return next_config


def build_loop_config(config: FewShotExperimentConfig) -> NativeFewShotLoopConfig:
    """构造训练闭环运行时配置。"""
    return NativeFewShotLoopConfig(
        checkpoint=config.model.checkpoint,
        output_root=config.train.output_root,
        label=config.eval.label,
        device=config.model.device,
        resolution=config.data.img_size,
        seed=config.train.seed,
        max_rounds=config.train.max_rounds,
        steps_per_round=config.train.steps_per_round,
        learning_rate=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        score_threshold=config.eval.score_threshold,
        iou_threshold=config.eval.iou_threshold,
        localization_error_threshold=config.eval.localization_error_threshold,
        iou_mode=config.eval.iou_mode,
        backbone_type=config.model.backbone_type,
        model_name=config.model.model_name,
        enable_segmentation=config.model.enable_segmentation,
    )


def build_adapter_config(config: FewShotExperimentConfig) -> NativeAdapterConfig:
    """构造少样本 adapter 配置。"""
    return NativeAdapterConfig(
        num_prompt_tokens=config.adapter.num_prompt_tokens,
        prompt_dim=config.adapter.prompt_dim,
        prompt_adapter_dim=config.adapter.prompt_adapter_dim,
        prompt_init_std=config.adapter.prompt_init_std,
        train_dot_prod_scoring=config.adapter.train_dot_prod_scoring,
        train_bbox_embed=config.adapter.train_bbox_embed,
        train_decoder_cross_attention=config.adapter.train_decoder_cross_attention,
    )


def build_loss_config(config: FewShotExperimentConfig) -> NativeLossConfig:
    """构造 SAM3 原生 loss 配置。"""
    return NativeLossConfig(
        cost_class=config.loss.cost_class,
        cost_bbox=config.loss.cost_bbox,
        cost_giou=config.loss.cost_giou,
        loss_ce=config.loss.loss_ce,
        loss_bbox=config.loss.loss_bbox,
        loss_giou=config.loss.loss_giou,
        presence_loss=config.loss.presence_loss,
        pos_weight=config.loss.pos_weight,
        alpha=config.loss.alpha,
        gamma=config.loss.gamma,
        use_presence=config.loss.use_presence,
        o2m_weight=config.loss.o2m_weight,
        o2m_matcher_alpha=config.loss.o2m_matcher_alpha,
        o2m_matcher_threshold=config.loss.o2m_matcher_threshold,
        o2m_matcher_topk=config.loss.o2m_matcher_topk,
        use_o2m_matcher_on_o2m_aux=config.loss.use_o2m_matcher_on_o2m_aux,
        use_masks=config.loss.use_masks,
        loss_mask=config.loss.loss_mask,
        loss_dice=config.loss.loss_dice,
    )


def save_fewshot_config(path: str | Path, config: FewShotExperimentConfig) -> None:
    """保存最终生效配置，方便复现实验。"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dump_yaml(config_to_dict(config)), encoding="utf-8")


def config_to_dict(config: FewShotExperimentConfig) -> dict[str, dict[str, Any]]:
    """转换为大写 YAML 字段。"""
    return {
        "DATA": _section_to_upper_dict(config.data),
        "MODEL": _section_to_upper_dict(config.model),
        "ADAPTER": _section_to_upper_dict(config.adapter),
        "TRAIN": _section_to_upper_dict(config.train),
        "EVAL": _section_to_upper_dict(config.eval),
        "LOSS": _section_to_upper_dict(config.loss),
    }


def _apply_section_overrides(
    section: Any,
    payload: Mapping[str, Any],
    *,
    section_name: str,
) -> Any:
    field_names = {field.name for field in fields(section)}
    updates: dict[str, Any] = {}
    for raw_key, value in payload.items():
        if value is None:
            continue
        attr_name = _normalize_key(raw_key).lower()
        if attr_name not in field_names:
            raise ValueError(f"unknown config key: {section_name}.{raw_key}")
        updates[attr_name] = value
    if not updates:
        return section
    return replace(section, **updates)


def _section_to_upper_dict(section: Any) -> dict[str, Any]:
    return {key.upper(): value for key, value in asdict(section).items()}


def _normalize_key(key: Any) -> str:
    return str(key).strip().upper()


def _read_yaml(path: str | Path) -> Any:
    yaml = _require_yaml()
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _dump_yaml(payload: Mapping[str, Any]) -> str:
    yaml = _require_yaml()
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)


def _require_yaml() -> Any:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - 只在缺依赖环境触发。
        raise ModuleNotFoundError(
            "PyYAML is required for --config YAML support. Install with: pip install pyyaml"
        ) from exc
    return yaml
