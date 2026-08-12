"""SAM3 原生 matcher/loss 的少样本封装。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils.torch import require_torch


@dataclass(frozen=True)
class NativeLossConfig:
    """复用 SAM3 官方 loss 时的轻量配置。"""

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


class NativeLossFactory:
    """构造 SAM3 原生 loss 的高层门面。"""

    def build(self, config: NativeLossConfig | None = None) -> Any:
        return build_native_loss(config)


def build_native_loss(config: NativeLossConfig | None = None) -> Any:
    """构建 SAM3 原生检测 loss。

    这里采用 local normalization，避免单卡少样本验证时触发分布式 all_reduce。
    """
    cfg = config or NativeLossConfig()
    require_torch()
    from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks
    from sam3.train.loss.sam3_loss import Sam3LossWrapper
    from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher

    matcher = BinaryHungarianMatcherV2(
        cost_class=cfg.cost_class,
        cost_bbox=cfg.cost_bbox,
        cost_giou=cfg.cost_giou,
        focal=True,
        alpha=cfg.alpha,
        gamma=cfg.gamma,
    )
    # EfficientSAM3 训练态 DAC decoder 会产生 one-to-many 分支；官方配置使用
    # BinaryOneToManyMatcher，否则 Sam3LossWrapper 遇到 pred_logits_o2m 会报错。
    o2m_matcher = BinaryOneToManyMatcher(
        alpha=cfg.o2m_matcher_alpha,
        threshold=cfg.o2m_matcher_threshold,
        topk=cfg.o2m_matcher_topk,
    )
    loss_fns: list[Any] = [
        Boxes(weight_dict={"loss_bbox": cfg.loss_bbox, "loss_giou": cfg.loss_giou}),
        IABCEMdetr(
            weak_loss=False,
            weight_dict={
                "loss_ce": cfg.loss_ce,
                "presence_loss": cfg.presence_loss,
            },
            pos_weight=cfg.pos_weight,
            alpha=cfg.alpha,
            gamma=cfg.gamma,
            use_presence=cfg.use_presence,
        ),
    ]
    if cfg.use_masks:
        loss_fns.append(
            Masks(
                weight_dict={
                    "loss_mask": cfg.loss_mask,
                    "loss_dice": cfg.loss_dice,
                }
            )
        )
    return Sam3LossWrapper(
        loss_fns_find=loss_fns,
        matcher=matcher,
        o2m_matcher=o2m_matcher,
        o2m_weight=cfg.o2m_weight,
        use_o2m_matcher_on_o2m_aux=cfg.use_o2m_matcher_on_o2m_aux,
        normalization="local",
    )
