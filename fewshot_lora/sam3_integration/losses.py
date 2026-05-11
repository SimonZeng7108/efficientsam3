"""构造 EfficientSAM3 原生 find loss。

这里不自定义检测/分割损失，而是复用仓库里的 `Sam3LossWrapper`、matcher、
box loss、classification/presence loss 和可选 mask loss。这样训练路径尽量贴近
EfficientSAM3 原生 interactive find / grounding 流程。
"""

from __future__ import annotations


def build_sam3_find_loss(enable_mask_loss: bool = True, normalization: str = "local"):
    """创建少样本 LoRA 训练用 loss。

    `normalization` 默认用 local，是因为少样本单机实验通常不启动分布式；
    如果使用默认 global，SAM3 loss 内部会尝试 distributed all-reduce。
    """

    # 延迟导入 SAM3 训练模块：本地无 torch 时仍可测试纯逻辑模块。
    from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks
    from sam3.train.loss.sam3_loss import Sam3LossWrapper
    from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher

    matcher = BinaryHungarianMatcherV2(
        focal=True,
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        alpha=0.25,
        gamma=2,
        stable=False,
    )
    # 最小稳定组合：框回归 + 单类别置信度/presence。
    loss_fns = [
        Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
        IABCEMdetr(
            weak_loss=False,
            weight_dict={"loss_ce": 20.0, "presence_loss": 20.0},
            pos_weight=10.0,
            alpha=0.25,
            gamma=2,
            use_presence=True,
            pos_focal=False,
            pad_n_queries=200,
            pad_scale_pos=1.0,
        ),
    ]
    if enable_mask_loss:
        # 有 polygon mask 时开启 mask loss，帮助后处理从 pred_masks 拟合更准确的 OBB。
        loss_fns.append(
            Masks(
                focal_alpha=0.25,
                focal_gamma=2.0,
                weight_dict={"loss_mask": 200.0, "loss_dice": 10.0},
                compute_aux=False,
            )
        )
    return Sam3LossWrapper(
        matcher=matcher,
        o2m_weight=2.0,
        o2m_matcher=BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4),
        use_o2m_matcher_on_o2m_aux=False,
        loss_fns_find=loss_fns,
        loss_fn_semantic_seg=None,
        normalization=normalization,
        scale_by_find_batch_size=True,
    )
