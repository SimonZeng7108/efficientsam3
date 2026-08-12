"""测试 EfficientSAM3 原生少样本 adapter 的配置和冻结策略。"""

from fewshot_adapter.native.adapter import (
    NativeAdapterConfig,
    _empty_language_feature_shapes,
    should_train_parameter,
)


def test_should_train_parameter_always_keeps_task_prompt_trainable():
    """任务 prompt 是少样本快速学习的核心，必须始终可训练。"""
    config = NativeAdapterConfig(train_dot_prod_scoring=False)

    assert should_train_parameter("task_prompt_tokens", config)
    assert should_train_parameter("prompt_adapter.down.weight", config)


def test_should_train_parameter_enables_dot_product_scoring_optionally():
    """EfficientSAM3 当前活跃打分头是 dot_prod_scoring，不是 class_embed。"""
    config = NativeAdapterConfig(train_dot_prod_scoring=True)

    assert should_train_parameter("model.dot_prod_scoring.prompt_proj.weight", config)
    assert not should_train_parameter(
        "model.backbone.vision_backbone.trunk.model.patch_embed.weight",
        config,
    )


def test_should_train_parameter_can_enable_bbox_embed():
    """需要更强定位适配时，可以只开放 decoder bbox_embed 小模块。"""
    config = NativeAdapterConfig(train_bbox_embed=True)

    assert should_train_parameter("model.transformer.decoder.bbox_embed.0.weight", config)
    assert not should_train_parameter("model.transformer.encoder.layers.0.linear1.weight", config)


def test_empty_language_feature_shapes_follow_batch_size():
    """多图 batch 时 language_mask 必须和 text_ids 可索引的 prompt 数一致。"""
    shapes = _empty_language_feature_shapes(batch_size=2, hidden_dim=256)

    assert shapes["language_features"] == (0, 2, 256)
    assert shapes["language_mask"] == (2, 0)
    assert shapes["language_embeds"] == (2, 256)
