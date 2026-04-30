"""测试少样本 YAML 配置加载。"""

from pytest import approx, raises

from fewshot_adapter.config import (
    build_adapter_config,
    build_loop_config,
    build_loss_config,
    load_fewshot_config,
)


def test_load_fewshot_config_uses_official_detection_defaults(tmp_path):
    """默认参数尽量贴近 SAM3 官方 detection fine-tune 配置。"""
    config_path = tmp_path / "fewshot.yaml"
    config_path.write_text(
        "\n".join(
            [
                "DATA:",
                "  FULL_GROUND_TRUTH: dataset_json/full_gt.json",
                "  IMAGE_MAP: dataset_json/image_map.json",
                "MODEL:",
                "  CHECKPOINT: sam3_checkpoints/efficient_sam3_efficientvit_s.pt",
                "TRAIN:",
                "  OUTPUT_ROOT: runs/native_fewshot",
                "EVAL:",
                "  LABEL: obj",
            ]
        ),
        encoding="utf-8",
    )

    config = load_fewshot_config(config_path)

    assert config.data.img_size == 1008
    assert config.model.backbone_type == "efficientvit"
    assert config.model.model_name == "b0"
    assert config.model.enable_segmentation is False
    assert config.train.learning_rate == approx(8e-5)
    assert config.train.weight_decay == approx(0.1)
    assert config.eval.score_threshold == approx(0.3)
    assert config.loss.cost_class == approx(2.0)
    assert config.loss.cost_bbox == approx(5.0)
    assert config.loss.cost_giou == approx(2.0)
    assert config.loss.loss_ce == approx(20.0)
    assert config.loss.loss_bbox == approx(5.0)
    assert config.loss.loss_giou == approx(2.0)
    assert config.loss.presence_loss == approx(20.0)
    assert config.loss.pos_weight == approx(10.0)
    assert config.loss.o2m_weight == approx(2.0)
    assert config.loss.o2m_matcher_alpha == approx(0.3)
    assert config.loss.o2m_matcher_threshold == approx(0.4)
    assert config.loss.o2m_matcher_topk == 4
    assert config.loss.use_o2m_matcher_on_o2m_aux is False
    assert config.loss.use_masks is False


def test_load_fewshot_config_rejects_unknown_keys(tmp_path):
    config_path = tmp_path / "fewshot.yaml"
    config_path.write_text("MODEL:\n  UNKNOWN_FIELD: 1\n", encoding="utf-8")

    with raises(ValueError, match="UNKNOWN_FIELD"):
        load_fewshot_config(config_path)


def test_config_builds_runtime_dataclasses(tmp_path):
    config_path = tmp_path / "fewshot.yaml"
    config_path.write_text(
        "\n".join(
            [
                "DATA:",
                "  FULL_GROUND_TRUTH: gt.json",
                "  IMAGE_MAP: image_map.json",
                "  IMG_SIZE: 512",
                "MODEL:",
                "  CHECKPOINT: ckpt.pt",
                "  DEVICE: cuda:0",
                "  MODEL_NAME: b0",
                "TRAIN:",
                "  OUTPUT_ROOT: runs/test",
                "  STEPS_PER_ROUND: 2",
                "ADAPTER:",
                "  NUM_PROMPT_TOKENS: 4",
                "LOSS:",
                "  USE_MASKS: true",
                "  O2M_WEIGHT: 1.5",
                "  O2M_MATCHER_TOPK: 2",
                "EVAL:",
                "  LABEL: obj",
                "  SCORE_THRESHOLD: 0.2",
            ]
        ),
        encoding="utf-8",
    )

    config = load_fewshot_config(config_path)
    loop_config = build_loop_config(config)
    adapter_config = build_adapter_config(config)
    loss_config = build_loss_config(config)

    assert loop_config.checkpoint == "ckpt.pt"
    assert loop_config.output_root == "runs/test"
    assert loop_config.device == "cuda:0"
    assert loop_config.resolution == 512
    assert loop_config.steps_per_round == 2
    assert loop_config.score_threshold == approx(0.2)
    assert loop_config.backbone_type == "efficientvit"
    assert loop_config.model_name == "b0"
    assert loop_config.enable_segmentation is False
    assert adapter_config.num_prompt_tokens == 4
    assert loss_config.use_masks is True
    assert loss_config.o2m_weight == approx(1.5)
    assert loss_config.o2m_matcher_topk == 2
