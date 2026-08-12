from dataclasses import fields, is_dataclass

from fewshot_lora import config


def test_every_config_field_has_chinese_help_metadata():
    config_classes = [
        config.ModelConfig,
        config.LoRAConfig,
        config.TrainingConfig,
        config.EvaluationConfig,
        config.FewShotLoRAConfig,
    ]

    missing = []
    for cls in config_classes:
        assert is_dataclass(cls)
        for field in fields(cls):
            help_text = field.metadata.get("help_zh")
            if not help_text or not any("\u4e00" <= char <= "\u9fff" for char in help_text):
                missing.append(f"{cls.__name__}.{field.name}")

    assert missing == []
