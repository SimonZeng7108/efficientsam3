def test_refactored_package_structure_exposes_expected_modules():
    """结构重整后，核心模块应该从新的职责子包导入。"""

    from fewshot_lora.data.dataset import load_detect_dataset
    from fewshot_lora.eval.geometry import OrientedBox
    from fewshot_lora.runtime.loop import run_dataset_loop
    from fewshot_lora.sam3_integration.factory import _build_trainable_model

    assert callable(load_detect_dataset)
    assert OrientedBox.__name__ == "OrientedBox"
    assert callable(run_dataset_loop)
    assert callable(_build_trainable_model)
