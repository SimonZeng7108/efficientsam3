def test_refactored_package_structure_exposes_expected_modules():
    """结构重整后，核心模块应该从新的职责子包导入。"""

    from fewshot_lora.data.dataset import load_detect_dataset
    from fewshot_lora.eval.geometry import OrientedBox
    from fewshot_lora.runtime.loop import run_dataset_loop
    from fewshot_lora.sam3_integration.factory import build_trainable_model
    from fewshot_lora.types import TrainRoundOutput

    assert callable(load_detect_dataset)
    assert OrientedBox.__name__ == "OrientedBox"
    assert callable(run_dataset_loop)
    assert callable(build_trainable_model)
    assert TrainRoundOutput.__name__ == "TrainRoundOutput"


def test_sam3_integration_does_not_import_runtime_package():
    """SAM3 集成层不能反向依赖 runtime，避免形成分层环。"""

    import ast
    from pathlib import Path

    integration_dir = Path("fewshot_lora/sam3_integration")
    offenders = []
    for path in integration_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                is_absolute_runtime = module.startswith("fewshot_lora.runtime")
                is_relative_runtime = node.level > 0 and module.startswith("runtime")
                if is_absolute_runtime or is_relative_runtime:
                    offenders.append(path.as_posix())
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("fewshot_lora.runtime"):
                        offenders.append(path.as_posix())

    assert offenders == []
