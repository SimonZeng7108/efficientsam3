"""PyTorch 依赖的懒加载工具。

fewshot_adapter 的数据解析和几何工具不需要 PyTorch；只有真正训练或
调用 EfficientSAM3 时才需要导入 torch。这个小工具让轻量环境也能先跑
数据转换和单元测试。
"""

from __future__ import annotations

from typing import Any


def require_torch() -> Any:
    """懒加载 torch，并在缺失时给出明确安装提示。"""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for native EfficientSAM3 few-shot training. "
            "Install PyTorch in the GPU environment before running training."
        ) from exc
    return torch
