"""测试 SAM3 原生 loss 封装的依赖提示。"""

import pytest

from fewshot_adapter.native.loss import build_native_loss


def test_build_native_loss_requires_torch_in_lightweight_environment():
    """没有 PyTorch 时应清晰失败，而不是在深层 SAM3 import 处报晦涩错误。"""
    with pytest.raises(ModuleNotFoundError, match="PyTorch is required"):
        build_native_loss()
