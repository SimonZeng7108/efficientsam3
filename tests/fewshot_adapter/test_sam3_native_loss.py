"""测试 SAM3 原生 loss 封装的依赖提示。"""

import builtins
import sys
from types import ModuleType

import pytest

from fewshot_adapter.native.loss import NativeLossConfig, build_native_loss


def test_build_native_loss_requires_torch_in_lightweight_environment(monkeypatch):
    """没有 PyTorch 时应清晰失败，而不是在深层 SAM3 import 处报晦涩错误。"""
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError, match="PyTorch is required"):
        build_native_loss()


def test_build_native_loss_rejects_mask_loss_until_mask_targets_exist():
    """当前 DataTrain 训练 batch 尚未生成 mask target，开启 mask loss 应清晰失败。"""
    with pytest.raises(ValueError, match="USE_MASKS"):
        build_native_loss(NativeLossConfig(use_masks=True))


def test_build_native_loss_configures_o2m_matcher(monkeypatch):
    """训练态 EfficientSAM3 会输出 o2m 分支，loss wrapper 必须配置 o2m matcher。"""

    class Boxes:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class IABCEMdetr:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Masks:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Sam3LossWrapper:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class BinaryHungarianMatcherV2:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class BinaryOneToManyMatcher:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    loss_fns_module = ModuleType("sam3.train.loss.loss_fns")
    loss_fns_module.Boxes = Boxes
    loss_fns_module.IABCEMdetr = IABCEMdetr
    loss_fns_module.Masks = Masks

    sam3_loss_module = ModuleType("sam3.train.loss.sam3_loss")
    sam3_loss_module.Sam3LossWrapper = Sam3LossWrapper

    matcher_module = ModuleType("sam3.train.matcher")
    matcher_module.BinaryHungarianMatcherV2 = BinaryHungarianMatcherV2
    matcher_module.BinaryOneToManyMatcher = BinaryOneToManyMatcher

    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "sam3.train.loss.loss_fns", loss_fns_module)
    monkeypatch.setitem(sys.modules, "sam3.train.loss.sam3_loss", sam3_loss_module)
    monkeypatch.setitem(sys.modules, "sam3.train.matcher", matcher_module)

    loss = build_native_loss()

    assert isinstance(loss.kwargs["matcher"], BinaryHungarianMatcherV2)
    assert isinstance(loss.kwargs["o2m_matcher"], BinaryOneToManyMatcher)
    assert loss.kwargs["o2m_weight"] == pytest.approx(2.0)
    assert loss.kwargs["use_o2m_matcher_on_o2m_aux"] is False
    assert loss.kwargs["o2m_matcher"].kwargs == {
        "alpha": pytest.approx(0.3),
        "threshold": pytest.approx(0.4),
        "topk": 4,
    }
