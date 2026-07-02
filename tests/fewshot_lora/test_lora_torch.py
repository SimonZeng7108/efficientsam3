from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from fewshot_lora.sam3_integration.lora import (
    LoRAConv2d,
    LoRAInjectionReport,
    freeze_non_lora,
    inject_lora_conv2d_by_suffix,
    save_lora_adapter,
)


class TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Module()
        self.qkv.conv = torch.nn.Conv2d(3, 6, 1, bias=False)
        self.proj = torch.nn.Module()
        self.proj.conv = torch.nn.Conv2d(6, 3, 1, bias=False)
        self.other = torch.nn.Conv2d(3, 3, 1, bias=False)


def test_inject_lora_conv2d_by_suffix_replaces_only_requested_modules():
    model = TinyModule()

    injected = inject_lora_conv2d_by_suffix(model, ("qkv.conv", "proj.conv"), rank=2, alpha=4)

    assert injected == ["qkv.conv", "proj.conv"]
    assert isinstance(model.qkv.conv, LoRAConv2d)
    assert isinstance(model.proj.conv, LoRAConv2d)
    assert isinstance(model.other, torch.nn.Conv2d)
    assert not model.qkv.conv.base.weight.requires_grad


def test_freeze_non_lora_leaves_only_lora_parameters_trainable():
    model = TinyModule()
    inject_lora_conv2d_by_suffix(model, ("qkv.conv",), rank=2, alpha=4)

    freeze_non_lora(model)

    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable == ["qkv.conv.lora_down.weight", "qkv.conv.lora_up.weight"]


def test_save_lora_adapter_writes_only_lora_state_dict(tmp_path: Path):
    model = TinyModule()
    injected = inject_lora_conv2d_by_suffix(model, ("qkv.conv", "proj.conv"), rank=2, alpha=4)
    report = LoRAInjectionReport(
        injected_names=tuple(injected),
        trainable_parameter_count=1,
        total_parameter_count=2,
    )

    path = tmp_path / "adapter.pt"
    save_lora_adapter(model, path, report)

    payload = torch.load(path, map_location="cpu")
    state_keys = set(payload["state_dict"])
    assert state_keys
    assert all(".lora_down." in key or ".lora_up." in key for key in state_keys)
    assert not any(".base." in key or key.endswith(".other.weight") for key in state_keys)
