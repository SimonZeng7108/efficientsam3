import pytest

torch = pytest.importorskip("torch")

from fewshot_lora.sam3_integration.lora import LoRAConv2d, freeze_non_lora, inject_lora_conv2d_by_suffix


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
