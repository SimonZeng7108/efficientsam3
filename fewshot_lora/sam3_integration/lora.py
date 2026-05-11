"""EfficientViT LiteMLA 的 Conv-LoRA 注入工具。

核心策略：
- 只替换 `LiteMLA.qkv.conv` 和 `LiteMLA.proj.conv` 这类 `nn.Conv2d`。
- 原始卷积权重完全冻结。
- 只训练 `lora_down` 和 `lora_up` 两个低秩分支。

这样 adapter 很小，适合每个子数据集保存一份任务特定权重。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn


class LoRAConv2d(nn.Module):
    """Conv2d 的低秩残差 adapter。

    前向结果 = 原始卷积输出 + LoRA 分支输出 * (alpha / rank)。
    """

    def __init__(
        self,
        base: nn.Conv2d,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank 必须为正数")
        if base.groups != 1:
            raise ValueError("LoRAConv2d 当前只支持 groups=1 的卷积")

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Conv2d(
            in_channels=base.in_channels,
            out_channels=rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            bias=False,
        )
        self.lora_up = nn.Conv2d(rank, base.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scaling


@dataclass(frozen=True)
class LoRAInjectionReport:
    """LoRA 注入后的统计信息。"""

    injected_names: tuple[str, ...]
    trainable_parameter_count: int
    total_parameter_count: int


def inject_lora_conv2d_by_suffix(
    model: nn.Module,
    target_suffixes: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> list[str]:
    """按模块名后缀替换 Conv2d。

    例如目标后缀 `qkv.conv` 会命中 `...context_module.main.qkv.conv`。
    只按后缀匹配是为了不依赖 EfficientViT 具体 stage 名称。
    """

    suffixes = tuple(target_suffixes)
    injected: list[str] = []
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if not isinstance(child, nn.Conv2d) or isinstance(child, LoRAConv2d):
                continue
            if any(full_name.endswith(suffix) for suffix in suffixes):
                setattr(module, child_name, LoRAConv2d(child, rank=rank, alpha=alpha, dropout=dropout))
                injected.append(full_name)
    return injected


def freeze_non_lora(model: nn.Module) -> None:
    """冻结非 LoRA 参数，只留下 adapter 分支可训练。"""

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(("lora_down" in name) or ("lora_up" in name))


def apply_efficientvit_lora(
    model: nn.Module,
    target_suffixes: Iterable[str] = ("qkv.conv", "proj.conv"),
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> LoRAInjectionReport:
    """对 EfficientSAM3 模型应用 EfficientViT LoRA 注入。"""

    injected_names = inject_lora_conv2d_by_suffix(
        model,
        target_suffixes=target_suffixes,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    if not injected_names:
        raise RuntimeError(
            "没有任何 Conv2d 模块命中 LoRA 目标后缀："
            + ", ".join(tuple(target_suffixes))
        )
    freeze_non_lora(model)
    return LoRAInjectionReport(
        injected_names=tuple(injected_names),
        trainable_parameter_count=count_trainable_parameters(model),
        total_parameter_count=count_parameters(model),
    )


def count_parameters(model: nn.Module) -> int:
    """统计总参数量。"""

    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """统计当前可训练参数量。"""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """只导出 LoRA 参数，避免 adapter checkpoint 包含完整 base model。"""

    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if ".lora_down." in name or ".lora_up." in name
    }


def save_lora_adapter(
    model: nn.Module,
    path: Path,
    report: LoRAInjectionReport,
    extra: dict | None = None,
) -> None:
    """保存 LoRA adapter 和少量元信息。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": lora_state_dict(model),
            "injected_names": report.injected_names,
            "trainable_parameter_count": report.trainable_parameter_count,
            "total_parameter_count": report.total_parameter_count,
            "extra": extra or {},
        },
        path,
    )


def load_lora_adapter(model: nn.Module, path: Path, strict: bool = False):
    """加载 adapter 权重到已经注入相同 LoRA 结构的模型。"""

    payload = torch.load(path, map_location="cpu")
    return model.load_state_dict(payload["state_dict"], strict=strict)
