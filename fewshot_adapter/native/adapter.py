"""EfficientSAM3 原生少样本 prompt / adapter 包装器。

这里不新建检测 decoder，也不走候选框列表；包装器只给 SAM3 原生
`forward_grounding` 链路注入任务级视觉 prompt，并控制少量参数可训练。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..utils.torch import require_torch

try:  # 允许无 torch 环境导入配置和纯函数。
    import torch
except ModuleNotFoundError:  # pragma: no cover - 由轻量测试覆盖调用路径。
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class NativeAdapterConfig:
    """少样本原生 adapter 配置。"""

    num_prompt_tokens: int = 8
    prompt_dim: int = 256
    prompt_adapter_dim: int = 64
    prompt_init_std: float = 0.02
    train_dot_prod_scoring: bool = True
    train_bbox_embed: bool = False
    train_decoder_cross_attention: bool = False


def should_train_parameter(name: str, config: NativeAdapterConfig) -> bool:
    """判断某个参数是否属于少样本可训练集合。"""
    normalized = name.removeprefix("module.")
    if normalized.startswith("task_prompt_tokens"):
        return True
    if "prompt_adapter" in normalized:
        return True
    if config.train_dot_prod_scoring and ".dot_prod_scoring." in f".{normalized}":
        return True
    if config.train_bbox_embed and ".bbox_embed" in normalized:
        return True
    if config.train_decoder_cross_attention:
        is_decoder = ".transformer.decoder." in f".{normalized}"
        is_cross_attention = ".cross_attn." in normalized or ".ca_text." in normalized
        if is_decoder and is_cross_attention:
            return True
    return False


def freeze_for_fewshot(module: Any, config: NativeAdapterConfig) -> list[str]:
    """冻结大部分参数，只开放少样本 adapter 相关参数。"""
    trainable: list[str] = []
    for name, parameter in module.named_parameters():
        enabled = should_train_parameter(name, config)
        parameter.requires_grad = enabled
        if enabled:
            trainable.append(name)
    return trainable


def _empty_language_feature_shapes(*, batch_size: int, hidden_dim: int) -> dict[str, tuple[int, ...]]:
    """返回无文本分支需要注入的占位张量形状。"""
    return {
        "language_features": (0, batch_size, hidden_dim),
        "language_mask": (batch_size, 0),
        "language_embeds": (batch_size, hidden_dim),
    }


if torch is not None:

    class TaskPromptAdapter(torch.nn.Module):
        """对任务 prompt 做轻量残差适配的小 MLP。"""

        def __init__(self, hidden_dim: int, bottleneck_dim: int):
            super().__init__()
            self.down = torch.nn.Linear(hidden_dim, bottleneck_dim)
            self.act = torch.nn.GELU()
            self.up = torch.nn.Linear(bottleneck_dim, hidden_dim)
            # 初始时尽量接近恒等映射，避免一开始破坏 SAM3 预训练行为。
            torch.nn.init.zeros_(self.up.weight)
            torch.nn.init.zeros_(self.up.bias)

        def forward(self, tokens: Any) -> Any:
            return tokens + self.up(self.act(self.down(tokens)))


    class NativeEfficientSAM3FewShotModel(torch.nn.Module):
        """将任务 prompt 注入 SAM3 原生 grounding 链路的包装器。"""

        def __init__(self, model: Any, config: NativeAdapterConfig):
            super().__init__()
            self.model = model
            self.config = config
            self.task_prompt_tokens = torch.nn.Parameter(
                torch.randn(
                    config.num_prompt_tokens,
                    1,
                    config.prompt_dim,
                )
                * config.prompt_init_std
            )
            self.prompt_adapter = TaskPromptAdapter(
                hidden_dim=config.prompt_dim,
                bottleneck_dim=config.prompt_adapter_dim,
            )
            self.trainable_names = freeze_for_fewshot(self, config)

        def adapter_state_dict(self) -> dict[str, Any]:
            """只保存少样本任务相关权重，便于后续按任务切换。"""
            names = set(self.trainable_names)
            return {
                name: value.detach().cpu()
                for name, value in self.state_dict().items()
                if name in names or name.startswith("prompt_adapter")
            }

        def task_prompt(self, *, batch_size: int, device: Any) -> tuple[Any, Any]:
            """生成 `(T, B, C)` prompt token 和 `(B, T)` attention mask。"""
            tokens = self.task_prompt_tokens.to(device).expand(-1, batch_size, -1)
            tokens = self.prompt_adapter(tokens)
            mask = torch.zeros(
                batch_size,
                tokens.shape[0],
                dtype=torch.bool,
                device=device,
            )
            return tokens, mask

        def forward_batch(self, batch: Any) -> dict[str, Any]:
            """对 `NativeSam3Batch` 执行训练前向。"""
            image_batch = batch.image_batch
            return self.forward_image_batch(
                images=image_batch.images,
                find_stage=batch.find_stage,
                find_target=batch.find_target,
            )

        def forward_image_batch(
            self,
            *,
            images: Any,
            find_stage: Any,
            find_target: Any | None = None,
        ) -> dict[str, Any]:
            """直接复用 SAM3 backbone/encoder/decoder/mask head。"""
            backbone_out = {"img_batch_all_stages": images}
            backbone_out.update(self.model.backbone.forward_image(images))
            geometric_prompt = self.model._get_dummy_prompt(num_prompts=images.shape[0])
            return self.forward_grounding_with_visual_prompt(
                backbone_out=backbone_out,
                find_input=find_stage,
                geometric_prompt=geometric_prompt,
                find_target=find_target,
            )

        def forward_grounding_with_visual_prompt(
            self,
            *,
            backbone_out: dict[str, Any],
            find_input: Any,
            geometric_prompt: Any,
            find_target: Any | None = None,
        ) -> dict[str, Any]:
            """SAM3 `forward_grounding` 的 visual-prompt 版本。

            原始 `_encode_prompt` 在 `encode_text=False` 时仍会读取
            `language_features`，所以这里注入长度为 0 的语言特征，确保不
            运行文本编码器，同时保持 SAM3 内部张量索引逻辑不变。
            """
            batch_size = int(find_input.img_ids.numel())
            device = find_input.img_ids.device
            self._ensure_empty_language_features(
                backbone_out,
                batch_size=batch_size,
                device=device,
            )
            visual_prompt, visual_mask = self.task_prompt(
                batch_size=batch_size,
                device=device,
            )
            prompt, prompt_mask, backbone_out = self.model._encode_prompt(
                backbone_out,
                find_input,
                geometric_prompt,
                visual_prompt_embed=visual_prompt,
                visual_prompt_mask=visual_mask,
                encode_text=False,
            )
            backbone_out, encoder_out, _ = self.model._run_encoder(
                backbone_out,
                find_input,
                prompt,
                prompt_mask,
            )
            out: dict[str, Any] = {
                "encoder_hidden_states": encoder_out["encoder_hidden_states"],
                "prev_encoder_out": {
                    "encoder_out": encoder_out,
                    "backbone_out": backbone_out,
                },
            }
            out, hs = self.model._run_decoder(
                memory=out["encoder_hidden_states"],
                pos_embed=encoder_out["pos_embed"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=prompt,
                prompt_mask=prompt_mask,
                encoder_out=encoder_out,
            )
            self.model._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=find_input.img_ids,
                vis_feat_sizes=encoder_out["vis_feat_sizes"],
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
            )
            if find_target is not None:
                self.model._compute_matching(out, self.model.back_convert(find_target))
            return out

        def _ensure_empty_language_features(
            self,
            backbone_out: dict[str, Any],
            *,
            batch_size: int,
            device: Any,
        ) -> None:
            if "language_features" in backbone_out and "language_mask" in backbone_out:
                return
            hidden_dim = self.config.prompt_dim
            shapes = _empty_language_feature_shapes(batch_size=batch_size, hidden_dim=hidden_dim)
            backbone_out["language_features"] = torch.zeros(
                *shapes["language_features"],
                dtype=self.task_prompt_tokens.dtype,
                device=device,
            )
            backbone_out["language_mask"] = torch.zeros(
                *shapes["language_mask"],
                dtype=torch.bool,
                device=device,
            )
            backbone_out["language_embeds"] = torch.zeros(
                *shapes["language_embeds"],
                dtype=self.task_prompt_tokens.dtype,
                device=device,
            )

else:

    class TaskPromptAdapter:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            require_torch()


    class NativeEfficientSAM3FewShotModel:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            require_torch()


def build_native_fewshot_model(
    *,
    checkpoint_path: str | Path,
    config: NativeAdapterConfig,
    device: str = "cuda",
    backbone_type: str = "efficientvit",
    model_name: str = "b0",
    resolution: int = 1008,
) -> Any:
    """加载完整 EfficientSAM3，并包装为少样本 adapter 模型。"""
    require_torch()
    from sam3.model_builder import build_efficientsam3_image_model

    model = build_efficientsam3_image_model(
        checkpoint_path=str(checkpoint_path),
        backbone_type=backbone_type,
        model_name=model_name,
        device=device,
        load_from_HF=False,
        eval_mode=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )
    wrapper = NativeEfficientSAM3FewShotModel(model=model, config=config).to(device)
    wrapper.resolution = resolution
    return wrapper


def save_native_adapter(path: str | Path, wrapper: Any) -> None:
    """保存少样本 adapter checkpoint。"""
    torch_mod = require_torch()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch_mod.save(
        {
            "model_type": "native_efficientsam3_fewshot_adapter",
            "config": asdict(wrapper.config),
            "state_dict": wrapper.adapter_state_dict(),
            "trainable_names": list(wrapper.trainable_names),
        },
        output,
    )
