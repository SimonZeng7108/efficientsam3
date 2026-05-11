"""单轮 LoRA 微调。

训练阶段用 GT 模拟用户标注：已加入训练集的图片会把全部 GT 实例作为
geometric prompt 和监督目标。模型本体权重冻结，只更新 LoRA 参数。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import TYPE_CHECKING

from ..config import FewShotLoRAConfig
from ..data.dataset import PreparedImage
from ..data.preprocess import load_image_tensor, resize_mask
from ..runtime.loop import TrainRoundOutput

if TYPE_CHECKING:
    from .lora import LoRAInjectionReport
    from .sam3_batch import FindBatchSample


@dataclass(frozen=True)
class TrainRoundResult(TrainRoundOutput):
    """真实训练回调返回的单轮统计。"""

    pass


def train_lora_round(
    model,
    loss_fn,
    lora_report: "LoRAInjectionReport",
    images: list[PreparedImage],
    train_image_ids: tuple[str, ...],
    round_index: int,
    adapter_path: Path,
    config: FewShotLoRAConfig,
) -> TrainRoundResult:
    """执行一轮 LoRA 训练并保存 adapter。

    同一子数据集内会跨轮继续训练当前 adapter；新增失败图后，下一轮使用累计
    标注集继续优化。
    """

    # 延迟导入 torch，便于在无 torch 的开发机上运行纯逻辑测试。
    import torch

    from .lora import save_lora_adapter
    from .sam3_batch import build_batched_datapoint

    id_to_image = {image.image_name: image for image in images}
    train_images = [id_to_image[image_id] for image_id in train_image_ids]
    samples = [_to_find_batch_sample(image, config) for image in train_images]
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    use_amp = config.training.use_amp and config.device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    device = torch.device(config.device)
    start = time.perf_counter()
    steps = 0
    model.train()

    # 少样本交互场景通常只有 1 到几张标注图，这里每 step 使用累计标注集做一个小 batch。
    while steps < config.training.max_steps_per_round:
        if time.perf_counter() - start >= config.training.max_seconds_per_round:
            break
        batch = build_batched_datapoint(samples, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(batch)
            losses = _compute_find_losses(model, loss_fn, outputs, batch)
            loss = losses["core_loss"]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        steps += 1

    train_seconds = time.perf_counter() - start
    save_lora_adapter(
        model,
        adapter_path,
        lora_report,
        extra={"round_index": round_index, "train_image_ids": list(train_image_ids)},
    )
    return TrainRoundResult(
        adapter_path=adapter_path,
        train_image_count=len(train_images),
        train_instance_count=sum(len(image.instances) for image in train_images),
        train_seconds=train_seconds,
        train_steps=steps,
    )


def _compute_find_losses(model, loss_fn, outputs, batch):
    """把 SAM3 原生 target dataclass 转成 loss 需要的 dict 后再计算 loss。

    `Sam3Image.forward()` 内部用 `BatchedFindTarget` 做 matching，但
    `Sam3LossWrapper` 里的具体 loss 函数访问的是 dict，例如
    `targets["boxes_xyxy"]` 和 `targets["object_ids_padded"]`。因此训练时
    必须复用模型自己的 `back_convert()`，保持和原生路径一致。
    """

    converted_targets = [model.back_convert(batch.find_targets[0])]
    return loss_fn(outputs, converted_targets)


def _to_find_batch_sample(image: PreparedImage, config: FewShotLoRAConfig) -> "FindBatchSample":
    from .sam3_batch import FindBatchSample

    """把准备好的图片索引转换为 SAM3 batch 样本。"""

    image_tensor = load_image_tensor(image.path, config.image_size)
    target_boxes = [instance.box_cxcywh for instance in image.instances]
    target_masks = [resize_mask(instance.mask, config.image_size) for instance in image.instances]
    text = _image_text_prompt(image, config)
    return FindBatchSample(
        image=image_tensor,
        image_id=image.image_name,
        original_size=(image.height, image.width),
        target_boxes=target_boxes,
        target_masks=target_masks,
        prompt_boxes=target_boxes,
        text=text,
    )


def _image_text_prompt(image: PreparedImage, config: FewShotLoRAConfig) -> str:
    """训练时优先使用标注 label 作为文本 prompt。"""

    if image.instances:
        return image.instances[0].label or config.evaluation.text_prompt
    return config.evaluation.text_prompt
