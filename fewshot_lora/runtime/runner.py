"""批量实验运行器。

这个模块把所有组件串起来：
1. 从 txt 列表读取多个子数据集。
2. 每个子数据集重新构建 EfficientSAM3 + 注入 LoRA。
3. 在该子数据集内部执行多轮训练/评估/选样闭环。
4. 输出 summary.json 和每轮 adapter。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import FewShotLoRAConfig
from ..data.dataset import PreparedDataset, load_detect_dataset
from ..data.datatrain import read_dataset_list
from ..sam3_integration.factory import build_trainable_model
from ..sam3_integration.inference import evaluate_images
from ..sam3_integration.training import TrainRoundResult, train_lora_round
from .loop import DatasetRunSummary, run_dataset_loop
from .reports import write_dataset_summary


@dataclass(frozen=True)
class BatchRunSummary:
    """批量运行的总摘要。"""

    dataset_summaries: tuple[DatasetRunSummary, ...]


def run_from_dataset_list(dataset_list_path: Path, config: FewShotLoRAConfig) -> BatchRunSummary:
    """从 txt 列表批量运行多个子数据集。"""

    dataset_summaries = [
        run_one_dataset(dataset_dir, config)
        for dataset_dir in read_dataset_list(dataset_list_path)
    ]
    return BatchRunSummary(dataset_summaries=tuple(dataset_summaries))


def run_one_dataset(dataset_dir: Path, config: FewShotLoRAConfig) -> DatasetRunSummary:
    """运行单个子数据集。"""

    dataset = load_detect_dataset(dataset_dir, annotation_filename=config.annotation_filename)
    positive_images = [image for image in dataset.images if image.instances]
    if not positive_images:
        raise ValueError(f"{dataset_dir} 的 {config.annotation_filename} 中没有任何正样本图片")

    # 每个子数据集是一类新目标，因此重新构建模型并注入一份新的 LoRA adapter。
    model, loss_fn, lora_report = build_trainable_model(config)
    output_dir = config.dataset_output_dir(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_results: dict[int, TrainRoundResult] = {}

    def train_round(train_image_ids, round_index, adapter_path):
        """闭包：把通用 loop 的训练回调接到真实 LoRA 训练。"""

        result = train_lora_round(
            model=model,
            loss_fn=loss_fn,
            lora_report=lora_report,
            images=dataset.images,
            train_image_ids=tuple(train_image_ids),
            round_index=round_index,
            adapter_path=adapter_path,
            config=config,
        )
        train_results[round_index] = result
        return result

    def evaluate_round(round_index, adapter_path):
        """闭包：把通用 loop 的评估回调接到真实 text-only 全量评估。"""

        if config.evaluation.reload_adapter_for_eval:
            # 默认走“原始 base model + 磁盘 LoRA adapter”的恢复路径。
            # 这样每一轮评估都能顺带验证刚保存的 adapter.pt 是否可加载、可复现实例化。
            eval_model, _loss_fn, _lora_report = build_trainable_model(config)
            load_lora_adapter(eval_model, adapter_path)
            return evaluate_images(eval_model, dataset.images, config)
        return evaluate_images(model, dataset.images, config)

    summary = run_dataset_loop(
        dataset_name=output_dir.name,
        image_ids=[image.image_name for image in dataset.images],
        config=config,
        train_round=train_round,
        evaluate_round=evaluate_round,
        initial_selector=lambda _ids: positive_images[0].image_name,
    )
    write_dataset_summary(output_dir / "summary.json", config, summary, dataset.issues)
    return summary


def load_lora_adapter(model, adapter_path: Path):
    """延迟加载 LoRA adapter，避免 runner 顶层导入 torch 依赖。"""

    from ..sam3_integration.lora import load_lora_adapter as _load_lora_adapter

    return _load_lora_adapter(model, adapter_path)
