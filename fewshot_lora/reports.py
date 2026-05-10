"""实验 JSON 报告输出。

每个子数据集都会写一个 `summary.json`，里面包含：
- 本次运行配置。
- 每轮训练样本数、训练耗时、指标和下一轮选图。
- DetectTrainData 或图片解析中的数据问题。

报告使用中文友好的 `ensure_ascii=False`，后续可以直接打开查看中文路径和说明。
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

from .config import FewShotLoRAConfig
from .datatrain import DataIssue
from .loop import DatasetRunSummary


def write_dataset_summary(
    path: Path,
    config: FewShotLoRAConfig,
    summary: DatasetRunSummary,
    data_issues: list[DataIssue],
) -> Path:
    """把单个子数据集的运行结果写成 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": summary.dataset_name,
        "success": summary.success,
        "config": _jsonable(config),
        "rounds": [_jsonable(round_summary) for round_summary in summary.rounds],
        # 数据问题不阻断主流程，但必须显式落盘，避免标注数量不一致或图片歧义被静默吞掉。
        "data_issues": [_jsonable(issue) for issue in data_issues],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _jsonable(value: Any) -> Any:
    """把 dataclass、Path、tuple 等 Python 对象转成 JSON 可序列化对象。"""

    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
