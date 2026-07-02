"""跨子包共享的轻量数据结构。

这个文件不能依赖 `data`、`eval`、`runtime` 或 `sam3_integration`，否则很容易
重新形成分层环。适合放多个子包都需要引用、但不属于某个具体运行阶段的 DTO。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainRoundOutput:
    """单轮训练输出统计。

    这个 DTO 同时被 `runtime.loop` 的回调协议和 `sam3_integration.training`
    的真实训练结果使用，所以放在根级 `types.py`，避免 SAM3 集成层反向依赖
    runtime 调度层。
    """

    adapter_path: Path
    train_image_count: int
    train_instance_count: int
    train_seconds: float
    train_steps: int
