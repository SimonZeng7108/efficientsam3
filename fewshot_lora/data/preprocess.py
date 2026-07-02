"""图片和 mask 预处理。

这里刻意只做最小必要预处理：
- 图片 resize 到 EfficientSAM3 默认方图尺寸。
- 按 SAM3 Processor 的 mean/std=0.5 做归一化。
- mask 用最近邻 resize，避免插值制造半透明边界。

实现上显式贴近 `sam3.model.sam3_image_processor.Sam3Processor`：
它同样使用固定方图 resize，并采用 mean=[0.5, 0.5, 0.5]、
std=[0.5, 0.5, 0.5] 的归一化。这里保留轻量函数形式，是为了训练 batch
构造时可以直接得到 CHW tensor，而不引入交互式 processor 的 state 管理。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image_tensor(path: Path, image_size: int):
    """读取 RGB 图片并转换为 CHW float tensor。"""

    # 延迟导入 torch，方便没有 torch 的本地环境运行解析/几何/CLI 测试。
    import torch

    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    # 与 Sam3Processor 保持一致：Resize 到固定方图后使用 mean/std = 0.5。
    array = (array - 0.5) / 0.5
    chw = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(chw).float()


def resize_mask(mask: np.ndarray, image_size: int):
    """把原图尺寸 bool mask 缩放到模型输入尺寸。"""

    import torch

    pil_mask = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = pil_mask.resize((image_size, image_size), Image.NEAREST)
    return torch.from_numpy(np.asarray(resized) > 0)
