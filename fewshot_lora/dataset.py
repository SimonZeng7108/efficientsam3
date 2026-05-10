"""把 DetectTrainData 文本标注和真实图片合成训练/评估索引。

`datatrain.py` 只解析文本，不知道图片尺寸；但训练需要归一化框，mask
栅格化也需要宽高。因此本文件负责第二阶段准备：
1. 解析标注文件。
2. 容错解析图片路径和实际尺寸。
3. 把每个 OBB 实例转换成 AABB、归一化 cxcywh、polygon mask 和评估 OBB。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .datatrain import DataIssue, parse_detect_train_data
from .geometry import (
    AABB,
    OrientedBox,
    Polygon4,
    aabb_to_normalized_cxcywh,
    polygon_to_aabb,
    polygon_to_mask,
    polygon_to_obb,
)
from .images import ImageResolutionError, resolve_image_path


@dataclass(frozen=True)
class PreparedInstance:
    """准备好的单个目标实例。

    同一个实例同时保留多种表示，是为了训练和评估都能复用同一份索引。
    """

    polygon: Polygon4
    label: str
    aabb: AABB
    box_cxcywh: tuple[float, float, float, float]
    mask: np.ndarray
    obb: OrientedBox


@dataclass(frozen=True)
class PreparedImage:
    """准备好的一张图片及其全部实例。"""

    image_name: str
    path: Path
    width: int
    height: int
    instances: list[PreparedInstance]


@dataclass(frozen=True)
class PreparedDataset:
    """准备好的子数据集。"""

    dataset_dir: Path
    annotation_path: Path
    images: list[PreparedImage]
    issues: list[DataIssue]


def load_detect_dataset(
    dataset_dir: Path,
    annotation_filename: str = "DetectTrainData.txt",
) -> PreparedDataset:
    """加载一个子数据集目录。

    这个函数不会因为单张图片缺失就直接失败；缺失/歧义图片会写入 `issues`，
    其余可用图片继续参与实验。
    """

    annotation_path = dataset_dir / annotation_filename
    parsed = parse_detect_train_data(annotation_path)
    images: list[PreparedImage] = []
    issues = list(parsed.issues)

    for record in parsed.records:
        try:
            resolved = resolve_image_path(dataset_dir, record.image_name)
        except ImageResolutionError as exc:
            # 图片缺失或歧义属于数据问题，记录下来，便于 summary.json 中追踪。
            issues.append(
                DataIssue(
                    kind=f"image_{exc.kind}",
                    image_name=record.image_name,
                    line_number=record.line_number,
                    message=exc.message,
                )
            )
            continue

        instances = [
            _prepare_instance(instance.polygon, instance.label, resolved.width, resolved.height)
            for instance in record.instances
        ]
        images.append(
            PreparedImage(
                image_name=record.image_name,
                path=resolved.path,
                width=resolved.width,
                height=resolved.height,
                instances=instances,
            )
        )

    return PreparedDataset(
        dataset_dir=dataset_dir,
        annotation_path=annotation_path,
        images=images,
        issues=issues,
    )


def _prepare_instance(
    polygon: Polygon4,
    label: str,
    width: int,
    height: int,
) -> PreparedInstance:
    """把单个 polygon 标注转换成训练和评估都需要的格式。"""

    # EfficientSAM3 的 FindStage 和 BatchedFindTarget 都使用归一化 cx,cy,w,h；
    # OBB 只保留给评估匹配，训练监督仍走原生水平框 + polygon mask。
    aabb = polygon_to_aabb(polygon)
    box_cxcywh = aabb_to_normalized_cxcywh(aabb, width=width, height=height)
    mask = polygon_to_mask(polygon, width=width, height=height)
    obb = polygon_to_obb(polygon)
    return PreparedInstance(
        polygon=polygon,
        label=label,
        aabb=aabb,
        box_cxcywh=box_cxcywh,
        mask=mask,
        obb=obb,
    )
