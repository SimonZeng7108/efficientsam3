"""图片路径解析与大小写/复合后缀容错。

工业数据里常见 `.jpg.bmp`、`.bmp.bmp` 这样的复合后缀，也常见标注文件
里的大小写和磁盘真实文件名不一致。这个模块把路径查找单独封装起来，
避免数据解析、训练和评估模块里到处写重复的路径容错逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class ResolvedImage:
    """解析成功后的图片信息。"""

    path: Path
    width: int
    height: int
    match_kind: str


class ImageResolutionError(FileNotFoundError):
    """图片解析失败或歧义时抛出的异常。"""

    def __init__(self, kind: str, image_name: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.image_name = image_name
        self.message = message


def resolve_image_path(dataset_dir: Path, image_name: str) -> ResolvedImage:
    """在子数据集目录中解析图片路径。

    查找顺序：
    1. 精确文件名匹配。
    2. 大小写不敏感匹配。
    3. 去掉所有复合后缀后的 stem 唯一匹配。

    第 3 步只在唯一时生效；如果有多个候选，必须报歧义，不能偷偷选一个。
    """

    files = [p for p in dataset_dir.iterdir() if p.is_file()]
    exact_matches = [p for p in files if p.name == image_name]
    if len(exact_matches) == 1:
        return _resolved(exact_matches[0], "exact")
    if len(exact_matches) > 1:
        raise _ambiguous(image_name, exact_matches)

    image_name_lower = image_name.lower()
    case_matches = [p for p in files if p.name.lower() == image_name_lower]
    if len(case_matches) == 1:
        return _resolved(case_matches[0], "case_insensitive")
    if len(case_matches) > 1:
        raise _ambiguous(image_name, case_matches)

    requested_stem = _compound_stem(Path(image_name).name).lower()
    stem_matches = [p for p in files if _compound_stem(p.name).lower() == requested_stem]
    if len(stem_matches) == 1:
        return _resolved(stem_matches[0], "stem")
    if len(stem_matches) > 1:
        raise _ambiguous(image_name, stem_matches)

    raise ImageResolutionError(
        "missing",
        image_name,
        f"无法在目录 '{dataset_dir}' 下找到图片 '{image_name}'。",
    )


def _resolved(path: Path, match_kind: str) -> ResolvedImage:
    """读取图片尺寸并返回解析结果。"""

    with Image.open(path) as image:
        width, height = image.size
    return ResolvedImage(path=path, width=width, height=height, match_kind=match_kind)


def _compound_stem(name: str) -> str:
    """去掉所有后缀后的文件名主体。

    `Path.stem` 只会去掉最后一个后缀，例如 `a.jpg.bmp` 会得到 `a.jpg`；
    这里需要把 `.jpg.bmp` 整体视为复合后缀，所以使用 `suffixes` 全部移除。
    """

    path = Path(name)
    stem = path.name
    suffixes = "".join(path.suffixes)
    if suffixes:
        stem = stem[: -len(suffixes)]
    return stem


def _ambiguous(image_name: str, matches: list[Path]) -> ImageResolutionError:
    names = ", ".join(sorted(p.name for p in matches))
    return ImageResolutionError(
        "ambiguous",
        image_name,
        f"图片 '{image_name}' 存在多个候选，无法唯一确定：{names}。",
    )
