import glob
import hashlib
import io
import json
import logging
import os
import random
import sys
import traceback

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import pycocotools.mask as mask_utils
import torch

from PIL import Image as PILImage
from PIL.Image import DecompressionBombError

from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image,
    InferenceMetadata,
    Object,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class _ObjectRecord:
    bbox_xyxy: List[float]
    text: str
    area: float
    segmentation: Any
    category_id: int
    is_crowd: bool
    object_id: int
    source: str


@dataclass
class _SampleRecord:
    source_name: str
    image_path: Optional[str]
    image_payload: Any
    image_root: Optional[str]
    width: int
    height: int
    image_id: int
    file_name: str
    objects: List[_ObjectRecord]


def _choose_deterministic_index(num_items: int, key: str) -> int:
    if num_items <= 1:
        return 0
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_items


def _sample_records(
    records: List[_SampleRecord],
    max_samples: Optional[int],
    seed_key: str,
) -> List[_SampleRecord]:
    if max_samples is None or max_samples < 0 or len(records) <= max_samples:
        return records
    rng = random.Random(seed_key)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    indices = sorted(indices[:max_samples])
    return [records[index] for index in indices]


def _build_bbox_polygon(bbox_xyxy: List[float]) -> List[List[float]]:
    x1, y1, x2, y2 = bbox_xyxy
    return [[x1, y1, x2, y1, x2, y2, x1, y2]]


def _normalize_segmentation(
    segmentation: Any,
    height: int,
    width: int,
    bbox_xyxy: List[float],
) -> Any:
    if segmentation is None or segmentation == []:
        segmentation = _build_bbox_polygon(bbox_xyxy)

    if isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        if isinstance(counts, list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
            return mask_utils.merge(rle) if isinstance(rle, list) else rle
        return segmentation

    if isinstance(segmentation, list):
        if len(segmentation) == 0:
            segmentation = _build_bbox_polygon(bbox_xyxy)
        elif isinstance(segmentation[0], (int, float)):
            segmentation = [segmentation]
        rle = mask_utils.frPyObjects(segmentation, height, width)
        return mask_utils.merge(rle) if isinstance(rle, list) else rle

    return segmentation


def _is_valid_bbox(bbox: Any) -> bool:
    if bbox is None or len(bbox) != 4:
        return False
    for value in bbox:
        if value is None:
            return False
    return True


def _has_valid_segmentation(segmentation: Any) -> bool:
    if segmentation is None:
        return False
    if isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        size = segmentation.get("size")
        return counts is not None and size is not None
    if isinstance(segmentation, list):
        if len(segmentation) == 0:
            return False
        if isinstance(segmentation[0], (int, float)):
            return len(segmentation) >= 6
        return any(poly is not None and len(poly) >= 6 for poly in segmentation)
    return True


def _resolve_image_path(
    image_info: Dict[str, Any],
    ann_file: str,
    image_root: Optional[str],
    image_path_mode: str,
) -> str:
    file_name = image_info.get("file_name", "")
    if image_path_mode == "image_root_basename":
        return os.path.join(image_root, os.path.basename(file_name))
    if image_path_mode == "from_ann_parent":
        return os.path.join(os.path.dirname(ann_file), file_name)
    if image_path_mode == "from_coco_url_under_root":
        coco_url = image_info.get("coco_url", "")
        rel_path = "/".join(coco_url.split("/")[-2:]) if coco_url else file_name
        return os.path.join(image_root, rel_path)
    if image_root is None:
        return file_name
    return os.path.join(image_root, file_name)


class _JsonCategorySource:
    def __init__(
        self,
        name: str,
        ann_file: str,
        image_root: Optional[str],
        image_path_mode: str,
        use_synonyms: bool,
        max_samples: Optional[int],
        require_masks: bool,
    ) -> None:
        self.name = name
        self.ann_file = ann_file
        self.image_root = image_root
        self.image_path_mode = image_path_mode
        self.use_synonyms = use_synonyms
        self.max_samples = max_samples
        self.require_masks = require_masks

    def build_records(self) -> List[_SampleRecord]:
        with open(self.ann_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        categories = {}
        for category in payload.get("categories", []):
            texts = [category.get("name", "")]
            if self.use_synonyms:
                texts.extend(category.get("synonyms", []))
            texts = [text.strip() for text in texts if text and text.strip()]
            if not texts:
                texts = [f"category_{category['id']}"]
            categories[category["id"]] = texts

        images_by_id = {image["id"]: image for image in payload.get("images", [])}
        anns_by_image = defaultdict(list)
        for annotation in payload.get("annotations", []):
            anns_by_image[annotation["image_id"]].append(annotation)

        records: List[_SampleRecord] = []
        for image_id, annotations in anns_by_image.items():
            image_info = images_by_id.get(image_id)
            if image_info is None:
                continue

            width = int(image_info.get("width", 0))
            height = int(image_info.get("height", 0))
            objects: List[_ObjectRecord] = []
            for annotation in annotations:
                bbox = annotation.get("bbox")
                if not _is_valid_bbox(bbox):
                    continue
                x, y, w, h = [float(value) for value in bbox]
                if w <= 0 or h <= 0:
                    continue
                bbox_xyxy = [x, y, x + w, y + h]
                text_candidates = categories.get(annotation.get("category_id"), ["object"])
                category_key = f"{self.name}:{image_id}:{annotation.get('id', len(objects))}"
                text = text_candidates[
                    _choose_deterministic_index(len(text_candidates), category_key)
                ]
                segmentation = annotation.get("segmentation")
                if self.require_masks and not _has_valid_segmentation(segmentation):
                    continue
                objects.append(
                    _ObjectRecord(
                        bbox_xyxy=bbox_xyxy,
                        text=text,
                        area=float(annotation.get("area") or (w * h)),
                        segmentation=segmentation,
                        category_id=int(annotation.get("category_id", -1)),
                        is_crowd=bool(annotation.get("iscrowd", 0)),
                        object_id=len(objects),
                        source=self.name,
                    )
                )

            if not objects:
                continue

            image_path = _resolve_image_path(
                image_info=image_info,
                ann_file=self.ann_file,
                image_root=self.image_root,
                image_path_mode=self.image_path_mode,
            )
            records.append(
                _SampleRecord(
                    source_name=self.name,
                    image_path=image_path,
                    image_payload=None,
                    image_root=self.image_root,
                    width=width,
                    height=height,
                    image_id=int(image_id),
                    file_name=image_info.get("file_name", os.path.basename(image_path)),
                    objects=objects,
                )
            )

        return _sample_records(records, self.max_samples, f"{self.name}:{self.ann_file}")


class _RefCocoParquetSource:
    def __init__(
        self,
        name: str,
        parquet_paths: Iterable[str],
        max_samples: Optional[int],
        require_masks: bool,
    ) -> None:
        self.name = name
        self.parquet_paths = list(parquet_paths)
        self.max_samples = max_samples
        self.require_masks = require_masks

    def build_records(self) -> List[_SampleRecord]:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            LOGGER.warning("Skipping %s because pyarrow is unavailable.", self.name)
            return []

        records: List[_SampleRecord] = []
        for parquet_path in self.parquet_paths:
            try:
                schema = pq.read_schema(parquet_path)
                schema_names = set(schema.names)
            except Exception as e:
                LOGGER.warning("Skipping %s (%s): cannot read schema: %s", self.name, parquet_path, e)
                continue
            # Detect raw RefCOCO format (ref_id / sentences / image_path) vs
            # processed format (question_id / image / question).
            if "question_id" not in schema_names:
                LOGGER.warning(
                    "Skipping %s (%s): unsupported schema (missing question_id). "
                    "Expected the processed RefCOCO parquet format with embedded images.",
                    self.name, parquet_path,
                )
                continue
            try:
                table = pq.read_table(
                    parquet_path,
                    columns=[
                        "question_id",
                        "image",
                        "question",
                        "segmentation",
                        "bbox",
                        "iscrowd",
                        "file_name",
                    ],
                )
            except Exception as e:
                LOGGER.warning("Skipping %s (%s): failed to read columns: %s", self.name, parquet_path, e)
                continue
            for row in table.to_pylist():
                bbox = row.get("bbox")
                question = (row.get("question") or "").strip()
                if not _is_valid_bbox(bbox) or not question:
                    continue
                x, y, w, h = [float(value) for value in bbox]
                if w <= 0 or h <= 0:
                    continue
                segmentation = row.get("segmentation")
                if self.require_masks and not _has_valid_segmentation(segmentation):
                    continue
                bbox_xyxy = [x, y, x + w, y + h]
                records.append(
                    _SampleRecord(
                        source_name=self.name,
                        image_path=None,
                        image_payload=row.get("image"),
                        image_root=os.path.dirname(parquet_path),
                        width=0,
                        height=0,
                        image_id=len(records),
                        file_name=row.get("file_name", ""),
                        objects=[
                            _ObjectRecord(
                                bbox_xyxy=bbox_xyxy,
                                text=question,
                                area=w * h,
                                segmentation=segmentation,
                                category_id=-1,
                                is_crowd=bool(row.get("iscrowd", 0)),
                                object_id=0,
                                source=self.name,
                            )
                        ],
                    )
                )

        return _sample_records(records, self.max_samples, f"{self.name}:refcoco")


def _instantiate_sources(sources: List[Dict[str, Any]]) -> List[_SampleRecord]:
    records: List[_SampleRecord] = []
    for source_cfg in sources:
        kind = source_cfg["kind"]
        name = source_cfg["name"]
        source_records: List[_SampleRecord] = []

        if kind == "json_category":
            source_records = _JsonCategorySource(
                name=name,
                ann_file=source_cfg["ann_file"],
                image_root=source_cfg.get("image_root"),
                image_path_mode=source_cfg.get("image_path_mode", "image_root_join"),
                use_synonyms=bool(source_cfg.get("use_synonyms", False)),
                max_samples=source_cfg.get("max_samples"),
                require_masks=bool(source_cfg.get("require_masks", False)),
            ).build_records()
        elif kind == "json_category_glob":
            ann_globs = source_cfg.get("ann_globs")
            if ann_globs is None:
                ann_glob = source_cfg.get("ann_glob")
                ann_globs = [ann_glob] if ann_glob is not None else []
            ann_files: List[str] = []
            for pattern in ann_globs:
                ann_files.extend(glob.glob(pattern, recursive=True))
            ann_files = sorted(set(ann_files))
            if not ann_files:
                LOGGER.warning("Source %s matched no annotation files.", name)
            for ann_file in ann_files:
                source_records.extend(
                    _JsonCategorySource(
                        name=name,
                        ann_file=ann_file,
                        image_root=source_cfg.get("image_root"),
                        image_path_mode=source_cfg.get("image_path_mode", "from_ann_parent"),
                        use_synonyms=bool(source_cfg.get("use_synonyms", False)),
                        max_samples=None,
                        require_masks=bool(source_cfg.get("require_masks", False)),
                    ).build_records()
                )
            source_records = _sample_records(
                source_records,
                source_cfg.get("max_samples"),
                f"{name}:glob",
            )
        elif kind == "refcoco_parquet_glob":
            parquet_globs = source_cfg.get("parquet_globs")
            if parquet_globs is None:
                parquet_glob = source_cfg.get("parquet_glob")
                parquet_globs = [parquet_glob] if parquet_glob is not None else []
            parquet_paths: List[str] = []
            for pattern in parquet_globs:
                parquet_paths.extend(glob.glob(pattern, recursive=True))
            parquet_paths = sorted(set(parquet_paths))
            if not parquet_paths:
                LOGGER.warning("Source %s matched no parquet files.", name)
            source_records = _RefCocoParquetSource(
                name=name,
                parquet_paths=parquet_paths,
                max_samples=source_cfg.get("max_samples"),
                require_masks=bool(source_cfg.get("require_masks", False)),
            ).build_records()
        else:
            raise ValueError(f"Unsupported Stage3 source kind: {kind}")

        LOGGER.info("Stage3 source %s contributed %d samples", name, len(source_records))
        records.extend(source_records)

    if not records:
        raise RuntimeError("Stage3 mixed dataset found no usable samples.")
    return records


class Stage3MixedTextMaskDataset:
    def __init__(
        self,
        transforms,
        sources: List[Dict[str, Any]],
        max_ann_per_img: int,
        multiplier: int,
        training: bool,
        load_segmentation: bool = False,
        max_train_queries: int = 50000,
        max_val_queries: int = 50000,
    ) -> None:
        self._transforms = transforms
        self.training = training
        self.load_segmentation = load_segmentation
        self.max_ann_per_img = max_ann_per_img
        self.max_train_queries = max_train_queries
        self.max_val_queries = max_val_queries
        self.multiplier = max(1, int(multiplier))
        self.curr_epoch = 0
        self._MAX_RETRIES = 100

        self.records = _instantiate_sources(sources)
        self.repeat_factors = torch.ones(len(self.records), dtype=torch.float32)
        self.repeat_factors *= self.multiplier
        LOGGER.info("Stage3 mixed dataset built with %d base samples", len(self.records))

    def __len__(self) -> int:
        return len(self.records) * self.multiplier

    def set_epoch(self, epoch: int) -> None:
        self.curr_epoch = epoch

    def _load_pil_image(self, record: _SampleRecord) -> PILImage.Image:
        if record.image_path is not None:
            return PILImage.open(record.image_path).convert("RGB")

        payload = record.image_payload
        if isinstance(payload, dict):
            image_bytes = payload.get("bytes")
            image_path = payload.get("path")
            if image_bytes is not None:
                return PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
            if image_path:
                if os.path.isabs(image_path) and os.path.exists(image_path):
                    return PILImage.open(image_path).convert("RGB")
                candidate = os.path.join(record.image_root or "", image_path)
                if os.path.exists(candidate):
                    return PILImage.open(candidate).convert("RGB")

        if isinstance(payload, (bytes, bytearray)):
            return PILImage.open(io.BytesIO(payload)).convert("RGB")

        if record.file_name:
            candidate = os.path.join(record.image_root or "", record.file_name)
            if os.path.exists(candidate):
                return PILImage.open(candidate).convert("RGB")

        raise FileNotFoundError(f"Could not resolve image for Stage3 sample from {record.source_name}")

    def _build_datapoint(self, record: _SampleRecord) -> Datapoint:
        pil_image = self._load_pil_image(record)
        width, height = pil_image.size

        objects: List[Object] = []
        find_queries: List[FindQueryLoaded] = []
        for object_index, obj in enumerate(record.objects):
            segmentation = None
            if self.load_segmentation:
                segmentation = _normalize_segmentation(
                    segmentation=obj.segmentation,
                    height=height,
                    width=width,
                    bbox_xyxy=obj.bbox_xyxy,
                )

            bbox = torch.tensor(obj.bbox_xyxy, dtype=torch.float32)
            objects.append(
                Object(
                    bbox=bbox,
                    area=float(obj.area),
                    object_id=obj.object_id,
                    frame_index=0,
                    segment=segmentation,
                    is_crowd=obj.is_crowd,
                    source=obj.source,
                )
            )
            find_queries.append(
                FindQueryLoaded(
                    query_text=obj.text,
                    image_id=0,
                    object_ids_output=[object_index],
                    is_exhaustive=True,
                    query_processing_order=0,
                    input_bbox=bbox.view(1, 4),
                    input_bbox_label=torch.ones(1, dtype=torch.long),
                    input_points=None,
                    semantic_target=None,
                    is_pixel_exhaustive=True,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=-1 if self.training else record.image_id,
                        original_image_id=-1 if self.training else record.image_id,
                        original_category_id=obj.category_id,
                        original_size=(height, width),
                        object_id=obj.object_id,
                        frame_index=0,
                    ),
                )
            )

        return Datapoint(
            find_queries=find_queries,
            images=[Image(data=pil_image, objects=objects, size=(height, width))],
            raw_images=[pil_image],
        )

    def __getitem__(self, idx: int) -> Datapoint:
        if not self.records:
            raise RuntimeError("Stage3 mixed dataset has no samples.")

        base_idx = idx % len(self.records)
        for _ in range(self._MAX_RETRIES):
            try:
                datapoint = self._build_datapoint(self.records[base_idx])

                for query in datapoint.find_queries:
                    if len(query.object_ids_output) > self.max_ann_per_img:
                        raise DecompressionBombError(
                            f"Too many outputs ({len(query.object_ids_output)})"
                        )

                max_queries = self.max_train_queries if self.training else self.max_val_queries
                if len(datapoint.find_queries) > max_queries:
                    raise DecompressionBombError(
                        f"Too many find queries ({len(datapoint.find_queries)})"
                    )
                if len(datapoint.find_queries) == 0:
                    raise DecompressionBombError("No find queries")

                for transform in self._transforms:
                    datapoint = transform(datapoint, epoch=self.curr_epoch)
                return datapoint
            except (DecompressionBombError, FileNotFoundError, OSError, ValueError) as error:
                sys.stderr.write(f"ERROR: got loading error on datapoint {base_idx}\n")
                sys.stderr.write(f"Exception: {error}\n")
                sys.stderr.write(traceback.format_exc())
                base_idx = (base_idx + 1) % len(self.records)

        raise RuntimeError(f"Failed {self._MAX_RETRIES} times trying to load a Stage3 sample.")
