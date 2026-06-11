"""Smoke test for the SACap+SA-1B Stage 3 source. Avoids torch import."""
from __future__ import annotations

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# Stub the heavy/optional deps the module imports at top level. We only
# exercise `_SACapSA1BSource`, which doesn't actually call pycocotools or
# torch directly. The mask data is left as the raw RLE dict from the SA-1B
# JSON, so no decoding is needed for this smoke test.
import types  # noqa: E402

for stub_name in ("torch", "pycocotools", "pycocotools.mask"):
    if stub_name not in sys.modules:
        sys.modules[stub_name] = types.ModuleType(stub_name)

# `from sam3.train.data.sam3_image_dataset import ...` is needed by the
# module, but we only build records (no datapoints), so stub this too.
sam3_pkg = types.ModuleType("sam3")
sam3_train = types.ModuleType("sam3.train")
sam3_train_data = types.ModuleType("sam3.train.data")
sam3_image_dataset = types.ModuleType("sam3.train.data.sam3_image_dataset")
for cls_name in ("Datapoint", "FindQueryLoaded", "Image", "InferenceMetadata", "Object"):
    setattr(sam3_image_dataset, cls_name, type(cls_name, (), {}))
sys.modules["sam3"] = sam3_pkg
sys.modules["sam3.train"] = sam3_train
sys.modules["sam3.train.data"] = sam3_train_data
sys.modules["sam3.train.data.sam3_image_dataset"] = sam3_image_dataset

# Import the module file directly to bypass `stage3/__init__.py` (which
# imports torch).
_spec = importlib.util.spec_from_file_location(
    "_mixed_text_mask_dataset",
    os.path.join(HERE, "mixed_text_mask_dataset.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_SACapSA1BSource = _module._SACapSA1BSource

CFG = dict(
    name="sacap_sa1b_smoke",
    parquet_path=os.path.join(ROOT, "data/sa-1b-5p-sacap/anno.parquet"),
    image_root=os.path.join(ROOT, "data/SA-1B-5P/images"),
    annotation_root=os.path.join(ROOT, "data/SA-1B-5P/annotations"),
    max_samples=5,
    require_masks=True,
    use_prompt_boxes=False,
    val_holdout_frac=0.05,
    val_is_holdout=False,
    split_seed=123,
    max_segments_per_image=32,
)


def main() -> None:
    src = _SACapSA1BSource(**CFG)
    records = src.build_records()
    print(f"records: {len(records)}")
    for i, rec in enumerate(records):
        print(f"\n--- record {i}: {rec.file_name} ---")
        print(f"  image_path:     {rec.image_path}")
        print(f"  exists:         {os.path.exists(rec.image_path)}")
        print(f"  anno_path:      {rec.lazy_payload['anno_path']}")
        print(f"  anno_exists:    {os.path.exists(rec.lazy_payload['anno_path'])}")
        print(f"  segments:       {len(rec.lazy_payload['segments'])}")
        objects = rec.lazy_objects_fn(rec)
        print(f"  resolved objs:  {len(objects)}")
        for oi, obj in enumerate(objects[:3]):
            seg_repr = (
                list(obj.segmentation.keys())
                if isinstance(obj.segmentation, dict)
                else type(obj.segmentation).__name__
            )
            print(
                f"    obj{oi}: text={obj.text[:60]!r}, "
                f"bbox={obj.bbox_xyxy}, seg={seg_repr}"
            )
        print(f"  width/height:   {rec.width}x{rec.height}")


if __name__ == "__main__":
    main()
