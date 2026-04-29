"""数据层：统一管理标注模型、DataTrain 解析、JSON IO 和训练集采样。"""

from .datatrain import (
    DataTrainDataset,
    build_image_map,
    load_datatrain,
    parse_datatrain_line,
    save_image_map,
)
from .json_io import (
    AnnotationJsonIO,
    load_annotations,
    load_error_queue,
    load_predictions,
    save_annotations,
    save_error_queue,
    save_predictions,
)
from .models import (
    HBB,
    OBB,
    Annotation,
    Prediction,
    hbb_to_polygon,
    normalize_annotation,
    obb_to_polygon,
    polygon_to_hbb,
)
from .sampling import (
    InitialTrainSelector,
    TrainSetUpdater,
    add_selected_errors_to_train_set,
    create_initial_train_set,
)
from .sam3_batch import (
    NativeImageBatch,
    NativeSam3Batch,
    Sam3BatchBuilder,
    annotation_to_target_box,
    build_sam3_training_batch,
    group_annotations_by_image,
    hbb_to_cxcywh_norm,
    load_image_batch,
)

__all__ = [
    "Annotation",
    "AnnotationJsonIO",
    "DataTrainDataset",
    "HBB",
    "InitialTrainSelector",
    "OBB",
    "NativeImageBatch",
    "NativeSam3Batch",
    "Prediction",
    "Sam3BatchBuilder",
    "TrainSetUpdater",
    "add_selected_errors_to_train_set",
    "annotation_to_target_box",
    "build_image_map",
    "build_sam3_training_batch",
    "create_initial_train_set",
    "group_annotations_by_image",
    "hbb_to_polygon",
    "hbb_to_cxcywh_norm",
    "load_annotations",
    "load_datatrain",
    "load_error_queue",
    "load_image_batch",
    "load_predictions",
    "normalize_annotation",
    "obb_to_polygon",
    "parse_datatrain_line",
    "polygon_to_hbb",
    "save_annotations",
    "save_error_queue",
    "save_image_map",
    "save_predictions",
]
