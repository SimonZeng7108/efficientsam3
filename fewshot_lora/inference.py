"""评估阶段推理。

非常重要：评估未标注/测试图片时不能把 GT 框作为 prompt，否则会泄露目标位置。
因此这里构造空 `prompt_boxes=[]`，只依赖文本 prompt 和已经训练好的 LoRA 参数
完成同类目标查找。
"""

from __future__ import annotations

from .config import FewShotLoRAConfig
from .dataset import PreparedImage
from .metrics import ImageGroundTruth, evaluate_image
from .postprocess import PredictionArrays, postprocess_predictions
from .preprocess import load_image_tensor


def evaluate_images(model, images: list[PreparedImage], config: FewShotLoRAConfig):
    """对一个子数据集的全量图片进行评估。"""

    text_prompt = _dataset_text_prompt(images, config)
    return [
        evaluate_image(
            ground_truth=ImageGroundTruth(
                image_id=image.image_name,
                boxes=[instance.obb for instance in image.instances],
            ),
            prediction=predict_image(model, image, config, text_prompt=text_prompt),
            iou_threshold=config.evaluation.iou_threshold,
            localization_iou_threshold=config.evaluation.localization_iou_threshold,
        )
        for image in images
    ]


def predict_image(
    model,
    image: PreparedImage,
    config: FewShotLoRAConfig,
    text_prompt: str | None = None,
):
    """对单张图片做 text-only grounding 推理。"""

    import numpy as np
    import torch
    import torch.nn.functional as F

    from .sam3_batch import FindBatchSample, build_batched_datapoint

    device = torch.device(config.device)
    sample = FindBatchSample(
        image=load_image_tensor(image.path, config.image_size),
        image_id=image.image_name,
        original_size=(image.height, image.width),
        target_boxes=[],
        target_masks=[],
        # 评估阶段必须为空几何 prompt：不能把 GT 框作为测试图提示，否则会泄露目标位置。
        prompt_boxes=[],
        text=text_prompt or _prediction_text_prompt(image, config),
    )
    batch = build_batched_datapoint([sample], device=device)
    model.eval()
    with torch.inference_mode():
        outputs = model(batch)
        out = outputs[0]
        logits = out["pred_logits"][0, :, 0]
        scores = logits.sigmoid()
        if "presence_logit_dec" in out:
            scores = scores * out["presence_logit_dec"][0].sigmoid()
        boxes = out["pred_boxes"][0]
        masks = None
        if "pred_masks" in out:
            pred_masks = out["pred_masks"][0].unsqueeze(1)
            pred_masks = F.interpolate(
                pred_masks,
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1).sigmoid()
            masks = pred_masks.detach().cpu().numpy()

    arrays = PredictionArrays(
        scores=scores.detach().cpu().numpy().astype(np.float32),
        boxes_cxcywh=boxes.detach().cpu().numpy().astype(np.float32),
        masks=masks,
    )
    return postprocess_predictions(
        image_id=image.image_name,
        arrays=arrays,
        original_size=(image.height, image.width),
        score_threshold=config.evaluation.score_threshold,
        nms_iou_threshold=config.evaluation.nms_iou_threshold,
        mask_threshold=config.evaluation.mask_threshold,
    )


def _dataset_text_prompt(images: list[PreparedImage], config: FewShotLoRAConfig) -> str:
    """从子数据集正样本中提取单类别文本 prompt。

    单类别评估必须所有图片使用同一个 prompt。否则背景图没有 GT 时会退回
    generic `object`，导致正样本和负样本查询语义不一致。
    """

    for image in images:
        for instance in image.instances:
            if instance.label:
                return instance.label
    return config.evaluation.text_prompt


def _prediction_text_prompt(image: PreparedImage, config: FewShotLoRAConfig) -> str:
    """单张图独立推理时的兜底文本 prompt。"""

    for instance in image.instances:
        if instance.label:
            return instance.label
    return config.evaluation.text_prompt
