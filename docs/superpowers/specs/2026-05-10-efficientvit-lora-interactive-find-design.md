# EfficientViT LoRA Interactive Find Design

## Goal

Validate whether EfficientSAM3 with the EfficientViT student backbone can support a fast few-shot interactive single-class multi-instance detection workflow.

The offline experiment simulates the intended UI loop:

1. The user provides a folder of images.
2. The user selects one image containing the target objects.
3. The user annotates all target instances in that image with OBB boxes.
4. EfficientSAM3 learns the target appearance by updating LoRA parameters.
5. EfficientSAM3 finds same-class instances across the dataset.
6. If any image fails, the user corrects one failed image.
7. The corrected image and all previous annotated images are used for the next LoRA update.
8. The loop stops when all instances are detected correctly or `max_rounds` is reached.

In offline validation, dataset ground truth replaces UI annotation.

## Scope

This design focuses on EfficientSAM3's native interactive find / grounding path. It does not use sliding-window prompt generation, and it does not use the separate `fewshot_adapter` implementation.

The task is single-class, multi-instance detection:

- Each dataset has one target category such as `Sample` or `obj`.
- Each image can contain zero, one, or many instances.
- Success means every GT instance has a matched prediction and there are no extra predictions.

## Core Model Route

Use the EfficientSAM3 image model with the EfficientViT student backbone:

```python
build_efficientsam3_image_model(
    backbone_type="efficientvit",
    model_name="b0",
    enable_segmentation=True,
    eval_mode=False,
)
```

Source paths:

- `build_efficientsam3_image_model`: `sam3/sam3/model_builder.py`
- `_create_student_vision_backbone`: `sam3/sam3/model_builder.py`
- `ImageStudentEncoder`: `sam3/sam3/model_builder.py`
- `Sam3Image`: `sam3/sam3/model/sam3_image.py`

The intended forward path is:

```text
BatchedDatapoint
-> Sam3Image.forward()
-> Sam3Image.forward_grounding()
-> _encode_prompt()
-> _run_encoder()
-> _run_decoder()
-> _run_segmentation_heads()
-> pred_logits / pred_boxes / pred_masks
```

Source paths:

- `BatchedDatapoint`, `FindStage`, `BatchedFindTarget`: `sam3/sam3/model/data_misc.py`
- `Sam3Image.forward`: `sam3/sam3/model/sam3_image.py`
- `Sam3Image.forward_grounding`: `sam3/sam3/model/sam3_image.py`
- `Sam3Image._encode_prompt`: `sam3/sam3/model/sam3_image.py`
- `Sam3Image._run_encoder`: `sam3/sam3/model/sam3_image.py`
- `Sam3Image._run_decoder`: `sam3/sam3/model/sam3_image.py`
- `Sam3Image._run_segmentation_heads`: `sam3/sam3/model/sam3_image.py`

## Why This Is Not SAM-Style Sliding Prompt Detection

SAM-style segmentation uses:

```text
image_encoder -> prompt_encoder(box) -> mask_decoder -> mask
```

That route segments a prompted region. It does not natively enumerate same-class objects across the image.

This project should instead use EfficientSAM3's native find / grounding flow. In this flow, user OBB annotations become geometric prompts that describe the target. The model then predicts same-class instances across the image via its detector / grounding decoder.

Reference paths for the SAM-style route, useful only as background:

- `PromptEncoder`: `sam3/sam3/model/student_sam/modeling/prompt_encoder.py`
- `MaskDecoder`: `sam3/sam3/model/student_sam/modeling/mask_decoder.py`
- `TwoWayTransformer`: `sam3/sam3/model/student_sam/modeling/transformer.py`

These are not the primary route for this design.

## EfficientViT LoRA Strategy

The primary LoRA target is the EfficientViT student backbone, because the final requirement is a lightweight edge-deployable model.

EfficientViT attention blocks use `LiteMLA`, where QKV and output projection are convolutional:

```text
LiteMLA.qkv.conv
LiteMLA.proj.conv
```

Source paths:

- `EfficientViTBackbone`: `sam3/sam3/backbones/efficientvit/efficientvit/backbone.py`
- `EfficientViTBlock`: `sam3/sam3/backbones/efficientvit/nn/ops.py`
- `LiteMLA`: `sam3/sam3/backbones/efficientvit/nn/ops.py`
- `ConvLayer`: `sam3/sam3/backbones/efficientvit/nn/ops.py`

LoRA should be injected by replacing selected `nn.Conv2d` modules inside `LiteMLA.qkv` and `LiteMLA.proj` with a Conv-LoRA wrapper. The wrapper should freeze the base convolution and train only low-rank adapter parameters.

If backbone-only LoRA is not expressive enough, add a second trainable group in the EfficientSAM3 decoder:

- `linear1`
- `linear2`
- `out_proj`

Relevant source paths:

- Transformer encoder construction: `sam3/sam3/model_builder.py`
- Transformer decoder construction: `sam3/sam3/model_builder.py`
- Encoder layer definitions: `sam3/sam3/model/encoder.py`
- Decoder layer definitions: `sam3/sam3/model/decoder.py`

## Dataset Input Flow

The batch entry point is a plain text file. Each non-empty line is one dataset directory:

```text
/home/data/public/datasets/fewshot_test_20260429/24q4_machinery_circle
/home/data/public/datasets/fewshot_test_20260429/12356_工件定位
/home/data/public/datasets/fewshot_test_20260429/喷嘴有无
```

For each dataset directory, load annotations from `DetectTrainData.txt` by default. `DetectTrainData_sample5.txt` can be supported as an explicit override for quick smoke tests.

Annotation format:

```text
Version 1.0.0
20230922101406.jpg.bmp:6 R:4 604 423 504 362 671 86 772 148 "Sample" ...
90008300_c1s1_06_01.jpg.bmp:1 R:4 601 299 551 298 552 248 602 250 "obj"
```

Parsing rules:

- Skip `Version ...` lines.
- Treat the substring before the first `:` as the image name.
- Treat the integer after `:` as the declared instance count.
- Parse each `R:4 x1 y1 x2 y2 x3 y3 x4 y4 "label"` group as one OBB polygon.
- Preserve the dataset's original label string, but evaluate as single-class within that dataset.
- Report count mismatches instead of silently ignoring them.

Image path resolution must be tolerant:

- First try `dataset_dir / image_name`.
- Support compound suffixes such as `.jpg.bmp` and `.bmp.bmp`.
- If the exact path is missing, perform case-insensitive lookup within the dataset directory.
- If still missing, match by full filename stem only when the match is unique.
- Record missing or ambiguous images in a data validation report.

## Annotation Conversion

Each OBB polygon from `DetectTrainData.txt` should produce three internal targets:

```text
polygon: four source points
aabb: horizontal XYXY box enclosing the polygon
mask: rasterized polygon mask
```

Training input uses:

- AABB as `FindStage.input_boxes`.
- AABB or converted normalized box as `BatchedFindTarget.boxes`.
- Rasterized polygon mask as `BatchedFindTarget.segments` when mask loss is enabled.

Source paths for expected training structures:

- `FindStage`: `sam3/sam3/model/data_misc.py`
- `BatchedFindTarget`: `sam3/sam3/model/data_misc.py`
- `BatchedDatapoint`: `sam3/sam3/model/data_misc.py`

OBB evaluation should use polygon-derived OBB and rotated IoU. If a prediction only has a mask, derive OBB from the largest connected component of the thresholded mask.

## Iterative Learning Loop

For each dataset:

1. Build a full annotation index from `DetectTrainData.txt`.
2. Select the initial training image.
3. Add all GT instances from that image to the annotated set.
4. Build an EfficientSAM3 training batch from the annotated set.
5. Train only LoRA parameters for up to the per-round time budget.
6. Evaluate all images in the dataset.
7. Match predictions to GT by OBB IoU.
8. If precision and recall are both `1.0`, stop.
9. Otherwise select one failed image.
10. Add all GT instances from the selected image to the annotated set.
11. Repeat until success or `max_rounds`.

The selected failed image simulates a user correction in the UI.

## Training Forward

Construct a `BatchedDatapoint` that contains:

- `img_batch`: `(B_img, 3, H, W)`
- `find_text_batch`: a one-element target label list or a generic target phrase
- `find_inputs`: one or more `FindStage` records
- `find_targets`: one or more `BatchedFindTarget` records
- `find_metadatas`: metadata needed by the model path

Source paths:

- `BatchedDatapoint`: `sam3/sam3/model/data_misc.py`
- `FindStage`: `sam3/sam3/model/data_misc.py`
- `BatchedFindTarget`: `sam3/sam3/model/data_misc.py`
- `BatchedInferenceMetadata`: `sam3/sam3/model/data_misc.py`

For each annotated image, geometric prompts should include the user's OBB annotations converted to AABB boxes. The model should then use `Sam3Image.forward()` so that matching and training outputs remain aligned with the native EfficientSAM3 loss path.

## Loss Strategy

Use native EfficientSAM3 losses first:

- Box loss for detection alignment.
- Classification / presence loss for instance confidence.
- Mask loss for OBB-quality segmentation when `enable_segmentation=True`.

Source paths:

- `Sam3LossWrapper`: `sam3/sam3/train/loss/sam3_loss.py`
- `Boxes`: `sam3/sam3/train/loss/loss_fns.py`
- `IABCEMdetr`: `sam3/sam3/train/loss/loss_fns.py`
- `Masks`: `sam3/sam3/train/loss/loss_fns.py`
- `sigmoid_focal_loss`: `sam3/sam3/train/loss/loss_fns.py`
- `dice_loss`: `sam3/sam3/train/loss/loss_fns.py`
- `BinaryHungarianMatcherV2`: `sam3/sam3/train/matcher.py`
- `BinaryOneToManyMatcher`: `sam3/sam3/train/matcher.py`

The first experiments should keep the trainable parameter count small:

- Train Conv-LoRA parameters in EfficientViT.
- Freeze all base model weights.
- Use AMP on GPU.
- Cap each round by wall-clock time and/or step count.

## Evaluation

Run inference over every image in the dataset after each training round.

Prediction outputs to consume:

- `pred_logits`
- `pred_boxes`
- `pred_boxes_xyxy`
- `pred_masks` when segmentation is enabled

Source path:

- `Sam3Image.forward_grounding` output path: `sam3/sam3/model/sam3_image.py`

Postprocessing:

1. Convert `pred_logits` to scores.
2. Filter by score threshold.
3. Convert boxes to original image coordinates.
4. If masks exist, threshold them and fit OBB from the largest connected component.
5. Otherwise use the predicted box as an angle-zero fallback.
6. Run same-class NMS.
7. Match predictions to GT with OBB IoU.

Success condition:

```text
precision == 1.0
recall == 1.0
false_positive_count == 0
false_negative_count == 0
localization_error_count == 0
```

## Error Queue

Each round produces an ordered error queue:

- `false_negative`: a GT instance has no matching prediction.
- `localization_error`: a prediction overlaps a GT but does not reach the OBB IoU threshold.
- `false_positive`: a prediction does not match any GT.
- `low_confidence_true_positive`: optional, a correct match has low score.

Selection priority:

```text
false_negative > localization_error > false_positive > low_confidence_true_positive
```

When an image is selected, add every GT instance from that image to the next round's annotated set. This simulates a user correcting the whole failed image.

## Metrics And Artifacts

For each dataset and round, record:

- Training image count.
- Training instance count.
- Trainable parameter count.
- Training time in seconds.
- Precision, recall, F1.
- Mean matched OBB IoU.
- False positive count.
- False negative count.
- Localization error count.
- Selected next image.
- Adapter checkpoint path.

For the batch run, record:

- Number of datasets processed.
- Number of datasets reaching 100%.
- Average rounds to 100%.
- Average per-round training time.
- Failure reasons for datasets that do not converge.

## Performance Targets

Initial GPU validation targets:

- Per-round LoRA update time: less than 60 seconds.
- Successful datasets should report the number of rounds needed to reach 100%.
- Adapter size should remain small enough for per-task swapping.

NPU deployment constraints:

- Keep the base EfficientViT model unchanged.
- Keep task-specific state in LoRA adapter weights.
- Avoid introducing operations that are hard to export unless they are used only in offline training.
- Treat postprocessing as a separable deployment component.

## Open Design Decisions

These should be decided during implementation planning:

- Whether to reset LoRA from the base checkpoint each round or continue from the previous round.
- Which exact EfficientViT stages receive Conv-LoRA.
- Whether mask loss is enabled from the first experiment or added after box-only smoke tests.
- Whether selected false-positive images without GT should become hard negatives.
- The default OBB IoU and score thresholds.

## Spec Self-Review

- No dependency on `fewshot_adapter`.
- No sliding-window prompt generation in the main route.
- Uses EfficientSAM3 native find / grounding flow.
- Uses EfficientViT as the lightweight student backbone.
- Includes source paths for the key model, data, loss, and LoRA-relevant modules.
- Handles compound image suffixes such as `.jpg.bmp` and `.bmp.bmp`.
