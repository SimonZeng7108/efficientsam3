# EfficientViT LoRA 交互式同类目标查找设计

## 目标

验证 EfficientSAM3 在使用 EfficientViT 学生主干时，是否能够支持快速少样本交互式单类别多实例检测流程。

离线实验用于模拟真实交互界面中的流程：

1. 用户输入一批待检测图片。
2. 用户随机选择一张包含目标的图片。
3. 用户用 OBB 框标注该图中的所有目标实例。
4. EfficientSAM3 通过更新 LoRA 参数快速学习该目标类别特征。
5. EfficientSAM3 在整批图片中查找同类目标实例。
6. 如果存在漏检、误检或定位错误，用户修正其中一张失败图片。
7. 修正图片和历史标注图片一起进入下一轮 LoRA 微调。
8. 当所有实例均检测正确，或达到 `max_rounds` 后停止。

离线验证时，用数据集真值标注代替用户交互标注。

## 范围

本设计聚焦 EfficientSAM3 原生 interactive find / grounding 流程。主路线不使用滑窗候选 prompt，也不使用或参考单独的 `fewshot_adapter` 实现。

任务类型是单类别、多实例检测：

- 每个子数据集只有一个目标类别，例如 `Sample` 或 `obj`。
- 一张图片中可能有 0 个、1 个或多个目标实例。
- 成功条件是所有 GT 实例都有匹配预测，并且没有多余预测。

## 核心模型路线

使用带 EfficientViT 学生主干的 EfficientSAM3 image model：

```python
build_efficientsam3_image_model(
    backbone_type="efficientvit",
    model_name="b0",
    enable_segmentation=True,
    eval_mode=False,
)
```

源码定位：

- `build_efficientsam3_image_model`：`sam3/sam3/model_builder.py`
- `_create_student_vision_backbone`：`sam3/sam3/model_builder.py`
- `ImageStudentEncoder`：`sam3/sam3/model_builder.py`
- `Sam3Image`：`sam3/sam3/model/sam3_image.py`

计划使用的原生前向路径：

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

源码定位：

- `BatchedDatapoint`、`FindStage`、`BatchedFindTarget`：`sam3/sam3/model/data_misc.py`
- `Sam3Image.forward`：`sam3/sam3/model/sam3_image.py`
- `Sam3Image.forward_grounding`：`sam3/sam3/model/sam3_image.py`
- `Sam3Image._encode_prompt`：`sam3/sam3/model/sam3_image.py`
- `Sam3Image._run_encoder`：`sam3/sam3/model/sam3_image.py`
- `Sam3Image._run_decoder`：`sam3/sam3/model/sam3_image.py`
- `Sam3Image._run_segmentation_heads`：`sam3/sam3/model/sam3_image.py`

## 为什么不走 SAM-style 滑窗 prompt 检测

SAM-style 简化分割路线是：

```text
image_encoder -> prompt_encoder(box) -> mask_decoder -> mask
```

这条路线的语义是“给一个框 prompt，分割该 prompt 对应的目标区域”。它本身不会枚举整张图中的同类实例。

本项目要验证的是 EfficientSAM3 原生“用户标注一个或多个目标后，查找同类目标”的能力。用户 OBB 标注应作为 geometric prompt 描述目标类别，模型随后通过 grounding / detector decoder 在图片中输出同类实例。

需要特别注意：`Sam3Image.forward()` 中的 geometric prompt 是和当前 `find_input.img_ids` 对应的查询图片绑定的，不是一个可以直接跨图片复用的“参考图 embedding”。因此离线闭环中，用户标注图主要用于 LoRA 微调监督；全量评估未标注图片时不能把训练图坐标直接作为评估图 prompt 使用。评估阶段应显式测试 LoRA 微调后的跨图查找能力，默认使用目标文本和空几何 prompt，或只在同一张被用户修正的图片上使用该图片自己的标注 prompt。

仅作为背景参考的 SAM-style 相关源码：

- `PromptEncoder`：`sam3/sam3/model/student_sam/modeling/prompt_encoder.py`
- `MaskDecoder`：`sam3/sam3/model/student_sam/modeling/mask_decoder.py`
- `TwoWayTransformer`：`sam3/sam3/model/student_sam/modeling/transformer.py`

这些不是本设计的主路线。

## EfficientViT LoRA 策略

主要 LoRA 注入目标是 EfficientViT 学生主干，因为最终需求是轻量、适合边缘设备部署。

EfficientViT 的注意力模块使用 `LiteMLA`，其中 QKV 和输出投影是卷积结构：

```text
LiteMLA.qkv.conv
LiteMLA.proj.conv
```

源码定位：

- `EfficientViTBackbone`：`sam3/sam3/backbones/efficientvit/efficientvit/backbone.py`
- `EfficientViTBlock`：`sam3/sam3/backbones/efficientvit/nn/ops.py`
- `LiteMLA`：`sam3/sam3/backbones/efficientvit/nn/ops.py`
- `ConvLayer`：`sam3/sam3/backbones/efficientvit/nn/ops.py`

LoRA 注入方式：

- 查找 EfficientViT 主干中 `LiteMLA.qkv.conv` 和 `LiteMLA.proj.conv` 对应的 `nn.Conv2d`。
- 用 Conv-LoRA wrapper 替换这些卷积。
- 冻结原始卷积权重。
- 只训练低秩 LoRA 参数。

如果只训练 EfficientViT 主干 LoRA 表达能力不足，再考虑添加 EfficientSAM3 decoder 中的轻量可训练组：

- `linear1`
- `linear2`
- `out_proj`

相关源码定位：

- Transformer encoder 构建：`sam3/sam3/model_builder.py`
- Transformer decoder 构建：`sam3/sam3/model_builder.py`
- Encoder layer 定义：`sam3/sam3/model/encoder.py`
- Decoder layer 定义：`sam3/sam3/model/decoder.py`

## 数据集输入流程

批量实验入口是一个普通文本文件。每个非空行是一个子数据集目录：

```text
/home/data/public/datasets/fewshot_test_20260429/24q4_machinery_circle
/home/data/public/datasets/fewshot_test_20260429/12356_工件定位
/home/data/public/datasets/fewshot_test_20260429/喷嘴有无
```

每个子数据集目录默认读取 `DetectTrainData.txt`。`DetectTrainData_sample5.txt` 可作为显式配置的快速 smoke test 输入。

标注格式示例：

```text
Version 1.0.0
20230922101406.jpg.bmp:6 R:4 604 423 504 362 671 86 772 148 "Sample" ...
90008300_c1s1_06_01.jpg.bmp:1 R:4 601 299 551 298 552 248 602 250 "obj"
```

解析规则：

- 跳过 `Version ...` 行。
- 第一个 `:` 之前的内容是图片名。
- `:` 后的整数是该图片声明的实例数量。
- 每组 `R:4 x1 y1 x2 y2 x3 y3 x4 y4 "label"` 解析为一个 OBB polygon。
- 保留原始 label 字符串，但在每个子数据集内按单类别任务评估。
- 如果声明数量和实际解析出的 `R:4` 数量不一致，写入数据检查报告，不静默吞掉。

图片路径解析必须宽松：

- 优先尝试 `dataset_dir / image_name`。
- 支持 `.jpg.bmp`、`.bmp.bmp` 等复合后缀。
- 如果精确路径不存在，在子数据集目录内做大小写不敏感匹配。
- 如果仍不存在，可按完整 filename stem 匹配，但必须唯一。
- 对缺失图片和歧义匹配图片写入数据检查报告。

## 标注转换

每个 `DetectTrainData.txt` 中的 OBB polygon 需要转换成三类内部目标：

```text
polygon: 原始四点
aabb: polygon 外接水平 XYXY 框
mask: polygon 栅格化后的二值 mask
```

训练输入使用：

- AABB 转归一化 `cx,cy,w,h` 后作为 `FindStage.input_boxes`。
- AABB 转归一化 `cx,cy,w,h` 后作为 `BatchedFindTarget.boxes`。
- 启用 mask loss 时，polygon mask 作为 `BatchedFindTarget.segments`。

源码定位：

- `FindStage`：`sam3/sam3/model/data_misc.py`
- `BatchedFindTarget`：`sam3/sam3/model/data_misc.py`
- `BatchedDatapoint`：`sam3/sam3/model/data_misc.py`
- `GeometryEncoder` 对 box 格式的说明：`sam3/sam3/model/geometry_encoders.py`
- `box_cxcywh_to_xyxy`：`sam3/sam3/model/box_ops.py`

OBB 评估应使用 polygon 派生出的 OBB 和 rotated IoU。若预测只有 mask，则从阈值化 mask 的最大连通域拟合 OBB。

## 迭代学习闭环

对每个子数据集执行以下流程：

1. 从 `DetectTrainData.txt` 构建完整标注索引。
2. 选择初始训练图片。
3. 将该图片所有 GT 实例加入已标注集合。
4. 从已标注集合构造 EfficientSAM3 训练 batch。
5. 在每轮时间预算内只训练 LoRA 参数。
6. 对当前子数据集全量图片进行评估。
7. 用 OBB IoU 将预测和 GT 匹配。
8. 如果 precision 和 recall 均为 `1.0`，停止。
9. 否则选择一张失败图片。
10. 将该失败图片所有 GT 实例加入下一轮已标注集合。
11. 重复直到成功或达到 `max_rounds`。

被选中的失败图片模拟 UI 中的用户修正图片。

## 训练前向

构造 `BatchedDatapoint`，其中包含：

- `img_batch`：`(B_img, 3, H, W)`
- `find_text_batch`：目标 label 或通用目标描述
- `find_inputs`：一个或多个 `FindStage`
- `find_targets`：一个或多个 `BatchedFindTarget`
- `find_metadatas`：模型路径所需 metadata

源码定位：

- `BatchedDatapoint`：`sam3/sam3/model/data_misc.py`
- `FindStage`：`sam3/sam3/model/data_misc.py`
- `BatchedFindTarget`：`sam3/sam3/model/data_misc.py`
- `BatchedInferenceMetadata`：`sam3/sam3/model/data_misc.py`

对每张已标注图片，geometric prompt 应包含用户标注 OBB 转换得到的 AABB boxes。训练时调用 `Sam3Image.forward()`，让 matching 和 loss 保持在 EfficientSAM3 原生训练路径中。

坐标格式要求：

- `FindStage.input_boxes` 形状为 `(N_prompt_boxes, B_query, 4)`。
- `FindStage.input_boxes` 的 4 维 box 是归一化 `cx,cy,w,h`，不是像素 `xyxy`。
- `FindStage.input_boxes_mask` 形状为 `(B_query, N_prompt_boxes)`，`False` 表示有效框，`True` 表示 padding。
- `FindStage.input_boxes_label` 形状为 `(N_prompt_boxes, B_query)`。
- `BatchedFindTarget.boxes` 和 `boxes_padded` 同样使用归一化 `cx,cy,w,h`。

跨图评估约束：

- 训练图的标注框坐标不能直接作为其他图片的 prompt。
- 若评估未标注图片，默认应构造空几何 prompt，并依赖文本 prompt 与 LoRA 参数完成同类目标查找。
- 若需要模拟用户修正某一张失败图，可以在该失败图自己的查询中使用该图 GT 转换出的 prompt，然后把该图加入下一轮训练集。

## Loss 策略

优先使用 EfficientSAM3 原生 loss：

- Box loss：用于检测框对齐。
- Classification / presence loss：用于实例置信度。
- Mask loss：在 `enable_segmentation=True` 时用于提升 mask 和 OBB 拟合质量。

源码定位：

- `Sam3LossWrapper`：`sam3/sam3/train/loss/sam3_loss.py`
- `Boxes`：`sam3/sam3/train/loss/loss_fns.py`
- `IABCEMdetr`：`sam3/sam3/train/loss/loss_fns.py`
- `Masks`：`sam3/sam3/train/loss/loss_fns.py`
- `sigmoid_focal_loss`：`sam3/sam3/train/loss/loss_fns.py`
- `dice_loss`：`sam3/sam3/train/loss/loss_fns.py`
- `BinaryHungarianMatcherV2`：`sam3/sam3/train/matcher.py`
- `BinaryOneToManyMatcher`：`sam3/sam3/train/matcher.py`

第一批实验保持可训练参数量尽量小：

- 只训练 EfficientViT Conv-LoRA 参数。
- 冻结所有 base model 权重。
- GPU 上启用 AMP。
- 每轮训练由 wall-clock 时间和 step 数共同限制。

## 评估流程

每轮训练后，对当前子数据集中的所有图片执行推理评估。

评估时不得使用 GT 框作为普通测试图片的 prompt。否则会把目标位置直接泄露给模型，无法验证“少样本学习后查找同类目标”的能力。

需要消费的预测输出：

- `pred_logits`
- `pred_boxes`
- `pred_boxes_xyxy`
- `pred_masks`，当 segmentation head 启用时存在

源码定位：

- `Sam3Image.forward_grounding` 输出路径：`sam3/sam3/model/sam3_image.py`

后处理步骤：

1. 将 `pred_logits` 转为 score。
2. 按 score threshold 过滤。
3. 将预测框转换回原图坐标。
4. 如果存在 mask，阈值化 mask，并从最大连通域拟合 OBB。
5. 如果没有 mask，则用预测水平框作为 angle=0 的兜底 OBB。
6. 做单类别 NMS。
7. 用 OBB IoU 将预测和 GT 匹配。

成功条件：

```text
precision == 1.0
recall == 1.0
false_positive_count == 0
false_negative_count == 0
localization_error_count == 0
```

## 错误队列

每轮评估后生成有序错误队列：

- `false_negative`：某个 GT 实例没有匹配预测。
- `localization_error`：预测与 GT 有重叠，但 OBB IoU 未达到阈值。
- `false_positive`：预测没有匹配任何 GT。
- `low_confidence_true_positive`：可选，预测位置正确但分数偏低。

选样优先级：

```text
false_negative > localization_error > false_positive > low_confidence_true_positive
```

选中某张失败图片后，将该图片中的所有 GT 实例加入下一轮训练集合，模拟用户修正整张图片。

## 指标和产物

每个子数据集每一轮记录：

- 训练图片数量。
- 训练实例数量。
- 可训练参数数量。
- 训练耗时，单位秒。
- Precision、recall、F1。
- 已匹配实例的平均 OBB IoU。
- False positive 数量。
- False negative 数量。
- Localization error 数量。
- 下一轮选中的图片。
- Adapter checkpoint 路径。

批量运行整体记录：

- 处理的数据集数量。
- 达到 100% 的数据集数量。
- 达到 100% 的平均轮数。
- 平均每轮训练耗时。
- 未收敛数据集的失败原因。

## 性能目标

初始 GPU 验证目标：

- 每轮 LoRA 更新时间小于 60 秒。
- 成功数据集记录达到 100% 所需轮数。
- Adapter 体积足够小，便于按任务切换。

NPU 边缘部署约束：

- 保持 EfficientViT base model 不变。
- 任务特定状态保存在 LoRA adapter 权重中。
- 避免在推理路径中新增难以导出的算子；仅训练期使用的算子不受此限制。
- 后处理作为可独立部署的组件处理。

## 待实施阶段决定的问题

以下问题留到 implementation plan 阶段确定：

- 每轮是从 base checkpoint 重新初始化 LoRA，还是接着上一轮 adapter 继续训练。
- 哪些 EfficientViT stage 注入 Conv-LoRA。
- 第一批实验是否从一开始启用 mask loss，还是先做 box-only smoke test。
- 对没有 GT 的 false-positive 图片是否作为 hard negative 使用。
- 默认 OBB IoU 阈值和 score threshold。

## 设计自检

- 不依赖 `fewshot_adapter`。
- 主路线不使用滑窗候选 prompt。
- 使用 EfficientSAM3 原生 interactive find / grounding 流程。
- 使用 EfficientViT 作为轻量学生主干。
- 已为关键模型、数据结构、loss 和 LoRA 相关模块标注源码路径。
- 数据加载流程支持 `.jpg.bmp`、`.bmp.bmp` 等复合图片后缀。

## 2026-05-11 实现状态

当前仓库已新增独立的 `fewshot_lora` 包，代码和测试均不引用 `fewshot_adapter`。实现路线保持本设计文档的主路线：使用
`build_efficientsam3_image_model(backbone_type="efficientvit", model_name="b0", enable_segmentation=True)` 构建
EfficientSAM3 图像模型，并走原生 `BatchedDatapoint -> Sam3Image.forward()` 的 interactive find / grounding 流程。

已实现模块包括：

- 数据解析：读取批量数据集列表，解析 `DetectTrainData.txt` 的 `Version` 行、`R:4` OBB 四点标注、复合后缀图片名和大小写容错路径。
- 几何转换：OBB polygon 转归一化 `cx,cy,w,h`、polygon mask、评估用 OBB；预测 mask 从最大连通域拟合旋转 OBB。
- SAM3 batch：构造原生训练输入所需的 `BatchedDatapoint`、`FindStage`、`BatchedFindTarget`。
- LoRA 注入：默认目标为 EfficientViT `LiteMLA.qkv.conv` 和 `LiteMLA.proj.conv`，并冻结非 LoRA 参数。
- 训练闭环：用 GT 模拟用户标注，单轮只训练 LoRA 参数，记录训练耗时、样本数和 adapter 路径。
- 评估闭环：评估阶段不把 GT 框作为普通测试图 prompt，按 dataset 级别单类别 label 生成统一 text prompt。
- 错误队列：根据 false negative、localization error、false positive 的优先级选择下一轮失败图。
- 汇总报告：输出每轮 precision、recall、F1、OBB IoU、错误统计、下一轮选图和 adapter 路径。

本轮审查修复点：

- 原生 `Sam3LossWrapper` 需要 dict 形式 target。训练代码现在先调用 `model.back_convert(batch.find_targets[0])`，再将转换后的 target 传给 loss。
- `mask_to_obb()` 不再用水平外接框兜底主路径；当前会对最大连通域前景点构造凸包，并用 `cv2.minAreaRect` 拟合旋转 OBB。
- 负样本图不再退回 `"object"` prompt；同一子数据集所有图片统一使用正样本解析出的单类别 label。
- 新增 `pytest.ini` 固定仓库根目录到 `pythonpath`，保证 `pytest tests/fewshot_lora -q` 和 `python -m pytest tests/fewshot_lora -q` 行为一致。

当前验证结果：

```text
pytest tests/fewshot_lora -q
26 passed, 2 skipped

python -m pytest tests/fewshot_lora -q
26 passed, 2 skipped
```

其中 2 个 skipped 为本地环境缺少 PyTorch / SAM3 运行依赖时的条件跳过。后续在 Linux GPU 服务器上需要继续验证真实
EfficientSAM3 权重、真实 DetectTrainData 数据集和完整训练闭环耗时。
