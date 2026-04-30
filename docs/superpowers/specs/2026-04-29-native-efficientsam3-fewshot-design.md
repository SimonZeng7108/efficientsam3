# EfficientSAM3 原生少样本 Adapter 重构设计

## 目标

本次重构把前一版“区域特征 + proposal_candidates.json + 外置检测头”的实验路线移出主代码，新增并默认使用 EfficientSAM3 原生模型路线。主流程直接加载 `efficient_sam3_efficientvit_s.pt`，复用其中的 EfficientViT 图像编码器、SAM3 transformer decoder、DotProductScoring、bbox head 和 mask head，只训练少量任务级 prompt / adapter 参数。

目标产品仍然是少样本、交互式思想的离线验证闭环：先给一张或少量带真值的图片，在 GPU 上快速微调；推理全部图片；用全量真值自动找漏检、误检、定位错误；自动挑一张错误图片及其真值加入训练集；继续微调，直到全量检测正确或达到最大轮数。验证阶段不做交互界面，NPU 阶段只做固化后的推理。

## SAM3 代码依据

阅读 `sam3/` 后确认：

- `build_efficientsam3_image_model(...)` 会构建完整 `Sam3Image`，不是只构建图像编码器。
- `efficient_sam3_efficientvit_s.pt` 是已把 EfficientViT 图像学生合并进 SAM3 结构的 checkpoint。
- `Sam3Image.forward_grounding(...)` 会输出 `pred_logits`、`pred_boxes`、`pred_boxes_xyxy`、`pred_masks`。
- 当前 builder 默认启用 `DotProductScoring`，因此活跃的“分类/打分头”不是 `class_embed`，而是 `dot_prod_scoring`。
- `Sam3Processor` 的 box/mask 输出来自 `forward_grounding` 后处理，不需要用户准备 proposal 列表。
- SAM3 官方训练链路已经有 `BinaryHungarianMatcherV2`、`IABCEMdetr`、`Boxes`、`Masks` 等损失，可复用。

## 架构选择

推荐主线是“EfficientSAM3 原生模型 + 任务级视觉 prompt adapter”。

系统会冻结大部分 EfficientSAM3 参数。每个少样本任务新增一组可训练 `task_prompt_tokens`，维度与 SAM3 prompt token 一致，作为视觉任务提示注入 `_encode_prompt(..., visual_prompt_embed=...)`。由于原始 `_encode_prompt` 即使 `encode_text=False` 也会读取 `language_features`，包装层会注入空语言特征，避免真实文本编码器参与训练和推理。

第一版可训练参数包括：

- `task_prompt_tokens`：任务级目标记忆，表示“当前要找的目标”。
- 可选 `prompt_adapter`：小型 bottleneck MLP，对 prompt token 做残差适配。
- 可选 native 模块解冻：`dot_prod_scoring`，以及必要时 decoder 中很小范围的 score/bbox 相关参数。

不在第一版做的事：

- 不训练新的 YOLO 式检测头。
- 不要求用户提供候选区域 JSON。
- 不把 OBB 作为 SAM3 decoder 原生输出维度。
- 不做 NPU 在线训练。

## 数据流

输入数据仍然支持用户当前格式：

```text
{图片名称} 4 P/R x1 y1 x2 y2 ... "label" ; P/R ...
```

`DataTrain.txt` 解析后得到统一 `Annotation`。HBB、Polygon、未来扩展 OBB 都会补齐 HBB / polygon 派生字段。训练送给 SAM3 的目标框统一转为归一化 `cxcywh`，用于 matcher 和 bbox loss。当前 MVP 不生成 SAM3 mask target，`LOSS.USE_MASKS` 必须保持 `false`；后续如果要训练 mask loss，再补 polygon/HBB 到 mask 的栅格化目标。

每轮训练集是 annotation 列表，不是候选框列表。每轮推理会对全量图片直接跑 EfficientSAM3 原生输出，再经阈值、NMS、mask resize、box/mask/polygon/OBB 后处理生成 `predictions.json`。

## 训练流

训练入口加载完整 EfficientSAM3：

```python
build_efficientsam3_image_model(
    checkpoint_path="efficient_sam3_efficientvit_s.pt",
    backbone_type="efficientvit",
    model_name="b0",
    eval_mode=False,
    enable_segmentation=True,
)
```

包装器执行以下步骤：

1. 图像按 `Sam3Processor` 同款 resize / normalize 变换成 `(B, 3, resolution, resolution)`。
2. 调用 `model.backbone.forward_image` 得到原生 `backbone_out`。
3. 注入空语言特征，构造空几何 prompt。
4. 将 `task_prompt_tokens` 作为 `visual_prompt_embed` 注入 SAM3 encoder/decoder。
5. 调用 SAM3 原生 `_run_encoder`、`_run_decoder`、`_run_segmentation_heads`。
6. 使用 `model.matcher` 生成 Hungarian matching。
7. 使用 SAM3 原生检测/框/可选 mask 损失反向传播。

## 推理流

推理使用训练后的同一个 wrapper：

1. 对每张图做图像前向。
2. 注入训练好的 task prompt / adapter。
3. 得到原生 `pred_logits`、`pred_boxes`、`pred_masks`。
4. 将 `pred_logits.sigmoid()` 与 presence score 合并成最终置信度。
5. 过滤低分预测，归一化 box 转原图像素坐标。
6. mask resize 到原图尺寸。
7. 根据 mask 或 polygon 拟合 OBB。
8. 写出 `predictions.json`，供现有错误筛选模块使用。

## 自动闭环

闭环保留已有错误样本选择思想：

1. `train_round_0.json` 从全量 GT 随机或按 label 选一张含目标图片。
2. 训练 task prompt / adapter 若干步。
3. 推理全量图片。
4. 与全量 GT 做 HBB/Polygon/OBB IoU 匹配。
5. 自动选一张最有价值错误样本，把该图所有相关真值加入下一轮训练集。
6. 保存 adapter checkpoint 和 round summary。

## 文件边界

当前主线模块：

- `fewshot_adapter/data/sam3_batch.py`：把图片和 `TrainingSample` 转成 SAM3 训练 batch / target；支持正样本和 no-object 负样本。
- `fewshot_adapter/native/adapter.py`：任务 prompt、prompt adapter、冻结/解冻策略和 SAM3 原生 wrapper。
- `fewshot_adapter/native/loss.py`：轻量封装 SAM3 matcher/loss，同时配置 o2o / o2m matcher。
- `fewshot_adapter/native/predictor.py`：原生推理与预测后处理。
- `fewshot_adapter/native/trainer.py`：多轮自动训练、推理、筛错、补样本闭环。
- `fewshot_adapter/train_native_efficientsam3_fewshot.py`：兼容 CLI 入口，实际转发到 `fewshot_adapter/cli/train_native.py`。

已删除旧路线文件：

- `proposals.py`
- `candidate_features.py`
- `prototype_head.py`
- `torch_heads.py`
- `train_efficientvit_s_fewshot.py`

当前代码只保留原生 EfficientSAM3 少样本主线和必要的数据/评估基础模块。

## 验证

由于当前环境可能没有完整 PyTorch / CUDA / SAM3 checkpoint，单元测试优先覆盖：

- `DataTrain.txt` 到训练样本的转换。
- SAM3 target box 的归一化和 padded 表达。
- wrapper 冻结/解冻参数选择逻辑。
- CLI 在缺少 PyTorch 时给出清晰错误。
- 推理后处理能把 normalized box 转回像素预测。

真实训练验证由用户在 GPU 环境执行，命令会输出每轮 checkpoint、predictions、errors、summary，便于观察是否逐轮减少错误。
