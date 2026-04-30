# 基于 EfficientSAM3 轻量 Adapter 微调的少样本交互式 OBB 目标检测方案

## 1. 方案定位

本方案面向一个“少样本、给几张图就能找相同目标”的视觉产品。当前阶段优先使用 GPU 做离线验证：数据集已经带有真值标注，系统从少量初始样本开始训练，每轮自动检测全量图片、根据真值找出漏检/误检样本，并自动选取一张错误样本及其真值加入下一轮训练。等效果确认后，再考虑把模型固化、量化并部署到 NPU 边缘设备上。

这个方案不是直接把 EfficientSAM3 当成 YOLO-OBB 检测器使用，也不是在 NPU 上在线训练大模型。推荐路线是：

```text
GPU 验证阶段：
少量标注样本
  -> EfficientSAM3 提供视觉/分割基础能力
  -> 冻结大部分模型参数
  -> 只训练任务级 prompt / Adapter / 少量原生打分模块
  -> 检测全量图片
  -> 输出 mask / polygon / OBB
  -> 用数据集真值自动判定漏检和误检
  -> 自动选择一张错误样本及其真值加入训练集
  -> 继续轻量微调

NPU 部署阶段：
固化微调后的模型
  -> 导出 ONNX / NPU 格式
  -> 量化和算子适配
  -> NPU 只做前向推理
  -> CPU 做后处理、OBB、NMS、任务管理
```

一句话总结：**先在 GPU 上验证“EfficientSAM3 + 轻量可训练模块”能否少样本学会目标；如果有效，再把训练后的固定模型部署到 NPU。**

## 2. 产品目标

用户拥有一批待检测图片，验证阶段这批图片已经有真值标注。系统先从其中随机或按策略选取一张包含目标的图片作为初始训练样本，标注可能是普通矩形框、旋转框或多边形。系统根据少量标注样本快速学习目标外观，然后检测全部图片。检测后，系统将预测结果与全量真值比对，自动找出漏检、误检或定位错误样本，从中选一张最有价值的错误样本及其真值加入训练集，继续轻量微调并再次检测全部图片。这个闭环不断迭代，直到指标满足要求或达到最大轮数。

目标能力：

- 支持 1 到 N 张少样本标注启动任务。
- 支持普通矩形框、OBB、多边形标注。
- 支持输出 OBB，必要时同时输出 mask 和 polygon。
- 支持每轮自动加入一张错误样本后快速更新模型。
- 支持基于真值自动判定漏检、误检、定位错误和低置信样本。
- GPU 阶段可以训练；NPU 阶段只部署推理。

非目标：

- 不做端侧 NPU 在线训练。
- 不从零训练 EfficientSAM3。
- GPU 验证阶段不做交互界面，先用已有真值自动闭环。
- NPU 部署阶段不做在线增量训练，只加载 GPU 阶段固化后的模型。
- 不把 EfficientSAM3 改成完整 YOLO 式训练框架。

## 3. 为什么选择 Adapter 微调作为主方案

纯 prototype memory 的优点是简单、端侧友好，但它本质是“记忆 + 相似度匹配”，遇到视角变化、背景相似、目标形变、遮挡时可能不够稳。

全量微调 EfficientSAM3 的问题是参数量大、训练慢、过拟合风险高，而且后续难以迁移到 NPU。

轻量 Adapter / LoRA / 小型适配头是折中方案：

- EfficientSAM3 主体保留预训练能力。
- 大部分参数冻结，训练成本低。
- 少量样本也能较快收敛。
- 每轮自动加入错误样本后可以快速继续训练。
- 最终可以只保存 adapter 权重，便于任务级部署。
- 后续可把 adapter 合并或作为额外分支一起导出。

推荐优先级：

1. **训练任务级 visual prompt / prompt adapter**：不新增检测 decoder，最贴近 SAM3 原生链路，适合作为当前 MVP。
2. **开放 `dot_prod_scoring` 和少量 bbox 相关参数**：表达能力更强，但仍保持轻量。
3. **训练 decoder cross-attention LoRA / Adapter**：适合目标外观变化较大场景，但导出和 NPU 支持需要额外验证。

## 4. 总体架构

系统由 9 个模块组成：

```text
数据导入
  -> 标注统一
  -> EfficientSAM3 特征/分割骨干
  -> 轻量可训练适配模块
  -> 少样本训练器
  -> 全量推理器
  -> OBB 后处理
  -> 基于真值的错误判定与样本选择
  -> 自动增量训练闭环
```

### 4.1 数据导入模块

负责读取图片、标注和任务配置。

输入：

- 图片目录。
- 初始少样本标注。
- 后续补充标注。
- 类别名称或目标名称。
- 训练参数，例如学习率、训练步数、冻结策略、输入尺寸。

输出：

- 标准化图片列表。
- 标准化标注对象。
- 训练集、验证集、待检测集划分。
- 任务状态文件。

### 4.2 标注统一模块

数据集中可能同时存在三类标注：

- HBB：普通水平矩形框。
- OBB：旋转矩形框。
- Polygon：多边形。

内部建议统一保存为：

```text
Annotation {
  image_id
  object_id
  label
  source_type
  hbb
  obb
  polygon
  mask
}
```

转换规则：

- HBB 转 polygon：取四个角点。
- OBB 转 polygon：取旋转框四个角点。
- Polygon 转 mask：栅格化多边形。
- Mask 转 polygon：轮廓提取。
- Polygon 或 mask 转 OBB：最小外接旋转矩形。

训练时优先使用 `mask` 或 `polygon`，因为它们比框提供更精细的目标区域。如果只有 HBB 或 OBB，可以先生成粗 mask，再用 EfficientSAM3 的 prompt 能力生成伪 mask。

### 4.3 EfficientSAM3 骨干模块

EfficientSAM3 提供三类能力：

- 图像特征提取。
- prompt 条件分割。
- mask / region 表达。

需要注意：`efficient_sam3_efficientvit_s.pt` 是完整 EfficientSAM3 图像模型，不是单独的图像编码器。`efficientsam3_image_predictor_example.ipynb` 里的 `box/mask` 输出来自 `Sam3Processor._forward_grounding`，其底层会使用 SAM3 原生 `pred_logits`、`pred_boxes`、`pred_masks`。因此当前重构后的 MVP 不再要求用户准备候选区域，也不再把自动 proposal 作为主线；系统会训练任务级 visual prompt / adapter，然后直接使用 SAM3 原生 decoder 输出检测结果。

GPU 验证阶段可先使用仓库现有的 EfficientSAM3 image model 跑通功能，不急于做极致轻量化。等方案有效后，再选择适合 NPU 的版本，例如 EfficientViT-B0/B1 或 TinyViT 小模型。

推荐冻结策略：

- 冻结视觉 backbone 主体。
- 冻结大部分 SAM3 decoder / segmentation head。
- 只开放 adapter、LoRA 或小型 head。
- 如果效果不足，再逐步解冻靠后的少量层。

不要一开始就全量微调，否则少样本容易过拟合，训练成本也高。

### 4.4 轻量可训练适配模块

这是本方案的核心。

有三种可选实现。

#### 方案 B1：目标适配头

在 EfficientSAM3 特征后面加一个小型模块：

```text
image feature / mask feature
  -> small target head
  -> targetness score
  -> mask/box refinement
```

它可以是小型 MLP、1x1 conv、轻量 transformer block 或 metric classifier。

优点：

- 实现最快。
- 训练参数少。
- 最容易导出。
- 最适合作为 MVP。

缺点：

- 对复杂外观变化的适应能力有限。

#### 方案 B2：Adapter 模块

在 backbone 或 neck 的若干层之间插入小模块：

```text
x -> frozen layer -> adapter -> x + adapter(x)
```

Adapter 通常是 bottleneck 结构：

```text
Linear/Conv 降维
  -> 激活
  -> Linear/Conv 升维
  -> 残差加回原特征
```

优点：

- 比小 head 表达能力更强。
- 训练参数仍然可控。
- 适合少样本任务适配。

缺点：

- 需要改模型结构。
- 后续 NPU 导出要验证 adapter 算子支持。

#### 方案 B3：LoRA

在 attention 或 linear 层上插入低秩增量：

```text
W' = W + A * B
```

优点：

- 参数量很小。
- 对 transformer 层适配能力好。
- 适合保存多个任务的增量权重。

缺点：

- EfficientSAM3 的可插入位置需要仔细选择。
- NPU 部署时可能需要把 LoRA 合并进原权重，或者额外实现算子。

### 4.5 Prototype Memory 辅助模块

在本方案中，prototype memory 不是主学习机制，而是辅助机制。

用途：

- 用少量标注快速初始化目标特征。
- 帮助选择 hard negative 和高价值错误样本。
- 辅助 adapter 训练时构造正负样本。
- 在全量推理后作为 re-rank 分数。

每个标注目标会提取一个 prototype：

```text
Prototype {
  vector
  label
  source_image_id
  source_annotation_id
  quality_score
}
```

推理时可计算：

```text
final_score = adapter_score * 0.7 + prototype_similarity * 0.3
```

具体权重可通过验证集调节。

### 4.6 少样本训练器

每轮训练输入：

- 当前所有正样本标注。
- 用户确认的负样本或误检样本。
- 自动生成的 hard negatives。
- 可选伪 mask。

训练目标：

- 正样本区域得分高。
- 负样本区域得分低。
- 对纯背景 no-object 样本，presence 分支输出低置信度，避免背景图持续误检。
- 预测 mask/region 与标注区域对齐。
- OBB 或 polygon 后处理后与标注尽量一致。

推荐损失组合：

```text
total_loss =
  classification_loss
  + lambda_mask * mask_loss
  + lambda_box * box_or_obb_loss
  + lambda_metric * contrastive_loss
```

MVP 阶段可以简化为：

```text
total_loss =
  classification_loss
  + lambda_metric * contrastive_loss
```

如果使用 mask：

- 可用 BCE / Dice loss。
- 对 HBB/OBB 生成的粗 mask，边界区域降低权重。

如果使用 OBB：

- MVP 可先不直接回归 OBB。
- 先预测 mask/polygon，再后处理得到 OBB。
- 后续再加入 OBB regression head。

训练策略：

- 每轮自动加入新样本后训练 50 到 500 steps。
- 使用较小学习率。
- 使用 early stopping。
- 保留上一轮 adapter 权重继续训练。
- 维护最佳 checkpoint，避免新一轮增量训练导致退化。

### 4.7 全量推理器

每轮训练完成后，对全部图片推理。

输出：

- SAM3 原生 `pred_logits` 转换后的目标置信度。
- SAM3 原生 `pred_boxes` 转换后的像素 HBB。
- 可选 `pred_masks` 后处理得到的 mask / polygon。
- 开启 segmentation + mask loss 时，由预测 mask 的凸包拟合 OBB；没有 mask 或 mask 为空时，才回退到 HBB 派生的 angle=0 OBB 兼容字段。
- 风险标记和错误类型。

当前 MVP 默认走“完整 EfficientSAM3 -> task visual prompt / adapter -> SAM3 原生 decoder 输出”的路径。旧的 `proposal_candidates.json` / 外置 head 实现已经从代码中清理，不再是产品验证主线。每轮训练后直接对全量图片执行原生前向，预测结果写入 `predictions.json`，再由真值匹配模块筛选漏检、误检和定位错误。

为了提速，GPU 验证阶段建议先冻结 backbone、encoder、decoder 大部分权重，只训练 task prompt、prompt adapter 和可选 `dot_prod_scoring`。这种设置仍需要重新跑图像前向，但参数更新很轻；如果后续要进一步提速，可以增加图像特征缓存版本作为实验分支。

### 4.8 OBB 后处理模块

EfficientSAM3 更自然地产生 mask 或区域，不天然输出 OBB。因此 OBB 推荐通过后处理得到。

流程：

```text
predicted mask / polygon
  -> 连通域分析
  -> 轮廓提取
  -> 最小外接旋转矩形
  -> OBB
  -> rotated NMS
```

后处理规则：

- 过滤面积太小的目标。
- 过滤面积太大的异常目标。
- 根据目标先验过滤长宽比。
- 用 rotated IoU 做 NMS。
- 如果一个 mask 分裂成多个连通域，可按面积或分数拆分。

如果只有候选框没有 mask：

- 可以先输出 HBB 或粗 OBB。
- 再用 EfficientSAM3 prompt 分割细化。

### 4.9 基于真值的错误判定与样本选择模块

GPU 验证阶段假设数据集已有全量真值，因此系统不需要人工界面来判断错误。每轮全量推理后，系统将预测结果和真值标注做匹配，自动得到漏检、误检和定位错误样本，然后从错误样本中选择一张最有训练价值的图片及其真值加入下一轮训练集。

错误类型：

- 漏检：某个真值目标没有匹配到任何预测结果。
- 误检：某个预测结果没有匹配到任何真值目标。
- 定位错误：预测和真值类别匹配，但 OBB IoU 或 polygon/mask IoU 低于阈值。
- 低置信正确：预测匹配真值，但分数接近阈值，属于不稳定样本。

匹配规则：

- 如果最终输出是 OBB，优先使用 rotated IoU。
- 如果有 polygon 或 mask，额外计算 mask IoU 或 polygon IoU。
- 当多个预测匹配同一真值时，保留 IoU 最高者，其余视为重复误检。
- 默认匹配阈值可从 `OBB IoU >= 0.5` 开始，后续按业务调节。

自动样本选择策略：

- 优先选择漏检样本，因为它能补充模型完全没学到的外观。
- 其次选择定位错误样本，因为它能改善边界和角度。
- 再选择高置信误检样本，把它作为 hard negative。
- 如果同一轮错误很多，选择 `risk_score` 最高的一张。
- 如果需要更稳，可以每轮选 1 张正向错误样本和 1 张 hard negative。

输出错误队列：

```text
ErrorItem {
  image_id
  error_type
  risk_score
  reason
  predictions
  ground_truth
  selected_for_next_round
}
```

当前验证阶段默认每轮自动选择 1 张错误图片加入训练，不需要人工界面。

## 5. GPU 离线自动闭环流程

完整流程如下：

```text
第 0 轮：
  系统从带真值的数据集中选择一张含目标图片
  读取该图片的 HBB / OBB / Polygon 真值
  系统生成 mask / polygon / OBB
  系统初始化 prototype memory
  系统初始化 adapter/head

第 1 轮：
  GPU 轻量微调 adapter/head
  对全量图片推理
  输出预测结果
  与全量真值比对
  生成漏检、误检、定位错误队列
  自动选择一张错误样本及其真值加入训练集

第 N 轮：
  增量加入新样本
  继续微调 adapter/head
  重新全量推理
  更新错误队列和指标
  直到达到目标指标或最大迭代轮数
```

训练集中支持两类样本：

- 正样本：这里有目标。
- 负样本：这里不是目标，是误检。

负样本可由高置信误检自动生成，非常重要，能显著降低相似背景误检。

## 6. GPU 验证阶段 MVP

MVP 目标是验证技术路线，而不是一开始就做完整产品或交互界面。验证阶段默认数据集已有真值标注，系统可以自动判断每轮检测错误，并自动把一张错误图片和真值加入下一轮训练。

建议 MVP 功能：

- 图片目录导入。
- HBB / OBB / Polygon 标注读取。
- 标注统一转换。
- EfficientSAM3 特征提取。
- 小型 target adapter/head。
- 少样本训练循环。
- 全量图片推理。
- mask/polygon 转 OBB。
- rotated NMS。
- 预测和真值自动匹配。
- 漏检、误检、定位错误排序。
- 每轮自动选择一张错误样本加入训练后继续训练。

MVP 暂时不做：

- NPU 部署。
- 云端训练平台。
- 复杂 UI。
- 人工交互标注界面。
- 多用户任务管理。
- 多类别大规模训练。
- 端侧在线训练。

推荐先用命令行或简单 notebook 跑通：

```text
输入：
  images/
  annotations_round_0.json
  full_ground_truth.json

运行：
  train_adapter.py
  infer_all.py
  evaluate_predictions.py
  select_next_training_sample.py

输出：
  predictions_round_1.json
  errors_round_1.json
  train_set_round_1.json
  adapter_round_1.pth
```

## 7. NPU 部署阶段设计

等 GPU 验证效果通过后，再进入 NPU 部署。

部署原则：

- NPU 只做固定模型前向。
- 不在 NPU 上做训练。
- Adapter/LoRA 在 GPU 阶段训练完成后固化。
- CPU 负责数据管理和动态后处理。

部署流程：

```text
GPU 训练完成
  -> 导出 EfficientSAM3 + adapter
  -> 如使用 LoRA，优先合并进原权重
  -> 导出 ONNX
  -> NPU 编译工具转换
  -> INT8/FP16 量化
  -> 端侧验证精度和速度
```

NPU/CPU 分工：

```text
NPU：
  - 图像预处理后的模型前向
  - backbone / adapter / head

CPU：
  - 图片读写
  - 标注解析
  - polygon/mask/OBB 转换
  - prototype memory
  - rotated NMS
  - 基于真值的错误样本选择
  - 本地任务状态管理
```

如果 NPU 不支持完整 EfficientSAM3：

- 先导出 feature extractor + adapter/head。
- 把复杂 mask decoder 放到 GPU 验证阶段或 CPU fallback。
- 端侧先输出 OBB，mask 作为可选能力。

## 8. 数据格式建议

### 8.1 标注格式

```json
{
  "image_id": "img_0001",
  "label": "target",
  "source_type": "obb",
  "hbb": [110, 30, 180, 90],
  "obb": {
    "cx": 145.0,
    "cy": 60.0,
    "w": 72.0,
    "h": 45.0,
    "angle": -12.5
  },
  "polygon": [[120, 30], [180, 45], [170, 90], [110, 75]],
  "mask_path": "masks/img_0001_target_0.png"
}
```

### 8.2 预测格式

```json
{
  "image_id": "img_0002",
  "label": "target",
  "score": 0.87,
  "adapter_score": 0.84,
  "prototype_similarity": 0.91,
  "polygon": [[120, 30], [180, 45], [170, 90], [110, 75]],
  "obb": {
    "cx": 241.2,
    "cy": 133.7,
    "w": 81.4,
    "h": 39.2,
    "angle": 18.0
  },
  "risk_flags": []
}
```

### 8.3 错误队列格式

```json
{
  "image_id": "img_0034",
  "error_type": "false_negative",
  "risk_score": 0.78,
  "reason": "真值目标没有匹配到任何预测结果",
  "ground_truth_ids": ["gt_0034_0001"],
  "prediction_ids": [],
  "selected_for_next_round": true
}
```

### 8.4 原生 adapter checkpoint 格式

当前主线保存的是任务级 adapter 权重，不是 proposal 列表：

```json
{
  "model_type": "native_efficientsam3_fewshot_adapter",
  "config": {
    "num_prompt_tokens": 8,
    "prompt_dim": 256,
    "train_dot_prod_scoring": true
  },
  "state_dict": "<torch checkpoint 中的 adapter 权重>",
  "trainable_names": ["task_prompt_tokens", "prompt_adapter.*", "model.dot_prod_scoring.*"]
}
```

旧版 proposal / 外置 head 文件已经删除；默认产品验证流程只要求 `full_gt.json`、`image_map.json` 和 EfficientSAM3 checkpoint。

## 9. 评估指标

GPU 验证阶段需要量化每轮迭代是否真的变好。

检测指标：

- OBB mAP。
- OBB IoU。
- precision / recall。
- 漏检数。
- 误检数。

交互指标：

- 标注 1 张、2 张、3 张、5 张时的性能曲线。
- 达到目标 recall 所需标注张数。
- 每轮训练耗时。
- 每轮全量推理耗时。
- 自动错误样本选择后带来的指标增益。
- 漏检、误检、定位错误分别下降的速度。

部署指标：

- NPU 单图延迟。
- NPU 吞吐。
- CPU 后处理耗时。
- 内存占用。
- 量化前后精度差。

## 10. 主要风险

### 风险 1：少样本过拟合

应对：

- 冻结主干，只训练小模块。
- 使用数据增强。
- 使用负样本。
- 使用 early stopping。
- 保留历史最佳 checkpoint。

### 风险 2：只有框标注导致 mask 粗糙

应对：

- OBB/HBB 转粗 mask。
- 用 EfficientSAM3 prompt 生成伪 mask。
- mask loss 降低边界权重。
- 如果只有框标注但需要更精细监督，允许在数据中补充 polygon。

### 风险 3：OBB 输出不稳定

应对：

- 优先由 mask/polygon 拟合 OBB。
- 加入几何过滤。
- 加 rotated NMS。
- 后续再加 OBB regression head。

### 风险 4：Adapter 难以部署到 NPU

应对：

- MVP 先用小 head。
- Adapter 使用 NPU 友好算子，例如 conv、linear、relu。
- LoRA 部署前合并权重。
- 提前做 ONNX 导出验证。

### 风险 5：全量推理太慢

应对：

- 缓存冻结 backbone 特征。
- adapter/head 放在后层。
- 先粗筛，再精细分割。
- 高风险样本优先处理。

## 11. 推荐实施顺序

### 第一步：离线 GPU 原型

- 读取图片和标注。
- 实现 HBB/OBB/Polygon 统一转换。
- 用 `DataTrain.txt` 转出 `full_gt.json` 和 `image_map.json`。
- 直接加载完整 `efficient_sam3_efficientvit_s.pt`。
- 先跑 `max-rounds=1`、`steps-per-round=1` 的 smoke test。
- 只训练 task prompt / prompt adapter / dot_prod_scoring。
- 先用 HBB IoU 验证闭环是否能跑通。

### 第二步：交互闭环

- 实现全量推理。
- 实现预测与真值匹配。
- 实现漏检、误检、定位错误排序。
- 支持自动选择下一轮错误图片，并把该图片真值加入训练。
- 每轮继续微调。
- 记录每轮性能变化。

当前代码已实现这一步的正样本闭环、错误队列、指标统计和每轮可视化输出，并已补充 no-object hard negative：纯背景误检会以 `sample_type=negative` 写入 `next_train.json`，下一轮用 `num_boxes=0` 的 SAM3 target 参与训练。

### 第三步：增强模型

- 如果定位不足，尝试开放 `bbox_embed`。
- 如果类别/目标适配仍不足，再谨慎尝试 decoder cross-attention。
- 增加 mask / polygon 后处理，拟合真正 OBB。
- 继续增强 hard negative 策略，例如每轮同时选择一个正向错误和一个背景负样本，或限制负样本比例避免过度压低召回。
- 必要时再考虑 LoRA 或 prototype memory re-rank。

### 第四步：NPU 可部署性验证

- 固化模型结构。
- 导出 ONNX。
- 量化。
- 用目标 NPU 工具链转换。
- 对比 GPU/NPU 输出差异。

## 12. 最终推荐

当前阶段推荐采用：

```text
完整 EfficientSAM3 图像模型
  + efficient_sam3_efficientvit_s.pt
  + task visual prompt / prompt adapter 少量微调
  + 默认冻结主体，只开放 prompt、adapter、dot_prod_scoring
  + 可选开放 bbox_embed 或 decoder cross-attention
  + SAM3 原生 decoder / matcher / loss
  + 可选开启粗 mask loss，并通过预测 mask/polygon 后处理生成 OBB
  + GPU 端基于真值自动迭代微调验证
  + NPU 端只部署最终固定推理模型
```

这样既能验证“少样本交互式学习”的真实效果，又不会一开始被 NPU 训练能力限制住。等 GPU 阶段证明路线可行后，再进入模型压缩、导出、量化和 NPU 工程化部署。

## 13. GPU 验证操作指南

详细命令和排错步骤见：

```text
docs/fewshot_gpu_validation_guide.md
```

GPU 验证的最小流程是：

1. 在 GPU 环境安装项目和 PyTorch。
2. 用 `python -m fewshot_adapter.convert_datatrain` 把 `DataTrain.txt` 转成 `full_gt.json` 和 `image_map.json`。
3. 先用 `python -m fewshot_adapter.train_native_efficientsam3_fewshot --max-rounds 1 --steps-per-round 1` 跑 smoke test。
4. 再用 20 到 100 张图跑 5 轮左右的小规模验证。
5. 观察每轮 `summary.json` 里的 `error_count`、`prediction_count`、`selected_image_id`。

第一轮环境验证建议先使用 `--iou-mode hbb`，确认模型加载、训练、全量推理和错误队列都能跑通。要验证真实 OBB，请在 YAML 中同时打开 `MODEL.ENABLE_SEGMENTATION=true` 和 `LOSS.USE_MASKS=true`；此时训练会用 HBB/OBB/Polygon 生成粗 mask target，推理会优先由 `pred_masks` 拟合旋转 OBB。如果模型没有输出 mask 或 mask 为空，单条预测仍会回退到 angle=0 的兼容 OBB。
