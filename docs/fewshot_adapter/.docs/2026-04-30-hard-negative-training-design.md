# Hard Negative / No-Object 训练设计

## 目标

补齐纯背景误检无法更新模型的问题。闭环选中纯背景 `false_positive` 时，把该图片作为 no-object 负样本加入下一轮训练，让 SAM3 原生 loss 对该图片产生负类 / absence 监督，从而降低持续误检的概率。

## 设计

- 新增 `TrainingSample` 数据结构，按图片表示训练输入。
- 正样本：`sample_type=positive`，包含该图片上的一个或多个 `Annotation`。
- 负样本：`sample_type=negative`，只包含 `image_id`、`label` 和 `reason`，没有目标框。
- `train_round_0.json` 和 `next_train.json` 改为保存训练样本列表，而不是单纯 annotation 列表。
- SAM3 batch 构造支持负样本：对应图片仍进入 image batch，但 target 的 `num_boxes=0`，`boxes` / `object_ids` 为空。
- 当前默认 `USE_PRESENCE=true`，no-object 负样本主要通过 SAM3 decoder 的 `presence_loss` 学习“该图无目标”，推理后处理会把 box 分数与 presence 分数相乘来压低纯背景误检。
- 错误队列更新时：
  - 漏检 / 定位错误：继续加入对应真值正样本。
  - 有目标图片上的误检：加入该图同类真值，避免把有目标图错误当成整图负样本。
  - 纯背景误检：加入 no-object 负样本。
- `train_inputs/` 可视化中，正样本画绿色真值；负样本保存原图并写 `NEGATIVE no-object`，不画框。

## 风险和边界

- 需要 GPU smoke test 验证 SAM3 官方 loss 在 `num_boxes=0` 样本上 loss 有限，尤其检查 `presence_loss`、`loss_bbox`、`loss_giou` 是否非 NaN。
- 如果全量数据真值不完整，某些“误检”其实可能是真实未标目标；此时 hard negative 会压掉真实目标，所以必须只在真值可信的数据集上启用。
- 当前默认闭环使用全量真值验证，因此可以把无目标占位图片视为可信背景。
