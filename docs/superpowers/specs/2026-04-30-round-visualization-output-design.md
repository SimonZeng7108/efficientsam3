# 每轮可视化输出设计

## 目标

在原生 EfficientSAM3 少样本闭环训练中，每一轮 `round_xx/` 目录直接输出三类可视化图片，便于快速查看训练输入、筛选错误和全量检测结果，不需要再把 JSON 和原图手工对照。

## 输出结构

```text
runs/native_fewshot/
  round_00/
    adapter.pt
    predictions.json
    errors.json
    next_train.json
    summary.json
    train_inputs/
      image_gt.jpg
    errors_vis/
      image_error.jpg
    predictions_vis/
      image_pred.jpg
```

## 渲染规则

- `train_inputs/`：渲染当前轮实际输入训练的图片；正样本画绿色真值框或 polygon，纯背景 hard negative 写 `NEGATIVE no-object`，不画框。
- `errors_vis/`：只渲染本轮错误队列涉及的图片；绿色画真值，红色画预测，文字标出错误类型。
- `predictions_vis/`：按 `image_map.json` 渲染全量图片；有预测时画红色预测框、类别和分数；没有预测时保存原图。
- 当前模型主输出是 HBB；真值若来自 polygon，则按 polygon 轮廓画，便于观察旋转/四边形标注。
- 可视化输出不改变训练逻辑，只作为 GPU 验证和错误复查辅助；纯背景负样本真正参与训练的入口是 `next_train.json` 里的 `sample_type=negative`，训练 batch 中对应 `num_boxes=0`。

## 测试口径

- 用临时小图片验证三类目录和图片文件都会生成。
- 验证 `predictions_vis/` 会为没有预测的背景图保存原图。
- 验证 `errors_vis/` 会把错误类型写入图片并画出 GT / prediction。
