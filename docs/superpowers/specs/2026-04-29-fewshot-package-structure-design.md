# Fewshot Adapter 分包重构设计

## 目标

把 `fewshot_adapter` 从平铺文件重构为按功能分包的结构，并把核心数据处理能力整理成类。重构只改变代码组织方式和高层接口，不改变当前 EfficientSAM3 原生少样本训练路线。

## 目录设计

```text
fewshot_adapter/
  data/
    models.py
    datatrain.py
    json_io.py
    sampling.py
    sam3_batch.py
  geometry/
    ops.py
  evaluation/
    matching.py
  native/
    adapter.py
    loss.py
    predictor.py
    trainer.py
  cli/
    convert_datatrain.py
    train_native.py
  utils/
    torch.py
  __init__.py
  convert_datatrain.py
  train_native_efficientsam3_fewshot.py
```

根目录只保留公共导出和两个兼容 CLI 薄入口。真正实现放到对应功能文件夹。

## 类化边界

- `DataTrainDataset`：读取 `DataTrain.txt`、构建 `image_map`、保存 `full_gt.json` / `image_map.json`。
- `AnnotationJsonIO`：统一读写 annotation、prediction、error queue。
- `InitialTrainSelector`：选择第 0 轮训练样本。
- `TrainSetUpdater`：根据错误样本更新下一轮训练集。
- `GeometryOps`：HBB/OBB/polygon 转换、IoU、polygon 转 OBB。
- `DetectionMatcher`：预测与真值匹配、错误队列构建。
- `ErrorSelector`：选择下一轮错误样本。
- `Sam3BatchBuilder`：构造 SAM3 原生训练 batch。
- `NativeLossFactory`：构造 SAM3 原生 loss。
- `NativePredictor`：SAM3 原生输出后处理。
- `NativeFewShotTrainer`：完整少样本闭环训练入口。

## 兼容策略

保留 `python -m fewshot_adapter.convert_datatrain` 和 `python -m fewshot_adapter.train_native_efficientsam3_fewshot`。这两个根模块只调用 `fewshot_adapter.cli.*`，不放业务逻辑。

测试会迁移到与分包对应的目录下，旧的根路径导入改为新路径。`fewshot_adapter.__init__` 继续导出常用类和函数，方便 notebook 或后续智能体快速使用。
