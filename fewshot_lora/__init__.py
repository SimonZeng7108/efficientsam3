"""EfficientSAM3 少样本 LoRA 交互式同类目标查找工具包。

这个包是全新方案，和旧的 `fewshot_adapter` 完全隔离。根目录只保留轻量入口：
`cli.py` 负责命令行，`config.py` 负责配置。其余代码按职责拆到四个子包：

- `data`：DetectTrainData 解析、图片路径解析、数据准备和输入预处理。
- `sam3_integration`：EfficientSAM3 原生 batch、LoRA、loss、训练和推理。
- `eval`：OBB 几何、预测后处理、指标和错误队列。
- `runtime`：交互闭环、批量 runner 和 summary 输出。
"""
