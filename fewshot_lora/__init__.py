"""EfficientSAM3 少样本 LoRA 交互式同类目标查找工具包。

这个包是全新方案，和旧的 `fewshot_adapter` 完全隔离。主要模块分工：
- `datatrain.py`：解析 DetectTrainData 文本标注。
- `dataset.py`：结合图片尺寸准备训练/评估索引。
- `geometry.py`：处理 OBB、AABB、mask 和 IoU。
- `sam3_batch.py`：构造 EfficientSAM3 原生 BatchedDatapoint。
- `lora.py`：注入 EfficientViT LiteMLA Conv-LoRA。
- `training.py` / `inference.py` / `loop.py`：完成训练、评估和交互闭环。
"""
