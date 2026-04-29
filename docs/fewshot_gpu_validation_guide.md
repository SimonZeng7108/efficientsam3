# EfficientSAM3 少样本 GPU 验证指南

这份文档用于后续在 GPU 机器上验证当前少样本闭环方案。当前主线是：

```text
完整 EfficientSAM3 图像模型
  + efficient_sam3_efficientvit_s.pt
  + task visual prompt / prompt adapter 少量微调
  + SAM3 原生 decoder / matcher / loss
  + 自动筛错并补充下一轮真值
```

本阶段只验证 GPU 训练闭环效果，不做交互界面，不做 NPU 部署，不做最终量化。

## 1. 验证前准备

### 1.1 GPU 环境

建议优先使用 Linux 或 WSL2 + CUDA。Windows 原生也可以尝试，但 `mmcv`、CUDA 扩展和 SAM3 依赖在 Windows 上更容易踩坑。

需要确认：

- Python 版本满足项目要求，当前 `pyproject.toml` 写的是 `>=3.12`。
- PyTorch 能正常识别 CUDA。
- 当前仓库已安装为 editable package。
- `efficient_sam3_efficientvit_s.pt` checkpoint 路径存在。

推荐从仓库根目录执行：

```powershell
python -m pip install -U pip
python -m pip install -e ".[stage1]"
```

如果 GPU 机器上 `mmcv` 安装失败，先按该机器 CUDA / PyTorch 版本安装匹配的 `mmcv` wheel，再重新执行 editable 安装。

### 1.2 CUDA 自检

```powershell
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

期望：

- `cuda: True`
- 能打印出 GPU 名称

如果这里是 `False`，不要继续跑训练，先修 PyTorch / CUDA 环境。

### 1.3 SAM3 包导入自检

```powershell
python -c "from sam3.model_builder import build_efficientsam3_image_model; print('sam3 import ok')"
python -c "from fewshot_adapter.native.trainer import NativeFewShotTrainer; print('fewshot import ok')"
```

如果报 `ModuleNotFoundError: sam3`，通常是没有在仓库根目录执行 `pip install -e ".[stage1]"`，或者当前 Python 环境不是刚才安装的那个环境。

## 2. 数据准备

你的原始数据应至少包含：

```text
dataset/
  images/
    img_001.jpg
    img_002.jpg
    ...
  DataTrain.txt
```

`DataTrain.txt` 每行格式类似：

```text
img_001.jpg 2 R 10 20 30 40 "target" ; P 1 2 5 2 5 6 1 6 "target"
```

说明：

- `2` 表示这张图有 2 个目标。
- `R` 表示水平矩形框，后面 4 个坐标。
- `P` 表示多边形，后面是偶数个坐标点。
- `"target"` 是类别名。多类别数据必须显式传 `--label target`，避免训练错目标。

把原始数据转换成闭环训练需要的 JSON：

```powershell
python -m fewshot_adapter.convert_datatrain `
  --datatrain dataset\DataTrain.txt `
  --image-dir dataset\images `
  --output-dir dataset_json
```

转换成功后会生成：

```text
dataset_json/
  full_gt.json
  image_map.json
```

含义：

- `full_gt.json`：全量真值标注，每个目标一条记录。
- `image_map.json`：图片名到图片实际路径的映射。

转换阶段会检查 `DataTrain.txt` 中的图片是否真实存在。如果报 `image file not found`，先修正图片目录或文件名，再进入训练。

注意：当前训练 loss 首先使用 SAM3 原生 box loss。多边形标注会先派生 HBB 参与训练；后续如果要验证真实 OBB / polygon 输出，需要继续增强预测后处理。

## 3. 先跑 1 步 Smoke Test

第一次不要直接跑 10 轮。先用 1 轮 1 步确认模型能加载、数据能读取、前向和反向能走通。

```powershell
python -m fewshot_adapter.train_native_efficientsam3_fewshot `
  --full-ground-truth dataset_json\full_gt.json `
  --image-map dataset_json\image_map.json `
  --checkpoint sam3_checkpoints\efficient_sam3_efficientvit_s.pt `
  --output-root runs\native_fewshot_smoke `
  --label target `
  --device cuda `
  --max-rounds 1 `
  --steps-per-round 1 `
  --score-threshold 0.3 `
  --iou-threshold 0.5 `
  --iou-mode hbb
```

Smoke test 成功后应看到：

```text
runs/native_fewshot_smoke/
  train_round_0.json
  summary.json
  round_00/
    adapter.pt
    predictions.json
    errors.json
    next_train.json
    summary.json
```

重点检查：

- `round_00/adapter.pt` 是否生成。
- `round_00/predictions.json` 是否生成。
- `round_00/errors.json` 是否生成。
- 根目录 `summary.json` 是否能打开。
- `summary.json` 里的 `last_loss.core_loss` 是否是有限数值，不应为 `NaN` 或 `inf`。

快速查看每轮摘要：

```powershell
python -c "import json; s=json.load(open('runs/native_fewshot_smoke/summary.json', encoding='utf-8')); print(json.dumps(s['rounds'], ensure_ascii=False, indent=2))"
```

## 4. 正式小规模验证

Smoke test 通过后，建议先用 20 到 100 张图的小数据子集验证闭环趋势。目标不是一开始追求最高精度，而是确认：

- 初始 1 张图训练后能产生合理预测。
- 错误队列能识别漏检、误检、定位错误。
- 下一轮训练集会自动增加被选中错误图片的真值。
- 多轮后 `error_count` 有下降趋势，或错误类型变得更少、更集中。

建议第一组正式参数：

```powershell
python -m fewshot_adapter.train_native_efficientsam3_fewshot `
  --full-ground-truth dataset_json\full_gt.json `
  --image-map dataset_json\image_map.json `
  --checkpoint sam3_checkpoints\efficient_sam3_efficientvit_s.pt `
  --output-root runs\native_fewshot_baseline `
  --label target `
  --device cuda `
  --resolution 1008 `
  --seed 0 `
  --max-rounds 5 `
  --steps-per-round 30 `
  --learning-rate 1e-3 `
  --score-threshold 0.3 `
  --iou-threshold 0.5 `
  --localization-error-threshold 0.1 `
  --iou-mode hbb `
  --num-prompt-tokens 8
```

如果 baseline 能跑通但定位不够好，再尝试开放 bbox head：

```powershell
python -m fewshot_adapter.train_native_efficientsam3_fewshot `
  --full-ground-truth dataset_json\full_gt.json `
  --image-map dataset_json\image_map.json `
  --checkpoint sam3_checkpoints\efficient_sam3_efficientvit_s.pt `
  --output-root runs\native_fewshot_bbox_embed `
  --label target `
  --device cuda `
  --resolution 1008 `
  --seed 0 `
  --max-rounds 5 `
  --steps-per-round 50 `
  --learning-rate 5e-4 `
  --score-threshold 0.3 `
  --iou-threshold 0.5 `
  --iou-mode hbb `
  --num-prompt-tokens 8 `
  --train-bbox-embed
```

`--train-decoder-cross-attention` 暂时不建议第一轮就打开。它可训练参数更多，可能更强，但也更容易过拟合和爆显存。只有在 prompt + bbox embed 明显不够时再试。

## 5. 输出文件怎么看

每一轮目录：

```text
round_00/
  adapter.pt
  predictions.json
  errors.json
  next_train.json
  summary.json
```

文件含义：

- `adapter.pt`：本轮训练后的少样本 adapter 权重，只保存任务相关权重。
- `predictions.json`：全量图片推理结果。
- `errors.json`：根据全量真值自动筛出的错误队列。
- `next_train.json`：下一轮训练集。
- `summary.json`：本轮统计信息。

根目录 `summary.json` 里最重要的是：

- `rounds[].train_count`：本轮训练目标数量。
- `rounds[].prediction_count`：本轮全量预测数量。
- `rounds[].error_count`：本轮错误数量。
- `rounds[].selected_image_id`：自动选入下一轮的错误图片。
- `rounds[].last_loss`：最后一步 loss 字典。

快速打印每轮错误数：

```powershell
python -c "import json; s=json.load(open('runs/native_fewshot_baseline/summary.json', encoding='utf-8')); [print(r['round'], 'train=', r['train_count'], 'pred=', r['prediction_count'], 'err=', r['error_count'], 'next=', r['selected_image_id']) for r in s['rounds']]"
```

判断是否值得继续：

- 如果 `prediction_count` 长期为 0，优先降低 `--score-threshold` 到 `0.1` 或检查模型输出后处理。
- 如果 `error_count` 不下降，但预测框位置大致对，尝试 `--train-bbox-embed`。
- 如果全是误检，先提高 `--score-threshold`，再检查 label 是否一致。
- 如果全是漏检，先降低 `--score-threshold`，并确认初始训练图片确实有该 label。

## 6. HBB、Polygon、OBB 当前验证策略

当前推荐先用：

```text
--iou-mode hbb
```

原因：

- 原始 `R` 框会直接作为 HBB。
- 原始 `P` 多边形会派生 HBB 参与第一版训练和匹配。
- 当前预测后处理主要输出 SAM3 box，对外写成 HBB，并补一个 angle=0 的 OBB 基线。

暂时不要把 `--iou-mode obb` 当成最终旋转框能力验证。当前预测写出的 OBB 是由 HBB 补出来的 angle=0 兼容字段，不是真实旋转框预测。它目前只能验证已有 OBB 字段的匹配逻辑，还不是完整旋转框产品能力。后续要做真正 OBB 产品，需要补：

- 从 mask 或 polygon 拟合 OBB。
- 或增加 OBB 后处理 / rotated NMS。
- 或在 adapter 之外增加 OBB regression 分支。

## 7. 常见问题

### 7.1 `PyTorch is required`

说明当前 Python 环境没有安装 torch，或者运行命令用的不是 GPU 环境的 Python。

先跑：

```powershell
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__)"
```

### 7.2 `cuda: False`

说明 PyTorch 没连上 CUDA。先不要改项目代码，优先修环境：

- 检查 NVIDIA 驱动。
- 检查安装的 torch 是否是 CUDA 版本。
- 检查是否进入了正确 conda / venv 环境。

### 7.3 `ModuleNotFoundError: sam3`

在仓库根目录重新安装：

```powershell
python -m pip install -e ".[stage1]"
```

### 7.4 `KeyError: image_id not found in image_map`

通常是 `DataTrain.txt` 里的图片名和 `--image-dir` 下的文件名对不上。

检查：

- `DataTrain.txt` 里的 `img_001.jpg` 是否真实存在。
- 图片扩展名大小写是否一致。
- 是否多了一层子目录。

### 7.5 CUDA out of memory

按顺序尝试：

```text
1. 不要打开 --train-decoder-cross-attention。
2. 把 --resolution 1008 降到 768 或 512。
3. 把 --steps-per-round 从 80 降到 20。
4. 把 --num-prompt-tokens 从 8 降到 4。
5. 先用更小的数据子集验证闭环。
```

### 7.6 训练能跑但效果差

按顺序排查：

- 确认 `--label` 和 `DataTrain.txt` 中的 label 完全一致。
- 先看 `predictions.json` 是否有预测。
- 降低 `--score-threshold` 到 `0.1` 看是否只是阈值过高。
- 检查第一轮 `train_round_0.json` 是否选到了有代表性的目标。
- 换 `--seed` 多跑几次，避免第一张样本太偏。
- 如果定位框大致对但 IoU 不够，尝试 `--train-bbox-embed`。

## 8. 推荐记录表

每次 GPU 验证建议记录：

```text
实验名：
checkpoint：
数据集图片数：
目标类别：
resolution：
max_rounds：
steps_per_round：
learning_rate：
score_threshold：
iou_threshold：
是否 train_bbox_embed：
是否 train_decoder_cross_attention：
每轮 error_count：
主要失败类型：
结论：
```

这样后面决定是否进入 NPU 部署时，有证据判断路线是否值得继续。

## 9. 进入 NPU 前的判断标准

建议至少满足下面条件，再考虑 NPU：

- GPU 上 smoke test 稳定通过。
- 小规模数据上多轮 `error_count` 有下降趋势。
- 误检和漏检能通过补样本继续改善。
- 输出后处理形式已经确定，是 HBB、mask 拟合 OBB，还是额外 OBB 分支。
- 已经明确哪些参数需要训练，哪些参数部署时固定。

NPU 阶段只部署最终固定推理图，不建议在无 GPU 的边缘设备上做在线训练。GPU 阶段负责少样本学习和 adapter 产出，NPU 阶段负责推理加速和工程部署。
