# EfficientSAM3 智能体接手说明

这份文档是给后续智能体看的项目交接笔记。它说明这个仓库是干什么的、主要模块怎么配合、每个文件或文件组大概负责什么，方便后续智能体不用重新通读全部源码就能接着干活。

## 项目是干什么的

EfficientSAM3 是一个研究型代码库，目标是通过“渐进式分层知识蒸馏”把 Meta 的 SAM3 做轻量化。核心思路是保留 SAM3 的可提示概念分割能力，但把昂贵的大模块换成更小的学生模型：

- Stage 1：把 SAM3 的图像编码器和文本编码器蒸馏成轻量 backbone。
- Stage 1 几何微调：让已蒸馏的图像编码器在点/框 prompt 条件下更适配 SAM3 后续分割模块。
- `sam3/`：内置了一份上游 SAM3 包，同时加入了 EfficientSAM3 的 builder、轻量 backbone、学生文本编码器等扩展。
- `data/`：数据集下载和重组脚本。
- `eval/`：轻量评估脚本，包括 COCO 图像 mask 评估和文本编码器相似度评估。

## 当前少样本产品重构状态

用户当前要做的是基于 `efficient_sam3_efficientvit_s.pt` 的少样本目标检测验证产品。最新方向已经从旧的“proposal 候选框 + 图像区域特征 + 外置 head”重构为“完整 EfficientSAM3 原生模型 + task visual prompt / adapter 微调”。

关键结论：

- `efficient_sam3_efficientvit_s.pt` 是完整 EfficientSAM3 图像模型，包含 EfficientViT 图像编码器、SAM3 decoder、`DotProductScoring`、box/mask head，不是单独图像编码器。
- 主流程不再要求 `proposal_candidates.json`。SAM3 原生输出 `pred_logits`、`pred_boxes`、`pred_masks`，后处理后写出 `predictions.json`。
- 少样本训练默认冻结大部分 EfficientSAM3，只训练 `task_prompt_tokens`、`prompt_adapter`、可选 `dot_prod_scoring`、可选少量 bbox/cross-attention 参数。
- 验证阶段没有交互界面。每轮训练后自动推理全量图片，用真值筛出漏检、误检、定位错误，再把被选中错误图片的真值加入下一轮训练。
- 数据入口支持用户真实 `DataTrain.txt` 格式：文件头 `Version 1.0.0` 会跳过；`图片名:数量 P:4/R:4 x1 y1 ... "label"` 会解析为 polygon；`.jpg.bmp`、`.bmp.bmp` 都按普通图片名处理；`1 1 1 1 1 1 1 1` 是无目标占位，不写入 `full_gt.json`，但图片仍保留在 `image_map.json` 参与全量推理和误检检查。

新增原生少样本主线文件已按职责分包：

- `fewshot_adapter/data/`：标注模型、DataTrain 解析、JSON IO、初始样本选择、训练集更新、SAM3 batch 构造。
- `fewshot_adapter/geometry/`：polygon/OBB 面积、IoU 和 polygon 转 OBB。
- `fewshot_adapter/evaluation/`：预测与真值匹配、错误队列生成、下一轮样本选择。
- `fewshot_adapter/native/`：EfficientSAM3 task prompt / adapter、原生 loss、预测后处理、自动闭环训练。
- `fewshot_adapter/cli/`：命令行实现层。
- `fewshot_adapter/convert_datatrain.py`：兼容入口，内部转发到 `fewshot_adapter.cli.convert_datatrain`。
- `fewshot_adapter/train_native_efficientsam3_fewshot.py`：兼容入口，内部转发到 `fewshot_adapter.cli.train_native`。
- `docs/superpowers/specs/2026-04-29-native-efficientsam3-fewshot-design.md`：本次原生重构规格。
- `docs/superpowers/plans/2026-04-29-native-efficientsam3-fewshot-refactor.md`：本次原生重构实施计划。
- `docs/fewshot_gpu_validation_guide.md`：后续在 GPU 机器上验证少样本闭环的命令、输出检查和常见问题排查。

旧的 proposal、候选特征、prototype head、外置 torch head 相关文件已经删除，避免后续智能体误走旧路线。

典型使用流程：

1. 在仓库根目录用 `pip install -e ".[stage1]"` 安装。
2. 把 SAM3 checkpoint 下载到 `sam3_checkpoints/`。
3. 用 `data/download_*.sh` 或 Python 下载脚本准备数据集。
4. 用 `stage1/save_embedding_*_stage1.py` 导出 teacher embedding。
5. 用 `stage1/train_*_encoder_stage1.py` 训练学生编码器。
6. 用 `stage1/convert_*_weights_stage1.py` 把学生权重转换/合并到 SAM3 兼容 checkpoint。
7. 可选：用 `stage1_geometry_finetune/train_geometry_finetune.py` 做几何感知微调。

## 关键入口

- `README.md`：项目主说明，包含项目背景、安装、推理示例、模型 zoo、训练/评估概览。
- `README_stage1.md`：Stage 1 编码器蒸馏的详细流程。
- `README_stage1_finetune.md`：几何感知微调的详细流程。
- `README_dataset.md`：数据集下载和目录结构说明。
- `pyproject.toml`：根包 `efficientsam3` 的安装配置，会包含 `stage1*` 和 `sam3*`。
- `sam3/sam3/model_builder.py`：SAM3 和 EfficientSAM3 图像/视频模型的主要构建入口。
- `stage1/model.py`：Stage 1 用到的 teacher/student 编码器构建逻辑。
- `stage1/train_image_encoder_stage1.py`：用已保存的 SAM3 图像 teacher embedding 训练轻量图像编码器。
- `stage1/train_text_encoder_stage1.py`：用已保存的 SAM3 文本 teacher embedding 训练 MobileCLIP 风格文本编码器。
- `stage1_geometry_finetune/model.py`：把可训练 student trunk 和冻结的 SAM3 组件包装起来，做 prompt 条件下的 mask 蒸馏。
- `stage1_geometry_finetune/train_geometry_finetune.py`：几何感知微调主训练循环。
- `eval/eval_coco.py`：在 COCO 上做点 prompt 图像分割评估。
- `eval/eval_text_encoder_similarity.py`：比较学生文本特征和 SAM3 teacher 文本特征。

## 架构地图

### Stage 1 蒸馏

`stage1/` 有两条蒸馏分支：

- 图像分支：SA-1B 图像经过 SAM3 图像 trunk，保存 teacher embedding，然后让 RepViT、TinyViT、EfficientViT 等学生 backbone 回归这些 embedding。
- 文本分支：Recap-DataComp 或文本标注数据经过 SAM3 文本编码器，保存 token 级 teacher feature，然后训练 MobileCLIP-S0、MobileCLIP-S1、MobileCLIP2-L 等学生文本编码器。

学生模型主要输出 SAM3 兼容的 256 通道特征表示。训练时使用 masked MSE 加 cosine loss。转换脚本会把学生编码器权重拼接到 SAM3 checkpoint 对应的 key 命名空间里。

### 几何感知微调

`stage1_geometry_finetune/` 接收一个 Stage 1 图像学生模型，用两个损失继续训练：

- Embedding loss：student trunk 输出要匹配已保存的 SAM3 teacher trunk embedding。
- Mask loss：student embedding 和 teacher embedding 都经过同一套冻结的 SAM3 FPN、geometry encoder、transformer、segmentation head，再用 BCE/Dice 风格损失对齐预测 mask。

这一步存在的原因是：SAM3 的几何 prompt 会从 backbone 特征里池化点/框信息，所以 prompt 兼容性不只取决于原始 embedding 相似度，还取决于特征能不能被后续几何模块正确使用。

### 内置 SAM3 包

`sam3/` 大部分是上游 SAM3 包，但这里不是纯第三方代码。EfficientSAM3 在里面扩展了：

- `sam3/sam3/model_builder.py`：EfficientSAM3 图像/视频 builder。
- `sam3/sam3/model/text_encoder_student.py`：学生文本编码器。
- `sam3/sam3/backbones/`：RepViT、TinyViT、EfficientViT、MobileCLIP 等轻量模块。
- `sam3/efficientsam3_examples/`：EfficientSAM3 专用示例。

后续改推理或模型构建时，不要只看根目录的 `stage1/`，一定也要看 `sam3/sam3/model_builder.py`。

## 顶层文件

- `.gitignore`：忽略 checkpoint、数据集、Python 缓存、实验输出和本地临时文件。
- `pyproject.toml`：根项目安装配置；声明 Stage 1 依赖，如 PyTorch、decord、pycocotools、hydra-core、fairscale、mmcv、`segment-anything`。
- `README.md`：EfficientSAM3 主文档，包含项目描述、安装、推理片段、模型 zoo、初步评估、路线图、引用和 license。
- `README_dataset.md`：数据集脚本说明，覆盖 COCO、DAVIS、LVIS、SA-1B、SA-V、LVOS、MOSE、YouTube-VOS、Recap-DataComp、Recap-COCO、SA-Co。
- `README_stage1.md`：Stage 1 蒸馏完整说明，包括 teacher embedding 导出、学生训练、checkpoint 转换。
- `README_stage1_finetune.md`：Stage 1 几何感知微调完整说明。
- `AGENT_CONTEXT.md`：当前这份智能体交接文档。

## `data/`

数据集下载、重组和文本标注辅助脚本。

- `download_coco.sh`：下载/解压 COCO 2017 图像和标注；脚本注释说明部分 `wget` 行可能需要手动取消注释。
- `download_datacomp.py`：从 Hugging Face 下载 1% Recap-DataComp-1B 子集。
- `download_davis.sh`：下载 DAVIS 2016/2017 trainval 和 unsupervised 数据。
- `download_lvis.sh`：下载 LVIS v1 标注和 COCO train/val 图像。
- `download_lvos.sh`：通过 `gdown` 下载 LVOS v2 train/val。
- `download_mose.sh`：从 Hugging Face 下载 MOSE/MOSEv2，并处理 multipart archive。
- `download_recap_coco.sh`：下载 Recap-COCO-30K parquet 数据。
- `download_recap_datacomp.sh`：Recap-DataComp 子集下载的 shell 辅助脚本。
- `download_rf100.py`：Roboflow 100-VL 相关下载/设置的小工具。
- `download_sa_v.sh`：基于 `sa-v.txt` 调用 SA archive 下载逻辑的薄封装。
- `download_sa1b.sh`：从 TSV 列表并行/可续传下载 SA-1B archive。
- `download_ytvos.sh`：通过 `gdown` 下载 YouTube-VOS 2019。
- `reorg_sa1b.py`：把解压后的 SA-1B archive 重组成 Stage 1 期望的 train/val 图像和标注目录。
- `reorg_sav_text.py`：整理 SA-V text/SA-Co 风格标注，供本地评估或训练使用。
- `sa-1b.txt`：完整 SA-1B archive TSV 列表。
- `sa-1b-1p.txt`：1% SA-1B 子集 TSV 列表。
- `sa-1b-10p.txt`：10% SA-1B 子集 TSV 列表。
- `sa-v.txt`：SA-V archive TSV/checksum 来源列表。
- `sa-v-text/sa-co-veval/combine_veval_noun_phrases_only.py`：为 SA-Co/VEval 文本编码器评估提取/合并 noun phrases。
- `sa-v-text/sa-co-veval/saco_veval_noun_phrases.json`：`eval/eval_text_encoder_similarity.py` 使用的 noun phrase 列表。

## `eval/`

- `eval_coco.py`：加载 EfficientSAM3 图像模型，从 COCO mask 采样点 prompt，预测 mask，并报告 IoU/mIoU。
- `eval_text_encoder_similarity.py`：构建 SAM3 teacher 文本编码器和一个或多个学生文本编码器，计算 noun phrases 上的 token 级 cosine similarity。

## `images/`

README 使用的项目媒体资源。

- `efficientsam3.svg`：简版架构图。
- `efficientsam3_full.svg`：README 中使用的完整架构/训练流程图。
- `es-ev-s-teaser.jpg`：EfficientViT-S 图像模型 teaser 结果。
- `es-tv-mc-m-teaser.png`：TinyViT + MobileCLIP 文本 prompt 模型 teaser 结果。

## `personal-site/`

- `personal-site/efficientsam3/.gitignore`：项目网页目录的占位/复制元数据。本次 checkout 中没有实质站点源码。

## `sam3_checkpoints/`

- `config.json`：Hugging Face 风格的 SAM3 video model 配置，描述 detector、text、vision、geometry encoder、mask decoder、tracker 等设置。它不是大模型权重文件。

## `stage1/`

Stage 1 编码器蒸馏包。

- `__init__.py`：标记 `stage1` 为 Python 包。
- `config.py`：Stage 1 的 YACS 默认配置和 CLI override 合并逻辑。
- `convert_both_encoders_weights_stage1.py`：把训练好的图像学生和文本学生 checkpoint 合并成一个 SAM3 兼容 checkpoint。
- `convert_image_encoder_weights_stage1.py`：用训练好的图像学生替换 SAM3 图像编码器 key。
- `convert_text_encoder_weights_stage1.py`：用训练好的文本学生替换 SAM3 文本编码器 key。
- `logger.py`：按 rank 区分的日志工具。
- `lr_scheduler.py`：scheduler builder 和自定义 linear LR scheduler。
- `model.py`：构建 SAM3 teacher 编码器、图像/文本 student 编码器、轻量 backbone adapter 和 EfficientSAM3 wrapper。
- `my_meter.py`：记录 loss/timing 的 AverageMeter 工具。
- `optimizer.py`：optimizer builder 和 weight decay 参数分组。
- `save_embedding_image_stage1.py`：把 teacher 图像 embedding 导出为分片 key/value 二进制文件。
- `save_embedding_text_stage1.py`：把 teacher 文本 embedding 导出为分片 key/value 二进制文件。
- `train_image_encoder_stage1.py`：DDP/AMP 图像学生训练循环，使用 masked MSE/cosine loss。
- `train_text_encoder_stage1.py`：DDP/AMP 文本学生训练循环，使用 token 级 MSE/cosine loss，并支持可选 word permutation。
- `trim_weights.py`：过滤或检查 checkpoint state dict key 的工具。
- `utils.py`：checkpoint 读写、pretrained 加载、AMP scaler、分布式 reduction、loss helper、Git 信息、LR 分组等工具。

### `stage1/configs/`

- `base_stage1.yaml`：Stage 1 训练/导出共享默认配置。
- `es_ev_l.yaml`、`es_ev_m.yaml`、`es_ev_s.yaml`：EfficientViT-B2/B1/B0 图像学生配置。
- `es_rv_l.yaml`、`es_rv_m.yaml`、`es_rv_s.yaml`：RepViT-M2.3/M1.1/M0.9 图像学生配置。
- `es_tv_l.yaml`、`es_tv_m.yaml`、`es_tv_s.yaml`：TinyViT-21M/11M/5M 图像学生配置。
- `es_mc_l.yaml`、`es_mc_m.yaml`、`es_mc_s.yaml`：MobileCLIP2-L、MobileCLIP-S1、MobileCLIP-S0 文本学生配置。
- `es_mc_l_sav.yaml`、`es_mc_m_sav.yaml`、`es_mc_s_sav.yaml`：面向 SA-V/SA-Co 风格文本标注的文本学生配置。
- `es_mc_s_pretrained.yaml`：带 pretrained 初始化的 MobileCLIP-S0 配置。
- `text_l_ctx16.yaml`、`text_l_ctx32.yaml`：MobileCLIP2-L 的 16/32 token context 文本配置。
- `text_s0_ctx16.yaml`、`text_s0_ctx32.yaml`：MobileCLIP-S0 的 16/32 token context 文本配置。
- `text_s1_ctx16.yaml`、`text_s1_ctx32.yaml`：MobileCLIP-S1 的 16/32 token context 文本配置。

### `stage1/configs/teacher/`

- `sam_vit_huge_sa1b.yaml`：SAM3 图像 teacher embedding 导出配置。
- `sam_text_teacher.yaml`：Recap/DataComp 风格数据用的 SAM3 文本 teacher 配置。
- `sam_text_teacher_ctx16.yaml`、`sam_text_teacher_ctx32.yaml`：16/32 active context length 的 SAM3 文本 teacher 配置。
- `sam_text_teacher_sav.yaml`：SA-V/SA-Co 文本标注数据用的 SAM3 文本 teacher 配置。

### `stage1/data/`

- `__init__.py`：标记 data 包。
- `build.py`：根据 `DATA.DATASET` 构建 train/val dataset 和 dataloader。
- `coco_caption_dataset.py`：文本蒸馏用的 COCO captions dataset wrapper。
- `coco_dataset.py`：图像分割风格数据用的 COCO instance dataset wrapper。
- `recap_coco_dataset.py`：Recap-COCO parquet/text dataset wrapper。
- `recap_datacomp_dataset.py`：Recap-DataComp parquet dataset wrapper。
- `sa1b_dataset.py`：图像编码器蒸馏用的 SA-1B 图像/mask 标注 dataset。
- `sampler.py`：自定义 distributed sampler。
- `text_annotations_dataset.py`：SA-Co 类 noun phrase 数据使用的通用文本标注 dataset。
- `transforms.py`：resize、pad、box 格式转换等 transforms。
- `augmentation/aug_random.py`：控制 Python 和 NumPy 随机状态的 context manager。
- `augmentation/dataset_wrapper.py`：augmentation/repeat 管理用 dataset wrapper。
- `augmentation/manager.py`：用于保存或读取 augmentation 元数据的文本管理器。

### `stage1/scripts/`

- `save_image_embeddings.sh`：`save_embedding_image_stage1.py` 的 `torchrun` 启动脚本。
- `save_text_embeddings.sh`：`save_embedding_text_stage1.py` 的 `torchrun` 启动脚本。
- `train_image_student.sh`：图像学生训练的 `torchrun` 启动脚本。
- `train_text_student.sh`：文本学生训练的 `torchrun` 启动脚本。

## `stage1_geometry_finetune/`

面向 prompt 条件的图像编码器微调包。

- `__init__.py`：标记包。
- `config.py`：几何微调的 YACS 默认配置和 CLI override 合并逻辑。
- `convert_geometry_finetune.py`：把几何微调后的 student trunk 权重替换到已合并的 Stage 1 checkpoint 中。
- `losses.py`：Focal/CE/Dice/MSE/cosine loss，以及 `GeometryFinetuningLoss`。
- `model.py`：构建 student trunk，加载冻结 SAM3 组件，应用 FPN，构建点/框 prompt，并执行 dual-path mask prediction。
- `train_geometry_finetune.py`：DDP/AMP 训练和验证循环、prompt 采样、迭代 refinement 点采样、checkpoint 保存。
- `utils.py`：共享 CLI 参数工具。

### `stage1_geometry_finetune/configs/`

- `base_geometry_finetune.yaml`：几何微调共享默认配置。
- `base_geometry_finetune_edgesam_style.yaml`：EdgeSAM 风格的替代默认配置。
- `es_ev_l.yaml`、`es_ev_m.yaml`、`es_ev_s.yaml`：EfficientViT 图像几何微调配置。
- `es_rv_l.yaml`、`es_rv_m.yaml`、`es_rv_m_edgesam_style.yaml`、`es_rv_s.yaml`：RepViT 图像几何微调配置。
- `es_tv_l.yaml`、`es_tv_m.yaml`、`es_tv_s.yaml`：TinyViT 图像几何微调配置。

### `stage1_geometry_finetune/data/`

- `__init__.py`：标记 data 包。
- `build.py`：构建 SA-1B prompt 条件 dataloader。
- `sa1b_prompt_dataset.py`：从 SA-1B mask 中采样 box/point，供几何微调使用的 dataset。

### `stage1_geometry_finetune/scripts/`

- `train_geometry_finetune.sh`：几何微调的 `torchrun` 启动脚本。

## `sam3/`

内置的上游 SAM3 包，加上 EfficientSAM3 扩展。

### 包元数据和文档

- `sam3/.gitignore`：嵌套 SAM3 包的忽略规则。
- `sam3/CODE_OF_CONDUCT.md`：上游 SAM3 社区行为准则。
- `sam3/CONTRIBUTING.md`：上游贡献指南。
- `sam3/LICENSE`：上游 SAM3 license。
- `sam3/MANIFEST.in`：打包时包含 assets 的 manifest。
- `sam3/pyproject.toml`：嵌套 `sam3` 包的元数据和 optional dependency 组。
- `sam3/README.md`：上游 SAM3 用户文档。
- `sam3/README_TRAIN.md`：上游 SAM3 fine-tuning/evaluation 训练说明。
- `sam3/__init__.py`：仓库层面的嵌套 package marker。
- `sam3/.github/workflows/format.yml`：嵌套包格式化 workflow。

### `sam3/assets/`

静态资源和小 demo：

- `bpe_simple_vocab_16e6.txt.gz`：SAM3/CLIP 风格文本编码器使用的 BPE tokenizer 词表。
- `dog_person.jpeg`、`groceries.jpg`、`test_image.jpg`、`truck.jpg`：示例图片。
- `dog.gif`、`player.gif`、`model_diagram.png`、`sa_co_dataset.jpg`、`saco_gold_annotation.png`：README/demo 可视化资源。
- `veval/toy_gt_and_pred/*.json`：很小的 SA-Co/VEval toy ground truth、prediction、result JSON。
- `videos/bedroom.mp4`：示例视频。
- `videos/0001/*.jpg`：从示例视频抽出的连续帧，用于 notebook/demo。

### `sam3/efficientsam3_examples/`

EfficientSAM3 专用示例：

- `efficientsam3_for_sam1_task_example.py` / `.ipynb`：兼容 SAM1 风格任务的图像点/框 prompt 示例。
- `efficientsam3_for_sam2_video_task_example.ipynb`：兼容 SAM2 风格交互的视频任务示例。
- `efficientsam3_image_predictor_example.py` / `.ipynb`：EfficientSAM3 图像 predictor 示例。
- `efficientsam3_litetext_image_inference_example.py`：LiteText 图像推理示例。
- `efficientsam3_litetext_video_predictor_example.py`：LiteText 视频 predictor 示例。
- `efficientsam3_video_predictor_example.ipynb`：EfficientSAM3 视频 predictor 教程。
- `run_sam3_point_prompt.py`：最小点 prompt 推理脚本。
- `run_sam3_text_prompt.py`：最小文本 prompt 推理脚本。

### `sam3/examples/`

上游 SAM3 notebook 示例，覆盖图像/视频 predictor、SA-Co 可视化/评估、batched inference、交互式图像使用和 agent 风格工作流。

### `sam3/evaluation/`

- `eval_coco.py`：嵌套/旧版 COCO 评估入口，目的和顶层 `eval/eval_coco.py` 类似。

### `sam3/sam3/`

核心 Python 包。

- `__init__.py`：包元数据。
- `device.py`：选择 torch device 和 autocast dtype。
- `logger.py`：彩色 logger 工具。
- `model_builder.py`：构建 SAM3 图像/视频模型、EfficientSAM3 图像/视频模型、学生视觉 backbone、学生文本编码器、checkpoint 加载和 tracker wrapper。
- `visualization_utils.py`：mask、box、point、video、COCO/SA-Co 可视化工具。

### `sam3/sam3/agent/`

MLLM/SAM3 agent 工具：

- `agent_core.py`：多轮 agent 循环和 debug transcript 管理。
- `client_llm.py`：向 LLM endpoint 发送图像/文本请求。
- `client_sam3.py`：本地或服务化调用 SAM3 inference。
- `inference.py`：单图 agent inference wrapper。
- `viz.py`：agent 结果可视化。
- `helpers/*.py`：agent 工作流使用的 box、mask、RLE、keypoint、ROI align、color、visualization、overlap removal、zoom-in、memory 工具。
- `system_prompts/*.txt`：agent 循环使用的系统 prompt。

### `sam3/sam3/backbones/`

轻量 backbone 和上游兼容 backbone：

- `mobile_clip.py`：LiteText/学生文本编码器使用的 MobileCLIP text transformer 组件。
- `repvit.py`：RepViT backbone 定义。
- `tiny_vit.py`：TinyViT backbone 定义。
- `efficientvit/`：EfficientViT backbone、分类/分割/SAM wrapper、神经网络 op、norm、activation、drop、Triton RMS norm 和工具函数。

### `sam3/sam3/model/`

SAM3 模型内部模块：

- `decoder.py`、`encoder.py`：Transformer decoder/encoder 组件。
- `geometry_encoders.py`：点/框几何 prompt 编码。
- `maskformer_segmentation.py`：Pixel decoder 和 segmentation head。
- `memory.py`：tracking 用 mask memory 模块。
- `necks.py`、`vitdet.py`、`position_encoding.py`：视觉 trunk/neck/position encoding。
- `sam3_image.py`、`sam3_image_processor.py`：图像模型和高级图像 processor。
- `sam3_video_base.py`、`sam3_video_inference.py`、`sam3_video_predictor.py`：视频 inference 和 predictor wrapper。
- `sam3_tracker_base.py`、`sam3_tracker_utils.py`、`sam3_tracking_predictor.py`：tracking 模型和工具。
- `text_encoder_ve.py`、`text_encoder_student.py`、`tokenizer_ve.py`：SAM3 文本编码器、学生文本编码器和 tokenizer。
- `vl_combiner.py`：组合视觉 backbone 和语言 backbone。
- `sam1_task_predictor.py`：SAM1 风格交互 predictor 兼容层。
- `student_sam/`：学生 SAM/SAM1 兼容模型、automatic mask generator、predictor、config、backbone 和 modeling 工具。
- `utils/`：SAM1/SAM2 兼容工具。
- `box_ops.py`、`data_misc.py`、`edt.py`、`io_utils.py`、`model_misc.py`、`act_ckpt_utils.py` 等：数学、IO、activation checkpoint 等支撑工具。

### `sam3/sam3/eval/`

评估实现：

- COCO、cgF1、SA-Co/VEval、YouTube-VIS wrapper、输出 writer、postprocessor、格式转换工具。
- `hota_eval_toolkit/` 和 `teta_eval_toolkit/`：tracking/video segmentation 指标实现。

### `sam3/sam3/perflib/`

性能 kernel 和加速工具：

- `associate_det_trk.py`、`connected_components.py`、`masks_ops.py`、`nms.py`：mask、connected components、NMS 等优化操作。
- `compile.py`、`fa3.py`：编译/FlashAttention 相关工具。
- `triton/*.py`：connected components 和 NMS 的 Triton 实现。
- `tests/tests.py`：perflib 基础测试。

### `sam3/sam3/sam/`

SAM 风格 prompt encoder 和 mask decoder 组件：

- `common.py`、`mask_decoder.py`、`prompt_encoder.py`、`rope.py`、`transformer.py`。

### `sam3/sam3/train/`

上游 SAM3 训练框架：

- `train.py`：Hydra/submitit 入口。
- `trainer.py`：主 trainer，负责 AMP、checkpoint、logging、分布式编排。
- `matcher.py`、`masks_ops.py`、`nms_helper.py`：训练时 matching、mask op、NMS 工具。
- `configs/`：gold/silver image eval、ODinW13、Roboflow V100、SA-Co video eval 等配置。
- `data/`：COCO JSON loader、image/video dataset、collator、通用 torch dataset wrapper。
- `loss/`：SAM3 loss、mask sampling、focal loss。
- `optim/`：optimizer 和 scheduler 工具。
- `transforms/`：图像/视频数据增强、query 过滤、point sampling、segmentation transform。
- `utils/`：checkpoint、distributed、logging、train utility。

### `sam3/scripts/`

结果提取和评估数据准备脚本：

- `extract_odinw_results.py`、`extract_roboflow_vl100_results.py`：聚合 benchmark 结果。
- `eval/standalone_cgf1.py`：独立 cgF1 评估。
- `eval/gold/`：Gold benchmark 评估脚本和 README。
- `eval/silver/`：Silver benchmark 下载、预处理、抽帧脚本和 README。
- `eval/veval/`：SA-Co/VEval 下载、抽帧准备、标注更新脚本和 README。

## 常见修改入口

- 如果要继续少样本/自动闭环原型，优先看新增的 `fewshot_adapter/` 包。它是当前面向产品验证的新代码，不属于原 EfficientSAM3 论文训练流程。
- 如果要新增或修改 EfficientSAM3 推理行为，先看 `sam3/sam3/model_builder.py`。
- 如果要新增图像学生 backbone，要同时检查/修改 `stage1/model.py`、`stage1_geometry_finetune/model.py` 和 `sam3/sam3/model_builder.py` 里的 EfficientSAM3 builder。
- 如果要新增文本学生编码器变体，要检查/修改 `stage1/model.py`、`sam3/sam3/model_builder.py` 和 `stage1/configs/` 下的配置。
- 如果要改 Stage 1 loss 或 embedding 处理，先看 `stage1/train_image_encoder_stage1.py`、`stage1/train_text_encoder_stage1.py` 和 `stage1/utils.py`。
- 如果要改 prompt 条件微调，先看 `stage1_geometry_finetune/model.py`、`stage1_geometry_finetune/losses.py` 和 `stage1_geometry_finetune/train_geometry_finetune.py`。
- 如果要改数据路径或目录结构假设，检查 `README_dataset.md`、`data/reorg_*.py` 和 `stage1/data/` 下的 dataset wrapper。

## `fewshot_adapter/`

这是为“少样本、自动闭环、后续可接 EfficientSAM3 Adapter/NPU 部署”的产品验证新增的轻量包。当前版本已经清理掉旧的 proposal / 区域特征 / 外置 head 路线，主线只保留完整 EfficientSAM3 原生 decoder + task visual prompt / adapter 微调。

核心文件和类：

- `data/models.py`：统一 HBB、OBB、polygon、mask 标注和预测结构，并提供 HBB/OBB/polygon 互转辅助。
- `data/datatrain.py`：`DataTrainDataset`，解析 `DataTrain.txt`，支持 `Version 1.0.0` 文件头、`图片名:数量`、`P:4/R:4` 四点标注和无目标占位过滤，构建 `image_map`，保存 `full_gt.json` / `image_map.json`。
- `data/json_io.py`：`AnnotationJsonIO`，读取/保存标注、预测和错误队列 JSON。
- `data/sampling.py`：`InitialTrainSelector` 和 `TrainSetUpdater`，负责第 0 轮样本选择与增量训练集更新。
- `data/sam3_batch.py`：`Sam3BatchBuilder`，把图片和 `Annotation` 转成 SAM3 原生 batch / target。
- `geometry/ops.py`：`GeometryOps`，polygon/OBB 面积、IoU 和 polygon 转 OBB。
- `evaluation/matching.py`：`DetectionMatcher` 和 `ErrorSelector`，负责匹配、筛错、选择下一轮样本。
- `native/adapter.py`：task prompt、prompt adapter、冻结/解冻策略和 EfficientSAM3 原生 wrapper。
- `native/loss.py`：`NativeLossFactory`，封装 SAM3 原生 matcher/loss。
- `native/predictor.py`：`NativePredictor`，把 SAM3 原生输出转成项目 `Prediction`。
- `native/trainer.py`：`NativeFewShotTrainer`，完整多轮自动训练、推理、筛错、补样本闭环。
- `cli/convert_datatrain.py`：把 `DataTrain.txt + 图片目录` 一键转换为 `full_gt.json`、`image_map.json`。
- `cli/train_native.py`：当前推荐训练 CLI 的实现层。
- `utils/torch.py`：PyTorch 懒加载工具；无 torch 环境会给清晰错误。

如果原始数据只有图片目录和 `DataTrain.txt`，先运行转换：

```powershell
python -m fewshot_adapter.convert_datatrain `
  --datatrain DataTrain.txt `
  --image-dir images `
  --output-dir dataset_json
```

它会输出：

- `dataset_json/full_gt.json`：全量真值，每个目标一条 Annotation。
- `dataset_json/image_map.json`：图片名到图片实际路径的映射。

注意：`1 1 1 1 1 1 1 1` 表示无目标占位，只会让该图片进入 `image_map.json`，不会生成目标标注。

然后运行原生 EfficientSAM3 少样本闭环：

```powershell
python -m fewshot_adapter.train_native_efficientsam3_fewshot `
  --full-ground-truth full_gt.json `
  --image-map image_map.json `
  --checkpoint efficient_sam3_efficientvit_s.pt `
  --output-root runs/native_fewshot `
  --label target `
  --device cuda `
  --max-rounds 10 `
  --steps-per-round 80
```

当前本地环境没有安装 PyTorch，所以单元测试只覆盖轻量路径和清晰失败提示；真实训练需要用户在 GPU 环境验证。

GPU 验证时优先看 `docs/fewshot_gpu_validation_guide.md`。第一步先跑 `--max-rounds 1 --steps-per-round 1` 的 smoke test，确认模型加载、前向、反向、预测和错误队列都能走通，再扩大轮数和数据规模。

最近一次外部 code review 后已处理的点：

- `DataTrainDataset` 转换阶段会提前检查图片文件是否存在。
- 多类别数据如果没有显式传 `--label` 会报错，避免静默取第一类。
- 空文本语言特征占位 shape 已随 batch size 生成。
- 错误匹配按 `(image_id, label)` 分组，避免大数据量时全量交叉遍历。
- 单轮训练会按训练图片缓存 SAM3 batch，减少重复读图和 GPU 拷贝。
- 训练 loss 出现 NaN/inf 会直接报错。
- 预测 OBB 已在代码和文档中明确为 angle=0 基线，不是真实旋转框能力。

## 容易踩坑的地方

- 根包会同时安装 `stage1*` 和 `sam3*`；嵌套的 `sam3/` 包是运行路径的一部分。
- `sam3/sam3/model_builder.py` 同时有上游 SAM3 和 EfficientSAM3 函数。新增 builder 前先在这里找。
- 文本编码器的 context length 和 positional embedding table size 是故意分开的。当前默认是 fixed-table/slice 行为；插值只用于旧 checkpoint。
- 几何微调需要 SAM3 checkpoint、Stage 1 权重和 teacher embeddings。它不会重新导出 teacher embeddings。
- 很多脚本默认 Linux 风格环境：`bash`、`torchrun`、CUDA/DDP、大型本地数据集。在 Windows 上可能需要 WSL/Git Bash，或直接调用 Python 脚本。
- 大型数据集和 checkpoint 没有提交到仓库。`sam3_checkpoints/config.json` 只是配置，不是模型权重。
- 当前桌面环境里 `rg` 可能无法运行；可用 PowerShell 的 `Get-ChildItem` 和 `Select-String` 作为替代。
