### EfficientSAM3: Progressive Hierachical Knowledge Distillation (PhD) from SAM1, 2 and 3
[Chengxi Simon Zeng](https://simonzeng7108.github.io/about/)<sup>1,†</sup>, [Yuxuan Jiang](https://YuxuanJJ.github.io/)<sup>1</sup>, [Gao Ge](https://scholar.google.com/citations?user=j2_80ewAAAAJ&hl=en)<sup>1</sup>, [Shuai Wang](https://shuaiwang97.github.io/)<sup>2</sup>, [Duolikun Danier](https://danier97.github.io/)<sup>3</sup>, [Bin Zhu](https://binzhubz.github.io/)<sup>4</sup>, [Stevan Rudinac](https://stevanrudinac.com/)<sup>2</sup>, [David Bull](https://david-bull.github.io/)<sup>1</sup>, [Fan Aaron Zhang](https://fan-aaron-zhang.github.io/)<sup>1</sup>
<sup>1</sup>Visual Information Lab, University of Bristol; <sup>2</sup>MultiX lab, University of Amsterdam; <sup>3</sup>University of Edinburgh; <sup>4</sup>Singapore Management University

<sup>†</sup>Tech Lead & Corresponding Author

[![arXiv](https://img.shields.io/badge/arXiv-EfficientSAM3-b31b1b.svg)](https://arxiv.org/abs/2511.15833) [![arXiv](https://img.shields.io/badge/arXiv-SAM3--LiteText-b31b1b.svg)](https://arxiv.org/abs/2602.12173) [![Project Page](https://img.shields.io/badge/Project-Page-green)](https://simonzeng7108.github.io/efficientsam3/) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-EfficientSAM3-blue)](https://huggingface.co/Simon7108528/EfficientSAM3) [![Discord](https://img.shields.io/badge/Discord-Join-7289da?logo=discord&logoColor=white)](https://discord.gg/FMyaQca7xT)
---
## Updates
- **[2026/04/19]** **SAM3-LiteText** is live on HuggingFace main branch, [[Docs](https://huggingface.co/docs/transformers/main/en/model_doc/sam3_lite_text)], [[Model](https://huggingface.co/yonigozlan/sam3-litetext-s0)] and accepted by [ICMR2026](https://icmr2026.org/)- Same performance to SAM3 but much smaller! Thanks to @NielsRogge, @yonigozlan and HF integration team.
- **[2026/04/13]** **EfficientSAM3.1** and **SAM3.1-LiteText** image models were released on the [`stage1_sam3.1`](https://github.com/SimonZeng7108/efficientsam3/tree/data_engine) branch. SAM3-LiteText has also been officially merged into [HuggingFace Transformers](https://github.com/huggingface/transformers/pull/44320). Stage 3 data engine support is now available on the [`data_engine`](https://github.com/SimonZeng7108/efficientsam3/tree/data_engine/data_engine) branch.
- **[2026/02/18]** **SAM3-LiteText** released! SAM3-LiteText reduces text encoder parameters by up to 88% with similar performance to the original text encoder. [Paper](https://arxiv.org/abs/2602.12173) available on arXiv. Code available in [`sam3_litetext`](https://github.com/SimonZeng7108/efficientsam3/tree/sam3_litetext) branch and weights on [Hugging Face](https://huggingface.co/Simon7108528/EfficientSAM3/tree/main/sam3_litetext).
- **[2026/01/11]** Stage 1 geometry-prompt fine-tuned (**ft**) weights released/updated (image encoders on 1% SA-1B; text encoders fine-tuned on SA-Co Gold+Silver).
- **[2025/12/08]** Stage 1 text encoder weights released for all 3 variants (MobileCLIP S0, S1, and MobileCLIP2 L) - distilled on 1% Recap-DataComp-1B dataset.
- **[2025/12/02]** Stage 1 image encoder weights released for all 9 variants (RepViT, TinyViT, EfficientViT) - unsupervised distilled on 1% of SA-1B dataset.
- **[2025/11/25]** Teaser model released. See Above. More models are baking in the oven🔥.
- **[2025/10/18]** Project announced. Code and weights are not released yet; they will be published once SAM3 code is publicly available.
---


## Table of Contents

- [Table of Contents](#table-of-contents)
- [Updates](#updates)
- [Installation](#installation)
- [Inference](#inference)
- [Training and Evaluation](#training-and-evaluation)
- [Datasets](#datasets)
- [EfficientSAM3 Model Zoo \& Weight Release](#efficientsam3-model-zoo--weight-release)
- [Preliminary Evaluation](#preliminary-evaluation)
- [CoreML / ONNX Export](#coreml--onnx-export)
- [Web Demo](#web-demo)
- [Development To-Do List](#development-to-do-list)
- [Call for Pull Requests](#call-for-pull-requests)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Users](#users)

---

[SAM3](https://github.com/facebookresearch/sam3) (Segment Anything Model 3) has introduced powerful **Promptable Concept Segmentation (PCS)** capabilities, enabling semantic understanding and temporal object tracking beyond traditional mask generation. However, SAM3's massive vision backbone and dense memory bank make it impractical for real-time, on-device applications where computational resources and latency constraints are critical.

**EfficientSAM3** addresses this challenge by distilling SAM3's capabilities into lightweight architectures suitable for edge devices, enabling high-quality concept segmentation on mobile phones, embedded systems, and resource-constrained platforms.

<p align="center">
  <img src="images/efficientsam3_full.svg" alt="EfficientSAM3 Architecture" width="100%">
</p>


---



<details>
<summary>Supported Models and Architecture</summary>

| Component | Model/Backbone | Purpose |
|-----------|----------------|---------|
| **Teacher Models** | [SAM](https://github.com/facebookresearch/segment-anything) (Segment Anything Model) | Foundation for image-level encoder distillation |
| | [SAM2](https://github.com/facebookresearch/sam2) | Temporal memory and video tracking distillation |
| | [SAM3](https://github.com/facebookresearch/sam3) | Promptable Concept Segmentation (PCS) capabilities |
| **Datasets** | [SA-1B](https://ai.meta.com/datasets/segment-anything/) | Image segmentation dataset |
| | [SA-V](https://ai.meta.com/datasets/segment-anything-video/) | Video object segmentation dataset |
| | [SA-Co/Gold](https://huggingface.co/datasets/facebook/SACo-Gold) | Promptable concept segmentation benchmark |
| | [Recap-DataComp-1B](https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B) | Large-scale image-text dataset for text encoder distillation |
| **Student Backbones (Image)** | [RepViT](https://github.com/THU-MIG/RepViT) (M0.9, M1.1, M2.3) | Mobile-optimized Vision Transformer for highest throughput |
| | [TinyViT](https://github.com/wkcn/TinyViT) (5M, 11M, 21M) | Balanced efficiency and performance |
| | [EfficientViT](https://github.com/mit-han-lab/efficientvit) (B0, B1, B2) | Ultra-lightweight architectures for minimal latency |
| **Student Backbones (Text)** | [MobileCLIP](https://github.com/apple/ml-mobileclip) S0 | Lightweight text encoder (42.57M params) |
| | [MobileCLIP](https://github.com/apple/ml-mobileclip) S1 | Balanced text encoder (63.56M params) |
| | [MobileCLIP2](https://github.com/apple/ml-mobileclip) L | Larger text encoder (123.6M params) |


</details>

---

<details>
<summary>Three-Stage Progressive Training Curriculum</summary>

EfficientSAM3 is trained through a three-stage progressive distillation:

### Stage 1: Encoder Distillation (Image-Level Segmentation)

- Distill the SAM3 image encoder to nine student backbones (3 RepViT × 3 TinyViT × 3 EfficientViT variants)
- Distill the SAM3 text encoder to three student text encoders (MobileCLIP S0, S1, 2-L variants)
- Use [SA-1B](https://ai.meta.com/datasets/segment-anything/) dataset with Prompt-in-the-Loop Distillation for image encoder distillation
- Use [Recap-DataComp-1B](https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B) dataset for text encoder distillation
- Align student backbone features with teacher encoder outputs.

### Stage 2: Temporal Memory Distillation (Video Tracking)

- Replace SAM3's dense memory bank with a compact Perceiver-based memory module (adapted from EdgeTAM)
- Distill memory-conditioned mask predictions using [SA-V](https://ai.meta.com/datasets/segment-anything-video/) dataset
- Train the Perceiver module to compress and retrieve spatiotemporal features efficiently

### Stage 3: End-to-End Fine-Tuning (Concept Segmentation)

- Refine the complete EfficientSAM3 pipeline using SAM3 official dataset
- Joint optimization of distilled encoder + compressed memory + mask decoder
- Preserve Promptable Concept Segmentation capabilities while maintaining efficiency

### tl;dr
Stage 1: We distill the SAM3 encoder using SAM1 data. <br>
Stage 2: We align the distilled encoder to a perceiver and an efficient memory bank using SAM2 data. <br>
Stage 3: We fine-tune the complete pipeline using SAM3 data. <br>

</details>


---

## Installation

EfficientSAM3 purposely shares the same software contract as upstream SAM3:

- **Python** ≥ 3.12
- **PyTorch** 2.7.0
- **Device**: NVIDIA GPU (CUDA), Apple Silicon (MPS), or CPU

For non-CUDA platforms (MPS/CPU), install `scipy` for distance transform operations:
```bash
pip install scipy
```

Follow the exact environment setup from the [official SAM3 README](sam3/README.md) or use the condensed steps below:


```bash
git clone https://github.com/SimonZeng7108/efficientsam3.git
cd efficientsam3

conda create -n efficientsam3 python=3.12 -y
conda activate efficientsam3

pip install --upgrade pip

# Install PyTorch (choose one based on your device):
# CUDA (default):
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# MPS/CPU (Apple Silicon or CPU-only):
pip install torch==2.7.0 torchvision torchaudio

# Install repo dependencies via the root pyproject (brings in SAM3 + Stage-1 extras)
pip install -e ".[stage1]"

# Note: the Stage-1 extra includes the SAM1 package dependency
# (PyPI name: segment-anything, import name: segment_anything).
# If your environment cannot resolve it from PyPI, install the vendored repo instead:
# pip install -e ./segment-anything
```

---

## Inference

Download checkpoints from the [Model Zoo](#efficientsam3-model-zoo--weight-release) section. All Stage 1 image encoder weights are available via Google Drive and Hugging Face links in the table below.

**Quick Start (Image Segmentation):**
#### 🔥 Teaser Image Model
<p align="center">
  <img src="https://github.com/SimonZeng7108/efficientsam3/blob/main/images/es-ev-s-teaser.jpg" width="30%">
</p>

 **EfficientViT-S (0.68M params)** distilled from **SAM3 Encoder (461.84M)** — **99.85% smaller**, trained on **1% SA-1B**.
 
```python
from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load model
model = build_efficientsam3_image_model(
  checkpoint_path="efficient_sam3_efficientvit_s.pt",
  backbone_type="efficientvit",
  model_name="b0",
  enable_inst_interactivity=True,
)

# Process image and predict
processor = Sam3Processor(model)
inference_state = processor.set_image(image)

# Single positive point prompt (x, y) in pixels
points = [[image.size[0] / 2, image.size[1] / 2]]
labels = [1]
masks, scores, _ = model.predict_inst(
    inference_state, 
    point_coords=points, 
    point_labels=labels
)
```

#### 🔥 Teaser Text Prompt Model
<p align="center">
  <img src="https://github.com/SimonZeng7108/efficientsam3/blob/main/images/es-tv-mc-m-teaser.png" width="30%">
</p>

 **MobileCLIP-S1 (63.56M)** distilled from **SAM3 Text Encoder (353.72M)**.



```python
from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load model with student text encoder
# Default recommended setup:
# - fixed training checkpoint
# - slice-based inference
# - eval context matches training context
model = build_efficientsam3_image_model(
    checkpoint_path="efficient_sam3_tinyvit_m_mobileclip_s1.pt",
    backbone_type="tinyvit",
    model_name="11m",
    text_encoder_type="MobileCLIP-S1",
    text_encoder_context_length=16,
    text_encoder_pos_embed_table_size=16,
    interpolate_pos_embed=False,
)

# Process image and predict with text prompt
processor = Sam3Processor(model)
inference_state = processor.set_image(image)
inference_state = processor.set_text_prompt(prompt="shoe", state=inference_state)
masks = inference_state["masks"]
scores = inference_state["scores"]
print(len(scores), scores)
```

For an older interpolation-style checkpoint, use:

```python
model = build_efficientsam3_image_model(
    checkpoint_path="legacy_interp_checkpoint.pt",
    backbone_type="tinyvit",
    model_name="11m",
    text_encoder_type="MobileCLIP-S1",
    text_encoder_context_length=16,
    text_encoder_pos_embed_table_size=77,
    interpolate_pos_embed=True,
)
```

For detailed examples including point/box prompts, batched inference, and more, see [sam3/efficientsam3_examples/efficientsam3_for_sam1_task_example.py](sam3/efficientsam3_examples/efficientsam3_for_sam1_task_example.py). For text prompt inference, see [sam3/efficientsam3_examples/efficientsam3_image_predictor_example.ipynb](sam3/efficientsam3_examples/efficientsam3_image_predictor_example.ipynb).

---

## Training and Evaluation

**Training:**
- For Stage 1 encoder distillation training details, see [README_stage1.md](README_stage1.md). For Stage 1 geometry fine-tuning, check the `stage1_geometry_finetune` branch.
- Stage 2 and Stage 3 training details coming soon.

**Evaluation:**
- To evaluate models on COCO dataset:
  ```bash
  python eval/eval_coco.py --coco_root data/coco --output_dir output
  ```

- To evaluate text encoder quality (token-level cosine similarity vs SAM3 teacher):
  ```bash
  python eval/eval_text_encoder_similarity.py \
    --student-ckpt /path/to/student_text_encoder_1.pth /path/to/student_text_encoder_2.pth \
    --np-json data/sa-v-text/sa-co-veval/saco_veval_noun_phrases.json \
    --device cuda
  # Optional: override teacher checkpoint
  #   --teacher-ckpt /path/to/sam3_teacher_checkpoint.pt
  ```

---

## Datasets

For dataset setup and download scripts (`data/download_*.sh`) covering COCO, DAVIS, LVIS, SA-1B, SA-V, LVOS, MOSE, and YouTube-VOS, see:

- [README_dataset.md](README_dataset.md)

---


## EfficientSAM3 Model Zoo & Weight Release

### SAM3 Text Encoder + EfficientSAM3 Image Encoder Models

| Model Name | Backbone | Parameters | Stage 1 Weights<br/>(Encoder Distilled) | Stage 2 Weights<br/>(Memory Module Trained) | Stage 3 Weights<br/>(End-to-End Fine-Tuned) |
|------------|----------|------------|----------------------------------------|---------------------------------------------|---------------------------------------------|
| **ES-RV-S** | RepViT-M0.9 | 4.72M | [GDrive](https://drive.google.com/file/d/1lVvPPoIVDhCFGte-1E_dr4X5EKbE5xKq/view?usp=drive_link) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_repvit_s.pt) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-RV-M** | RepViT-M1.1 | 7.77M | [GDrive](https://drive.google.com/file/d/1JW3KiTnYF2r8nIijf8UXrKXJwf5D5s-5/view?usp=drive_link) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_repvit_m.pt) (ft: [GDrive](https://drive.google.com/file/d/1FYpWqSOY_iZcfk_Q07s4laUZXgFUK5tZ/view?usp=sharing), [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_repvit_m_geo_ft.pt)) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-RV-L** | RepViT-M2.3 | 22.40M | [GDrive](https://drive.google.com/file/d/1ocAkz6DgkaKCKpLdalq2Ya8X6VIMrLLI/view?usp=drive_link) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_repvit_l.pt) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-TV-S** | TinyViT-5M | 5.07M | [GDrive](https://drive.google.com/file/d/1CDfJTd2fTKJTV5nsfYLAV_CGMfQ-AWXS/view?usp=drive_link) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_tinyvit_s.pt) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-TV-M** | TinyViT-11M | 10.55M | [GDrive](https://drive.google.com/file/d/1TX70zw7SduQRZP6hce6MIxEOsdoooZFB/view?usp=drive_link) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_tinyvit_m.pt) (ft: [GDrive](https://drive.google.com/file/d/1NgctD7y_ylE1P6ULjXXywcfmNZPJ_lA1/view?usp=sharing), [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_tinyvit_m_geo_ft.pt)) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-TV-L** | TinyViT-21M | 20.62M | [GDrive](https://drive.google.com/file/d/19hyKjjZ4_8ldmxIAm6D8e8z89xX-M3hZ/view?usp=drive_link) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_tinyvit_l.pt) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-EV-S** | EfficientViT-B0 | 0.68M | [GDrive](https://drive.google.com/file/d/1EnA581iSExZRRWlI6oY-wXTgX4gESijG/view?usp=drive_link) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_efficientvit_s.pt) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-EV-M** | EfficientViT-B1 | 4.64M | [GDrive](https://drive.google.com/file/d/14CRA3LhquUkf8prrKfI1INyHtCw6buvm/view?usp=sharing) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_efficientvit_m.pt) (ft: [GDrive](https://drive.google.com/file/d/1cNF1mB2of7tyP32vs5M2kS8DfW1NkTAG/view?usp=sharing), [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_efficientvit_m_geo_ft.pt)) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-EV-L** | EfficientViT-B2 | 14.98M | [GDrive](https://drive.google.com/file/d/1Zg0Er0LwYYNCFJezSUSlQ8L645cR1OhN/view?usp=drive_link) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_efficientvit_l.pt) | $$\text{Planned}$$ | $$\text{Planned}$$ |

> **Note (2025/12/02):** The current Stage 1 image encoder weights are distilled on 1% of the SA-1B dataset.

> **Note (2026/01/11):** The fine-tuned (**ft**) models use geometry-prompt fine-tuning on the same 1% subset of SA-1B; see training details in the `stage1_geometry_finetune` branch.

### EfficientSAM3 Text Encoder + EfficientSAM3 Image Encoder Models

| Model Name | Backbone | Parameters | Stage 1 Weights<br/>(Encoder Distilled) | Stage 2 Weights<br/>(Memory Module Trained) | Stage 3 Weights<br/>(End-to-End Fine-Tuned) |
|------------|----------|------------|----------------------------------------|---------------------------------------------|---------------------------------------------|
| **ES-RV-S-MC-S1** | RepViT-M0.9 & MobileCLIP-S1 | 4.72M + 63.56M | [GDrive](https://drive.google.com/file/d/1SvBPDqeEYCKpOui79tCIl3c1nMnQijL-/view?usp=sharing) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_repvit-m0_9_mobileclip_s1.pth) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-RV-M-MC-S1** | RepViT-M1.1 & MobileCLIP-S1 | 7.77M + 63.56M | [GDrive](https://drive.google.com/file/d/10VB-1IYAO3iqGq3U63xcM9uGyrqGGvAi/view?usp=sharing) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_repvit-m1_1_mobileclip_s1.pth) (ft: [GDrive](https://drive.google.com/file/d/1HL0qSgB8Z5NjJJHdSfdZUaLCKJPCZR_x/view?usp=sharing), [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth)) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-RV-L-MC-S1** | RepViT-M2.3 & MobileCLIP-S1 | 22.40M + 63.56M | [GDrive](https://drive.google.com/file/d/1IxcVq1BBlMF2LNJ2uljQ8ajLpOWKFpUr/view?usp=sharing) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_repvit-m2_3_mobileclip_s1.pth) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-TV-S-MC-S1** | TinyViT-5M & MobileCLIP-S1 | 5.07M + 63.56M | [GDrive](https://drive.google.com/file/d/1EtG6j3pGtaf-taxo5NLGCqZ8QvOgdLAn/view?usp=sharing) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_tinyvit_5m_mobileclip_s1.pth) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-TV-M-MC-S1** | TinyViT-11M & MobileCLIP-S1 | 10.55M + 63.56M | [GDrive](https://drive.google.com/file/d/1dz5bl0RkCbEUjeK54azREbkEQA-hW_IG/view?usp=sharing) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_tinyvit_11m_mobileclip_s1.pth) (ft: [GDrive](https://drive.google.com/file/d/1rH-tAKNfhrIPCGDbdLPdWWxxx2GHnaaS/view?usp=sharing), [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_tiny_vit_11m_mobileclip_s1_ft.pth)) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-TV-L-MC-S1** | TinyViT-21M & MobileCLIP-S1 | 20.62M + 63.56M | [GDrive](https://drive.google.com/file/d/1DIeJmFle_tHAUKWbycxrNQAW0peZAuy4/view?usp=sharing) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_tinyvit_21m_mobileclip_s1.pth) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-EV-S-MC-S1** | EfficientViT-B0 & MobileCLIP-S1 | 0.68M + 63.56M | [GDrive](https://drive.google.com/file/d/1pa4wKJysp2dUVkUMTrJ7Rs8ZVPIHntFL/view?usp=sharing) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_efficientvit-b0_mobileclip_s1.pth) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-EV-M-MC-S1** | EfficientViT-B1 & MobileCLIP-S1 | 4.64M + 63.56M | [GDrive](https://drive.google.com/file/d/1Ds8AMZIw3DkWw4J3ml82Ke1RKBpFGT8T/view?usp=sharing) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_efficientvit-b1_mobileclip_s1.pth) (ft: [GDrive](https://drive.google.com/file/d/1jMiRuMj6aHDg7mncHxNT3I1URD4-H9el/view?usp=sharing), [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_efficientvit_b1_mobileclip_s1_ft.pth)) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-EV-L-MC-S1** | EfficientViT-B2 & MobileCLIP-S1 | 14.98M + 63.56M | [GDrive](https://drive.google.com/file/d/1d_dqJvaAm8rYYoYQIrpSpi9iWpDeDS2q/view?usp=sharing) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_efficientvit-b2_mobileclip_s1.pth) | $$\text{Planned}$$ | $$\text{Planned}$$ |
> **Note (2025/12/08):** The current Stage 1 text encoder weights are distilled on 1% of the Recap-DataComp-1B dataset combined with all 9 image encoder variants. We notice a performance degradation, this is expected as the text encoder are not aligning with the light image encoders in stage1. We will release the stage1+ fine-tuned weights in the future.

> **Note (2025/12/08):** We have also uploaded standalone text encoder weights trained on 1% Recap-DataComp-1B dataset: [MobileCLIP-S1](https://drive.google.com/file/d/1As_lkYTyxnu3nshEd3_3s50NT_m2XMul/view?usp=sharing) and [MobileCLIP2-L](https://drive.google.com/file/d/16C2-PB3-oU7uway3PdXtuVoSxvrLXjdw/view?usp=sharing). You can merge with stage 1 trained image encoder weights to get the full model.

> **Note (2026/01/11):** The fine-tuned (**ft**) text encoder models are fine-tuned on SA-Co Gold+Silver text annotations. Standalone fine-tuned text encoder weights: [MobileCLIP-S0](https://drive.google.com/file/d/1JtbqC2d_F0i9pN-skGuUfSnJx50aAKsd/view?usp=sharing), [MobileCLIP-S1](https://drive.google.com/file/d/14x9iwLnVq282fGPy8JypnCWsB9da_1Mh/view?usp=sharing), and [MobileCLIP2-L](https://drive.google.com/file/d/1xdyDkGaBwUYALZ1u7U7sGS7yq-gIEnDz/view?usp=sharing).

### SAM3-LiteText Models

SAM3-LiteText replaces the SAM3 text encoder with a lightweight distilled text encoder, reducing text encoder parameters by up to **88%** with comparable performance. See the [SAM3-LiteText paper](https://arxiv.org/abs/2602.12173) for details.

Recommended default for SAM3-LiteText:

- Use a **fixed** checkpoint (`pos_embed_table_size == context_length`)
- Run inference with **slice** behavior by default
- Only turn on interpolation for older legacy checkpoints that were trained with a larger positional table

| Model | Text Encoder | Ctx | Text Params | Weights |
|-------|--------------|-----|-------------|---------|
| **SAM3-LiteText-S0-16** | MobileCLIP-S0 | 16 | 42.54M | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt)/ [GDrive](https://drive.google.com/file/d/1Eo81WYzfozFSvgvwlScGorUAIfMVPAFm/view?usp=sharing) |
| **SAM3-LiteText-S1-16** | MobileCLIP-S1 | 16 | 63.53M | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/efficient_sam3_image_encoder_mobileclip_s1_ctx16.pt)/ [GDrive](https://drive.google.com/file/d/1zL6x91PzvupHtZdA68jYip6yAUel8MMV/view?usp=sharing) |
| **SAM3-LiteText-L-16** | MobileCLIP2-L | 16 | 123.80M | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/efficient_sam3_image_encoder_mobileclip2_l_ctx16.pt)/ [GDrive](https://drive.google.com/file/d/1Mc4pk0FNCWwPTGoj1CCdAhkkNz02CUyY/view?usp=sharing) |

> All models use the **SAM3 ViT-H image encoder** (461.84M vision params). The text encoder parameters shown represent the distilled student replacing the original 353.72M text encoder, achieving up to **88% parameter reduction**.

Example:

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model(
    checkpoint_path="efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt",
    load_from_HF=False,
    text_encoder_type="MobileCLIP-S0",
    text_encoder_context_length=16,
    text_encoder_pos_embed_table_size=16,
    interpolate_pos_embed=False,
    enable_segmentation=True,
    enable_inst_interactivity=False,
)

processor = Sam3Processor(model)
inference_state = processor.set_image(image)
inference_state = processor.set_text_prompt(prompt="dog", state=inference_state)
print(inference_state["scores"])
```


---

## Preliminary Evaluation

<details>
<summary>Stage 1 Image Model Evaluation Results (COCO val2017)</summary>

| Model Name | Backbone | Parameters | COCO mIoU | Test Time (s) |
|------------|----------|------------|-----------|---------------|
| **ES-RV-S** | RepViT-M0.9 | 4.72M | 64.80% | 407.23 |
| **ES-RV-M** | RepViT-M1.1 | 7.77M | 65.28% (ft 65.60%) | 413.38  |
| **ES-RV-L** | RepViT-M2.3 | 22.40M | 65.53% | 466.66 |
| **ES-TV-S** | TinyViT-5M | 5.07M | 65.51% | 430.52 |
| **ES-TV-M** | TinyViT-11M | 10.55M | 65.45% (ft 65.69%) | 443.45  |
| **ES-TV-L** | TinyViT-21M | 20.62M | 66.29% | 452.14 |
| **ES-EV-S** | EfficientViT-B0 | 0.68M | 61.62% | 419.57 |
| **ES-EV-M** | EfficientViT-B1 | 4.64M | 64.82% (ft 64.94%) | 434.45  |
| **ES-EV-L** | EfficientViT-B2 | 14.98M | 66.30% | 450.36 |

> **Note:** The evaluation is done with a single NVIDIA 4070 Ti.

</details>


<details>
<summary>Stage 1 Text Encoder Evaluation Results (SA-Co/VEval Noun Phrases)</summary>

Metric: average token-level cosine similarity between student text features and SAM3 text encoder features.

**Pretrained on 1% Recap-DataComp-1B**

| Model Name | Text Backbone | Avg Cos Similarity | Eval Set |
|------------|--------------|-------------------|----------|
| **ES-MC-S0 (Recap-DC1B 1% pt)** | MobileCLIP-S0 | 0.864846 | 5184 noun phrases |
| **ES-MC-S1 (Recap-DC1B 1% pt)** | MobileCLIP-S1 | 0.854405 | 5184 noun phrases |
| **ES-MC2-L (Recap-DC1B 1% pt)** | MobileCLIP2-L | 0.850976 | 5184 noun phrases |

**Fine-tuned on SA-Co Gold+Silver text annotations**

| Model Name | Text Backbone | Avg Cos Similarity | Eval Set |
|------------|--------------|-------------------|----------|
| **ES-MC-S0 (SA-Co ft)** | MobileCLIP-S0 | 0.938915 | 5184 noun phrases |
| **ES-MC-S1 (SA-Co ft)** | MobileCLIP-S1 | 0.947152 | 5184 noun phrases |
| **ES-MC2-L (SA-Co ft)** | MobileCLIP2-L | 0.952901 | 5184 noun phrases |

> **Note:** Evaluation is done with [eval_text_encoder_similarity.py](eval/eval_text_encoder_similarity.py) using `data/sa-v-text/sa-co-veval/saco_veval_noun_phrases.json`. Pretrained models are trained on Recap-DataComp-1B (1%), and fine-tuned models are trained on SA-Co Gold+Silver text annotations.

</details>

---


## CoreML / ONNX Export

Coming soon: export pipelines to ONNX and CoreML for cross-platform deployment.

---

## Web Demo

Coming soon: an interactive web demo for real-time concept segmentation and tracking.

---
## Development To-Do List

- [x] **Release Stage 1 Image Encoder Weights**: Distilled image encoder weights from SAM3 image encoder for all 9 variants (RepViT, TinyViT, EfficientViT)
- [x] **Release Stage 1 Text Encoder Weights**: Distill SAM3 text encoder weights to MobileCLIP-S1 combined with all 9 image encoder variants
- [x] **Release Stage 1+ Fine-Tuned Encoder Weights**: Prompt-in-the-loop supervised fine-tuning for improved encoder performance
- [x] **Release SAM3-LiteText Weights**: Distilled a lightweight MobileCLIP text encoder that is competitive to the SAM3 text encoder for efficient vision-language segmentation
- [ ] **Release Stage 2 Memory Bank Aligned Model Weights**: Models with Perceiver-based memory compression trained on SA-V dataset
- [ ] **Release Stage 3 Fine-Tuned Model Weights**: End-to-end fine-tuned models on SAM3 dataset with full PCS capabilities
- [ ] **ONNX/CoreML Export**: Export models to ONNX and CoreML formats for cross-platform deployment
- [ ] **Web Demo**: Interactive web demonstration for real-time concept segmentation and tracking

---

## Call for Pull Requests
The idea for this repository originated from my work on SAM2 at Amazon, particularly as part of the research described in [this paper](https://ieeexplore.ieee.org/abstract/document/11084428). Since company policy, I cannot share the codebase. This year I am super excited to work on making SAM3 more efficient and accessible to the community.

We welcome contributions to EfficientSAM3! Please feel free to submit pull requests to improve the codebase, add new features, or fix bugs. Particularly, we are looking for:
- Efficient MedSAM3 integration (see [MedSAM2 by Bo Wang Lab](https://github.com/bowang-lab/MedSAM2))
- A Gradio demo (e.g. [EfficientTAM on Hugging Face Spaces](https://huggingface.co/spaces/yunyangx/EfficientTAM))
- A web demo deployed with Vercel (e.g. [Segment Anything Web UI](https://segment-anything-webui.vercel.app/))
- Annotation tools, such as [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) and [AnyLabeling](https://github.com/vietanhdev/anylabeling)
- An iOS or Android app (e.g. [Cutcha Photo on the App Store](https://apps.apple.com/us/app/cutcha-photo/id6478521132))
- An NVCC-based desktop application
- Anything else that you think is cool!
---

All meaningful contributions will be acknowledged and integrated into both the repository and the associated paper. We warmly welcome all contributors to the repository and happily offer co-authorship to those whose work merits inclusion in the paper.

## Citation

If you use EfficientSAM3 in your research, please cite:

```bibtex
@misc{zeng2025efficientsam3progressivehierarchicaldistillation,
  title={EfficientSAM3: Progressive Hierarchical Distillation for Video Concept Segmentation from SAM1, 2, and 3}, 
  author={Chengxi Zeng and Yuxuan Jiang and Aaron Zhang},
  year={2025},
  eprint={2511.15833},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2511.15833}, 
}
```

```bibtex
@misc{zeng2026sam3litetextanatomicalstudysam3,
      title={SAM3-LiteText: An Anatomical Study of the SAM3 Text Encoder for Efficient Vision-Language Segmentation}, 
      author={Chengxi Zeng and Yuxuan Jiang and Ge Gao and Shuai Wang and Duolikun Danier and Bin Zhu and Stevan Rudinac and David Bull and Fan Zhang},
      year={2026},
      eprint={2602.12173},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.12173}, 
}
```

## License

This repository is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

This project builds upon [SAM](https://github.com/facebookresearch/segment-anything), [SAM2](https://github.com/facebookresearch/sam2), [SAM3](https://github.com/facebookresearch/sam3), [EdgeSAM](https://github.com/chongzhou96/EdgeSAM), [EdgeTAM](https://github.com/facebookresearch/EdgeTAM), [EfficientTAM](https://github.com/yformer/EfficientTAM), [RepViT](https://github.com/THU-MIG/RepViT), [TinyViT](https://github.com/wkcn/TinyViT), [EfficientViT](https://github.com/mit-han-lab/efficientvit), and [MobileCLIP](https://github.com/apple/ml-mobileclip). Please refer to their respective licenses for usage terms.


## Acknowledgments

We gratefully acknowledge the [University of Bristol Isambard-AI supercomputer cluster](https://www.bristol.ac.uk/research/centres/bristol-supercomputing/articles/2025/isambard-ai-is-11th-fastest-supercomputer-in-the-world.html) for providing computational resources to this project. Special thanks to [Dr. Fan Aaron Zhang](https://fan-aaron-zhang.github.io/) for allocating resources and supporting this research.

---

## Users

Organizations and projects using EfficientSAM3:

<table>
  <tr>
    <td align="center" width="20%">
      <img src="https://github.com/SimonZeng7108/simonzeng7108.github.io/blob/main/efficientsam3/static/images/esa.png" alt="European Space Agency" height="80"><br>
      <a href="https://www.esa.int/Applications/Observing_the_Earth/Phsat-2/Introducing_Phsat-2">European Space Agency</a>
    </td>
  </tr>
</table>

> **Note:** If you're using EfficientSAM3 in your work, please acknowledge us in your publications or projects. We're happy to promote your work here! Contact us to be featured in this section.

