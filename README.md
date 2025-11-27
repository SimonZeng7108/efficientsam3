### EfficientSAM3: Progressive Hierachical Knowledge Distillation (PhD) from SAM1, 2 and 3
[Chengxi Simon Zeng](https://simonzeng7108.github.io/about/)<sup>1,†</sup>, [Yuxuan Jiang](https://YuxuanJJ.github.io/)<sup>1</sup>, [Gao Ge](https://scholar.google.com/citations?user=j2_80ewAAAAJ&hl=en)<sup>1</sup>, [Shuai Wang](https://shuaiwang97.github.io/)<sup>2</sup>, [Fan Aaron Zhang](https://fan-aaron-zhang.github.io/)<sup>1</sup>
<sup>1</sup>Visual Information Lab, University of Bristol; <sup>2</sup>MultiX lab, University of Amsterdam

<sup>†</sup>Tech Lead & Corresponding Author

[[Paper](https://arxiv.org/abs/2511.15833)] [[Project Page](https://simonzeng7108.github.io/efficientsam3/)] [[Hugging Face](https://huggingface.co/Simon7108528/EfficientSAM3)]

---
## 🔥 Teaser Image Model
<p align="center">
  <img src="https://github.com/SimonZeng7108/efficientsam3/blob/main/images/es-ev-s-teaser.jpg" width="30%">
</p>

 **EfficientViT-B0 (0.68M params)** distilled from **SAM3 Encoder (461.84M)** — **99.85% smaller**, trained on **1% SA-1B**.

**Download Weight:** [Google Drive](https://drive.google.com/file/d/1MqtnQBaZWgtmURgBgQEEphnCNiLrPCjn/view?usp=drive_link). **Visualisation:** [Script](https://github.com/SimonZeng7108/efficientsam3/blob/stage1/sam3/efficientsam3_examples/efficientsam3_for_sam1_task_example.py) (Switch to Branch Stage1).

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Updates](#updates)
- [Installation](#installation)
- [Inference](#inference)
- [Training and Evaluation](#training-and-evaluation)
- [Datasets](#datasets)
- [EfficientSAM3 Model Zoo \& Weight Release](#efficientsam3-model-zoo--weight-release)
- [CoreML / ONNX Export](#coreml--onnx-export)
- [Web Demo](#web-demo)
- [Development To-Do List](#development-to-do-list)
- [Call for Pull Requests](#call-for-pull-requests)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Updates
- **[2025/11/25]** Teaser model released. See Above. More models are baking in the oven🔥.
- **[2025/10/18]** Project announced. Code and weights are not released yet; they will be published once SAM3 code is publicly available.


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
| | SAM3 Data | Concept segmentation dataset |
| **Student Backbones** | [RepViT](https://github.com/THU-MIG/RepViT) (M0.9, M1.1, M2.3) | Mobile-optimized Vision Transformer for highest throughput |
| | [TinyViT](https://github.com/wkcn/TinyViT) (5M, 11M, 21M) | Balanced efficiency and performance |
| | [EfficientViT](https://github.com/mit-han-lab/efficientvit) (B0, B1, B2) | Ultra-lightweight architectures for minimal latency |
| **Reference Modules** | [EdgeSAM](https://github.com/chongzhou96/EdgeSAM) | Lightweight encoder distillation techniques |
| | [EdgeTAM](https://github.com/facebookresearch/EdgeTAM) | Perceiver-based memory compression for video tracking |
| | [EfficientSAM](https://github.com/yformer/EfficientSAM) | Lightweight memory bank design |

</details>

---

<details>
<summary>Three-Stage Progressive Training Curriculum</summary>

EfficientSAM3 is trained through a three-stage progressive distillation:

### Stage 1: Encoder Distillation (Image-Level Segmentation)

- Distill the SAM3 encoder to nine student backbones (3 RepViT × 3 TinyViT × 3 EfficientViT variants)
- Use [SA-1B](https://ai.meta.com/datasets/segment-anything/) dataset with Prompt-in-the-Loop Distillation
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
- **PyTorch** 2.7.0 (CUDA 12.6 build recommended)
- **CUDA**-capable GPUs with drivers that support CUDA ≥ 12.6

Follow the exact environment setup from the [official SAM3 README](sam3/README.md) or use the condensed steps below (single-node example):

```bash
git clone https://github.com/SimonZeng7108/efficientsam3.git
cd efficientsam3

conda create -n efficientsam3 python=3.12 -y
conda activate efficientsam3

pip install --upgrade pip
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install repo dependencies via the root pyproject (brings in SAM3 + Stage-1 extras)
pip install -e ".[stage1]"
```

---

## Inference
Coming soon (previsional example):
Download checkpoints (refer to [Checkpoints](#efficientsam3-model-zoo--weight-release) for more details):

```bash
mkdir -p weights
# Example: download an ES-TV-S Stage 1 encoder checkpoint
wget -P weights/ https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/weights/es_tv_s_stage1.pth
```

Use EfficientSAM3 in Python:

```python
from efficientsam3 import SamPredictor, sam_model_registry

# Load model
sam = sam_model_registry["es_tv_s"](checkpoint="weights/es_tv_s_stage1.pth")
predictor = SamPredictor(sam)
predictor.set_image(<your_image>)

# Generate masks with prompts
masks, _, _ = predictor.predict(<input_prompts>)
```

---

## Training and Evaluation

Coming soon: training and evaluation details will be published once SAM3 is publicly available.

---

## Datasets

For dataset setup and download scripts (`data/download_*.sh`) covering COCO, DAVIS, LVIS, SA-1B, SA-V, LVOS, MOSE, and YouTube-VOS, see:

- [README_dataset.md](README_dataset.md)

---


## EfficientSAM3 Model Zoo & Weight Release

| Model Name | Backbone | Parameters | Stage 1 Weights<br/>(Encoder Distilled) | Stage 2 Weights<br/>(Memory Module Trained) | Stage 3 Weights<br/>(End-to-End Fine-Tuned) |
|------------|----------|------------|----------------------------------------|---------------------------------------------|---------------------------------------------|
| **ES-RV-S** | RepViT-M0.9 | 4.72M | $$\text{Planned}$$ | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-RV-M** | RepViT-M1.1 | 7.77M | $$\text{Planned}$$ | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-RV-L** | RepViT-M2.3 | 22.40M | $$\text{Planned}$$ | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-TV-S** | TinyViT-5M | 5.07M | $$\text{Planned}$$ | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-TV-M** | TinyViT-11M | 10.55M | $$\text{Planned}$$ | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-TV-L** | TinyViT-21M | 20.62M | $$\text{Planned}$$ | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-EV-S** | EfficientViT-B0 | 0.68M | $$\text{Planned}$$ | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-EV-M** | EfficientViT-B1 | 4.64M | $$\text{Planned}$$ | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-EV-L** | EfficientViT-B2 | 14.98M | $$\text{Planned}$$ | $$\text{Planned}$$ | $$\text{Planned}$$ |

---


## CoreML / ONNX Export

Coming soon: export pipelines to ONNX and CoreML for cross-platform deployment.

---

## Web Demo

Coming soon: an interactive web demo for real-time concept segmentation and tracking.

---
## Development To-Do List

- [ ] **Release Stage 1 Encoder Weights**: Distilled encoder weights for all 9 variants (RepViT, TinyViT, EfficientViT) — pending SAM3 public release
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
@misc{efficientsam3,
  title={EfficientSAM3: Progressive Hierachical Knowledge Distillation (PhD) from SAM1, 2 and 3},
  author={Zeng, Chengxi Simon and Jiang, Yuxuan and Ge, Gao  and Wang, Shuai and Zhang, Fan Aaron},
  institution={University of Bristol},
  year={2025},
  howpublished={\url{https://github.com/SimonZeng7108/efficientsam3}}
}
```

## License

This repository is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

This project builds upon [SAM](https://github.com/facebookresearch/segment-anything), [SAM2](https://github.com/facebookresearch/sam2), [SAM3](https://github.com/facebookresearch/sam3), [EdgeSAM](https://github.com/chongzhou96/EdgeSAM), [EdgeTAM](https://github.com/facebookresearch/EdgeTAM), [EfficientTAM](https://github.com/yformer/EfficientTAM), [RepViT](https://github.com/THU-MIG/RepViT), [TinyViT](https://github.com/wkcn/TinyViT), and [EfficientViT](https://github.com/mit-han-lab/efficientvit). Please refer to their respective licenses for usage terms.

## Acknowledgments

This work is inspired by and builds upon:
- **[SAM](https://github.com/facebookresearch/segment-anything)** (Meta AI) - Foundation segmentation model
- **[SAM2](https://github.com/facebookresearch/sam2)** - Video object segmentation capabilities
- **[SAM3](https://github.com/facebookresearch/sam3)** - Promptable Concept Segmentation
- **[EdgeSAM](https://github.com/chongzhou96/EdgeSAM)** - Efficient encoder distillation techniques
- **[EdgeTAM](https://github.com/facebookresearch/EdgeTAM)** - Perceiver-based memory compression for tracking
- **[EfficientTAM](https://github.com/yformer/EfficientTAM)** - Efficient temporal attention mechanisms
- **[RepViT](https://github.com/THU-MIG/RepViT)** - Mobile-optimized Vision Transformer backbones
- **[TinyViT](https://github.com/wkcn/TinyViT)** - Tiny Vision Transformer architectures
- **[EfficientViT](https://github.com/mit-han-lab/efficientvit)** - Efficient Vision Transformer models


