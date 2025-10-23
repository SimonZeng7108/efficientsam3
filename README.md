### EfficientSAM3: Progressive Hierachical Knowledge Distillation (PhD) from SAM1, 2 and 3
[Chengxi Simon Zeng](https://simonzeng7108.github.io/about/), [Yuxuan Jiang](https://pikapi22.github.io/), [Aaron Zhang](https://fan-aaron-zhang.github.io/)

Visual Information Lab, University of Bristol

[[Paper](#)] [[Project Page](#)] [[Hugging Face](#)]

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Updates](#updates)
  - [Stage 1: Encoder Distillation (Image-Level Segmentation)](#stage-1-encoder-distillation-image-level-segmentation)
  - [Stage 2: Temporal Memory Distillation (Video Tracking)](#stage-2-temporal-memory-distillation-video-tracking)
  - [Stage 3: End-to-End Fine-Tuning (Concept Segmentation)](#stage-3-end-to-end-fine-tuning-concept-segmentation)
  - [tl;dr](#tldr)
- [Installation](#installation)
- [Inference](#inference)
- [Training and Evaluation](#training-and-evaluation)
- [Datasets](#datasets)
- [EfficientSAM3 Model Zoo \& Weight Release](#efficientsam3-model-zoo--weight-release)
- [CoreML / ONNX Export](#coreml--onnx-export)
- [Web Demo](#web-demo)
- [Development To-Do List](#development-to-do-list)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Updates

- **[2025/10/18]** 🎉 We first release EfficientSAM3 weights for all 9 encoder variants (Stage 1: Encoder Distillation)

---

[SAM3](https://ai.meta.com/sam3) (Segment Anything Model 3) has introduced powerful **Promptable Concept Segmentation (PCS)** capabilities, enabling semantic understanding and temporal object tracking beyond traditional mask generation. However, SAM3's massive vision backbone and dense memory bank make it impractical for real-time, on-device applications where computational resources and latency constraints are critical.

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
| | SAM3 | Promptable Concept Segmentation (PCS) capabilities |
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
Stage 1: We distill the SAM3 encoder using SAM1 data.
Stage 2: We align the distilled encoder to a perceiver and an efficient memory bank using SAM2 data.
Stage 3: We fine-tune the complete pipeline using SAM3 data.

</details>


---

## Installation

The code requires `python>=3.8`. We recommend `torch==2.0.0` and `torchvision==0.15.1`. Please refer to the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

Clone the repository locally:

```bash
git clone https://github.com/SimonZeng7108/efficientsam3.git && cd efficientsam3
```

Install additional dependencies:

```bash
pip install -r requirements.txt
```

Install EfficientSAM3:

```bash
pip install -e .
```

---

## Inference

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

Please refer to [README_TRAIN.md](README_TRAIN.md) for training and evaluation details.

---

## Datasets

For dataset setup and download scripts (`data/download_*.sh`) covering COCO, DAVIS, LVIS, SA-1B, SA-V, LVOS, MOSE, and YouTube-VOS, see:

- [README_dataset.md](README_dataset.md)

---


## EfficientSAM3 Model Zoo & Weight Release

| Model Name | Backbone | Parameters | Stage 1 Weights<br/>(Encoder Distilled) | Stage 2 Weights<br/>(Memory Module Trained) | Stage 3 Weights<br/>(End-to-End Fine-Tuned) |
|------------|----------|------------|----------------------------------------|---------------------------------------------|---------------------------------------------|
| **ES-RV-S** | RepViT-M0.9 | 5.1M | $$\text{Planned}$$ | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-RV-M** | RepViT-M1.1 | 6.8M | $$\text{Planned}$$ | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-RV-L** | RepViT-M2.3 | 8.2M | $$\text{Planned}$$ | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-TV-S** | TinyViT-5M | 5.4M | $$\text{Planned}$$ | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-TV-M** | TinyViT-11M | 11M | $$\text{Planned}$$ | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-TV-L** | TinyViT-21M | 21M | $$\text{Planned}$$ | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-EV-S** | EfficientViT-B0 | 0.7M | $$\text{Planned}$$ | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-EV-M** | EfficientViT-B1 | 4.8M | $$\text{Planned}$$ | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-EV-L** | EfficientViT-B2 | 15M | $$\text{Planned}$$ | $$\text{Planned}$$ | $$\text{Planned}$$ |

---


## CoreML / ONNX Export

Coming soon: export pipelines to ONNX and CoreML for cross-platform deployment.

---

## Web Demo

Coming soon: an interactive web demo for real-time concept segmentation and tracking.

---
## Development To-Do List

- [x] **Release Stage 1 Encoder Weights**: Distilled encoder weights for all 9 variants (RepViT, TinyViT, EfficientViT)
- [ ] **Release Stage 2 Memory Bank Aligned Model Weights**: Models with Perceiver-based memory compression trained on SA-V dataset
- [ ] **Release Stage 3 Fine-Tuned Model Weights**: End-to-end fine-tuned models on SAM3 dataset with full PCS capabilities
- [ ] **ONNX/CoreML Export**: Export models to ONNX and CoreML formats for cross-platform deployment
- [ ] **Web Demo**: Interactive web demonstration for real-time concept segmentation and tracking

---


## Citation

If you use EfficientSAM3 in your research, please cite:

```bibtex
@misc{efficientsam3,
  title={EfficientSAM3: Progressive Hierachical Knowledge Distillation (PhD) from SAM1, 2 and 3},
  author={Zeng, Chengxi Simon and Jiang, Yuxuan and Zhang, Aaron},
  institution={University of Bristol},
  year={2025},
  howpublished={\url{https://github.com/SimonZeng7108/efficientsam3}}
}
```

## License

This repository is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

This project builds upon [SAM](https://github.com/facebookresearch/segment-anything), [SAM2](https://github.com/facebookresearch/sam2), SAM3, [EdgeSAM](https://github.com/chongzhou96/EdgeSAM), [EdgeTAM](https://github.com/facebookresearch/EdgeTAM), [EfficientTAM](https://github.com/yformer/EfficientTAM), [RepViT](https://github.com/THU-MIG/RepViT), [TinyViT](https://github.com/wkcn/TinyViT), and [EfficientViT](https://github.com/mit-han-lab/efficientvit). Please refer to their respective licenses for usage terms.

## Acknowledgments

This work is inspired by and builds upon:
- **[SAM](https://github.com/facebookresearch/segment-anything)** (Meta AI) - Foundation segmentation model
- **[SAM2](https://github.com/facebookresearch/sam2)** - Video object segmentation capabilities
- **SAM3** - Promptable Concept Segmentation
- **[EdgeSAM](https://github.com/chongzhou96/EdgeSAM)** - Efficient encoder distillation techniques
- **[EdgeTAM](https://github.com/facebookresearch/EdgeTAM)** - Perceiver-based memory compression for tracking
- **[EfficientTAM](https://github.com/yformer/EfficientTAM)** - Efficient temporal attention mechanisms
- **[RepViT](https://github.com/THU-MIG/RepViT)** - Mobile-optimized Vision Transformer backbones
- **[TinyViT](https://github.com/wkcn/TinyViT)** - Tiny Vision Transformer architectures
- **[EfficientViT](https://github.com/mit-han-lab/efficientvit)** - Efficient Vision Transformer models

