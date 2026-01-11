### EfficientSAM3: Progressive Hierachical Knowledge Distillation (PhD) from SAM1, 2 and 3
[Chengxi Simon Zeng](https://simonzeng7108.github.io/about/)<sup>1,†</sup>, [Yuxuan Jiang](https://YuxuanJJ.github.io/)<sup>1</sup>, [Gao Ge](https://scholar.google.com/citations?user=j2_80ewAAAAJ&hl=en)<sup>1</sup>, [Shuai Wang](https://shuaiwang97.github.io/)<sup>2</sup>, [Fan Aaron Zhang](https://fan-aaron-zhang.github.io/)<sup>1</sup>
<sup>1</sup>Visual Information Lab, University of Bristol; <sup>2</sup>MultiX lab, University of Amsterdam

<sup>†</sup>Tech Lead & Corresponding Author

[📄 Paper](https://arxiv.org/abs/2511.15833) | [🌐 Project Page](https://simonzeng7108.github.io/efficientsam3/) | [🤗 Hugging Face](https://huggingface.co/Simon7108528/EfficientSAM3) | [💬 Discord](https://discord.gg/FMyaQca7xT)
---
## Updates
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
- **PyTorch** 2.7.0 (CUDA 12.6 build recommended)
- **CUDA**-capable GPUs with drivers that support CUDA ≥ 12.6

Follow the exact environment setup from the [official SAM3 README](sam3/README.md) or use the condensed steps below:


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
    checkpoint_path="efficient_sam3_tinyvit_s.pt",
    backbone_type="tinyvit",
    model_name="5m"
)

# Process image and predict
processor = Sam3Processor(model)
inference_state = processor.set_image(image)
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

 **MobileCLIP-S1 (63.56M)** distilled from **SAM3 Text Encoder (353.72M)** — trained on **1% Recap-DataComp-1B**.

```python
from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load model with text encoder
model = build_efficientsam3_image_model(
    checkpoint_path="efficient_sam3_tinyvit_m_mobileclip_s1.pt",
    backbone_type="tinyvit",
    model_name="11m",
    text_encoder_type="MobileCLIP-S1"
)

# Process image and predict with text prompt
processor = Sam3Processor(model)
inference_state = processor.set_image(image)
inference_state = processor.set_text_prompt(inference_state, prompt="shoe")
masks, scores, _ = model.predict_inst(inference_state)
```

For detailed examples including point/box prompts, batched inference, and more, see [sam3/efficientsam3_examples/efficientsam3_for_sam1_task_example.py](sam3/efficientsam3_examples/efficientsam3_for_sam1_task_example.py). For text prompt inference, see [sam3/efficientsam3_examples/efficientsam3_image_predictor_example.ipynb](sam3/efficientsam3_examples/efficientsam3_image_predictor_example.ipynb).

---

## Training and Evaluation

**Training:**
- For Stage 1 encoder distillation training details, see [README_stage1.md](README_stage1.md).
- Stage 2 and Stage 3 training details coming soon.

**Evaluation:**
- To evaluate models on COCO dataset:
  ```bash
  python eval/eval_coco.py --coco_root data/coco --output_dir output
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
| **ES-RV-M** | RepViT-M1.1 | 7.77M | [GDrive](https://drive.google.com/file/d/1JW3KiTnYF2r8nIijf8UXrKXJwf5D5s-5/view?usp=drive_link) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_repvit_m.pt) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-RV-L** | RepViT-M2.3 | 22.40M | [GDrive](https://drive.google.com/file/d/1ocAkz6DgkaKCKpLdalq2Ya8X6VIMrLLI/view?usp=drive_link) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_repvit_l.pt) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-TV-S** | TinyViT-5M | 5.07M | [GDrive](https://drive.google.com/file/d/1CDfJTd2fTKJTV5nsfYLAV_CGMfQ-AWXS/view?usp=drive_link) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_tinyvit_s.pt) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-TV-M** | TinyViT-11M | 10.55M | [GDrive](https://drive.google.com/file/d/1TX70zw7SduQRZP6hce6MIxEOsdoooZFB/view?usp=drive_link) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_tinyvit_m.pt) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-TV-L** | TinyViT-21M | 20.62M | [GDrive](https://drive.google.com/file/d/19hyKjjZ4_8ldmxIAm6D8e8z89xX-M3hZ/view?usp=drive_link) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_tinyvit_l.pt) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-EV-S** | EfficientViT-B0 | 0.68M | [GDrive](https://drive.google.com/file/d/1EnA581iSExZRRWlI6oY-wXTgX4gESijG/view?usp=drive_link) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_efficientvit_s.pt) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-EV-M** | EfficientViT-B1 | 4.64M | [GDrive](https://drive.google.com/file/d/14CRA3LhquUkf8prrKfI1INyHtCw6buvm/view?usp=sharing) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_efficientvit_m.pt) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-EV-L** | EfficientViT-B2 | 14.98M | [GDrive](https://drive.google.com/file/d/1Zg0Er0LwYYNCFJezSUSlQ8L645cR1OhN/view?usp=drive_link) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_efficientvit_l.pt) | $$\text{Planned}$$ | $$\text{Planned}$$ |

> **Note (2025/12/02):** The current Stage 1 image encoder weights are distilled on 1% of the SA-1B dataset.

### EfficientSAM3 Text Encoder + Image Encoder Models

| Model Name | Backbone | Parameters | Stage 1 Weights<br/>(Encoder Distilled) | Stage 2 Weights<br/>(Memory Module Trained) | Stage 3 Weights<br/>(End-to-End Fine-Tuned) |
|------------|----------|------------|----------------------------------------|---------------------------------------------|---------------------------------------------|
| **ES-RV-S-MC-S1** | RepViT-M0.9 & MobileCLIP-S1 | 4.72M + 63.56M | [GDrive](https://drive.google.com/file/d/1SvBPDqeEYCKpOui79tCIl3c1nMnQijL-/view?usp=sharing) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_repvit-m0_9_mobileclip_s1.pth) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-RV-M-MC-S1** | RepViT-M1.1 & MobileCLIP-S1 | 7.77M + 63.56M | [GDrive](https://drive.google.com/file/d/10VB-1IYAO3iqGq3U63xcM9uGyrqGGvAi/view?usp=sharing) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_repvit-m1_1_mobileclip_s1.pth) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-RV-L-MC-S1** | RepViT-M2.3 & MobileCLIP-S1 | 22.40M + 63.56M | [GDrive](https://drive.google.com/file/d/1IxcVq1BBlMF2LNJ2uljQ8ajLpOWKFpUr/view?usp=sharing) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_repvit-m2_3_mobileclip_s1.pth) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-TV-S-MC-S1** | TinyViT-5M & MobileCLIP-S1 | 5.07M + 63.56M | [GDrive](https://drive.google.com/file/d/1EtG6j3pGtaf-taxo5NLGCqZ8QvOgdLAn/view?usp=sharing) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_tinyvit_5m_mobileclip_s1.pth) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-TV-M-MC-S1** | TinyViT-11M & MobileCLIP-S1 | 10.55M + 63.56M | [GDrive](https://drive.google.com/file/d/1dz5bl0RkCbEUjeK54azREbkEQA-hW_IG/view?usp=sharing) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_tinyvit_11m_mobileclip_s1.pth) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-TV-L-MC-S1** | TinyViT-21M & MobileCLIP-S1 | 20.62M + 63.56M | [GDrive](https://drive.google.com/file/d/1DIeJmFle_tHAUKWbycxrNQAW0peZAuy4/view?usp=sharing) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_tinyvit_21m_mobileclip_s1.pth) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-EV-S-MC-S1** | EfficientViT-B0 & MobileCLIP-S1 | 0.68M + 63.56M | [GDrive](https://drive.google.com/file/d/1pa4wKJysp2dUVkUMTrJ7Rs8ZVPIHntFL/view?usp=sharing) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_efficientvit-b0_mobileclip_s1.pth) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-EV-M-MC-S1** | EfficientViT-B1 & MobileCLIP-S1 | 4.64M + 63.56M | [GDrive](https://drive.google.com/file/d/1Ds8AMZIw3DkWw4J3ml82Ke1RKBpFGT8T/view?usp=sharing) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_efficientvit-b1_mobileclip_s1.pth) | $$\text{Planned}$$ | $$\text{Planned}$$ |
| **ES-EV-L-MC-S1** | EfficientViT-B2 & MobileCLIP-S1 | 14.98M + 63.56M | [GDrive](https://drive.google.com/file/d/1d_dqJvaAm8rYYoYQIrpSpi9iWpDeDS2q/view?usp=sharing) / [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/efficient_sam3_efficientvit-b2_mobileclip_s1.pth) | $$\text{Planned}$$ | $$\text{Planned}$$ |
> **Note (2025/12/08):** The current Stage 1 text encoder weights are distilled on 1% of the Recap-DataComp-1B dataset combined with all 9 image encoder variants. We notice a performance degradation, this is expected as the text encoder are not aligning with the light image encoders in stage1. We will release the stage1+ fine-tuned weights in the future.

> **Note (2025/12/08):** We have also uploaded standalone text encoder weights trained on 1% Recap-DataComp-1B dataset: [MobileCLIP-S1](https://drive.google.com/file/d/1As_lkYTyxnu3nshEd3_3s50NT_m2XMul/view?usp=sharing) and [MobileCLIP2-L](https://drive.google.com/file/d/16C2-PB3-oU7uway3PdXtuVoSxvrLXjdw/view?usp=sharing). You can merge with stage 1 trained image encoder weights to get the full model.


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
- [ ] **Release Stage 1+ Fine-Tuned Encoder Weights**: Prompt-in-the-loop supervised fine-tuning for improved encoder performance
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
  author={Chengxi Zeng and Yuxuan Jiang and Gao Ge and Shuai Wang and Fan Aaron Zhang},
  year={2025},
  eprint={2511.15833},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2511.15833}, 
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
      <a href="https://www.esa.int/">European Space Agency</a>
    </td>
  </tr>
</table>

> **Note:** If you're using EfficientSAM3 in your work, please acknowledge us in your publications or projects. We're happy to promote your work here! Contact us to be featured in this section.

