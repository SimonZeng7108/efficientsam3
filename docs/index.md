---
layout: page
title: Home
permalink: /
---

# EfficientSAM3

Progressive Hierarchical Knowledge Distillation from SAM1, 2 and 3

**EfficientSAM3** compresses SAM3's 1.4B parameter model into lightweight variants (< 15M params) while maintaining competitive performance.

**SAM3-LiteText** reduces the text encoder by 88% with similar performance.

![Segmentation Example](https://raw.githubusercontent.com/SimonZeng7108/efficientsam3/main/images/dog_person_example_dog.png)

## Quick Start

```bash
git clone https://github.com/SimonZeng7108/efficientsam3
cd efficientsam3
pip install -e ".[stage1]"
```

## Model Zoo

### EfficientSAM3 Full Models

| Model | Image Encoder | Text Encoder | Total Params | vs SAM3 |
|-------|---------------|--------------|---------------|---------|
| **EV-M** | EfficientViT-B1 (4.6M) | MobileCLIP-S0 (4.07M) | ~10M | 99.3% smaller |
| **RV-M** | RepViT-M1.1 (7.8M) | MobileCLIP-S0 (4.07M) | ~12M | 99.1% smaller |
| **TV-M** | TinyViT-11M (10.6M) | MobileCLIP-S0 (4.07M) | ~15M | 98.9% smaller |

### SAM3-LiteText Models

| Model | Text Encoder | Params | vs SAM3 Text |
|-------|-------------|--------|--------------|
| **LiteText-S0** | MobileCLIP-S0 | 4.07M | 98.9% smaller |
| **LiteText-S1** | MobileCLIP-S1 | 4.69M | 98.7% smaller |
| **LiteText-L** | MobileCLIP2-L | 42.38M | 88.0% smaller |

## Installation

```bash
pip install -e ".[stage1]"
```

Requirements:
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+

## Usage

```python
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model(
    checkpoint_path="efficient_sam3_tvm_m_mobileclip_s0_ctx16_5p_full.pt",
    enable_segmentation=True,
)
processor = Sam3Processor(model)
state = processor.set_image(image)
state = processor.set_text_prompt(state, "dog")
```

## Links

- [arXiv Paper](https://arxiv.org/abs/2511.15833)
- [HuggingFace Models](https://huggingface.co/Simon7108528/EfficientSAM3)
- [Project Page](https://simonzeng7108.github.io/efficientsam3/)
