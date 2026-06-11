---
layout: default
title: Home
---

# EfficientSAM3

Progressive Hierarchical Knowledge Distillation from SAM1, 2 and 3

**EfficientSAM3** compresses SAM3's 1.4B parameter model into lightweight variants (~614M total) while maintaining competitive performance on downstream benchmarks.

**SAM3-LiteText** keeps the SAM3 vision encoder but replaces the heavy text encoder with lightweight MobileCLIP variants (~25% model size reduction).

<p align="center">
  <img src="https://raw.githubusercontent.com/SimonZeng7108/efficientsam3/main/images/dog_person_example_dog.png" width="45%">
  <img src="https://raw.githubusercontent.com/SimonZeng7108/efficientsam3/main/images/dog_person_example_person.png" width="45%">
</p>

## News

- **[2026/06/11]** **Stage 3 Fine-tuned Models Released!** EfficientSAM3 full models (EV-M, RV-M, TV-M) fine-tuned on 5% SA1B data & SACap labels.
- **[2026/04/19]** **SAM3-LiteText** accepted by [ICMR2026](https://icmr2026.org/)!

## Quick Start

```bash
git clone https://github.com/SimonZeng7108/efficientsam3
cd efficientsam3
pip install -e ".[stage1]"
```

## Model Zoo

### EfficientSAM3 Full Models (Lightweight Image + Text Encoders)

| Model | Vision | Text | Decoder | Total | vs SAM3 (1.4B) |
|-------|--------|------|---------|-------|-----------------|
| **EV-M** | 4.6M | 4.07M | ~605M | ~614M | **56% smaller** |
| **RV-M** | 7.8M | 4.07M | ~605M | ~617M | **56% smaller** |
| **TV-M** | 10.6M | 4.07M | ~605M | ~620M | **56% smaller** |

> **SAM3 Teacher**: 1.4B total (Vision: 461M + Text: 354M + Decoder/Heads: ~600M)

### SAM3-LiteText Models (Lightweight Text Encoder Only)

SAM3-LiteText keeps the SAM3 vision encoder but replaces the text encoder with lightweight MobileCLIP variants.

| Model | Text Encoder | Context | Full Model | vs SAM3 (1.4B) |
|-------|-------------|---------|------------|-----------------|
| **LiteText-S0-16** | MobileCLIP-S0 | ctx16 | ~1050M | **25% smaller** |
| **LiteText-S0-32** | MobileCLIP-S0 | ctx32 | ~1050M | **25% smaller** |
| **LiteText-S1-16** | MobileCLIP-S1 | ctx16 | ~1051M | **25% smaller** |
| **LiteText-S1-32** | MobileCLIP-S1 | ctx32 | ~1051M | **25% smaller** |
| **LiteText-L-16** | MobileCLIP2-L | ctx16 | ~1088M | **22% smaller** |
| **LiteText-L-32** | MobileCLIP2-L | ctx32 | ~1088M | **22% smaller** |

> Full model = Vision (~1050M) + Text encoder + Decoder (~600M). Text encoder: 4.07M (S0), 4.69M (S1), 42.38M (L) replaces 354M SAM3 text encoder.

## Usage

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image

# Load model (TV-M example)
model = build_sam3_image_model(
    checkpoint_path="efficientsam3_tinyvit_11m_mobileclip_s0_ctx16_5p_full.pt",
    load_from_HF=False,
)

# Process image
processor = Sam3Processor(model)
image = Image.open("your_image.jpg").convert("RGB")
state = processor.set_image(image)

# Text prompt segmentation
state = processor.set_text_prompt("dog", state)

# Get masks
masks = state["masks"]
scores = state["scores"]
print(f"Found {len(masks)} masks")
```

## Training

- **Stage 1:** Encoder distillation ([README_stage1.md](https://github.com/SimonZeng7108/efficientsam3/blob/main/README_stage1.md))
- **Stage 3:** Full fine-tuning ([README_stage3.md](https://github.com/SimonZeng7108/efficientsam3/blob/main/README_stage3.md))

## Links

- [arXiv Paper](https://arxiv.org/abs/2511.15833)
- [SAM3-LiteText Paper](https://arxiv.org/abs/2602.12173)
- [HuggingFace Models](https://huggingface.co/Simon7108528/EfficientSAM3)
- [Documentation](https://github.com/SimonZeng7108/efficientsam3/tree/main/docs)
