# EfficientSAM3: Progressive Hierarchical Knowledge Distillation from SAM1, 2 and 3

[Chengxi Simon Zeng](https://simonzeng7108.github.io/about/)<sup>1,†</sup>, [Yuxuan Jiang](https://YuxuanJJ.github.io/)<sup>1</sup>, [Gao Ge](https://scholar.google.com/citations?user=j2_80ewAAAAJ&hl=en)<sup>1</sup>, [Shuai Wang](https://shuaiwang97.github.io/)<sup>2</sup>, [Duolikun Danier](https://danier97.github.io/)<sup>3</sup>, [Bin Zhu](https://binzhubz.github.io/)<sup>4</sup>, [Stevan Rudinac](https://stevanrudinac.com/)<sup>2</sup>, [David Bull](https://david-bull.github.io/)<sup>1</sup>, [Fan Aaron Zhang](https://fan-aaron-zhang.github.io/)<sup>1</sup>

<sup>1</sup>Visual Information Lab, University of Bristol; <sup>2</sup>MultiX lab, University of Amsterdam; <sup>3</sup>University of Edinburgh; <sup>4</sup>Singapore Management University

<sup>†</sup>Tech Lead & Corresponding Author

[![arXiv](https://img.shields.io/badge/arXiv-EfficientSAM3-b31b1b.svg)](https://arxiv.org/abs/2511.15833) [![arXiv](https://img.shields.io/badge/arXiv-SAM3--LiteText-b31b1b.svg)](https://arxiv.org/abs/2602.12173) [![Project Page](https://img.shields.io/badge/Project-Page-green)](https://SimonZeng7108.github.io/efficientsam3/) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-EfficientSAM3-blue)](https://huggingface.co/Simon7108528/EfficientSAM3) [![Discord](https://img.shields.io/badge/Discord-Join-7289da?logo=discord&logoColor=white)](https://discord.gg/FMyaQca7xT)

---

## Table of Contents
1. [Highlights](#highlights)
2. [Model Zoo](#model-zoo)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Training and Evaluation](#training-and-evaluation)
6. [Citations](#citations)

---

## Highlights

- **Efficient Vision Encoders**: Distilled into RepViT, TinyViT, and EfficientViT families (22-28M params vs SAM3's 463M)
- **Efficient Text Encoders**: Distilled into MobileCLIP variants (42-124M vs SAM3's 354M)
- **Full PCS Models**: Image + text encoders distilled for promptable concept segmentation
- **LiteText Models**: Keep SAM3 vision encoder, replace text encoder only

---

## Model Zoo

### EfficientSAM3 Full Models (Lightweight Image + Text Encoders)

EfficientSAM3 compresses both SAM3's vision encoder and text encoder into lightweight student models while maintaining competitive performance on downstream benchmarks.

<p align="center">
  <img src="https://raw.githubusercontent.com/SimonZeng7108/efficientsam3/main/images/dog_person_example_dog.png" width="45%">
  <img src="https://raw.githubusercontent.com/SimonZeng7108/efficientsam3/main/images/dog_person_example_person.png" width="45%">
</p>

| Model | Vision | Text | Decoder | Other | Params | vs ImageSAM3 | Download |
|-------|--------|------|-------------|-------|--------|--------------|----------|
| **EV-M** | 22.2M | 42.5M | 21.0M | 3.5M | **89.2M** | **90% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_efficientvit.pt?download=true) |
| **RV-M** | 25.6M | 42.5M | 21.0M | 3.5M | **92.7M** | **89% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_repvit.pt?download=true) |
| **TV-M** | 28.3M | 42.5M | 21.0M | 3.5M | **95.3M** | **89% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_tinyvit.pt?download=true) |

> **Note:** "Text" is the distilled text encoder. "Transformer" is the mask decoder. "Other" includes segmentation head + scoring. ImageSAM3 (for comparison): Vision: 463M + Text: 354M + Transformer: 30.3M + Other: 14.2M = **861.5M**

### SAM3-LiteText Models (Lightweight Text Encoder Only)

SAM3-LiteText keeps the SAM3 vision encoder but replaces the text encoder with lightweight MobileCLIP variants.

| Model | Vision | Text | Decoder | Other | Params | vs ImageSAM3 | Download |
|-------|--------|------|-------------|-------|--------|--------------|----------|
| **LiteText-S0-16** | 463.0M | 42.5M | 30.3M | 14.2M | **550.0M** | **36% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip_s0_ctx16.pt) |
| **LiteText-S0-32** | 463.0M | 42.5M | 30.3M | 14.2M | **550.0M** | **36% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip_s0_ctx32.pt) |
| **LiteText-S1-16** | 463.0M | 63.5M | 30.3M | 14.2M | **571.0M** | **34% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip_s1_ctx16.pt) |
| **LiteText-S1-32** | 463.0M | 63.5M | 30.3M | 14.2M | **571.0M** | **34% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip_s1_ctx32.pt) |
| **LiteText-L-16** | 463.0M | 123.8M | 30.3M | 14.2M | **631.3M** | **27% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip2_l_ctx16.pt) |
| **LiteText-L-32** | 463.0M | 123.8M | 30.3M | 14.2M | **631.3M** | **27% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip2_l_ctx32.pt) |

> **Note:** "Text" is the distilled text encoder (42.5M-123.8M). SAM3-LiteText keeps SAM3's ViT-H vision encoder (~463M) but replaces the text encoder. "Other" includes geometry encoder + segmentation head + scoring.

---

## Installation

```bash
git clone https://github.com/SimonZeng7108/efficientsam3
cd efficientsam3
pip install -e ".[stage1]"
```

**Prerequisites:**
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)

---

## Quick Start

### EfficientSAM3 (Full Models with Lightweight Encoders)

EfficientSAM3 replaces both the SAM3 vision encoder and text encoder with lightweight student models (EfficientViT/RepViT/TinyViT + MobileCLIP).

```python
from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image

# Load EfficientSAM3 TV-M model (uses TinyViT vision encoder + MobileCLIP-S0 text encoder)
model = build_efficientsam3_image_model(
    checkpoint_path="efficientsam3_tinyvit.pt",
    backbone_type="tinyvit",
    model_name="11m",
    text_encoder_type="MobileCLIP-S0",
    text_encoder_context_length=16,
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

### SAM3-LiteText

SAM3-LiteText keeps the SAM3 vision encoder but replaces the heavy text encoder with a lightweight MobileCLIP variant.

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image

# Build model with LiteText encoder (keeps SAM3 ViT, replaces text encoder)
model = build_sam3_image_model(
    checkpoint_path="sam3_litetext_mobileclip_s0_ctx16.pt",
    text_encoder_type="MobileCLIP-S0",
    text_encoder_context_length=16,
    load_from_HF=False,
)

# Use as normal
processor = Sam3Processor(model)
image = Image.open("your_image.jpg").convert("RGB")
state = processor.set_image(image)
state = processor.set_text_prompt("person", state)
masks = state["masks"]
```

---

## Training and Evaluation

**Training:**
- **Stage 1:** Encoder distillation training details in [README_stage1.md](README_stage1.md)
- **Stage 3:** Full fine-tuning details in [README_stage3.md](README_stage3.md)

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
  python eval/eval_text_encoder_similarity.py \
    --teacher-ckpt /path/to/teacher.pth \
    --student-ckpt /path/to/student.pth \
    --np-json data/sa-v-text/sa-co-veval/saco_veval_noun_phrases.json \
    --device cuda
  ```

---

## Citations

If you find EfficientSAM3 useful in your research, please cite:

```bibtex
@misc{zeng2025efficientsam3,
      title={EfficientSAM3: Progressive Hierarchical Distillation for Video Concept Segmentation from SAM1, 2, and 3},
      author={Chengxi Zeng and Yuxuan Jiang and Gao Ge and Shuai Wang and Duolikun Danier and Bin Zhu and Stevan Rudinac and David Bull and Fan Zhang},
      year={2025},
      eprint={2511.15833},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.15833},
}

@misc{zeng2026sam3litetext,
      title={SAM3-LiteText: An Anatomical Study of the SAM3 Text Encoder for Efficient Vision-Language Segmentation},
      author={Chengxi Zeng and Yuxuan Jiang and Gao Ge and Shuai Wang and Duolikun Danier and Bin Zhu and Stevan Rudinac and David Bull and Fan Zhang},
      year={2026},
      eprint={2602.12173},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.12173},
}
```
