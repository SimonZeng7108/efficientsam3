# EfficientSAM3: Progressive Hierarchical Knowledge Distillation from SAM1, 2 and 3

[Chengxi Simon Zeng](https://simonzeng7108.github.io/about/)<sup>1,†</sup>, [Yuxuan Jiang](https://YuxuanJJ.github.io/)<sup>1</sup>, [Gao Ge](https://scholar.google.com/citations?user=j2_80ewAAAAJ&hl=en)<sup>1</sup>, [Shuai Wang](https://shuaiwang97.github.io/)<sup>2</sup>, [Duolikun Danier](https://danier97.github.io/)<sup>3</sup>, [Bin Zhu](https://binzhubz.github.io/)<sup>4</sup>, [Stevan Rudinac](https://stevanrudinac.com/)<sup>2</sup>, [David Bull](https://david-bull.github.io/)<sup>1</sup>, [Fan Aaron Zhang](https://fan-aaron-zhang.github.io/)<sup>1</sup>

<sup>1</sup>Visual Information Lab, University of Bristol; <sup>2</sup>MultiX lab, University of Amsterdam; <sup>3</sup>University of Edinburgh; <sup>4</sup>Singapore Management University

<sup>†</sup>Tech Lead & Corresponding Author

[![arXiv](https://img.shields.io/badge/arXiv-EfficientSAM3-b31b1b.svg)](https://arxiv.org/abs/2511.15833) [![arXiv](https://img.shields.io/badge/arXiv-SAM3--LiteText-b31b1b.svg)](https://arxiv.org/abs/2602.12173) [![Project Page](https://img.shields.io/badge/Project-Page-green)](https://simonzeng7108.github.io/efficientsam3/) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-EfficientSAM3-blue)](https://huggingface.co/Simon7108528/EfficientSAM3) [![Discord](https://img.shields.io/badge/Discord-Join-7289da?logo=discord&logoColor=white)](https://discord.gg/FMyaQca7xT)

---

## Updates

- **[2026/06/11]** **Stage 3 Fine-tuned Models Released!** EfficientSAM3 full models (EV-M, RV-M, TV-M) fine-tuned on 5% SA1B data & SACap labels. Checkpoints on [HuggingFace](https://huggingface.co/Simon7108528/EfficientSAM3/tree/main/efficientsam3_ft). See [README_stage3.md](README_stage3.md).
- **[2026/04/19]** **SAM3-LiteText** live on HuggingFace, accepted by [ICMR2026](https://icmr2026.org/)! [[Docs](https://huggingface.co/docs/transformers/main/en/model_doc/sam3_lite_text)] [[Demo](https://huggingface.co/spaces/nielsr/sam-3-lite-text-vs-sam-3)] Thanks @NielsRogge, @yonigozlan!

<details>
<summary><b>Older updates</b></summary>

- **[2026/02/18]** **SAM3-LiteText** released! Reduces text encoder by 88% with similar performance. [Paper](https://arxiv.org/abs/2602.12173)
- **[2026/01/11]** Stage 1 geometry-prompt fine-tuned weights released.
- **[2025/12/08]** Stage 1 text encoder weights released (MobileCLIP S0, S1, MobileCLIP2 L).
- **[2025/12/02]** Stage 1 image encoder weights released (RepViT, TinyViT, EfficientViT).

</details>

---

## Model Zoo

### EfficientSAM3 Full Models (Image + Text Encoders)

EfficientSAM3 compresses both SAM3's vision encoder and text encoder into lightweight student models while maintaining competitive performance on downstream benchmarks.

<p align="center">
  <img src="https://raw.githubusercontent.com/SimonZeng7108/efficientsam3/main/images/dog_person_example_dog.png" width="45%">
  <img src="https://raw.githubusercontent.com/SimonZeng7108/efficientsam3/main/images/dog_person_example_person.png" width="45%">
</p>

| Model | Vision | Text | Decoder | Total | vs SAM3 (1.4B) | Download |
|-------|--------|------|---------|-------|-----------------|----------|
| **EV-M** | 4.6M | 4.07M | ~605M | **~614M** | **56% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_efficientvit.pt?download=true) |
| **RV-M** | 7.8M | 4.07M | ~605M | **~617M** | **56% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_repvit.pt?download=true) |
| **TV-M** | 10.6M | 4.07M | ~605M | **~620M** | **56% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_tinyvit.pt?download=true) |

> **SAM3 Teacher**: 1.4B total (Vision: 461M + Text: 354M + Decoder/Heads: ~600M)

### SAM3-LiteText Models (Text Encoder Only)

SAM3-LiteText keeps the SAM3 vision encoder but replaces the text encoder with lightweight MobileCLIP variants.

| Model | Text Encoder | Context | Params | vs SAM3 Text (354M) | Download |
|-------|-------------|---------|--------|---------------------|----------|
| **LiteText-S0-16** | MobileCLIP-S0 | ctx16 | **4.07M** | **98.9% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip_s0_ctx16.pt) |
| **LiteText-S0-32** | MobileCLIP-S0 | ctx32 | **4.07M** | **98.9% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip_s0_ctx32.pt) |
| **LiteText-S1-16** | MobileCLIP-S1 | ctx16 | **4.69M** | **98.7% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip_s1_ctx16.pt) |
| **LiteText-S1-32** | MobileCLIP-S1 | ctx32 | **4.69M** | **98.7% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip_s1_ctx32.pt) |
| **LiteText-L-16** | MobileCLIP2-L | ctx16 | **42.38M** | **88.0% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip2_l_ctx16.pt) |
| **LiteText-L-32** | MobileCLIP2-L | ctx32 | **42.38M** | **88.0% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip2_l_ctx32.pt) |

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

### EfficientSAM3

EfficientSAM3 replaces both the SAM3 vision encoder and text encoder with lightweight student models.

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image

# Load model (TV-M example - student vision + student text)
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

---

## Citation

If you use EfficientSAM3 or SAM3-LiteText in your research, please cite:

```bibtex
@article{zeng2025efficientsam3,
  title={EfficientSAM3: Progressive Hierarchical Knowledge Distillation from SAM1, 2 and 3},
  author={Zeng, Chengxi and Jiang, Yuxuan and Ge, Gao and others},
  journal={arXiv preprint arXiv:2511.15833},
  year={2025}
}

@article{zeng2025sam3litetext,
  title={SAM3-LiteText: An Anatomical Study of the SAM3 Text Encoder for Efficient Vision-Language Segmentation},
  author={Zeng, Chengxi and Jiang, Yuxuan and Ge, Gao and others},
  journal={arXiv preprint arXiv:2602.12173},
  year={2025}
}
```

---

## License

This project is licensed under the terms specified by the original SAM3 model and Meta AI research.
