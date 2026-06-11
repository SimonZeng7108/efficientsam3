# EfficientSAM3: Progressive Hierarchical Knowledge Distillation from SAM1, 2 and 3

[Chengxi Simon Zeng](https://simonzeng7108.github.io/about/)<sup>1,†</sup>, [Yuxuan Jiang](https://YuxuanJJ.github.io/)<sup>1</sup>, [Gao Ge](https://scholar.google.com/citations?user=j2_80ewAAAAJ&hl=en)<sup>1</sup>, [Shuai Wang](https://shuaiwang97.github.io/)<sup>2</sup>, [Duolikun Danier](https://danier97.github.io/)<sup>3</sup>, [Bin Zhu](https://binzhubz.github.io/)<sup>4</sup>, [Stevan Rudinac](https://stevanrudinac.com/)<sup>2</sup>, [David Bull](https://david-bull.github.io/)<sup>1</sup>, [Fan Aaron Zhang](https://fan-aaron-zhang.github.io/)<sup>1</sup>

<sup>1</sup>Visual Information Lab, University of Bristol; <sup>2</sup>MultiX lab, University of Amsterdam; <sup>3</sup>University of Edinburgh; <sup>4</sup>Singapore Management University

<sup>†</sup>Tech Lead & Corresponding Author

[![arXiv](https://img.shields.io/badge/arXiv-EfficientSAM3-b31b1b.svg)](https://arxiv.org/abs/2511.15833) [![arXiv](https://img.shields.io/badge/arXiv-SAM3--LiteText-b31b1b.svg)](https://arxiv.org/abs/2602.12173) [![Project Page](https://img.shields.io/badge/Project-Page-green)](https://simonzeng7108.github.io/efficientsam3/) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-EfficientSAM3-blue)](https://huggingface.co/Simon7108528/EfficientSAM3) [![Discord](https://img.shields.io/badge/Discord-Join-7289da?logo=discord&logoColor=white)](https://discord.gg/FMyaQca7xT)

---

## Updates

<details>
<summary><b>Click to expand all updates</b></summary>

- **[2026/04/19]** **SAM3-LiteText** is live on HuggingFace main branch, [[Docs](https://huggingface.co/docs/transformers/main/en/model_doc/sam3_lite_text)], [[Model](https://huggingface.co/yonigozlan/sam3-litetext-s0)], [[Demo](https://huggingface.co/spaces/nielsr/sam-3-lite-text-vs-sam-3)], and accepted by [ICMR2026](https://icmr2026.org/)- Same performance to SAM3 but much smaller! Thanks to @NielsRogge, @yonigozlan and HF integration team.
- **[2026/04/13]** **EfficientSAM3.1** and **SAM3.1-LiteText** image models were released on the [`stage1_sam3.1`](https://github.com/SimonZeng7108/efficientsam3/tree/data_engine) branch. SAM3-LiteText has also been officially merged into [HuggingFace Transformers](https://github.com/huggingface/transformers/pull/44320). Stage 3 data engine support is now available on the [`data_engine`](https://github.com/SimonZeng7108/efficientsam3/tree/data_engine/data_engine) branch.
- **[2026/02/18]** **SAM3-LiteText** released! SAM3-LiteText reduces text encoder parameters by up to 88% with similar performance to the original text encoder. [Paper](https://arxiv.org/abs/2602.12173) available on arXiv.
- **[2026/01/11]** Stage 1 geometry-prompt fine-tuned (**ft**) weights released/updated (image encoders on 1% SA-1B; text encoders fine-tuned on SA-Co Gold+Silver).
- **[2025/12/08]** Stage 1 text encoder weights released for all 3 variants (MobileCLIP S0, S1, and MobileCLIP2 L) - distilled on 1% Recap-DataComp-1B dataset.
- **[2025/12/02]** Stage 1 image encoder weights released for all 9 variants (RepViT, TinyViT, EfficientViT) - unsupervised distilled on 1% of SA-1B dataset.
- **[2025/11/25]** Teaser model released. See Above. More models are baking in the oven🔥.
- **[2025/10/18]** Project announced. Code and weights are not released yet; they will be published once SAM3 code is publicly available.

</details>

---

## Model Zoo

### EfficientSAM3 Full Models (Image + Text Encoders)

EfficientSAM3 compresses both SAM3's vision encoder and text encoder into lightweight student models while maintaining competitive performance on downstream benchmarks.

<p align="center">
  <img src="https://raw.githubusercontent.com/SimonZeng7108/efficientsam3/main/images/dog_person_example_dog.png" width="45%">
  <img src="https://raw.githubusercontent.com/SimonZeng7108/efficientsam3/main/images/dog_person_example_person.png" width="45%">
</p>

| Model | Image Encoder | Text Encoder | Total Params | vs SAM3 (1.4B) | Download |
|-------|---------------|--------------|---------------|------------------|----------|
| **EV-M** | EfficientViT-B1 (4.6M) | MobileCLIP-S0 (4.07M) | **~10M** | **99.3% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_efficientvit/efficientsam3_efficientvit_b1_mobileclip_s0_ctx16_5p_full.pt) |
| **RV-M** | RepViT-M1.1 (7.8M) | MobileCLIP-S0 (4.07M) | **~12M** | **99.1% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_repvit/efficientsam3_repvit_m1_1_mobileclip_s0_ctx16_5p_full.pt) |
| **TV-M** | TinyViT-11M (10.6M) | MobileCLIP-S0 (4.07M) | **~15M** | **98.9% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_tinyvit/efficientsam3_tinyvit_11m_mobileclip_s0_ctx16_5p_full.pt) |

> **SAM3 Teacher**: 1.4B total params (Vision: 461M, Text: 354M, Decoder/Heads: ~600M)

### SAM3-LiteText Models (Text Encoder Only)

SAM3-LiteText replaces the heavy SAM3 text encoder (~354M params) with lightweight MobileCLIP variants, reducing text encoder parameters by up to **88%** while maintaining similar text-query segmentation performance.

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

### Image Segmentation with EfficientSAM3

```python
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image

# Load model
model = build_sam3_image_model(
    checkpoint_path="efficient_sam3_tvm_m_mobileclip_s0_ctx16_5p_full.pt",
    enable_segmentation=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Load and process image
processor = Sam3Processor(model)
image = Image.open("your_image.jpg").convert("RGB")
state = processor.set_image(image)

# Text prompt segmentation
state = processor.set_text_prompt(state, "dog")

# Get masks
masks = state["masks"]
scores = state["scores"]
print(f"Found {len(masks)} masks")
```

### Text Encoder Replacement with SAM3-LiteText

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Build model with LiteText encoder
model = build_sam3_image_model(
    checkpoint_path="sam3_litetext_mobileclip_s0_ctx16.pt",
    text_encoder_type="MobileCLIP-S0",
    text_encoder_context_length=16,
    enable_segmentation=True,
)

# Use as normal
processor = Sam3Processor(model)
```

---

## Directory Structure

```
efficientsam3/
├── sam3/                    # SAM3 model code
│   ├── model_builder.py     # Model building utilities
│   ├── model/               # Model components
│   └── assets/              # BPE vocab, sample images
├── stage1/                  # Stage 1: Encoder distillation
├── stage2/                  # Stage 2: Text encoder distillation
├── stage3/                  # Stage 3: Joint fine-tuning
├── scripts/                  # Training and evaluation scripts
└── data/                    # Dataset utilities
```

---

## Examples

See [sam3/examples/](sam3/examples/) for interactive Jupyter notebooks:

- `sam3_image_interactive.ipynb` - Interactive image segmentation
- `sam3_video_predictor_example.ipynb` - Video segmentation
- `sam3_agent.ipynb` - SAM3 as an agent for complex tasks

---

## Development

### Active Development Branches

| Branch | Purpose |
|--------|---------|
| `main` | Stable release (EfficientSAM3 + SAM3-LiteText) |
| `stage1_sam3.1` | SAM3.1 image encoder models |
| `data_engine` | Stage 3 data engine and training |
| `sam3_litetext` | SAM3-LiteText development |

### TODO

- [ ] Stage 3 full fine-tuning on downstream benchmarks
- [ ] Video segmentation optimization
- [ ] ONNX/TensorRT export
- [ ] Web demo deployment
- [ ] Additional backbone support (EfficientNet, etc.)

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

For major changes, please open an issue first to discuss what you would like to change.

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
