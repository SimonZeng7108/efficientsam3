---
layout: page
title: Documentation
permalink: /docs/
nav_order: 3
---

# Documentation

## Quick Links

- [Installation]({{ site.baseurl }}/#installation) - Setup instructions
- [Models]({{ site.baseurl }}/models/) - Available model checkpoints
- [Usage Examples]({{ site.baseurl }}/#quick-start) - Code examples
- [GitHub Repository](https://github.com/SimonZeng7108/efficientsam3) - Full source code

## Installation

```bash
git clone https://github.com/SimonZeng7108/efficientsam3
cd efficientsam3
pip install -e ".[stage1]"
```

**Requirements:**
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)

## Training Stages

### Stage 1: Encoder Distillation

Compress SAM3's vision and text encoders into lightweight backbones. See [README_stage1.md](https://github.com/SimonZeng7108/efficientsam3/blob/main/README_stage1.md).

**Vision Encoders:**
- TinyViT (5M, 11M, 21M params)
- RepViT (2M, 8M, 19M params)
- EfficientViT (1M, 5M, 9M params)

**Text Encoders:**
- MobileCLIP-S0 (4.07M params)
- MobileCLIP-S1 (4.69M params)
- MobileCLIP2-L (42.38M params)

### Stage 1+: Geometry Fine-tuning

Fine-tune with prompt-in-the-loop for improved encoder performance. See the `stage1_geometry_finetune` branch.

### Stage 3: Joint Fine-Tuning

End-to-end fine-tuning on SAM3 dataset with full PCS capabilities. See [README_stage3.md](https://github.com/SimonZeng7108/efficientsam3/blob/main/README_stage3.md).

## Usage

### EfficientSAM3 (Full Model)

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

### SAM3-LiteText

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image

model = build_sam3_image_model(
    checkpoint_path="sam3_litetext_mobileclip_s0_ctx16.pt",
    text_encoder_type="MobileCLIP-S0",
    text_encoder_context_length=16,
    load_from_HF=False,
)

processor = Sam3Processor(model)
image = Image.open("your_image.jpg").convert("RGB")
state = processor.set_image(image)
state = processor.set_text_prompt("person", state)
masks = state["masks"]
```

## Dataset Preparation

For dataset setup and download scripts covering COCO, DAVIS, LVIS, SA-1B, SA-V, LVOS, MOSE, and YouTube-VOS, see [README_dataset.md](README_dataset.md).

## Evaluation

### COCO Evaluation

```bash
python eval/eval_coco.py --coco_root data/coco --output_dir output
```

### Text Encoder Similarity

```bash
python eval/eval_text_encoder_similarity.py \
  --student-ckpt /path/to/student_text_encoder_1.pth /path/to/student_text_encoder_2.pth \
  --np-json data/sa-v-text/sa-co-veval/saco_veval_noun_phrases.json \
  --device cuda
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or use a smaller model variant.

### Checkpoint Not Found

Download checkpoints from [HuggingFace](https://huggingface.co/Simon7108528/EfficientSAM3).

## Citation

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
