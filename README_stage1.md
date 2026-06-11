## Stage 1 — Encoder Distillation

Stage 1 compresses SAM3's vision encoder and text encoder into lightweight student backbones using knowledge distillation.

### Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 1: ENCODER DISTILLATION           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Vision: SAM3 ViT-H → RepViT / TinyViT / EfficientViT    │
│  Text:   SAM3 Text Encoder → MobileCLIP variants           │
│                                                             │
│  Loss: MSE + Cosine Similarity on embedding maps           │
└─────────────────────────────────────────────────────────────┘
```

### Parameter Comparison

| Model | Vision Encoder | Text Encoder | Total Params | vs SAM3 (1.4B) |
|-------|---------------|--------------|---------------|------------------|
| **ES-EV-S** | EfficientViT-B0 (0.68M) | - | 0.68M | 99.95% smaller |
| **ES-EV-M** | EfficientViT-B1 (4.61M) | - | 4.61M | 99.67% smaller |
| **ES-EV-L** | EfficientViT-B2 (9.12M) | - | 9.12M | 99.35% smaller |
| **ES-RV-S** | RepViT-M0.9 (1.92M) | - | 1.92M | 99.86% smaller |
| **ES-RV-M** | RepViT-M1.1 (7.81M) | - | 7.81M | 99.44% smaller |
| **ES-RV-L** | RepViT-M2.3 (19.3M) | - | 19.3M | 98.62% smaller |
| **ES-TV-S** | TinyViT-5M (5.29M) | - | 5.29M | 99.62% smaller |
| **ES-TV-M** | TinyViT-11M (10.6M) | - | 10.6M | 99.24% smaller |
| **ES-TV-L** | TinyViT-21M (21.4M) | - | 21.4M | 98.47% smaller |
| **ES-MC-S** | - | MobileCLIP-S0 (42.57M) | 42.57M | 87.96% smaller |
| **ES-MC-M** | - | MobileCLIP-S1 (63.56M) | 63.56M | 82.03% smaller |
| **ES-MC-L** | - | MobileCLIP2-L (123.6M) | 123.6M | 65.06% smaller |

### Training Pipeline

1. **Save Teacher Embeddings** - Run SAM3 teacher to save image/text embeddings
2. **Train Student Encoders** - Distill embeddings to lightweight backbones
3. **Merge with SAM3** - Splice student weights into full SAM3 checkpoint

### Quick Start

```bash
# Save teacher embeddings
bash stage1/scripts/save_image_embeddings.sh

# Train vision student (e.g., TinyViT)
bash stage1/scripts/train_image_student.sh \
  CFG=stage1/configs/es_tv_m.yaml \
  OUTPUT=output/stage1/es_tv_m

# Train text student (MobileCLIP)
bash stage1/scripts/train_text_student.sh \
  CFG=stage1/configs/es_mc_s.yaml \
  OUTPUT=output/stage1_text/mobileclip_s

# Merge both encoders into SAM3
python stage1/convert_both_encoders_weights_stage1.py \
  --image-student-ckpt output/stage1/es_tv_m/ckpt_epoch_49.pth \
  --text-student-ckpt output/stage1_text/mobileclip_s/ckpt_epoch_49.pth \
  --sam3-ckpt sam3_checkpoints/sam3.pt \
  --image-model-name tiny_vit_11m \
  --text-model-name mobileclip_s0
```

### Configuration Files

| Model | Vision Config | Text Config |
|-------|--------------|--------------|
| ES-EV-* | `stage1/configs/es_ev_*.yaml` | - |
| ES-RV-* | `stage1/configs/es_rv_*.yaml` | - |
| ES-TV-* | `stage1/configs/es_tv_*.yaml` | - |
| ES-MC-S0 | - | `stage1/configs/es_mc_s_pretrained.yaml` |
| ES-MC-S1 | - | `stage1/configs/es_mc_m.yaml` |
| ES-MC-L | - | `stage1/configs/es_mc_l.yaml` |

### Output Structure

```
output/
├── stage1_teacher/           # Teacher embeddings
│   └── embeddings/
├── stage1/                   # Vision student checkpoints
│   └── es_tv_m/
│       └── ckpt_epoch_*.pth
├── stage1_text/              # Text student checkpoints
│   └── mobileclip_s/
│       └── ckpt_epoch_*.pth
└── efficient_sam3_*.pth      # Merged models
```

### Prerequisites

- **SAM3 checkpoint**: Download from [HuggingFace](https://huggingface.co/facebook/sam3/tree/main)
- **SA-1B dataset**: For vision distillation
- **Recap-DataComp-1B**: For text distillation

See [README_dataset.md](README_dataset.md) for dataset download instructions.
