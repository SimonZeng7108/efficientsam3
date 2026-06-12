---
layout: default
title: Models
permalink: /models/
nav_order: 2
---

# Model Zoo

## EfficientSAM3 Full Models (Lightweight Image + Text Encoders)

EfficientSAM3 compresses both SAM3's vision encoder and text encoder into lightweight student models while maintaining competitive performance on downstream benchmarks.

| Model | Vision | Text | Transformer | Other | Params | vs ImageSAM3 | Download |
|-------|--------|------|-------------|-------|--------|--------------|----------|
| **EV-M** | 22.2M | 42.5M | 21.0M | 3.5M | **89.2M** | **90% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_efficientvit.pt) |
| **RV-M** | 25.6M | 42.5M | 21.0M | 3.5M | **92.7M** | **89% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_repvit.pt) |
| **TV-M** | 28.3M | 42.5M | 21.0M | 3.5M | **95.3M** | **89% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_tinyvit.pt) |

> **Note:** "Text" is the distilled text encoder. "Transformer" is the mask decoder. "Other" includes segmentation head + scoring. ImageSAM3 (for comparison): Vision: 463M + Text: 354M + Transformer: 30.3M + Other: 14.2M = **861.5M**

## SAM3-LiteText Models (Lightweight Text Encoder Only)

SAM3-LiteText keeps the SAM3 vision encoder but replaces the text encoder with lightweight MobileCLIP variants.

| Model | Vision | Text | Transformer | Other | Params | vs ImageSAM3 | Download |
|-------|--------|------|-------------|-------|--------|--------------|----------|
| **LiteText-S0-16** | 463.0M | 42.5M | 30.3M | 14.2M | **550.0M** | **36% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip_s0_ctx16.pt) |
| **LiteText-S0-32** | 463.0M | 42.5M | 30.3M | 14.2M | **550.0M** | **36% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip_s0_ctx32.pt) |
| **LiteText-S1-16** | 463.0M | 63.5M | 30.3M | 14.2M | **571.0M** | **34% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip_s1_ctx16.pt) |
| **LiteText-S1-32** | 463.0M | 63.5M | 30.3M | 14.2M | **571.0M** | **34% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip_s1_ctx32.pt) |
| **LiteText-L-16** | 463.0M | 123.8M | 30.3M | 14.2M | **631.3M** | **27% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip2_l_ctx16.pt) |
| **LiteText-L-32** | 463.0M | 123.8M | 30.3M | 14.2M | **631.3M** | **27% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip2_l_ctx32.pt) |

> **Note:** "Text" is the distilled text encoder (42.5M-123.8M). SAM3-LiteText keeps SAM3's ViT-H vision encoder (~463M) but replaces the text encoder. "Other" includes geometry encoder + segmentation head + scoring.

## Comparison with SAM3

| Model | Vision Encoder | Text Encoder | Transformer | Other | Params | vs ImageSAM3 |
|-------|---------------|-------------|-------------|-------|--------|--------------|
| **ImageSAM3 Teacher** | ViT-H (463M) | SAM3 Text (354M) | 30.3M | 14.2M | **861.5M** | - |
| **EfficientSAM3 EV-M** | EfficientViT (22.2M) | Distilled (42.5M) | 21.0M | 3.5M | **89.2M** | **90% smaller** |
| **SAM3-LiteText S0** | ViT-H (463M) | Distilled (42.5M) | 30.3M | 14.2M | **550.0M** | **36% smaller** |
