---
layout: default
title: Models
permalink: /models/
nav_order: 2
---

# Model Zoo

## EfficientSAM3 Full Models (Lightweight Image + Text Encoders)

EfficientSAM3 compresses both SAM3's vision encoder and text encoder into lightweight student models while maintaining competitive performance on downstream benchmarks.

| Model | Vision | Text | Decoder | Total | vs SAM3 (1.4B) | Download |
|-------|--------|------|---------|-------|-----------------|----------|
| **EV-M** | 4.6M | 4.07M | ~605M | ~614M | **56% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_efficientvit.pt) |
| **RV-M** | 7.8M | 4.07M | ~605M | ~617M | **56% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_repvit.pt) |
| **TV-M** | 10.6M | 4.07M | ~605M | ~620M | **56% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_tinyvit.pt) |

> **SAM3 Teacher**: 1.4B total (Vision: 461M + Text: 354M + Decoder/Heads: ~600M)

## SAM3-LiteText Models (Lightweight Text Encoder Only)

SAM3-LiteText keeps the SAM3 vision encoder but replaces the text encoder with lightweight MobileCLIP variants.

| Model | Text Encoder | Context | Full Model | vs SAM3 (1.4B) | Download |
|-------|-------------|---------|------------|-----------------|----------|
| **LiteText-S0-16** | MobileCLIP-S0 | ctx16 | ~1050M | **25% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip_s0_ctx16.pt) |
| **LiteText-S0-32** | MobileCLIP-S0 | ctx32 | ~1050M | **25% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip_s0_ctx32.pt) |
| **LiteText-S1-16** | MobileCLIP-S1 | ctx16 | ~1051M | **25% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip_s1_ctx16.pt) |
| **LiteText-S1-32** | MobileCLIP-S1 | ctx32 | ~1051M | **25% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip_s1_ctx32.pt) |
| **LiteText-L-16** | MobileCLIP2-L | ctx16 | ~1088M | **22% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip2_l_ctx16.pt) |
| **LiteText-L-32** | MobileCLIP2-L | ctx32 | ~1088M | **22% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip2_l_ctx32.pt) |

> Full model = Vision (~1050M) + Text encoder + Decoder (~600M). Text encoder: 4.07M (S0), 4.69M (S1), 42.38M (L) replaces 354M SAM3 text encoder.

## Comparison with SAM3

| Model | Vision Encoder | Text Encoder | Decoder/Heads | Total Params | vs SAM3 (1.4B) |
|-------|---------------|--------------|---------------|--------------|-----------------|
| **SAM3 Teacher** | ViT-H (461M) | SAM3 Text (354M) | ~600M | ~1.4B | - |
| **EfficientSAM3 EV-M** | EfficientViT (4.6M) | MobileCLIP-S0 (4.07M) | ~605M | ~614M | **56% smaller** |
| **SAM3-LiteText S0** | ViT-H (461M) | MobileCLIP-S0 (4.07M) | ~600M | ~1050M | **25% smaller** |
