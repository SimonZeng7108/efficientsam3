---
layout: page
title: Models
permalink: /models/
---

# Model Zoo

## EfficientSAM3 Full Models

These models replace both the vision encoder and text encoder with lightweight students.

| Model | Image Encoder | Text Encoder | Total Params | vs SAM3 (1.4B) | Download |
|-------|---------------|--------------|---------------|------------------|----------|
| **EV-M** | EfficientViT-B1 (4.6M) | MobileCLIP-S0 (4.07M) | ~10M | 99.3% smaller | [HuggingFace](https://huggingface.co/Simon7108528/EfficientSAM3/tree/main/efficientsam3_ft/efficientsam3_efficientvit) |
| **RV-M** | RepViT-M1.1 (7.8M) | MobileCLIP-S0 (4.07M) | ~12M | 99.1% smaller | [HuggingFace](https://huggingface.co/Simon7108528/EfficientSAM3/tree/main/efficientsam3_ft/efficientsam3_repvit) |
| **TV-M** | TinyViT-11M (10.6M) | MobileCLIP-S0 (4.07M) | ~15M | 98.9% smaller | [HuggingFace](https://huggingface.co/Simon7108528/EfficientSAM3/tree/main/efficientsam3_ft/efficientsam3_tinyvit) |

## SAM3-LiteText Models

These models replace only the text encoder with lightweight MobileCLIP variants.

| Model | Text Encoder | Context | Params | vs SAM3 Text (354M) | Download |
|-------|-------------|---------|--------|---------------------|----------|
| **LiteText-S0-16** | MobileCLIP-S0 | ctx16 | 4.07M | 98.9% smaller | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip_s0_ctx16.pt) |
| **LiteText-S0-32** | MobileCLIP-S0 | ctx32 | 4.07M | 98.9% smaller | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip_s0_ctx32.pt) |
| **LiteText-S1-16** | MobileCLIP-S1 | ctx16 | 4.69M | 98.7% smaller | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip_s1_ctx16.pt) |
| **LiteText-S1-32** | MobileCLIP-S1 | ctx32 | 4.69M | 98.7% smaller | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip_s1_ctx32.pt) |
| **LiteText-L-16** | MobileCLIP2-L | ctx16 | 42.38M | 88.0% smaller | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip2_l_ctx16.pt) |
| **LiteText-L-32** | MobileCLIP2-L | ctx32 | 42.38M | 88.0% smaller | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip2_l_ctx32.pt) |

## Comparison with SAM3

| Model | Vision Encoder | Text Encoder | Total Params | vs SAM3 (1.4B) |
|-------|---------------|--------------|---------------|------------------|
| **SAM3 Teacher** | ViT-H (461M) | SAM3 Text (354M) | ~1.4B | - |
| **EfficientSAM3** | ~10-15M | ~4-42M | ~15-60M | **98.9-99.3% smaller** |
