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

## Training Stages

### Stage 1: Encoder Distillation

Compress SAM3's vision and text encoders into lightweight backbones.

**Vision Encoders:**
- TinyViT (5M, 11M, 21M params)
- RepViT (2M, 8M, 19M params)
- EfficientViT (1M, 5M, 9M params)

**Text Encoders:**
- MobileCLIP-S0 (4.07M params)
- MobileCLIP-S1 (4.69M params)
- MobileCLIP2-L (42.38M params)

### Stage 2: Text Encoder Distillation

Fine-tune text encoders with improved positional embeddings.

### Stage 3: Joint Fine-Tuning

Combine trained encoders with SAM3 decoder heads for full model fine-tuning.

## API Reference

### build_sam3_image_model

```python
def build_sam3_image_model(
    checkpoint_path: str,
    text_encoder_type: str = "MobileCLIP-S0",
    text_encoder_context_length: int = 16,
    enable_segmentation: bool = True,
    device: str = "cuda",
):
```

### Sam3Processor

```python
processor = Sam3Processor(model)
state = processor.set_image(image)
state = processor.set_text_prompt(state, "dog")
masks = state["masks"]
```

## Dataset Preparation

See [README_dataset.md](README_dataset.md) for dataset download instructions.

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or use a smaller model variant.

### Checkpoint Not Found

Download checkpoints from [HuggingFace](https://huggingface.co/Simon7108528/EfficientSAM3).
