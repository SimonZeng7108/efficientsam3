# EfficientSAM3 v0.4.0 — Stage 3 Fine-Tuned Models & Full PCS Release

**EfficientSAM3 — v0.4.0 Fine-Tuned PCS Models (2026-06-11)**

We're excited to announce the release of **EfficientSAM3 Stage 3 fine-tuned models**, bringing full Promptable Concept Segmentation (PCS) capabilities to lightweight student models!

## What's New

### Stage 3 Fine-Tuned Full Models Released

EfficientSAM3 full models (EV-M, RV-M, TV-M) are now fine-tuned on 5% SA1B data & SACap labels with complete PCS capabilities:

| Model | Vision | Text | Total | vs SAM3 (1.4B) |
|-------|--------|------|-------|----------------|
| **EV-M** | EfficientViT-B1 (4.6M) | MobileCLIP-S0 (4.07M) | ~614M | **56% smaller** |
| **RV-M** | RepViT-M1.1 (7.8M) | MobileCLIP-S0 (4.07M) | ~617M | **56% smaller** |
| **TV-M** | TinyViT-11M (10.6M) | MobileCLIP-S0 (4.07M) | ~620M | **56% smaller** |

Checkpoints available on [HuggingFace](https://huggingface.co/Simon7108528/EfficientSAM3/tree/main/efficientsam3_ft).

### Three-Stage Progressive Distillation

EfficientSAM3 now delivers the complete distillation pipeline:

- **Stage 1**: Compact encoder distillation on SA-1B (image) + Recap-DataComp-1B (text)
- **Stage 2**: Temporal memory alignment on SA-V (coming soon)
- **Stage 3**: End-to-end fine-tuning on SAM3 data for full PCS quality

### HuggingFace Integration

SAM3-LiteText is officially integrated into [HuggingFace Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/sam3_lite_text). Try the [live demo](https://huggingface.co/spaces/nielsr/sam-3-lite-text-vs-sam-3)!

## Contributors

Thanks to our community contributors:

- [@NielsRogge](https://github.com/NielsRogge), [@yonigozlan](https://github.com/yonigozlan): SAM3-LiteText HuggingFace integration
- [@colinlin1982](https://github.com/colinlin1982): Model trimming script, EfficientSAM3.1
- [@clcl777](https://github.com/clcl777): Multi-device support

## Get Started

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image

# Load fine-tuned model
model = build_sam3_image_model(
    checkpoint_path="efficientsam3_tinyvit_11m_mobileclip_s0_ctx16_5p_full.pt",
    load_from_HF=False,
)

processor = Sam3Processor(model)
image = Image.open("your_image.jpg").convert("RGB")
state = processor.set_image(image)
state = processor.set_text_prompt("dog", state)
masks = state["masks"]
print(f"Found {len(masks)} masks")
```

## Documentation

- [Full README](README.md) — Installation, quick start, training
- [Stage 1 README](README_stage1.md) — Encoder distillation training
- [Stage 3 README](README_stage3.md) — Full fine-tuning pipeline
- [Dataset Guide](README_dataset.md) — Data preparation
- [Project Page](https://SimonZeng7108.github.io/efficientsam3/) — Live demo & model zoo

## What's Next

- Stage 2 memory bank aligned models
- ONNX/CoreML export for mobile deployment
- Interactive web demo
