### EfficientSAM3: Progressive Hierachical Knowledge Distillation (PhD) from SAM1, 2 and 3
[Chengxi Simon Zeng](https://simonzeng7108.github.io/about/)<sup>1,†</sup>, [Yuxuan Jiang](https://YuxuanJJ.github.io/)<sup>1</sup>, [Gao Ge](https://scholar.google.com/citations?user=j2_80ewAAAAJ&hl=en)<sup>1</sup>, [Shuai Wang](https://shuaiwang97.github.io/)<sup>2</sup>, [Duolikun Danier](https://danier97.github.io/)<sup>3</sup>, [Bin Zhu](https://binzhubz.github.io/)<sup>4</sup>, [Stevan Rudinac](https://stevanrudinac.com/)<sup>2</sup>, [David Bull](https://david-bull.github.io/)<sup>1</sup>, [Fan Aaron Zhang](https://fan-aaron-zhang.github.io/)<sup>1</sup>

<sup>1</sup>Visual Information Lab, University of Bristol; <sup>2</sup>MultiX lab, University of Amsterdam; <sup>3</sup>University of Edinburgh; <sup>4</sup>Singapore Management University

<sup>†</sup>Tech Lead & Corresponding Author

[![arXiv](https://img.shields.io/badge/arXiv-EfficientSAM3-b31b1b.svg)](https://arxiv.org/abs/2511.15833) [![arXiv](https://img.shields.io/badge/arXiv-SAM3--LiteText-b31b1b.svg)](https://arxiv.org/abs/2602.12173) [![Project Page](https://img.shields.io/badge/Project-Page-green)](https://simonzeng7108.github.io/efficientsam3/) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-EfficientSAM3-blue)](https://huggingface.co/Simon7108528/EfficientSAM3) [![Discord](https://img.shields.io/badge/Discord-Join-7289da?logo=discord&logoColor=white)](https://discord.gg/FMyaQca7xT)

---

## Project Overview

This repository contains two related projects:

1. **EfficientSAM3** - Progressive Hierarchical Knowledge Distillation from SAM1, 2 and 3
2. **SAM3-LiteText** - An Anatomical Study of the SAM3 Text Encoder for Efficient Vision-Language Segmentation

## Updates

- **[2026/04/19]** **SAM3-LiteText** is live on HuggingFace main branch, [[Docs](https://huggingface.co/docs/transformers/main/en/model_doc/sam3_lite_text)], [[Model](https://huggingface.co/yonigozlan/sam3-litetext-s0)], [[Demo](https://huggingface.co/spaces/nielsr/sam-3-lite-text-vs-sam-3)], and accepted by [ICMR2026](https://icmr2026.org/)- Same performance to SAM3 but much smaller! Thanks to @NielsRogge, @yonigozlan and HF integration team.
- **[2026/04/13]** **EfficientSAM3.1** and **SAM3.1-LiteText** image models were released on the [`stage1_sam3.1`](https://github.com/SimonZeng7108/efficientsam3/tree/data_engine) branch. SAM3-LiteText has also been officially merged into [HuggingFace Transformers](https://github.com/huggingface/transformers/pull/44320). Stage 3 data engine support is now available on the [`data_engine`](https://github.com/SimonZeng7108/efficientsam3/tree/data_engine/data_engine) branch.
- **[2026/02/18]** **SAM3-LiteText** released! SAM3-LiteText reduces text encoder parameters by up to 88% with similar performance to the original text encoder. [Paper](https://arxiv.org/abs/2602.12173) available on arXiv. Code available in [`sam3_litetext`](https://github.com/SimonZeng7108/efficientsam3/tree/sam3_litetext) branch and weights on [Hugging Face](https://huggingface.co/Simon7108528/EfficientSAM3/tree/main/sam3_litetext).
- **[2026/01/11]** Stage 1 geometry-prompt fine-tuned (**ft**) weights released/updated (image encoders on 1% SA-1B; text encoders fine-tuned on SA-Co Gold+Silver).
- **[2025/12/08]** Stage 1 text encoder weights released for all 3 variants (MobileCLIP S0, S1, and MobileCLIP2 L) - distilled on 1% Recap-DataComp-1B dataset.
- **[2025/12/02]** Stage 1 image encoder weights released for all 9 variants (RepViT, TinyViT, EfficientViT) - unsupervised distilled on 1% of SA-1B dataset.
- **[2025/11/25]** Teaser model released. See Above. More models are baking in the oven🔥.
- **[2025/10/18]** Project announced. Code and weights are not released yet; they will be published once SAM3 code is publicly available.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Zoo](#efficientsam3-model-zoo--weight-release)
- [Stage 1: Image Encoder Distillation](#stage-1-image-encoder-distillation)
- [Stage 2: Text Encoder Distillation](#stage-2-text-encoder-distillation)
- [Stage 3: Joint Fine-Tuning](#stage-3-joint-fine-tuning)
- [Examples](#examples)
- [Citation](#citation)

## Installation

```bash
git clone https://github.com/SimonZeng7108/efficientsam3
cd efficientsam3
pip install -e ".[stage1]"
```

Download checkpoints from the [Model Zoo](#efficientsam3-model-zoo--weight-release) section. All Stage 1 image encoder weights are available via Google Drive and Hugging Face links in the table below.

**Quick Start (Image Segmentation):**

### 🔥 Teaser Image Model

**EfficientViT-S (0.68M params)** distilled from **SAM3 Encoder (461.84M)** — **99.85% smaller**, trained on **1% SA-1B**.

```python
from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load model
model = build_efficientsam3_image_model(
  checkpoint_path="efficient_sam3_efficientvit_s.pt",
  backbone_type="efficientvit",
  model_name="b0",
  enable_inst_interactivity=True,
)

# Process image and predict
processor = Sam3Processor(model)
inference_state = processor.set_image(image)

# Single positive point prompt (x, y) in pixels
points = [[image.size[0] / 2, image.size[1] / 2]]
labels = [1]
masks, scores, _ = model.predict_inst(
    inference_state, 
    point_coords=points, 
    point_labels=labels
)
```

### 🔥 Teaser Video Model

**EfficientViT-S (0.68M params) + MobileCLIP-S0 (4.07M params)** — combined **93.18% smaller** than SAM3.

```python
from sam3.model_builder import build_efficientsam3_video_model
from sam3.model.sam3_video_predictor import SAM3VideoPredictor

# Load model
predictor = SAM3VideoPredictor(
    build_efficientsam3_video_model(
        image_encoder_ckpt="efficient_sam3_efficientvit_s.pt",
        text_encoder_ckpt="efficient_sam3_mobileclip_s0.pt",
        backbone_type="efficientvit",
        image_model_name="b0",
        enable_inst_interactivity=True,
        enable_video=True,
    )
)

# Video inference
 inference_state = predictor.init_inference_state(video_images)
 inference_state = predictor.add_text_prompt(
     inference_state,
     points=[[[100, 100]]],
     labels=[[1]],
     text_prompts=["a dog"]
 )
 output_dict = predictor.get_segmentation(inference_state)
```

## EfficientSAM3 Model Zoo & Weight Release

### Full Models (Image + Text Encoders)

| Model | Image Encoder | Text Encoder | Checkpoint |
|-------|---------------|--------------|------------|
| **TV-M** | TinyViT-11M | MobileCLIP-S0 (4.07M) | `output/full_models/efficient_sam3_tvm_m_mobileclip_s0_ctx16_5p_full.pt` |
| **RV-M** | RepViT-M1.1 (7.8M) | MobileCLIP-S0 (4.07M) | `output/full_models/efficient_sam3_rvm_m_mobileclip_s0_ctx16_5p_full.pt` |
| **EV-M** | EfficientViT-B1 (4.6M) | MobileCLIP-S0 (4.07M) | `output/full_models/efficient_sam3_evm_m_mobileclip_s0_ctx16_5p_full.pt` |

### SAM3-LiteText Models

| Model | Text Encoder | Context | Checkpoint |
|-------|-------------|---------|------------|
| **LiteText-S0** | MobileCLIP-S0 (4.07M) | ctx16 | `output/sam3_litetext/sam3_litetext_mobileclip_s0_ctx16.pt` |
| **LiteText-S1** | MobileCLIP-S1 (4.69M) | ctx16 | `output/sam3_litetext/sam3_litetext_mobileclip_s1_ctx16.pt` |
| **LiteText-L** | MobileCLIP2-L (42.38M) | ctx16 | `output/sam3_litetext/sam3_litetext_mobileclip2_l_ctx16.pt` |
| **LiteText-S0-32** | MobileCLIP-S0 (4.07M) | ctx32 | `output/sam3_litetext/sam3_litetext_mobileclip_s0_ctx32.pt` |
| **LiteText-S1-32** | MobileCLIP-S1 (4.69M) | ctx32 | `output/sam3_litetext/sam3_litetext_mobileclip_s1_ctx32.pt` |
| **LiteText-L-32** | MobileCLIP2-L (42.38M) | ctx32 | `output/sam3_litetext/sam3_litetext_mobileclip2_l_ctx32.pt` |

### Stage 1 Image Encoders (distilled from SAM3 ViT-H)

| Encoder | Params | Checkpoint |
|---------|--------|------------|
| TinyViT-S (5M) | 5.29M | `output/stage1_image_5p/es_tv_s/ckpt_epoch_49.pth` |
| TinyViT-M (11M) | 10.6M | `output/stage1_image_2p/es_tv_m/ckpt_epoch_49.pth` |
| TinyViT-L (21M) | 21.4M | `output/stage1_image/es_tv_l/ckpt_epoch_49.pth` |
| RepViT-S (2M) | 1.92M | `output/stage1_image_2p/es_rv_s/ckpt_epoch_49.pth` |
| RepViT-M (8M) | 7.81M | `output/stage1_image_5p/es_rv_m/ckpt_epoch_49.pth` |
| RepViT-L (19M) | 19.3M | `output/stage1_image/es_rv_l/ckpt_epoch_49.pth` |
| EfficientViT-S (1M) | 0.68M | `output/stage1_image_2p/es_ev_s/ckpt_epoch_49.pth` |
| EfficientViT-M (5M) | 4.61M | `output/stage1_image_5p/es_ev_m/ckpt_epoch_49.pth` |
| EfficientViT-L (9M) | 9.12M | `output/stage1_image/es_ev_l/ckpt_epoch_49.pth` |

### Stage 1 Text Encoders (distilled from SAM3 text encoder)

| Encoder | Params | Context | Checkpoint |
|---------|--------|---------|------------|
| MobileCLIP-S0 | 4.07M | ctx16 | `output/stage1_text/mobileclip_s0_5dataset_ctx16_fixed/ckpt_epoch_79.pth` |
| MobileCLIP-S0 | 4.07M | ctx32 | `output/stage1_text/mobileclip_s0_5dataset_ctx32_fixed/ckpt_epoch_79.pth` |
| MobileCLIP-S1 | 4.69M | ctx16 | `output/stage1_text/mobileclip_s1_5dataset_ctx16_fixed/ckpt_epoch_79.pth` |
| MobileCLIP-S1 | 4.69M | ctx32 | `output/stage1_text/mobileclip_s1_5dataset_ctx32_fixed/ckpt_epoch_79.pth` |
| MobileCLIP2-L | 42.38M | ctx16 | `output/stage1_text/mobileclip2_l_5dataset_ctx16_fixed/ckpt_epoch_79.pth` |
| MobileCLIP2-L | 42.38M | ctx32 | `output/stage1_text/mobileclip2_l_5dataset_ctx32_fixed/ckpt_epoch_79.pth` |

## Stage 1: Image Encoder Distillation

See [README_stage1.md](README_stage1.md) for detailed instructions.

## Stage 2: Text Encoder Distillation

See [README_stage2.md](README_stage2.md) for detailed instructions.

## Stage 3: Joint Fine-Tuning

See [README_stage3.md](README_stage3.md) for detailed instructions.

## Examples

See [sam3/examples/](sam3/examples/) for Jupyter notebooks with interactive examples:

- `sam3_image_interactive.ipynb` - Interactive image segmentation
- `sam3_video_predictor_example.ipynb` - Video segmentation
- `sam3_agent.ipynb` - SAM3 as an agent for complex tasks
- `sam3_for_sam1_task_example.ipynb` - Using SAM3 for SAM1-style tasks
- `sam3_for_sam2_video_task_example.ipynb` - SAM3 for video tasks

## Citation

```bibtex
@article{zeng2025efficientsam3,
  title={EfficientSAM3: Progressive Hierarchical Knowledge Distillation from SAM1, 2 and 3},
  author={Zeng, Chengxi and others},
  journal={arXiv preprint arXiv:2511.15833},
  year={2025}
}

@article{zeng2025sam3litetext,
  title={SAM3-LiteText: An Anatomical Study of the SAM3 Text Encoder for Efficient Vision-Language Segmentation},
  author={Zeng, Chengxi and others},
  journal={arXiv preprint arXiv:2602.12173},
  year={2025}
}
```
