### EfficientSAM3: Progressive Hierachical Knowledge Distillation (PhD) from SAM1, 2 and 3
[Chengxi Simon Zeng](https://simonzeng7108.github.io/about/)<sup>1,†</sup>, [Yuxuan Jiang](https://YuxuanJJ.github.io/)<sup>1</sup>, [Gao Ge](https://scholar.google.com/citations?user=j2_80ewAAAAJ&hl=en)<sup>1</sup>, [Shuai Wang](https://shuaiwang97.github.io/)<sup>2</sup>, [Duolikun Danier](https://danier97.github.io/)<sup>3</sup>, [Bin Zhu](https://binzhubz.github.io/)<sup>4</sup>, [Stevan Rudinac](https://stevanrudinac.com/)<sup>2</sup>, [David Bull](https://david-bull.github.io/)<sup>1</sup>, [Fan Aaron Zhang](https://fan-aaron-zhang.github.io/)<sup>1</sup>
<sup>1</sup>Visual Information Lab, University of Bristol; <sup>2</sup>MultiX lab, University of Amsterdam; <sup>3</sup>University of Edinburgh; <sup>4</sup>Singapore Management University

<sup>†</sup>Tech Lead & Corresponding Author

[![arXiv](https://img.shields.io/badge/arXiv-EfficientSAM3-b31b1b.svg)](https://arxiv.org/abs/2511.15833) [![arXiv](https://img.shields.io/badge/arXiv-SAM3--LiteText-b31b1b.svg)](https://arxiv.org/abs/2602.12173) [![Project Page](https://img.shields.io/badge/Project-Page-green)](https://simonzeng7108.github.io/efficientsam3/) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-EfficientSAM3-blue)](https://huggingface.co/Simon7108528/EfficientSAM3) [![Discord](https://img.shields.io/badge/Discord-Join-7289da?logo=discord&logoColor=white)](https://discord.gg/FMyaQca7xT)
---
## Updates
- **[2026/04/13]** **EfficientSAM3.1** and **SAM3.1-LiteText** image models were released on the `stage1_sam3.1` branch. SAM3-LiteText has also been officially merged into [Hugging Face Transformers](https://github.com/huggingface/transformers/pull/44320). Stage 3 data engine support is now available on the [data_engine](https://github.com/SimonZeng7108/efficientsam3/tree/data_engine/data_engine) branch.
- **[2026/02/18]** **SAM3-LiteText** released! SAM3-LiteText reduces text encoder parameters by up to 88% with similar performance to the original text encoder. [Paper](https://arxiv.org/abs/2602.12173) available on arXiv. Code available in [`sam3_litetext`](https://github.com/SimonZeng7108/efficientsam3/tree/sam3_litetext) branch and weights on [Hugging Face](https://huggingface.co/Simon7108528/EfficientSAM3/tree/main/sam3_litetext).
- **[2026/01/11]** Stage 1 geometry-prompt fine-tuned (**ft**) weights released/updated (image encoders on 1% SA-1B; text encoders fine-tuned on SA-Co Gold+Silver).
- **[2025/12/08]** Stage 1 text encoder weights released for all 3 variants (MobileCLIP S0, S1, and MobileCLIP2 L) - distilled on 1% Recap-DataComp-1B dataset.
- **[2025/12/02]** Stage 1 image encoder weights released for all 9 variants (RepViT, TinyViT, EfficientViT) - unsupervised distilled on 1% of SA-1B dataset.
- **[2025/11/25]** Teaser model released. See Above. More models are baking in the oven🔥.
- **[2025/10/18]** Project announced. Code and weights are not released yet; they will be published once SAM3 code is publicly available.
---


## Table of Contents

- [Table of Contents](#table-of-contents)
- [Updates](#updates)
- [Installation](#installation)
- [Inference](#inference)
- [EfficientSAM3.1 Model Zoo](#efficientsam31-model-zoo)
- [Evaluation](#evaluation)
- [Development To-Do List](#development-to-do-list)
- [Call for Pull Requests](#call-for-pull-requests)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Users](#users)

---

[SAM3](https://github.com/facebookresearch/sam3) (Segment Anything Model 3) has introduced powerful **Promptable Concept Segmentation (PCS)** capabilities, enabling semantic understanding and temporal object tracking beyond traditional mask generation. The latest SAM3.1 release adds **Object Multiplex**, a shared-memory approach for more efficient multi-object video tracking. However, SAM3's massive vision backbone and dense memory bank still make full-scale deployment impractical for real-time, on-device applications where computational resources and latency constraints are critical.

**EfficientSAM3** addresses this challenge by distilling SAM3's capabilities into lightweight architectures suitable for edge devices, enabling high-quality concept segmentation on mobile phones, embedded systems, and resource-constrained platforms.

<p align="center">
  <img src="images/efficientsam3_full.svg" alt="EfficientSAM3 Architecture" width="100%">
</p>



---

## Installation

EfficientSAM3.1 follows the latest SAM3.1 runtime stack while keeping Stage-1 tooling compatible with Python 3.10+:

- **Python** ≥ 3.10 (3.12 recommended)
- **PyTorch** ≥ 2.10.0
- **Device**: NVIDIA GPU (CUDA), Apple Silicon (MPS), or CPU

For non-CUDA platforms (MPS/CPU), install `scipy` for distance transform operations:
```bash
pip install scipy
```

Follow the exact environment setup from the [official SAM3 README](sam3/README.md) or use the condensed steps below:


```bash
git clone https://github.com/SimonZeng7108/efficientsam3.git
cd efficientsam3

conda create -n efficientsam3 python=3.12 -y
conda activate efficientsam3

pip install --upgrade pip

# Install PyTorch (choose one based on your device):
# CUDA (default):
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# MPS/CPU (Apple Silicon or CPU-only):
pip install torch==2.10.0 torchvision torchaudio

# Install repo dependencies via the root pyproject (brings in SAM3 + Stage-1 extras)
pip install -e ".[stage1]"

# Optional dependencies for faster SAM3.1 multiplex inference (FlashAttention 3, CUDA/Linux)
pip install -e ".[sam31-runtime]"
pip install git+https://github.com/ronghanghu/cc_torch.git

# Note: the Stage-1 extra includes the SAM1 package dependency
# (PyPI name: segment-anything, import name: segment_anything).
# If your environment cannot resolve it from PyPI, install the vendored repo instead:
# pip install -e ./segment-anything
```

If you run SAM3.1 multiplex with `use_fa3=True`, the runtime imports `flash_attn_interface`.
If `flash_attn_interface` is unavailable in your environment, either:

- install the SAM3.1 runtime extra via `.[sam31-runtime]` (Linux/CUDA), or
- set `use_fa3=False` when constructing the SAM3.1 multiplex predictor.

---

## Inference

This branch is focused on the SAM3.1 runtime update for both EfficientSAM3 and SAM3-LiteText. The two reference examples below are the recommended entry points.

### EfficientSAM3.1 for SAM1-Style Image Tasks

Use [sam3/efficientsam3_examples/efficientsam3_for_sam1_task_example.ipynb](sam3/efficientsam3_examples/efficientsam3_for_sam1_task_example.ipynb) for point prompts, box prompts, multimask outputs, and batched SAM1-style interactive image segmentation with an EfficientSAM3 SAM3.1 checkpoint.

```python
import numpy as np
from pathlib import Path
from PIL import Image

from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

bpe_path = "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
checkpoint_path = Path(
    "checkpoints/stage1_sam3p1/efficient_sam3p1_tinyvit_m_mobileclip_s0_ctx16.pt"
)

model = build_efficientsam3_image_model(
    bpe_path=bpe_path,
    checkpoint_path=str(checkpoint_path),
    load_from_HF=False,
    enable_inst_interactivity=True,
    backbone_type="tinyvit",
    model_name="m",
    text_encoder_type="mobileclip-s0",
    text_encoder_context_length=16,
    text_encoder_pos_embed_table_size=16,
    interpolate_pos_embed=False,
)

image = Image.open("sam3/assets/images/truck.jpg")
processor = Sam3Processor(model)
inference_state = processor.set_image(image)

input_point = np.array([[520, 375]])
input_label = np.array([1])

masks, scores, _ = model.predict_inst(
    inference_state,
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
print(scores)
```

The notebook also includes batched prompt examples through `predict_inst_batch` and visualization helpers for promptable image segmentation workflows.

### SAM3.1 LiteText for Image Segmentation

Use [sam3/efficientsam3_examples/efficientsam3_litetext_image_inference_example.py](sam3/efficientsam3_examples/efficientsam3_litetext_image_inference_example.py) for text-prompt image segmentation with the SAM3.1 LiteText checkpoints.

```bash
PYTHONPATH=sam3 python sam3/efficientsam3_examples/efficientsam3_litetext_image_inference_example.py \
  --checkpoint checkpoints/sam3p1_litetext/efficient_sam3p1_litetext_mobileclip_s0_ctx16.pt \
  --image sam3/assets/dog_person.jpeg \
  --prompt dog \
  --output vis/litetext_image_inference.png
```

The script wraps the common flow below and saves a visualization with the predicted masks and scores:

```python
from sam3p1_demo_utils import run_text_prompt_demo

result = run_text_prompt_demo(
    checkpoint_path="checkpoints/sam3p1_litetext/efficient_sam3p1_litetext_mobileclip_s0_ctx16.pt",
    image_path="sam3/assets/dog_person.jpeg",
    prompt="dog",
    output_path="vis/litetext_image_inference.png",
)
print(result["scores"])
```

For SAM3.1 multiplex inference with `use_fa3=True`, ensure the optional runtime dependencies are installed so `flash_attn_interface` is available.

---

## EfficientSAM3.1 Model Zoo

This branch currently provides two SAM3.1 checkpoint groups:

- Stage-1 EfficientSAM3.1 image+text student checkpoints
- SAM3.1 LiteText checkpoints that keep the SAM3.1 backbone and replace the text encoder with lightweight MobileCLIP variants

### Student Text + Image Encoders (MobileCLIP-S0 ctx16)

All weights are available in the Hugging Face folder here: [stage1_sam3p1](https://huggingface.co/Simon7108528/EfficientSAM3/tree/main/stage1_sam3p1).

| Model | Backbone | Text Encoder | COCO mIoU | Checkpoint | Weights |
|-------|----------|--------------|-----------|------------|---------|
| ES-RV-S-SAM3.1 | RepViT-S | MobileCLIP-S0 ctx16 | 0.6731 | `efficient_sam3p1_repvit_s_mobileclip_s0_ctx16.pt` | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_sam3p1/efficient_sam3p1_repvit_s_mobileclip_s0_ctx16.pt) |
| ES-RV-M-SAM3.1 | RepViT-M | MobileCLIP-S0 ctx16 | 0.6782 | `efficient_sam3p1_repvit_m_mobileclip_s0_ctx16.pt` | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_sam3p1/efficient_sam3p1_repvit_m_mobileclip_s0_ctx16.pt) |
| ES-RV-L-SAM3.1 | RepViT-L | MobileCLIP-S0 ctx16 | 0.6927 | `efficient_sam3p1_repvit_l_mobileclip_s0_ctx16.pt` | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_sam3p1/efficient_sam3p1_repvit_l_mobileclip_s0_ctx16.pt) |
| ES-TV-S-SAM3.1 | TinyViT-S | MobileCLIP-S0 ctx16 | 0.6825 | `efficient_sam3p1_tinyvit_s_mobileclip_s0_ctx16.pt` | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_sam3p1/efficient_sam3p1_tinyvit_s_mobileclip_s0_ctx16.pt) |
| ES-TV-M-SAM3.1 | TinyViT-M | MobileCLIP-S0 ctx16 | 0.6906 | `efficient_sam3p1_tinyvit_m_mobileclip_s0_ctx16.pt` | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_sam3p1/efficient_sam3p1_tinyvit_m_mobileclip_s0_ctx16.pt) |
| ES-TV-L-SAM3.1 | TinyViT-L | MobileCLIP-S0 ctx16 | 0.6967 | `efficient_sam3p1_tinyvit_l_mobileclip_s0_ctx16.pt` | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_sam3p1/efficient_sam3p1_tinyvit_l_mobileclip_s0_ctx16.pt) |
| ES-EV-S-SAM3.1 | EfficientViT-S | MobileCLIP-S0 ctx16 | 0.6476 | `efficient_sam3p1_efficientvit_s_mobileclip_s0_ctx16.pt` | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_sam3p1/efficient_sam3p1_efficientvit_s_mobileclip_s0_ctx16.pt) |
| ES-EV-M-SAM3.1 | EfficientViT-M | MobileCLIP-S0 ctx16 | 0.6721 | `efficient_sam3p1_efficientvit_m_mobileclip_s0_ctx16.pt` | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_sam3p1/efficient_sam3p1_efficientvit_m_mobileclip_s0_ctx16.pt) |
| ES-EV-L-SAM3.1 | EfficientViT-L | MobileCLIP-S0 ctx16 | 0.6889 | `efficient_sam3p1_efficientvit_l_mobileclip_s0_ctx16.pt` | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_sam3p1/efficient_sam3p1_efficientvit_l_mobileclip_s0_ctx16.pt) |

### SAM3.1 LiteText

All LiteText weights are available in the Hugging Face folder here: [sam3p1_litetext](https://huggingface.co/Simon7108528/EfficientSAM3/tree/main/sam3p1_litetext).

| Model | Backbone | Text Encoder | Context Length | Avg CG_F1 | Checkpoint | Weights |
|-------|----------|--------------|----------------|-----------|------------|---------|
| SAM3.1-LiteText-S0-16 | SAM3.1 multiplex | MobileCLIP-S0 | 16 | 54.26 | `efficient_sam3p1_litetext_mobileclip_s0_ctx16.pt` | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3p1_litetext/efficient_sam3p1_litetext_mobileclip_s0_ctx16.pt) |
| SAM3.1-LiteText-S1-16 | SAM3.1 multiplex | MobileCLIP-S1 | 16 | 54.27 | `efficient_sam3p1_litetext_mobileclip_s1_ctx16.pt` | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3p1_litetext/efficient_sam3p1_litetext_mobileclip_s1_ctx16.pt) |
| SAM3.1-LiteText-L-16 | SAM3.1 multiplex | MobileCLIP2-L | 16 | 54.03 | `efficient_sam3p1_litetext_mobileclip_l_ctx16.pt` | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3p1_litetext/efficient_sam3p1_litetext_mobileclip_l_ctx16.pt) |
| SAM3.1-LiteText-S0-32 | SAM3.1 multiplex | MobileCLIP-S0 | 32 | 54.28 | `efficient_sam3p1_litetext_mobileclip_s0_ctx32.pt` | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3p1_litetext/efficient_sam3p1_litetext_mobileclip_s0_ctx32.pt) |
| SAM3.1-LiteText-S1-32 | SAM3.1 multiplex | MobileCLIP-S1 | 32 | 54.00 | `efficient_sam3p1_litetext_mobileclip_s1_ctx32.pt` | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3p1_litetext/efficient_sam3p1_litetext_mobileclip_s1_ctx32.pt) |
| SAM3.1-LiteText-L-32 | SAM3.1 multiplex | MobileCLIP2-L | 32 | 54.17 | `efficient_sam3p1_litetext_mobileclip_l_ctx32.pt` | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3p1_litetext/efficient_sam3p1_litetext_mobileclip_l_ctx32.pt) |

---

## Development To-Do List

- [x] **Release Stage 1 Image Encoder Weights**: Distilled image encoder weights from SAM3 image encoder for all 9 variants (RepViT, TinyViT, EfficientViT)
- [x] **Release Stage 1 Text Encoder Weights**: Distill SAM3 text encoder weights to MobileCLIP-S1 combined with all 9 image encoder variants
- [x] **Release Stage 1+ Fine-Tuned Encoder Weights**: Prompt-in-the-loop supervised fine-tuning for improved encoder performance
- [x] **Release SAM3-LiteText Weights**: Distilled a lightweight MobileCLIP text encoder that is competitive to the SAM3 text encoder for efficient vision-language segmentation
- [x] **Release EfficientSAM3.1 and SAM3.1-LiteText Image Model Weights**: Released EfficientSAM3.1 and SAM3.1-LiteText image model checkpoints
- [ ] **Release Stage 2 Memory Bank Aligned Model Weights**: Models with Perceiver-based memory compression trained on SA-V dataset
- [ ] **Release Stage 3 Fine-Tuned Model Weights**: End-to-end fine-tuned models on SAM3 dataset with full PCS capabilities
- [ ] **ONNX/CoreML Export**: Export models to ONNX and CoreML formats for cross-platform deployment
- [ ] **Web Demo**: Interactive web demonstration for real-time concept segmentation and tracking

---

## Call for Pull Requests
The idea for this repository originated from my work on SAM2 at Amazon, particularly as part of the research described in [this paper](https://ieeexplore.ieee.org/abstract/document/11084428). Since company policy, I cannot share the codebase. This year I am super excited to work on making SAM3 more efficient and accessible to the community.

We welcome contributions to EfficientSAM3! Please feel free to submit pull requests to improve the codebase, add new features, or fix bugs. Particularly, we are looking for:
- Efficient MedSAM3 integration (see [MedSAM2 by Bo Wang Lab](https://github.com/bowang-lab/MedSAM2))
- A Gradio demo (e.g. [EfficientTAM on Hugging Face Spaces](https://huggingface.co/spaces/yunyangx/EfficientTAM))
- A web demo deployed with Vercel (e.g. [Segment Anything Web UI](https://segment-anything-webui.vercel.app/))
- Annotation tools, such as [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) and [AnyLabeling](https://github.com/vietanhdev/anylabeling)
- An iOS or Android app (e.g. [Cutcha Photo on the App Store](https://apps.apple.com/us/app/cutcha-photo/id6478521132))
- An NVCC-based desktop application
- Anything else that you think is cool!
---

All meaningful contributions will be acknowledged and integrated into both the repository and the associated paper. We warmly welcome all contributors to the repository and happily offer co-authorship to those whose work merits inclusion in the paper.

## Citation

If you use EfficientSAM3 in your research, please cite:

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

```bibtex
@misc{zeng2026sam3litetextanatomicalstudysam3,
      title={SAM3-LiteText: An Anatomical Study of the SAM3 Text Encoder for Efficient Vision-Language Segmentation}, 
      author={Chengxi Zeng and Yuxuan Jiang and Ge Gao and Shuai Wang and Duolikun Danier and Bin Zhu and Stevan Rudinac and David Bull and Fan Zhang},
      year={2026},
      eprint={2602.12173},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.12173}, 
}
```

## License

This repository is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

This project builds upon [SAM](https://github.com/facebookresearch/segment-anything), [SAM2](https://github.com/facebookresearch/sam2), [SAM3](https://github.com/facebookresearch/sam3), [EdgeSAM](https://github.com/chongzhou96/EdgeSAM), [EdgeTAM](https://github.com/facebookresearch/EdgeTAM), [EfficientTAM](https://github.com/yformer/EfficientTAM), [RepViT](https://github.com/THU-MIG/RepViT), [TinyViT](https://github.com/wkcn/TinyViT), [EfficientViT](https://github.com/mit-han-lab/efficientvit), and [MobileCLIP](https://github.com/apple/ml-mobileclip). Please refer to their respective licenses for usage terms.


## Acknowledgments

We gratefully acknowledge the [University of Bristol Isambard-AI supercomputer cluster](https://www.bristol.ac.uk/research/centres/bristol-supercomputing/articles/2025/isambard-ai-is-11th-fastest-supercomputer-in-the-world.html) for providing computational resources to this project. Special thanks to [Dr. Fan Aaron Zhang](https://fan-aaron-zhang.github.io/) for allocating resources and supporting this research.

---

## Users

Organizations and projects using EfficientSAM3:

<table>
  <tr>
    <td align="center" width="20%">
      <img src="https://github.com/SimonZeng7108/simonzeng7108.github.io/blob/main/efficientsam3/static/images/esa.png" alt="European Space Agency" height="80"><br>
      <a href="https://www.esa.int/Applications/Observing_the_Earth/Phsat-2/Introducing_Phsat-2">European Space Agency</a>
    </td>
  </tr>
</table>

> **Note:** If you're using EfficientSAM3 in your work, please acknowledge us in your publications or projects. We're happy to promote your work here! Contact us to be featured in this section.

