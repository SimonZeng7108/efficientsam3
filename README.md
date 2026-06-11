# EfficientSAM3: Progressive Hierarchical Knowledge Distillation from SAM1, 2 and 3

[Chengxi Simon Zeng](https://simonzeng7108.github.io/about/)<sup>1,†</sup>, [Yuxuan Jiang](https://YuxuanJJ.github.io/)<sup>1</sup>, [Gao Ge](https://scholar.google.com/citations?user=j2_80ewAAAAJ&hl=en)<sup>1</sup>, [Shuai Wang](https://shuaiwang97.github.io/)<sup>2</sup>, [Duolikun Danier](https://danier97.github.io/)<sup>3</sup>, [Bin Zhu](https://binzhubz.github.io/)<sup>4</sup>, [Stevan Rudinac](https://stevanrudinac.com/)<sup>2</sup>, [David Bull](https://david-bull.github.io/)<sup>1</sup>, [Fan Aaron Zhang](https://fan-aaron-zhang.github.io/)<sup>1</sup>

<sup>1</sup>Visual Information Lab, University of Bristol; <sup>2</sup>MultiX lab, University of Amsterdam; <sup>3</sup>University of Edinburgh; <sup>4</sup>Singapore Management University

<sup>†</sup>Tech Lead & Corresponding Author

[![arXiv](https://img.shields.io/badge/arXiv-EfficientSAM3-b31b1b.svg)](https://arxiv.org/abs/2511.15833) [![arXiv](https://img.shields.io/badge/arXiv-SAM3--LiteText-b31b1b.svg)](https://arxiv.org/abs/2602.12173) [![Project Page](https://img.shields.io/badge/Project-Page-green)](https://SimonZeng7108.github.io/efficientsam3/) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-EfficientSAM3-blue)](https://huggingface.co/Simon7108528/EfficientSAM3) [![Discord](https://img.shields.io/badge/Discord-Join-7289da?logo=discord&logoColor=white)](https://discord.gg/FMyaQca7xT)

---

## Table of Contents

- [Updates](#updates)
- [Model Zoo](#model-zoo)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training and Evaluation](#training-and-evaluation)
- [Datasets](#datasets)
- [Examples](#examples)
- [Development](#development)
- [Call for Pull Requests](#call-for-pull-requests)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Users](#users)

---

## Updates

- **[2026/06/11]** **Stage 3 Fine-tuned Models Released!** EfficientSAM3 full models (EV-M, RV-M, TV-M) fine-tuned on 5% SA1B data & SACap labels. Checkpoints on [HuggingFace](https://huggingface.co/Simon7108528/EfficientSAM3/tree/main/efficientsam3_ft). See [README_stage3.md](README_stage3.md).
- **[2026/04/19]** **SAM3-LiteText** live on HuggingFace, accepted by [ICMR2026](https://icmr2026.org/)! [[Docs](https://huggingface.co/docs/transformers/main/en/model_doc/sam3_lite_text)] [[Demo](https://huggingface.co/spaces/nielsr/sam-3-lite-text-vs-sam-3)] Thanks @NielsRogge, @yonigozlan!

<details>
<summary><b>Older updates</b></summary>

- **[2026/02/18]** **SAM3-LiteText** released! Reduces text encoder by 88% with similar performance. [Paper](https://arxiv.org/abs/2602.12173)
- **[2026/01/11]** Stage 1 geometry-prompt fine-tuned weights released.
- **[2025/12/08]** Stage 1 text encoder weights released (MobileCLIP S0, S1, MobileCLIP2 L).
- **[2025/12/02]** Stage 1 image encoder weights released (RepViT, TinyViT, EfficientViT).

</details>

---

## Model Zoo

### EfficientSAM3 Full Models (Lightweight Image + Text Encoders)

EfficientSAM3 compresses both SAM3's vision encoder and text encoder into lightweight student models while maintaining competitive performance on downstream benchmarks.

<p align="center">
  <img src="https://raw.githubusercontent.com/SimonZeng7108/efficientsam3/main/images/dog_person_example_dog.png" width="45%">
  <img src="https://raw.githubusercontent.com/SimonZeng7108/efficientsam3/main/images/dog_person_example_person.png" width="45%">
</p>

| Model | Vision | Text | Decoder | Total | vs SAM3 (1.4B) | Download |
|-------|--------|------|---------|-------|-----------------|----------|
| **EV-M** | 4.6M | 4.07M | ~605M | **~614M** | **56% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_efficientvit.pt?download=true) |
| **RV-M** | 7.8M | 4.07M | ~605M | **~617M** | **56% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_repvit.pt?download=true) |
| **TV-M** | 10.6M | 4.07M | ~605M | **~620M** | **56% smaller** | [HF](https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft/efficientsam3_tinyvit.pt?download=true) |

> **SAM3 Teacher**: 1.4B total (Vision: 461M + Text: 354M + Decoder/Heads: ~600M)

### SAM3-LiteText Models (Lightweight Text Encoder Only)

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

### EfficientSAM3

EfficientSAM3 replaces both the SAM3 vision encoder and text encoder with lightweight student models.

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image

# Load model (TV-M example - student vision + student text)
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

SAM3-LiteText keeps the SAM3 vision encoder but replaces the heavy text encoder with a lightweight MobileCLIP variant.

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image

# Build model with LiteText encoder (keeps SAM3 ViT, replaces text encoder)
model = build_sam3_image_model(
    checkpoint_path="sam3_litetext_mobileclip_s0_ctx16.pt",
    text_encoder_type="MobileCLIP-S0",
    text_encoder_context_length=16,
    load_from_HF=False,
)

# Use as normal
processor = Sam3Processor(model)
image = Image.open("your_image.jpg").convert("RGB")
state = processor.set_image(image)
state = processor.set_text_prompt("person", state)
masks = state["masks"]
```

---

## Training and Evaluation

**Training:**
- **Stage 1:** Encoder distillation training details in [README_stage1.md](README_stage1.md)
- **Stage 3:** Full fine-tuning details in [README_stage3.md](README_stage3.md)

**Evaluation:**
- To evaluate models on COCO dataset:
  ```bash
  python eval/eval_coco.py --coco_root data/coco --output_dir output
  ```

- To evaluate text encoder quality (token-level cosine similarity vs SAM3 teacher):
  ```bash
  python eval/eval_text_encoder_similarity.py \
    --student-ckpt /path/to/student_text_encoder_1.pth /path/to/student_text_encoder_2.pth \
    --np-json data/sa-v-text/sa-co-veval/saco_veval_noun_phrases.json \
    --device cuda
  # Optional: override teacher checkpoint
  #   --teacher-ckpt /path/to/sam3_teacher_checkpoint.pt
  ```

---

## Datasets

For dataset setup and download scripts (`data/download_*.sh`) covering COCO, DAVIS, LVIS, SA-1B, SA-V, LVOS, MOSE, and YouTube-VOS, see:

- [README_dataset.md](README_dataset.md)


---

### To-Do List

- [x] **Release Stage 1 Image Encoder Weights**: Distilled image encoder weights from SAM3 image encoder for all 9 variants (RepViT, TinyViT, EfficientViT)
- [x] **Release Stage 1 Text Encoder Weights**: Distill SAM3 text encoder weights to MobileCLIP-S1 combined with all 9 image encoder variants
- [x] **Release Stage 1+ Fine-Tuned Encoder Weights**: Prompt-in-the-loop supervised fine-tuning for improved encoder performance
- [x] **Release SAM3-LiteText Weights**: Distilled a lightweight MobileCLIP text encoder that is competitive to the SAM3 text encoder for efficient vision-language segmentation
- [ ] **Release Stage 2 Memory Bank Aligned Model Weights**: Models with Perceiver-based memory compression trained on SA-V dataset
- [x] **Release Stage 3 Fine-Tuned Model Weights**: End-to-end fine-tuned models on SAM3 dataset with full PCS capabilities
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

All meaningful contributions will be acknowledged and integrated into both the repository and the associated paper. We warmly welcome all contributors to the repository and happily offer co-authorship to those whose work merits inclusion in the paper.

---

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

---

## License

This repository is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

This project builds upon [SAM](https://github.com/facebookresearch/segment-anything), [SAM2](https://github.com/facebookresearch/sam2), [SAM3](https://github.com/facebookresearch/sam3), [EdgeSAM](https://github.com/chongzhou96/EdgeSAM), [EdgeTAM](https://github.com/facebookresearch/EdgeTAM), [EfficientTAM](https://github.com/yformer/EfficientTAM), [RepViT](https://github.com/THU-MIG/RepViT), [TinyViT](https://github.com/wkcn/TinyViT), [EfficientViT](https://github.com/mit-han-lab/efficientvit), and [MobileCLIP](https://github.com/apple/ml-mobileclip). Please refer to their respective licenses for usage terms.

---

## Acknowledgments

We gratefully acknowledge the [University of Bristol Isambard-AI supercomputer cluster](https://www.bristol.ac.uk/research/centres/bristol-supercomputing/articles/2025/isambard-ai-is-11th-fastest-supercomputer-in-the-world.html) for providing computational resources to this project. Special thanks to [Dr. Fan Aaron Zhang](https://fan-aaron-zhang.github.io/) for allocating resources and supporting this research.

---

## Users

Organizations and projects using EfficientSAM3:

| Organization | Description |
|-------------|-------------|
| <img src="https://github.com/SimonZeng7108/simonzeng7108.github.io/blob/main/efficientsam3/static/images/esa.png?raw=true" width="80"> | [European Space Agency](https://www.esa.int/Applications/Observing_the_Earth/Phsat-2/Introducing_Phsat-2) |

> **Note:** If you're using EfficientSAM3 in your work, please acknowledge us in your publications or projects. We're happy to promote your work here! Contact us to be featured in this section.
