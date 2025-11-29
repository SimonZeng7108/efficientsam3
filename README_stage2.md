## Stage 2 — Temporal Memory Distillation

Stage 2 focuses on compressing the temporal memory module of SAM3. While Stage 1 reduced the spatial encoder size, Stage 2 replaces the heavy memory attention mechanism with a lightweight **Hybrid Memory Module** inspired by EdgeTAM. This module combines a **Global Perceiver** and a **Spatial Perceiver** to efficiently handle long-term video context with significantly fewer parameters and computations.

### Objectives
- **Teacher**: Frozen SAM3 (ViT-Huge backbone + Standard Memory Attention).
- **Student**: EfficientSAM3 (EfficientViT/RepViT backbone + Hybrid Memory Module).
- **Goal**: Train the Hybrid Memory Module to mimic the memory representations and tracking performance of the teacher.

### Prerequisites

1.  **Environment**: Ensure the `efficientsam3` Conda environment is active.
2.  **Stage 1 Checkpoint**: You should have a merged Stage 1 checkpoint (e.g., `output/efficient_sam3_stage1_merged.pt`).
3.  **Dataset**: **SA-V (Segment Anything Video)**.
    *   Download the dataset using the links in `data/sa-v-1p.txt`.
    *   Extract the data to `data/sa-v/sav_train`.
    *   Ensure the structure is `data/sa-v/sav_train/sav_000/*.mp4`.

### Training

To train the Stage 2 model, run the `train_stage2.py` script. This script loads the frozen teacher, initializes the student with the Hybrid Memory Module, and trains it using distillation loss.

```bash
python stage2/scripts/train_stage2.py \
  --config stage2/configs/efficient_sam3_stage2.yaml \
  --teacher_checkpoint sam3_checkpoints/sam3.pt \
  --dataset_path data/sa-v/sav_train \
  --output_dir output/stage2
```

**Key Arguments:**
*   `--teacher_checkpoint`: Path to the pretrained SAM3 checkpoint.
*   `--dataset_path`: Path to the SA-V dataset directory.
*   `--memory_dim`: Dimension of the memory latents (default: 256).
*   `--num_global_latents`: Number of global context latents (default: 64).
*   `--num_spatial_latents`: Number of spatial context latents (default: 192).

### Weight Conversion (Final Assembly)

After training Stage 2, you have a checkpoint (`output/stage2/epoch_X.pth`) that contains the trained Memory Module but uses the original SAM3 backbone. To create the final EfficientSAM3 model, you must merge these weights with the **individual Stage 1 student checkpoints** (NOT the merged checkpoint).

**Important**: The conversion script requires **three separate checkpoints**:
1. **Stage 1 Image Student**: The vision encoder checkpoint (e.g., `output/stage1/es_ev_s/ckpt_epoch_0.pth`)
2. **Stage 1 Text Student**: The text encoder checkpoint (e.g., `output/stage1_text/mobileclip_s/ckpt_epoch_0.pth`)
3. **Stage 2 Checkpoint**: The trained memory module (e.g., `output/stage2/epoch_2.pth`)

Run the conversion script:

```bash
python stage2/scripts/convert_stage2_weights.py \
  --image_student_checkpoint output/stage1/es_ev_s/ckpt_epoch_0.pth \
  --text_student_checkpoint output/stage1_text/mobileclip_s/ckpt_epoch_0.pth \
  --stage2_checkpoint output/stage2/epoch_2.pth \
  --output_path output/efficient_sam3_final.pt
```

This produces `output/efficient_sam3_final.pt`, which is the fully compressed model with:
- EfficientViT-B0 vision backbone (from Stage 1 image student)
- MobileCLIP-S0 text encoder (from Stage 1 text student)
- Hybrid Memory Module (from Stage 2 training)
- SAM3 decoder components (transformer, mask decoder, prompt encoder)

### EfficientSAM3 API Usage

EfficientSAM3 provides the same API as SAM3 for three core capabilities:
1. **Image Segmentation** (points, boxes, masks)
2. **Video Object Tracking** (multi-frame propagation)
3. **Concept/Text-based Segmentation** (vision-language)

#### Model Initialization

```python
from sam3.model.efficient_sam3 import EfficientSam3Tracker
from sam3.model_builder import build_efficientsam3_image_model
import torch
import numpy as np

# 1. Build Student Backbone
dummy_student = build_efficientsam3_image_model(
    bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    checkpoint_path=None,
    load_from_HF=False,
    backbone_type="efficientvit",
    model_name="b0",
    text_encoder_type="MobileCLIP-S0",
    enable_inst_interactivity=True,
)

# 2. Initialize Tracker with Hybrid Memory
model = EfficientSam3Tracker(
    backbone=dummy_student.backbone,
    hybrid_memory_dim=256,
    num_global_latents=64,
    num_spatial_latents=192,
    window_size=8
)

# 3. Load Final Checkpoint
checkpoint = torch.load("output/efficient_sam3_final.pt", map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint, strict=True)
model.eval().cuda()
```

#### 1. Image Segmentation

```python
# Prepare image
image = ...  # Load your image (H, W, 3) numpy array
image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
image_tensor = image_tensor.cuda()

# Initialize inference state for single image
inference_state = model.init_state(
    video_height=image.shape[0],
    video_width=image.shape[1],
    num_frames=1
)
inference_state["images"] = {0: image_tensor}

# Example 1: Point-based segmentation
points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)  # Point coordinates
labels = np.array([1, 1], dtype=np.int32)  # 1 = foreground, 0 = background

frame_idx, obj_ids, low_res_masks, video_res_masks = model.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=1,
    points=points,
    labels=labels,
)

# video_res_masks: [num_objects, 1, H, W] - final segmentation masks
mask = (video_res_masks[0, 0] > 0).cpu().numpy()  # Binary mask

# Example 2: Box-based segmentation
box = np.array([x1, y1, x2, y2], dtype=np.float32)  # Bounding box coordinates

frame_idx, obj_ids, low_res_masks, video_res_masks = model.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=2,
    box=box,
)

# Example 3: Mask-based segmentation
mask_input = ...  # Binary mask (H, W)
model.add_new_mask(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=3,
    mask=mask_input,
)
```

#### 2. Video Object Tracking

```python
# Prepare video frames
video_frames = [...]  # List of (H, W, 3) numpy arrays

# Convert to tensors
video_tensors = {}
for i, frame in enumerate(video_frames):
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    video_tensors[i] = frame_tensor.cuda()

# Initialize inference state for video
video_state = model.init_state(
    video_height=video_frames[0].shape[0],
    video_width=video_frames[0].shape[1],
    num_frames=len(video_frames)
)
video_state["images"] = video_tensors

# Add initial prompt on first frame
initial_points = np.array([[x, y]], dtype=np.float32)
initial_labels = np.array([1], dtype=np.int32)

model.add_new_points_or_box(
    inference_state=video_state,
    frame_idx=0,
    obj_id=1,
    points=initial_points,
    labels=initial_labels,
)

# Propagate across entire video
video_segments = {}
for frame_idx, obj_ids, mask_logits in model.propagate_in_video(
    inference_state=video_state,
    start_frame_idx=0,
    max_frame_num_to_track=len(video_frames),
    reverse=False,
):
    video_segments[frame_idx] = {
        'obj_ids': obj_ids,
        'masks': (mask_logits > 0.0).cpu().numpy()
    }

# Access segmentation for any frame
frame_10_mask = video_segments[10]['masks'][0]  # First object's mask on frame 10
```

#### 3. Interactive Refinement

```python
# You can refine predictions on any frame by adding more prompts
refine_frame_idx = 15

# Add a negative click to remove false positives
negative_point = np.array([[x, y]], dtype=np.float32)
negative_label = np.array([0], dtype=np.int32)  # 0 = background

model.add_new_points_or_box(
    inference_state=video_state,
    frame_idx=refine_frame_idx,
    obj_id=1,
    points=negative_point,
    labels=negative_label,
    clear_old_points=False,  # Keep previous points
)

# Re-propagate to update all frames
video_segments = {}
for frame_idx, obj_ids, mask_logits in model.propagate_in_video(
    inference_state=video_state,
    start_frame_idx=0,
    max_frame_num_to_track=len(video_frames),
    reverse=False,
):
    video_segments[frame_idx] = {
        'obj_ids': obj_ids,
        'masks': (mask_logits > 0.0).cpu().numpy()
    }
```
