#!/usr/bin/env python3
"""
SAM3-LiteText Video Predictor Example
======================================
Demonstrates text-prompted dense video tracking using SAM3-LiteText.

Model setup (two-checkpoint workflow):
  1. Full SAM3 video model (tracker + ViT backbone) -- from sam3.pt or HuggingFace
  2. Student text encoder (MobileCLIP-S0) overlaid from a LiteText image checkpoint

Usage:
    python efficientsam3_litetext_video_predictor_example.py
    python efficientsam3_litetext_video_predictor_example.py --video /path/to/video.mp4
    python efficientsam3_litetext_video_predictor_example.py --prompt "person" --ctx 32
"""

import argparse
import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_sam3_repo_root = os.path.abspath(os.path.join(_here, ".."))   # .../sam3/  (contains the sam3 package)
_project_root = os.path.abspath(os.path.join(_sam3_repo_root, ".."))   # workspace root
if _sam3_repo_root not in sys.path:
    sys.path.insert(0, _sam3_repo_root)
# _sam3_pkg_dir for asset/checkpoint relative paths
_sam3_pkg_dir = _sam3_repo_root

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SAM3-LiteText video predictor example")
    p.add_argument(
        "--video",
        default=os.path.join(_sam3_pkg_dir, "assets", "videos", "0001"),
        help="Path to a JPEG frame folder or an .mp4 file",
    )
    p.add_argument(
        "--prompt", default="person",
        help="Text prompt for tracking (default: 'person')"
    )
    p.add_argument(
        "--sam3-checkpoint",
        default=os.path.join(_project_root, "sam3_checkpoints", "sam3.pt"),
        help="Path to the full SAM3 video checkpoint (tracker + ViT). "
             "If not found, will attempt to download from HuggingFace.",
    )
    p.add_argument(
        "--litetext-checkpoint",
        default=os.path.join(
            _project_root, "output", "sam3_litetext",
            "efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt"
        ),
        help="Path to the LiteText image checkpoint (student text encoder weights).",
    )
    p.add_argument(
        "--backbone-type",
        default="MobileCLIP-S0",
        choices=["MobileCLIP-S0", "MobileCLIP-S1", "MobileCLIP2-L"],
        help="Student text encoder variant (default: MobileCLIP-S0)",
    )
    p.add_argument(
        "--ctx", type=int, default=16,
        help="Context length the checkpoint was trained with (default: 16)",
    )
    p.add_argument(
        "--frame-stride", type=int, default=30,
        help="Save every N-th frame to the output dir (default: 30)",
    )
    p.add_argument(
        "--output-dir",
        default=os.path.join(_project_root, "output", "litetext_video"),
        help="Directory to save visualised frames",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_video_frames(video_path: str):
    """Return a list of RGB numpy arrays (H, W, 3) from a folder or .mp4."""
    if video_path.endswith(".mp4") or video_path.endswith(".avi"):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
    else:
        # JPEG folder
        paths = sorted(
            glob.glob(os.path.join(video_path, "*.jpg")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
            if os.path.splitext(os.path.basename(p))[0].isdigit()
            else p,
        )
        return [np.array(Image.open(p).convert("RGB")) for p in paths]


def visualize_frame(frame_rgb, outputs, prompt: str, frame_idx: int, save_path: str):
    """Overlay masks from model outputs onto the frame and save."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.imshow(frame_rgb)

    if outputs:
        cmap = plt.cm.get_cmap("tab10")
        legend_patches = []
        for obj_idx, (obj_id, obj_out) in enumerate(outputs.items()):
            mask = obj_out.get("mask")
            if mask is None:
                continue
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            mask = mask.squeeze()
            color = cmap(obj_idx % 10)
            colored = np.zeros((*mask.shape, 4), dtype=float)
            colored[mask > 0] = [*color[:3], 0.5]
            ax.imshow(colored)
            label = f"obj {obj_id}"
            legend_patches.append(mpatches.Patch(color=color[:3], label=label))
        if legend_patches:
            ax.legend(handles=legend_patches, loc="upper right", fontsize=9)

    ax.set_title(f"Frame {frame_idx} | prompt: '{prompt}'")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # 1. Build predictor
    #    - checkpoint_path    : full SAM3 video model (tracker + ViT)
    #    - text_encoder_type  : swap language backbone to student encoder
    #    - student_text_encoder_checkpoint: load student text encoder weights
    #    - text_encoder_context_length: truncate to training context window
    # -----------------------------------------------------------------------
    from sam3.model_builder import build_sam3_video_predictor

    bpe_path = os.path.join(_sam3_pkg_dir, "assets", "bpe_simple_vocab_16e6.txt.gz")

    sam3_ckpt = args.sam3_checkpoint if os.path.exists(args.sam3_checkpoint) else None
    load_from_HF = sam3_ckpt is None
    if load_from_HF:
        print("sam3.pt not found locally — will download base video model from HuggingFace.")
    else:
        print(f"Base video model  : {sam3_ckpt}")

    litetext_ckpt = args.litetext_checkpoint
    if not os.path.exists(litetext_ckpt):
        print(f"ERROR: LiteText checkpoint not found: {litetext_ckpt}")
        return
    print(f"LiteText checkpoint: {litetext_ckpt}")
    print(f"Text encoder       : {args.backbone_type}  ctx={args.ctx}")

    gpus_to_use = list(range(torch.cuda.device_count())) if device == "cuda" else None

    print("\nBuilding SAM3-LiteText video predictor...")
    predictor = build_sam3_video_predictor(
        gpus_to_use=gpus_to_use,
        checkpoint_path=sam3_ckpt,
        load_from_HF=load_from_HF,
        bpe_path=bpe_path,
        # LiteText options
        text_encoder_type=args.backbone_type,
        text_encoder_context_length=args.ctx,
        student_text_encoder_checkpoint=litetext_ckpt,
        strict_state_dict_loading=False,
    )
    print("Predictor ready.\n")

    # -----------------------------------------------------------------------
    # 2. Load video frames for visualization
    # -----------------------------------------------------------------------
    video_path = args.video
    print(f"Loading video: {video_path}")
    frames = load_video_frames(video_path)
    if not frames:
        print(f"ERROR: No frames found at {video_path}")
        return
    print(f"Loaded {len(frames)} frames  ({frames[0].shape[1]}x{frames[0].shape[0]})")

    # -----------------------------------------------------------------------
    # 3. Start session
    # -----------------------------------------------------------------------
    response = predictor.handle_request({"type": "start_session", "resource_path": video_path})
    session_id = response["session_id"]
    print(f"Session started: {session_id}")

    # -----------------------------------------------------------------------
    # 4. Add text prompt on frame 0
    # -----------------------------------------------------------------------
    prompt = args.prompt
    print(f"\nAdding text prompt '{prompt}' on frame 0...")
    response = predictor.handle_request({
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": 0,
        "text": prompt,
    })
    frame0_out = response["outputs"]
    print(f"  Detections on frame 0: {len(frame0_out)} object(s)")

    # -----------------------------------------------------------------------
    # 5. Propagate through entire video
    # -----------------------------------------------------------------------
    print("\nPropagating through video...")
    outputs_per_frame = {}
    for resp in predictor.handle_stream_request({
        "type": "propagate_in_video",
        "session_id": session_id,
    }):
        outputs_per_frame[resp["frame_index"]] = resp["outputs"]

    print(f"Propagated {len(outputs_per_frame)} frames.")

    # -----------------------------------------------------------------------
    # 6. Save visualizations
    # -----------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    stride = args.frame_stride
    saved = []
    all_frame_idxs = sorted(outputs_per_frame.keys())
    for i, fidx in enumerate(all_frame_idxs):
        if i % stride != 0:
            continue
        if fidx >= len(frames):
            continue
        save_path = os.path.join(args.output_dir, f"frame_{fidx:05d}.png")
        visualize_frame(frames[fidx], outputs_per_frame[fidx], prompt, fidx, save_path)
        saved.append(save_path)

    print(f"\nSaved {len(saved)} visualized frames to: {args.output_dir}")
    if saved:
        print(f"  First: {saved[0]}")
        print(f"  Last : {saved[-1]}")

    # -----------------------------------------------------------------------
    # 7. Clean up
    # -----------------------------------------------------------------------
    predictor.handle_request({"type": "close_session", "session_id": session_id})
    print("\nDone.")


if __name__ == "__main__":
    main()
