"""Extract teacher trunk embeddings from SAM3 for Stage 3 fine-tuning.

Runs SAM3's Hiera image encoder on every SA-1B image and saves trunk output
tensors (B, 1024, 72, 72) as individual .pt files keyed by image_id.

Usage:
    python stage3/scripts/extract_teacher_embeddings.py \
        --sa1b-root data/sa-1b-1p_reorg \
        --sam3-checkpoint sam3_checkpoints/sam3.pt \
        --output-dir output/stage3_teacher_embeddings \
        --split train
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stage1.data.transforms import ResizeLongestSide

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_SIZE = 1008
PIXEL_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
PIXEL_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)


def preprocess_image(img_path: str, transform: ResizeLongestSide) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    img_t = pil_to_tensor(img).float()
    img_t = transform.apply_image_torch(img_t[None]).squeeze(0)
    img_t = (img_t - PIXEL_MEAN) / PIXEL_STD
    h, w = img_t.shape[1:]
    img_t = F.pad(img_t, (0, IMG_SIZE - w, 0, IMG_SIZE - h))
    return img_t


def main():
    parser = argparse.ArgumentParser(description="Extract SAM3 teacher trunk embeddings")
    parser.add_argument("--sa1b-root", type=str, required=True)
    parser.add_argument("--sam3-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", help="Save embeddings in float16")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading SAM3 from {args.sam3_checkpoint}...")
    from sam3.model_builder import build_sam3_image_model

    sam3 = build_sam3_image_model(
        checkpoint_path=args.sam3_checkpoint,
        load_from_HF=False,
        eval_mode=True,
        device=args.device,
        enable_segmentation=False,
        enable_inst_interactivity=False,
        compile=False,
        enable_text_encoder=False,
    )
    sam3.eval()

    vision_backbone = sam3.backbone.vision_backbone
    trunk = vision_backbone.trunk

    transform = ResizeLongestSide(IMG_SIZE)

    img_pattern = os.path.join(args.sa1b_root, "images", args.split, "*.jpg")
    img_paths = sorted(glob.glob(img_pattern))
    print(f"Found {len(img_paths)} images in {args.split}")

    already_done = set()
    for f in glob.glob(os.path.join(args.output_dir, "*.pt")):
        already_done.add(Path(f).stem)

    remaining = [p for p in img_paths if Path(p).stem not in already_done]
    print(f"Skipping {len(already_done)} already extracted; {len(remaining)} remaining")

    batches = [remaining[i : i + args.batch_size] for i in range(0, len(remaining), args.batch_size)]

    save_dtype = torch.float16 if args.fp16 else torch.float32

    with torch.no_grad():
        for batch_paths in tqdm(batches, desc="Extracting embeddings"):
            imgs = []
            keys = []
            for p in batch_paths:
                imgs.append(preprocess_image(p, transform))
                keys.append(Path(p).stem)

            batch = torch.stack(imgs).to(args.device)

            features = trunk(batch)
            if isinstance(features, (list, tuple)):
                features = features[-1]

            for i, key in enumerate(keys):
                emb = features[i].cpu().to(save_dtype)
                out_path = os.path.join(args.output_dir, f"{key}.pt")
                torch.save(emb, out_path)

    print(f"Done. Embeddings saved to {args.output_dir}")


if __name__ == "__main__":
    main()
