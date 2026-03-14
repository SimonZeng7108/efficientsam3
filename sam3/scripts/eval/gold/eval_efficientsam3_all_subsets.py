#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Evaluation script for EfficientSAM3 models on all SA-Co Gold subsets.
Supports distributed inference via torchrun.

Usage:
  torchrun --nproc_per_node=4 eval_efficientsam3_all_subsets.py \
    --checkpoint <path> --backbone-type MobileCLIP-S0 \
    --data-root <sa-co-gold/all> --gt-folder <gt-annotations> \
    --output-dir <output_dir>
"""

import argparse
import json
import os
import time
import traceback
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from pycocotools import mask as mask_utils

from sam3.eval.cgf1_eval import CGF1Evaluator
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

# SA-Co Gold subsets and their GT files (same as eval_sam3.py)
SACO_GOLD_GTS = {
    "metaclip_nps": [
        "gold_metaclip_merged_a_release_test.json",
        "gold_metaclip_merged_b_release_test.json",
        "gold_metaclip_merged_c_release_test.json",
    ],
    "sa1b_nps": [
        "gold_sa1b_merged_a_release_test.json",
        "gold_sa1b_merged_b_release_test.json",
        "gold_sa1b_merged_c_release_test.json",
    ],
    "crowded": [
        "gold_crowded_merged_a_release_test.json",
        "gold_crowded_merged_b_release_test.json",
        "gold_crowded_merged_c_release_test.json",
    ],
    "fg_food": [
        "gold_fg_food_merged_a_release_test.json",
        "gold_fg_food_merged_b_release_test.json",
        "gold_fg_food_merged_c_release_test.json",
    ],
    "fg_sports_equipment": [
        "gold_fg_sports_equipment_merged_a_release_test.json",
        "gold_fg_sports_equipment_merged_b_release_test.json",
        "gold_fg_sports_equipment_merged_c_release_test.json",
    ],
    "attributes": [
        "gold_attributes_merged_a_release_test.json",
        "gold_attributes_merged_b_release_test.json",
        "gold_attributes_merged_c_release_test.json",
    ],
    "wiki_common": [
        "gold_wiki_common_merged_a_release_test.json",
        "gold_wiki_common_merged_b_release_test.json",
        "gold_wiki_common_merged_c_release_test.json",
    ],
}


def setup_distributed():
    """Initialize distributed process group if running with torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        rank = 0
        world_size = 1
        device = "cuda"
    return rank, world_size, device


def get_image_path(data_root, file_name):
    """Resolve image path: sa1b files go to sa1b-images/, else metaclip-images/."""
    if file_name.startswith("sa_"):
        return os.path.join(data_root, "sa1b-images", file_name)
    else:
        return os.path.join(data_root, "metaclip-images", file_name)


def run_inference_subset(processor, images_info, data_root, device):
    """Run inference on a list of image entries, grouping by file_name for efficiency."""
    predictions = []

    # Group entries by file_name to reuse image encoding
    groups = OrderedDict()
    for img_info in images_info:
        fn = img_info["file_name"]
        if fn not in groups:
            groups[fn] = []
        groups[fn].append(img_info)

    total = len(images_info)
    processed = 0

    for file_name, img_infos in groups.items():
        img_path = get_image_path(data_root, file_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            processed += len(img_infos)
            continue

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(image)

            for img_info in img_infos:
                state = processor.set_text_prompt(img_info["text_input"], state)

                masks = state.get("masks")
                scores = state.get("scores")

                if masks is not None and len(scores) > 0:
                    for i in range(len(scores)):
                        mask_np = masks[i, 0].cpu().numpy().astype(np.uint8)
                        rle = mask_utils.encode(np.asfortranarray(mask_np))
                        rle["counts"] = rle["counts"].decode("utf-8")
                        predictions.append(
                            {
                                "image_id": img_info["id"],
                                "category_id": 1,
                                "segmentation": rle,
                                "score": float(scores[i].cpu()),
                            }
                        )

                processor.reset_all_prompts(state)
                processed += 1

        if processed % 500 == 0:
            print(f"  [{device}] Processed {processed}/{total} entries")

    return predictions


def save_and_merge_predictions(predictions, rank, world_size, output_dir, subset_name):
    """Each rank saves its predictions to disk, then rank 0 merges all."""
    rank_dir = os.path.join(output_dir, f"gold_{subset_name}", "ranks")
    os.makedirs(rank_dir, exist_ok=True)
    rank_file = os.path.join(rank_dir, f"rank_{rank}.json")
    with open(rank_file, "w") as f:
        json.dump(predictions, f)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        all_predictions = []
        for r in range(world_size):
            with open(os.path.join(rank_dir, f"rank_{r}.json")) as f:
                all_predictions.extend(json.load(f))
        return all_predictions

    return []


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EfficientSAM3 on SA-Co Gold subsets"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to merged checkpoint")
    parser.add_argument(
        "--backbone-type",
        required=False,
        default=None,
        choices=["MobileCLIP-S0", "MobileCLIP-S1", "MobileCLIP2-L", "SAM3", None],
        help="Student text encoder type. Use 'SAM3' or omit for the original SAM3 text encoder.",
    )
    parser.add_argument(
        "--data-root", required=True, help="Path to sa-co-gold/all directory"
    )
    parser.add_argument(
        "--gt-folder", required=True, help="Path to gt-annotations directory"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--batch-size", type=int, default=4, help="(unused, kept for CLI compat)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="(unused, kept for CLI compat)"
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.5, help="Detection threshold"
    )
    parser.add_argument(
        "--resolution", type=int, default=1008, help="Input resolution"
    )
    parser.add_argument(
        "--context-length", type=int, default=77,
        help="Text encoder context length (e.g. 32 for ctx32 models, 77 for default)"
    )
    parser.add_argument(
        "--pos-embed-table-size",
        type=int,
        default=None,
        help="Student text encoder positional embedding table size. Defaults to"
             " --context-length (fixed/slice default). Set 77 for legacy interp"
             " checkpoints when using --interpolate-pos-embed.",
    )
    parser.add_argument(
        "--subsets", type=str, default=None,
        help="Comma-separated list of subset names to evaluate (default: all). "
             "E.g. metaclip_nps,sa1b_nps,crowded,fg_food"
    )
    parser.add_argument(
        "--interpolate-pos-embed", action="store_true", default=False,
        help="Optional inference mode that interpolates the positional table to"
             " --context-length. Default inference slices/truncates the table."
    )
    args = parser.parse_args()

    rank, world_size, device = setup_distributed()

    if rank == 0:
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Text encoder: {args.backbone_type} (resolved: {'SAM3/MetaCLIP (original)' if args.backbone_type in (None, 'SAM3') else args.backbone_type})")
        print(f"Context length: {args.context_length}")
        print(f"Pos-embed table size: {args.pos_embed_table_size if args.pos_embed_table_size is not None else args.context_length}")
        print(f"Interpolate pos-embed: {args.interpolate_pos_embed}")
        print(f"World size: {world_size}")
        print(f"Data root: {args.data_root}")
        print(f"Output dir: {args.output_dir}")

    # BPE path (relative to sam3 package)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bpe_path = os.path.join(
        script_dir, "..", "..", "..", "assets", "bpe_simple_vocab_16e6.txt.gz"
    )
    bpe_path = os.path.normpath(bpe_path)

    if rank == 0:
        print(f"BPE path: {bpe_path}")
        print("Loading model...")

    # "SAM3" or None both mean: use the original SAM3 text encoder (MetaCLIP)
    text_encoder_type = args.backbone_type if args.backbone_type not in (None, "SAM3") else None

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=args.checkpoint,
        text_encoder_type=text_encoder_type,
        text_encoder_context_length=args.context_length,
        text_encoder_pos_embed_table_size=args.pos_embed_table_size,
        interpolate_pos_embed=args.interpolate_pos_embed,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        load_from_HF=False,
        device="cuda",  # torch.cuda.set_device already called in setup_distributed
        eval_mode=True,
    )

    processor = Sam3Processor(
        model,
        resolution=args.resolution,
        device=device,  # use rank-specific device for processor
        confidence_threshold=args.confidence_threshold,
    )

    if rank == 0:
        print("Model loaded successfully.\n")

    results_str = ""

    # Filter subsets if --subsets was specified
    subsets_to_run = SACO_GOLD_GTS
    if args.subsets:
        requested = [s.strip() for s in args.subsets.split(",")]
        subsets_to_run = {k: v for k, v in SACO_GOLD_GTS.items() if k in requested}
        if rank == 0:
            print(f"Running subset(s): {list(subsets_to_run.keys())}")

    for subset_name, gts in subsets_to_run.items():
        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"Subset: {subset_name}")
            print(f"{'=' * 60}")

        # Load GT (use first annotator file for image list)
        gt_path = os.path.join(args.gt_folder, gts[0])
        with open(gt_path) as f:
            gt_data = json.load(f)

        images_info = gt_data["images"]
        # Sort by id for deterministic sharding across ranks
        images_info.sort(key=lambda x: x["id"])

        # Shard across ranks
        shard_size = len(images_info) // world_size
        remainder = len(images_info) % world_size
        start = rank * shard_size + min(rank, remainder)
        end = start + shard_size + (1 if rank < remainder else 0)
        local_images = images_info[start:end]

        if rank == 0:
            print(
                f"Total entries: {len(images_info)}, "
                f"per-rank shard: ~{len(local_images)}"
            )

        t0 = time.time()
        local_preds = run_inference_subset(
            processor, local_images, args.data_root, device
        )
        t1 = time.time()
        print(
            f"[Rank {rank}] {subset_name}: "
            f"{len(local_preds)} predictions in {t1 - t0:.1f}s"
        )

        # Merge predictions from all ranks
        all_preds = save_and_merge_predictions(
            local_preds, rank, world_size, args.output_dir, subset_name
        )

        if rank == 0:
            # Save final merged predictions
            pred_dir = os.path.join(
                args.output_dir,
                f"gold_{subset_name}",
                "dumps",
                f"gold_{subset_name}",
            )
            os.makedirs(pred_dir, exist_ok=True)
            pred_path = os.path.join(pred_dir, "coco_predictions_segm.json")
            with open(pred_path, "w") as f:
                json.dump(all_preds, f)
            print(f"Saved {len(all_preds)} predictions to {pred_path}")

            # Evaluate
            gt_paths = [os.path.join(args.gt_folder, gt) for gt in gts]
            try:
                evaluator = CGF1Evaluator(
                    gt_path=gt_paths, verbose=True, iou_type="segm"
                )
                summary = evaluator.evaluate(pred_path)

                cgf1 = str(round(summary["cgF1_eval_segm_cgF1"] * 100, 2))
                il_mcc = str(round(summary["cgF1_eval_segm_IL_MCC"], 2))
                pmf1 = str(
                    round(summary["cgF1_eval_segm_positive_micro_F1"] * 100, 2)
                )
                result = f"{cgf1},{il_mcc},{pmf1}"
                results_str += f"{subset_name}: {result}\n"
                print(f"  >> CGF1={cgf1}, IL_MCC={il_mcc}, pmF1={pmf1}")
            except Exception as e:
                print(f"  >> Evaluation FAILED for {subset_name}: {e}")
                traceback.print_exc()
                results_str += f"{subset_name}: EVAL_ERROR ({e})\n"

        if world_size > 1:
            dist.barrier()

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("All Results (Subset: CGF1, IL_MCC, pmF1)")
        print(f"{'=' * 60}")
        print(results_str)

        # Save summary
        os.makedirs(args.output_dir, exist_ok=True)
        summary_path = os.path.join(args.output_dir, "results_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Text encoder: {args.backbone_type}\n")
            f.write(f"Context length: {args.context_length}\n")
            f.write(f"Interpolate pos-embed: {args.interpolate_pos_embed}\n")
            f.write(f"Confidence threshold: {args.confidence_threshold}\n\n")
            f.write("Subset: CGF1, IL_MCC, pmF1\n")
            f.write(results_str)
        print(f"Summary saved to {summary_path}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
