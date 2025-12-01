import argparse
import glob
import os
import sys
import json
from typing import List, Tuple, Optional

import torch
import torch.distributed as dist
from tqdm import tqdm
from pycocotools import mask as mask_utils
import numpy as np

# Add repo root for relative imports when executed via python -m
sys.path.append(os.getcwd())

from sam3.model_builder import build_sam3_video_model


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        dist.barrier()
        return rank, world_size, torch.device(f"cuda:{local_rank}")
    
    # Fallback for non-distributed execution
    print("Running in single process mode (no DDP)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, device


def _to_sequence(feat: torch.Tensor) -> torch.Tensor:
    # input [B, C, H, W] -> [HW, B, C]
    return feat.flatten(2).permute(2, 0, 1).contiguous()


def load_frame_features(
    frame_path: str, device: torch.device
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple[int, int]]]:
    data = torch.load(frame_path, map_location="cpu")
    feats = []
    poses = []
    feat_sizes: List[Tuple[int, int]] = []
    for lvl in ("f0", "f1", "f2"):
        feat = data[lvl].unsqueeze(0).to(device=device, dtype=torch.float32)
        _, _, h, w = feat.shape
        feat_sizes.append((h, w))
        feats.append(_to_sequence(feat))
    for lvl in ("p0", "p1", "p2"):
        pos = data[lvl].unsqueeze(0).to(device=device, dtype=torch.float32)
        poses.append(_to_sequence(pos))
    return feats, poses, feat_sizes


class GTMaskLoader:
    def __init__(self, dataset_path, video_name, height, width, device):
        self.video_name = video_name
        self.height = height
        self.width = width
        self.device = device
        self.masklets = []
        self.start_frames = []
        self.obj_idx = -1
        
        if not dataset_path:
            return

        # Try manual first, then auto
        json_path = os.path.join(dataset_path, "extracted_frames", video_name, f"{video_name}_manual.json")
        if not os.path.exists(json_path):
            json_path = os.path.join(dataset_path, "extracted_frames", video_name, f"{video_name}_auto.json")
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                self.masklets = data.get("masklet", [])
                self.start_frames = data.get("masklet_first_appeared_frame", [])
                
                if self.masklets and self.start_frames:
                    # Find object with longest track
                    max_len = -1
                    for idx, m in enumerate(self.masklets):
                        if len(m) > max_len:
                            max_len = len(m)
                            self.obj_idx = idx
                    
                    if self.obj_idx >= 0:
                        print(f"[{video_name}] Selected object {self.obj_idx} with track length {max_len}")
            except Exception as e:
                print(f"Error loading JSON {json_path}: {e}")

    def get_mask_at_frame(self, frame_idx: int) -> Optional[torch.Tensor]:
        if self.obj_idx < 0:
            return None
            
        start = int(self.start_frames[self.obj_idx])
        rel_idx = frame_idx - start
        
        if 0 <= rel_idx < len(self.masklets[self.obj_idx]):
            rle_obj = self.masklets[self.obj_idx][rel_idx]
            try:
                mask_np = mask_utils.decode(rle_obj)
                mask_tensor = torch.from_numpy(mask_np).float().to(self.device)
                
                if mask_tensor.shape[0] != self.height or mask_tensor.shape[1] != self.width:
                     mask_tensor = torch.nn.functional.interpolate(
                         mask_tensor.unsqueeze(0).unsqueeze(0),
                         size=(self.height, self.width),
                         mode="nearest"
                     ).squeeze(0).squeeze(0)
                
                return mask_tensor.unsqueeze(0).unsqueeze(0)
            except Exception as e:
                print(f"Error decoding mask for {self.video_name} at {frame_idx}: {e}")
                return None
        
        return None


def save_embeddings(args):
    rank, world_size, device = setup_distributed()
    if rank == 0:
        print(f"[Stage2] Loading SAM3 teacher from {args.teacher_checkpoint}")
    try:
        teacher_model = build_sam3_video_model(
            checkpoint_path=args.teacher_checkpoint,
            apply_temporal_disambiguation=True,
            device=device,
            compile=False,
        )
    except Exception as e:
        if rank == 0:
            print(f"Error loading checkpoint: {e}")
            print("Using random weights for testing structure...")
        teacher_model = build_sam3_video_model(
            checkpoint_path=None,
            apply_temporal_disambiguation=True,
            device=device,
            compile=False,
        )
    teacher_tracker = teacher_model.tracker
    teacher_tracker.eval()
    # Enable memory saving
    teacher_tracker.trim_past_non_cond_mem_for_eval = True
    teacher_tracker.offload_output_to_cpu_for_eval = True
    teacher_tracker.max_cond_frames_in_attn = 8

    feature_videos = sorted(
        [d for d in glob.glob(os.path.join(args.features_dir, "*")) if os.path.isdir(d)]
    )
    my_videos = feature_videos[rank::world_size]
    if rank == 0:
        print(f"[Stage2] Found {len(feature_videos)} feature directories")
    os.makedirs(args.output_dir, exist_ok=True)

    iterator = tqdm(my_videos, disable=world_size > 1 and rank != 0)
    
    chunk_size = 24 # Process video in chunks to allow independent training samples

    with torch.inference_mode():
        for video_dir in iterator:
            video_name = os.path.basename(video_dir.rstrip("/"))
            out_dir = os.path.join(args.output_dir, video_name)
            os.makedirs(out_dir, exist_ok=True)
            
            # Simple check if done (count files)
            # For robust check, we might want to count chunks, but simple check is fine for now.
            if (
                os.path.exists(out_dir)
                and len(os.listdir(out_dir)) > 0
                and not args.overwrite
            ):
                if rank == 0:
                    print(f"[Stage2][{video_name}] skip (already cached)")
                continue

            frame_files = sorted(glob.glob(os.path.join(video_dir, "*.pt")))
            if not frame_files:
                continue

            gt_loader = GTMaskLoader(args.dataset_path, video_name, args.image_size, args.image_size, device)
            num_frames = len(frame_files)
            
            # Process in chunks
            for chunk_start in range(0, num_frames, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_frames)
                
                # Reset teacher state for each chunk
                teacher_out_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
                
                # Check for mask at start of chunk
                start_mask = gt_loader.get_mask_at_frame(chunk_start)
                
                if start_mask is None:
                    # If no GT mask, we can't effectively track the object in this chunk.
                    # We could skip saving this chunk, or save with a dummy init.
                    # For distillation, we want valid tracks. 
                    # If the object is not present, maybe we track "nothing"?
                    # Let's use a dummy empty mask (background) so the student learns to predict background.
                    start_mask = torch.zeros(
                        1, 1, args.image_size, args.image_size, device=device, dtype=torch.float32
                    )
                    # But wait, if we initialize with empty mask, track_step might treat it as "no object".
                    # That's fine.
                
                for frame_idx in range(chunk_start, chunk_end):
                    frame_path = frame_files[frame_idx]
                    
                    feats, poses, feat_sizes = load_frame_features(frame_path, device)
                    
                    is_init = (frame_idx == chunk_start)
                    mask_inputs = start_mask if is_init else None
                    
                    last_feat = feats[-1]
                    last_pos = poses[-1]
                    last_size = [feat_sizes[-1]]
                    
                    # If init, we use raw features. If tracking, we use memory fused features.
                    if is_init:
                        base_feat = feats[-1]
                        H2, W2 = feat_sizes[-1]
                        teacher_feat = base_feat.permute(1, 2, 0).view(
                            base_feat.size(1), base_feat.size(2), H2, W2
                        )
                    else:
                        teacher_feat = teacher_tracker._prepare_memory_conditioned_features(
                            frame_idx=frame_idx,
                            is_init_cond_frame=is_init,
                            current_vision_feats=[last_feat],
                            current_vision_pos_embeds=[last_pos],
                            feat_sizes=last_size,
                            output_dict=teacher_out_dict,
                            num_frames=num_frames, # This tells positional encoding the max range
                        )

                    teacher_out = teacher_tracker.track_step(
                        frame_idx=frame_idx,
                        is_init_cond_frame=is_init,
                        current_vision_feats=feats,
                        current_vision_pos_embeds=poses,
                        feat_sizes=feat_sizes,
                        image=None,
                        point_inputs=None,
                        mask_inputs=mask_inputs,
                        output_dict=teacher_out_dict,
                        num_frames=num_frames,
                        run_mem_encoder=True,
                    )

                    if is_init:
                        teacher_out_dict["cond_frame_outputs"][frame_idx] = teacher_out
                    else:
                        teacher_out_dict["non_cond_frame_outputs"][frame_idx] = teacher_out

                    # Save output
                    save_payload = {
                        "teacher_feat": teacher_feat.squeeze(0).cpu().to(torch.float16),
                        "mask_high_res": teacher_out["pred_masks_high_res"].detach().cpu().to(torch.float16),
                        "object_scores": teacher_out["object_score_logits"].detach().cpu().to(torch.float16),
                        "is_init": is_init, # This allows training loop to know when to reset
                    }
                    
                    # Optional: Save chunk info? 
                    # Actually, `is_init` is sufficient. 
                    # If `is_init` is True, the student should treat it as start of sequence.

                    torch.save(save_payload, os.path.join(out_dir, f"{frame_idx:05d}.pt"))
                    
                    del feats, poses, last_feat, last_pos, teacher_feat, teacher_out
                
                # Explicit cleanup after chunk
                del teacher_out_dict
                torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cache teacher memory-conditioned embeddings for Stage 2"
    )
    parser.add_argument(
        "--teacher_checkpoint", type=str, required=True, help="Path to SAM3 checkpoint"
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        required=True,
        help="Directory with per-frame feature tensors",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/stage2_teacher",
        help="Destination directory for cached embeddings",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/sa-v",
        help="Path to SA-V dataset root (containing extracted_frames)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=1008,
        help="Input image resolution used by SAM3",
    )
    parser.add_argument(
        "--init_box",
        type=int,
        default=128,
        help="Side length for the synthetic initialization mask",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Regenerate embeddings if present"
    )
    args = parser.parse_args()
    save_embeddings(args)
