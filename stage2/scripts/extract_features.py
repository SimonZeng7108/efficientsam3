import argparse
import os
import torch
import torch.distributed as dist
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(os.getcwd())

from sam3.model_builder import build_sam3_video_model

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
        
        print(f"Initialized process {rank}/{world_size} on GPU {local_rank}")
        return rank, world_size, torch.device(f"cuda:{local_rank}")
    else:
        print("Running in single-process mode")
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(args):
    rank, world_size, device = setup_distributed()
    
    if rank == 0:
        print(f"Loading Teacher Model from {args.teacher_checkpoint}...")
    
    try:
        teacher_model = build_sam3_video_model(
            checkpoint_path=args.teacher_checkpoint,
            apply_temporal_disambiguation=True,
            device=device
        )
    except Exception as e:
        if rank == 0:
            print(f"Error loading checkpoint: {e}")
            print("Using random weights for testing structure...")
        teacher_model = build_sam3_video_model(
            checkpoint_path=None,
            apply_temporal_disambiguation=True,
            device=device
        )
        
    teacher_model.to(device)
    teacher_model.eval()
    
    # We need the backbone
    backbone = teacher_model.detector.backbone
    
    # Find processed videos
    all_video_dirs = sorted(glob.glob(os.path.join(args.dataset_path, "*")))
    
    # Filter only directories (exclude files like .txt or .json if any)
    all_video_dirs = [d for d in all_video_dirs if os.path.isdir(d)]
    
    # Partition work among GPUs
    my_video_dirs = all_video_dirs[rank::world_size]
    
    if rank == 0:
        print(f"Found {len(all_video_dirs)} total videos in {args.dataset_path}")
    print(f"Rank {rank}: Processing {len(my_video_dirs)} videos")
    
    # Image normalization constants
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # Only show progress bar on rank 0 to avoid clutter, or use position
    iterator = tqdm(my_video_dirs, position=rank, desc=f"Rank {rank}") if world_size > 1 else tqdm(my_video_dirs)

    for video_dir in iterator:
        video_name = os.path.basename(video_dir)
        output_file = os.path.join(video_dir, "features.pt")
        
        if os.path.exists(output_file) and not args.overwrite:
            continue
            
        # Load frames
        frame_paths = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
        if not frame_paths:
            continue
            
        # Process in batches
        all_feats = []
        all_pos = []
        
        batch_size = args.batch_size
        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i:i+batch_size]
            batch_imgs = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                img = img.resize((1008, 1008)) # Resize to SAM input 1008x1008
                batch_imgs.append(np.array(img))
            
            # [B, H, W, C] -> [B, C, H, W]
            batch_tensor = torch.from_numpy(np.stack(batch_imgs)).permute(0, 3, 1, 2).float() / 255.0
            batch_tensor = batch_tensor.to(device)
            
            # Normalize
            batch_tensor = (batch_tensor - mean) / std
            
            with torch.no_grad():
                # Forward backbone
                out_dict = backbone.forward_image(batch_tensor)
                
                # We want sam2_backbone_out["backbone_fpn"][2] (Scale 1.0)
                if "sam2_backbone_out" in out_dict and out_dict["sam2_backbone_out"] is not None:
                    feats = out_dict["sam2_backbone_out"]["backbone_fpn"][2]
                    pos = out_dict["sam2_backbone_out"]["vision_pos_enc"][2]
                else:
                    # Fallback if sam2 neck not used (should be used)
                    # print(f"Warning: sam2_backbone_out not found for {video_name}")
                    feats = out_dict["vision_features"]
                    pos = out_dict["vision_pos_enc"]
                    # Assuming index 2 is correct for fallback too
                    if isinstance(feats, list):
                        feats = feats[2]
                        pos = pos[2]

                all_feats.append(feats.cpu())
                all_pos.append(pos.cpu())
        
        # Concatenate all batches
        final_feats = torch.cat(all_feats, dim=0)
        final_pos = torch.cat(all_pos, dim=0)
        
        # Save as dictionary
        torch.save({
            "feat": final_feats,
            "pos": final_pos
        }, output_file)
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_checkpoint", type=str, required=True, help="Path to SAM3 teacher checkpoint")
    parser.add_argument("--dataset_path", type=str, default="data/sa-v/train", help="Path to processed SA-V dataset")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    extract_features(args)
