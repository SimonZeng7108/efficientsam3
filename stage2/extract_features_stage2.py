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
        print(f"Loading SAM3 teacher from {args.teacher_checkpoint}...")
    
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
    tracker = teacher_model.tracker
    backbone = teacher_model.detector.backbone
    
    # Find processed videos
    all_video_dirs = sorted(glob.glob(os.path.join(args.dataset_path, "*")))
    
    # Filter only directories (exclude files like .txt or .json if any)
    all_video_dirs = [d for d in all_video_dirs if os.path.isdir(d)]
    
    # Partition work among GPUs
    my_video_dirs = all_video_dirs[rank::world_size]
    
    if rank == 0:
        print(f"Discovered {len(all_video_dirs)} videos under {args.dataset_path}")
    print(f"[Rank {rank}] assigned {len(my_video_dirs)} videos")
    
    # Image normalization constants
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # Only show progress bar on rank 0 to avoid clutter, or use position
    iterator = tqdm(my_video_dirs, position=rank, desc=f"Rank {rank}") if world_size > 1 else tqdm(my_video_dirs)

    os.makedirs(args.output_dir, exist_ok=True)

    for video_dir in iterator:
        video_name = os.path.basename(video_dir)
        
        # Create output directory for this video
        video_output_dir = os.path.join(args.output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Check if already done (check last frame?)
        # For now, we just overwrite or check if dir exists and is not empty if not overwrite
        if (
            os.path.exists(video_output_dir)
            and len(os.listdir(video_output_dir)) > 0
            and not args.overwrite
        ):
            if rank == 0:
                print(f"[Stage2][{video_name}] skip (already extracted)")
            continue
            
        # Load frames
        frame_paths = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
        if not frame_paths:
            continue
            
        # Process in batches
        batch_size = args.batch_size
        for i in range(0, len(frame_paths), batch_size):
            if rank == 0 and (i % 50 == 0):
                print(f"[Stage2][{video_name}] processed {i}/{len(frame_paths)} frames")
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
                if rank == 0 and i == 0:
                    print(f"[Stage2] backbone: {type(backbone)}")
                backbone_out = backbone.forward_image(batch_tensor)
                if "sam2_backbone_out" in backbone_out and backbone_out["sam2_backbone_out"] is not None:
                    feats_list = backbone_out["sam2_backbone_out"]["backbone_fpn"]
                    pos_list = backbone_out["sam2_backbone_out"]["vision_pos_enc"]
                else:
                    feats_list = backbone_out["backbone_fpn"]
                    pos_list = backbone_out["vision_pos_enc"]

                feats_list[0] = tracker.sam_mask_decoder.conv_s0(feats_list[0])
                feats_list[1] = tracker.sam_mask_decoder.conv_s1(feats_list[1])

                feats_list = feats_list[-3:]
                pos_list = pos_list[-3:]
                if rank == 0 and i == 0:
                    print(f"[Stage2] feature levels: {len(feats_list)}")
                    for k, f in enumerate(feats_list):
                        print(f"  Level {k}: {tuple(f.shape)}")

                # Save each frame
                for j in range(len(batch_paths)):
                    frame_idx = i + j
                    output_file_frame = os.path.join(
                        video_output_dir, f"{frame_idx:05d}.pt"
                    )
                    
                    # feats_list[k] is [B, C, H, W]
                    f0 = feats_list[0][j].detach().cpu().to(torch.float16)
                    f1 = feats_list[1][j].detach().cpu().to(torch.float16)
                    f2 = feats_list[2][j].detach().cpu().to(torch.float16)

                    p0 = pos_list[0][j].detach().cpu().to(torch.float16)
                    p1 = pos_list[1][j].detach().cpu().to(torch.float16)
                    p2 = pos_list[2][j].detach().cpu().to(torch.float16)

                    torch.save(
                        {
                            "f0": f0,
                            "f1": f1,
                            "f2": f2,
                            "p0": p0,
                            "p1": p1,
                            "p2": p2,
                        },
                        output_file_frame,
                    )
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_checkpoint", type=str, required=True, help="Path to SAM3 teacher checkpoint")
    parser.add_argument("--dataset_path", type=str, default="data/sa-v/train", help="Path to processed SA-V dataset")
    parser.add_argument("--output_dir", type=str, default="output/stage2_features", help="Path to save extracted features")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    extract_features(args)
