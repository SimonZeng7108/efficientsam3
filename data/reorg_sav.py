import os
import glob
import argparse
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
import shutil
import json

def process_video(args):
    video_path, output_root = args
    try:
        video_path = Path(video_path)
        video_name = video_path.stem
        
        # Create output directory for this video
        video_out_dir = output_root / video_name
        video_out_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Extract Frames using FFmpeg
        # -q:v 2 for high quality JPEG
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-q:v", "2",
            "-start_number", "0",
            str(video_out_dir / "%05d.jpg")
        ]
        
        # Run ffmpeg silently
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # 2. Copy Annotations
        # Look for json files with same stem in the same directory as video
        parent_dir = video_path.parent
        for json_file in parent_dir.glob(f"{video_name}*.json"):
            shutil.copy2(json_file, video_out_dir / json_file.name)
            
        return (True, video_name, None)
        
    except Exception as e:
        return (False, video_path.name, str(e))

def main():
    parser = argparse.ArgumentParser(description="Reorganize SA-V dataset: Extract frames and copy annotations.")
    parser.add_argument("--source_dir", type=str, default="data/sa-v", help="Root directory of SA-V raw data")
    parser.add_argument("--output_dir", type=str, default="data/sa-v/extracted_frames", help="Output directory")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    
    args = parser.parse_args()
    
    source_root = Path(args.source_dir)
    output_root = Path(args.output_dir)
    
    if not source_root.exists():
        print(f"Error: Source directory {source_root} does not exist.")
        return

    # Find all mp4 files recursively
    print(f"Scanning {source_root} for .mp4 files...")
    video_files = list(source_root.rglob("*.mp4"))
    
    if not video_files:
        print("No .mp4 files found.")
        return
        
    print(f"Found {len(video_files)} videos.")
    print(f"Processing to {output_root}...")
    
    num_workers = args.workers if args.workers else cpu_count()
    
    # Prepare args for workers
    task_args = [(v, output_root) for v in video_files]
    
    successful = 0
    failed = 0
    
    from tqdm import tqdm
    
    with Pool(num_workers) as pool:
        for success, name, error in tqdm(pool.imap_unordered(process_video, task_args), total=len(video_files)):
            if success:
                successful += 1
            else:
                failed += 1
                print(f"Failed {name}: {error}")
                
    print(f"\nProcessing Complete.")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()
