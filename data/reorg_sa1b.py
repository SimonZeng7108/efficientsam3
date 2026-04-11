import os
import shutil
import tarfile
from pathlib import Path
from typing import List, Tuple
import random
import json
from multiprocessing import Pool, cpu_count
import time

# ==========================================
# WORKER FUNCTIONS (Must be at module level)
# ==========================================

def _process_untar_pair(args):
    """
    Worker function to extract a single tar file.
    """
    tar_path, extract_root = args
    try:
        # Open the tar file
        with tarfile.open(tar_path, "r") as tar:
            # Extract contents to the directory
            # SA-1B tars usually contain the folder inside them, so we extract to root
            tar.extractall(path=extract_root)
        return (True, tar_path.name, None)
    except Exception as e:
        return (False, tar_path.name, str(e))

def _process_copy_pair(args):
    """
    Worker function to copy/move a single image-annotation pair.
    """
    img_path, ann_path, img_target_dir, ann_target_dir, move = args
    
    try:
        operation = shutil.move if move else shutil.copy2
        
        # Copy/move image
        img_target = Path(img_target_dir) / Path(img_path).name
        operation(str(img_path), str(img_target))
        
        # Copy/move annotation
        ann_target = Path(ann_target_dir) / Path(ann_path).name
        operation(str(ann_path), str(ann_target))
        
        return (True, Path(img_path).name, None)
    except Exception as e:
        return (False, Path(img_path).name, str(e))

# ==========================================
# MAIN LOGIC
# ==========================================

def extract_all_tars(source_dir: Path, num_workers: int = None):
    """
    Finds all .tar files in source_dir and extracts them in parallel.
    """
    if num_workers is None:
        num_workers = cpu_count()

    tar_files = list(source_dir.glob("*.tar"))
    
    if not tar_files:
        print("No .tar files found. Skipping extraction phase.")
        return

    print(f"\nFound {len(tar_files)} .tar files.")
    print(f"Extracting files using {num_workers} workers...")

    # Prepare args: (path_to_tar, directory_to_extract_into)
    # We extract into the same source_dir. 
    # The tars usually contain a folder (e.g. sa_000020/), so it won't clutter root.
    args_list = [(tar_file, source_dir) for tar_file in tar_files]

    start_time = time.time()
    successful = 0
    failed = 0

    with Pool(processes=num_workers) as pool:
        results = pool.imap_unordered(_process_untar_pair, args_list)
        
        for i, (success, filename, error) in enumerate(results, 1):
            if success:
                successful += 1
            else:
                failed += 1
                print(f"\n  Error extracting {filename}: {error}")
            
            # Progress reporting
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            print(f"  Extracted {i}/{len(tar_files)} tars ({rate:.1f} tars/sec)...", end='\r')

    elapsed = time.time() - start_time
    print(f"\n  Extraction complete: {successful} successful, {failed} failed in {elapsed:.1f}s")


def get_all_image_annotation_pairs(source_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Scan source_dir and its subdirectories to collect image-annotation pairs.
    """
    pairs = []
    
    # Directories to scan: source_dir itself + subdirectories
    dirs_to_scan = [source_dir]
    try:
        subdirs = [d for d in source_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        subdirs.sort()
        dirs_to_scan.extend(subdirs)
    except FileNotFoundError:
        return []

    # Filter out output dirs if they exist in source_dir
    excluded_names = ["images", "annotations", "SA-1B-10P", "SA-1B-1P", "train", "val"]
    dirs_to_scan = [d for d in dirs_to_scan if d.name not in excluded_names]

    print(f"\nScanning {len(dirs_to_scan)} directories for images...")
    
    for d in dirs_to_scan:
        dir_name = d.name if d != source_dir else "root"
        
        folder_pairs = 0
        # Get all jpg files in this directory
        for img_file in d.glob("*.jpg"):
            # Check if corresponding json exists
            json_file = img_file.with_suffix('.json')
            if json_file.exists():
                pairs.append((img_file, json_file))
                folder_pairs += 1
        
        if folder_pairs > 0:
            print(f"  {dir_name}: found {folder_pairs} pairs")
    
    return pairs


def create_directory_structure(output_dir: Path):
    """Create the target directory structure."""
    dirs_to_create = [
        output_dir / "images" / "train",
        output_dir / "images" / "val",
        output_dir / "annotations" / "train",
        output_dir / "annotations" / "val",
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        # print(f"Created directory: {dir_path}")


def split_train_val(pairs: List[Tuple[Path, Path]], 
                      val_ratio: float = 0.1,
                      seed: int = 42) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """Split pairs into train and validation sets."""
    random.seed(seed)
    shuffled_pairs = pairs.copy()
    random.shuffle(shuffled_pairs)
    
    val_size = int(len(shuffled_pairs) * val_ratio)
    val_pairs = shuffled_pairs[:val_size]
    train_pairs = shuffled_pairs[val_size:]
    
    return train_pairs, val_pairs


def copy_files(pairs: List[Tuple[Path, Path]], 
               output_dir: Path, 
               split: str,
               move: bool = False,
               num_workers: int = None):
    """Copy or move files to the target directory structure using multiprocessing."""
    if num_workers is None:
        num_workers = cpu_count()
    
    img_target_dir = output_dir / "images" / split
    ann_target_dir = output_dir / "annotations" / split
    
    operation_name = "Moving" if move else "Copying"
    
    print(f"\n{operation_name} {len(pairs)} files to {split} set using {num_workers} workers...")
    
    args_list = [
        (img_path, ann_path, str(img_target_dir), str(ann_target_dir), move)
        for img_path, ann_path in pairs
    ]
    
    start_time = time.time()
    successful = 0
    failed = 0
    
    with Pool(processes=num_workers) as pool:
        results = pool.imap_unordered(_process_copy_pair, args_list, chunksize=100)
        
        for i, (success, filename, error) in enumerate(results, 1):
            if success:
                successful += 1
            else:
                failed += 1
                print(f"\n  Error processing {filename}: {error}")
            
            if i % 500 == 0 or i == len(pairs):
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  Processed {i}/{len(pairs)} files ({rate:.1f} files/sec)...", end='\r')
    
    elapsed = time.time() - start_time
    print(f"\n  Completed {split} set: {successful} successful, {failed} failed in {elapsed:.1f}s")


def main():
    # ================= CONFIGURATION =================
    source_dir = Path("sa-1b-1p")   # Where the .tar files are
    output_dir = Path("SA-1B-1P")   # Where the Train/Val split goes
    val_ratio = 0.1 
    move_files = True                # Move after extracting?
    num_workers = cpu_count()        # Use all cores
    # =================================================

    print("=" * 60)
    print("SA-1B Dataset Extractor & Reorganizer")
    print("=" * 60)
    print(f"CPU cores available: {cpu_count()}")
    
    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' not found!")
        return

    # -------------------------------------------------
    # PHASE 1: CHECK & EXTRACT
    # -------------------------------------------------
    print(f"\nPhase 1: Checking for existing images/annotations...")
    pairs = get_all_image_annotation_pairs(source_dir)
    
    if len(pairs) > 0:
        print(f"Found {len(pairs)} existing pairs. Skipping extraction.")
    else:
        print(f"No existing pairs found. Proceeding to extraction.")
        print(f"Checking for .tar files in '{source_dir}'...")
        extract_all_tars(source_dir, num_workers)

        print(f"\nPhase 2: Scanning for extracted images/annotations...")
        pairs = get_all_image_annotation_pairs(source_dir)
    
    if len(pairs) == 0:
        print("Error: No image-annotation pairs found! Did the extraction work?")
        return

    # -------------------------------------------------
    # PHASE 3: SETUP & SPLIT
    # -------------------------------------------------
    print(f"\nPhase 3: Creating target structure in '{output_dir}'...")
    create_directory_structure(output_dir)
    
    print(f"Splitting data (train/val ratio: {1-val_ratio:.1%}/{val_ratio:.1%})...")
    train_pairs, val_pairs = split_train_val(pairs, val_ratio=val_ratio)
    print(f"Train set: {len(train_pairs)} pairs")
    print(f"Val set: {len(val_pairs)} pairs")

    # -------------------------------------------------
    # PHASE 4: REORGANIZE
    # -------------------------------------------------
    print(f"\nPhase 4: {'Moving' if move_files else 'Copying'} files to target structure...")
    copy_files(train_pairs, output_dir, "train", move=move_files, num_workers=num_workers)
    copy_files(val_pairs, output_dir, "val", move=move_files, num_workers=num_workers)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Check '{output_dir}' for your data.")

if __name__ == "__main__":
    main()
