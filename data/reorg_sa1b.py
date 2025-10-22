"""
Script to reorganize SA-1B dataset into train/val split structure.

Original structure:
    sa-1b/
        sa_000020/
            sa_234928.jpg
            sa_234928.json
            ...
        sa_000021/
            ...

Target structure:
    SA-1B/
        images/
            train/
                xxx.jpg
            val/
                yyy.jpg
        annotations/
            train/
                xxx.json
            val/
                yyy.json
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple
import random
import json
from multiprocessing import Pool, cpu_count
from functools import partial
import time


def get_all_image_annotation_pairs(source_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Scan all subdirectories in source_dir and collect image-annotation pairs.
    Automatically discovers and processes ALL subdirectories (sa_000000 through sa_999999).
    
    Args:
        source_dir: Path to sa-1b directory
        
    Returns:
        List of tuples (image_path, annotation_path)
    """
    pairs = []
    folder_stats = {}
    
    # Get all subdirectories and sort them
    subdirs = [d for d in source_dir.iterdir() if d.is_dir()]
    subdirs.sort()
    
    if not subdirs:
        print(f"Warning: No subdirectories found in {source_dir}")
        return pairs
    
    print(f"Found {len(subdirs)} folder(s) to process")
    
    # Iterate through ALL subdirectories (sa_000000, sa_000001, ..., sa_999999, etc.)
    for subdir in subdirs:
        print(f"Scanning {subdir.name}...", end=" ")
        
        folder_pairs = 0
        # Get all jpg files in this subdirectory
        for img_file in subdir.glob("*.jpg"):
            # Check if corresponding json exists
            json_file = img_file.with_suffix('.json')
            if json_file.exists():
                pairs.append((img_file, json_file))
                folder_pairs += 1
            else:
                print(f"\n  Warning: No annotation found for {img_file.name}", end="")
        
        folder_stats[subdir.name] = folder_pairs
        print(f"found {folder_pairs} pairs")
    
    # Print summary of all folders
    print(f"\nFolder Summary:")
    for folder_name, count in folder_stats.items():
        print(f"  {folder_name}: {count} pairs")
    
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
        print(f"Created directory: {dir_path}")


def split_train_val(pairs: List[Tuple[Path, Path]], 
                     val_ratio: float = 0.1,
                     seed: int = 42) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """
    Split pairs into train and validation sets.
    
    Args:
        pairs: List of (image_path, annotation_path) tuples
        val_ratio: Ratio of validation set (default: 0.1 = 10%)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_pairs, val_pairs)
    """
    random.seed(seed)
    shuffled_pairs = pairs.copy()
    random.shuffle(shuffled_pairs)
    
    val_size = int(len(shuffled_pairs) * val_ratio)
    val_pairs = shuffled_pairs[:val_size]
    train_pairs = shuffled_pairs[val_size:]
    
    return train_pairs, val_pairs


def _process_single_pair(args):
    """
    Helper function to process a single image-annotation pair.
    Must be at module level for multiprocessing to pickle it.
    
    Args:
        args: Tuple of (img_path, ann_path, img_target_dir, ann_target_dir, move)
        
    Returns:
        Tuple of (success: bool, file_name: str, error_msg: str or None)
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


def copy_files(pairs: List[Tuple[Path, Path]], 
               output_dir: Path, 
               split: str,
               move: bool = False,
               num_workers: int = None):
    """
    Copy or move files to the target directory structure using multiprocessing.
    
    Args:
        pairs: List of (image_path, annotation_path) tuples
        output_dir: Target SA-1B directory
        split: Either 'train' or 'val'
        move: If True, move files instead of copying
        num_workers: Number of parallel workers (default: CPU count)
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    img_target_dir = output_dir / "images" / split
    ann_target_dir = output_dir / "annotations" / split
    
    operation_name = "Moving" if move else "Copying"
    
    print(f"\n{operation_name} {len(pairs)} files to {split} set using {num_workers} workers...")
    
    # Prepare arguments for multiprocessing
    args_list = [
        (img_path, ann_path, str(img_target_dir), str(ann_target_dir), move)
        for img_path, ann_path in pairs
    ]
    
    # Process files in parallel
    start_time = time.time()
    successful = 0
    failed = 0
    
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for progress reporting
        results = pool.imap_unordered(_process_single_pair, args_list, chunksize=10)
        
        for i, (success, filename, error) in enumerate(results, 1):
            if success:
                successful += 1
            else:
                failed += 1
                print(f"\n  Error processing {filename}: {error}")
            
            # Progress reporting
            if i % 100 == 0 or i == len(pairs):
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  Processed {i}/{len(pairs)} files ({rate:.1f} files/sec)...", end='\r')
    
    elapsed = time.time() - start_time
    print(f"\n  Completed {split} set: {successful} successful, {failed} failed in {elapsed:.1f}s")


def main():
    """Main function to reorganize SA-1B dataset."""
    # Configuration
    source_dir = Path("sa-1b")  # Source directory
    output_dir = Path("SA-1B")  # Target directory
    val_ratio = 0.1  # 10% validation, 90% train
    move_files = False  # Set to True to move instead of copy
    num_workers = cpu_count()  # Number of parallel workers (None = auto-detect)
    
    print("=" * 60)
    print("SA-1B Dataset Reorganization Script (Multiprocessing)")
    print("=" * 60)
    print(f"CPU cores available: {cpu_count()}")
    print(f"Workers to use: {num_workers}")
    
    # Validate source directory exists
    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' not found!")
        return
    
    # Step 1: Collect all image-annotation pairs
    print(f"\nStep 1: Scanning source directory '{source_dir}'...")
    pairs = get_all_image_annotation_pairs(source_dir)
    print(f"Found {len(pairs)} image-annotation pairs")
    
    if len(pairs) == 0:
        print("Error: No image-annotation pairs found!")
        return
    
    # Step 2: Create target directory structure
    print(f"\nStep 2: Creating target directory structure in '{output_dir}'...")
    create_directory_structure(output_dir)
    
    # Step 3: Split into train and validation
    print(f"\nStep 3: Splitting data (train/val ratio: {1-val_ratio:.1%}/{val_ratio:.1%})...")
    train_pairs, val_pairs = split_train_val(pairs, val_ratio=val_ratio)
    print(f"Train set: {len(train_pairs)} pairs")
    print(f"Val set: {len(val_pairs)} pairs")
    
    # Step 4: Copy/move files to target structure
    print(f"\nStep 4: {'Moving' if move_files else 'Copying'} files to target structure...")
    copy_files(train_pairs, output_dir, "train", move=move_files, num_workers=num_workers)
    copy_files(val_pairs, output_dir, "val", move=move_files, num_workers=num_workers)
    
    # Summary
    print("\n" + "=" * 60)
    print("Dataset reorganization completed successfully!")
    print("=" * 60)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Train images: {len(train_pairs)}")
    print(f"Val images: {len(val_pairs)}")
    print(f"Total: {len(pairs)}")
    
    # Verify structure
    print("\nVerifying structure...")
    for split in ["train", "val"]:
        img_count = len(list((output_dir / "images" / split).glob("*.jpg")))
        ann_count = len(list((output_dir / "annotations" / split).glob("*.json")))
        print(f"  {split}: {img_count} images, {ann_count} annotations")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

