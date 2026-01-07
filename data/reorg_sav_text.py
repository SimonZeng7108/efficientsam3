#!/usr/bin/env python3
"""
Script to organize text annotations from sa-v-text and rf20-vl datasets.
Extracts unique text annotations for stage1_text_encoder training.

Datasets:
- sa-v-text: Contains JSON files with 'text_input' key in image entries
- rf20-vl: Contains COCO format JSONs with category names

Output: A single JSON file with unique text annotations.
"""

import json
import os
import glob
from pathlib import Path
from typing import Set, Dict, List
from tqdm import tqdm


def extract_text_from_sav_text(base_dir: str) -> Set[str]:
    """
    Extract text_input values from sa-v-text dataset.
    
    Structure:
    - sa-co-gold/gt-annotations/*.json
    - sa-co-silver/gt-annotations/*.json
    - sa-co-veval/*.json
    """
    text_annotations = set()
    
    # Define subdirectories to search (excluding sa-co-veval)
    subdirs = [
        "sa-co-gold/gt-annotations",
        "sa-co-silver/gt-annotations",
    ]
    
    for subdir in subdirs:
        json_dir = os.path.join(base_dir, subdir)
        if not os.path.exists(json_dir):
            print(f"Warning: Directory not found: {json_dir}")
            continue
        
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        print(f"\nProcessing {len(json_files)} files from {subdir}")
        
        for json_file in tqdm(json_files, desc=f"Processing {subdir}"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract text_input from images list
                if 'images' in data:
                    for img in data['images']:
                        if 'text_input' in img:
                            text_input = img['text_input'].strip()
                            if text_input:
                                text_annotations.add(text_input)
                                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
    
    return text_annotations


def extract_text_from_rf20vl(base_dir: str) -> Set[str]:
    """
    Extract category names from rf20-vl dataset (COCO format).
    
    Structure:
    - Each subdirectory has train/valid/test folders
    - Each folder contains _annotations.coco.json
    """
    text_annotations = set()
    
    # Find all _annotations.coco.json files
    pattern = os.path.join(base_dir, "**", "_annotations.coco.json")
    json_files = glob.glob(pattern, recursive=True)
    
    print(f"\nProcessing {len(json_files)} COCO annotation files from rf20-vl")
    
    for json_file in tqdm(json_files, desc="Processing rf20-vl"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract category names
            if 'categories' in data:
                for cat in data['categories']:
                    if 'name' in cat:
                        cat_name = cat['name'].strip()
                        if cat_name:
                            text_annotations.add(cat_name)
                            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    return text_annotations


def main():
    # Get the base directory (where this script is located)
    script_dir = Path(__file__).parent.resolve()
    
    # Define dataset paths
    sav_text_dir = script_dir / "sa-v-text"
    rf20vl_dir = script_dir.parent / "rf20-vl"
    
    print("=" * 60)
    print("Text Annotation Extraction for Stage1 Text Encoder Training")
    print("=" * 60)
    
    all_texts = set()
    
    # Extract from sa-v-text
    if sav_text_dir.exists():
        print(f"\n[1/2] Processing sa-v-text: {sav_text_dir}")
        sav_texts = extract_text_from_sav_text(str(sav_text_dir))
        print(f"Found {len(sav_texts)} unique text annotations from sa-v-text")
        all_texts.update(sav_texts)
    else:
        print(f"Warning: sa-v-text directory not found: {sav_text_dir}")
    
    # Extract from rf20-vl
    if rf20vl_dir.exists():
        print(f"\n[2/2] Processing rf20-vl: {rf20vl_dir}")
        rf20_texts = extract_text_from_rf20vl(str(rf20vl_dir))
        print(f"Found {len(rf20_texts)} unique text annotations from rf20-vl")
        all_texts.update(rf20_texts)
    else:
        print(f"Warning: rf20-vl directory not found: {rf20vl_dir}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total unique text annotations: {len(all_texts)}")
    
    # Sort and save
    sorted_texts = sorted(list(all_texts))
    
    # Output file
    output_file = script_dir / "text_annotations_combined.json"
    
    output_data = {
        "info": {
            "description": "Combined text annotations from sa-v-text and rf20-vl datasets",
            "purpose": "Stage1 text encoder training for distilling SAM3 text encoder",
            "num_annotations": len(sorted_texts),
            "sources": [
                "sa-v-text/sa-co-gold",
                "sa-v-text/sa-co-silver",
                "rf20-vl (all subdirectories)"
            ]
        },
        "text_annotations": sorted_texts
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nOutput saved to: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    # Also save a simple text file for quick reference
    txt_output = script_dir / "text_annotations_combined.txt"
    with open(txt_output, 'w', encoding='utf-8') as f:
        for text in sorted_texts:
            f.write(text + '\n')
    
    print(f"Text file saved to: {txt_output}")
    
    # Show some examples
    print("\n" + "=" * 60)
    print("SAMPLE ANNOTATIONS (first 20):")
    print("=" * 60)
    for i, text in enumerate(sorted_texts[:20], 1):
        print(f"  {i:3d}. {text}")
    
    if len(sorted_texts) > 20:
        print(f"  ... and {len(sorted_texts) - 20} more")


if __name__ == "__main__":
    main()
