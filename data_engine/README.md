# Stage 3 Data Engine

This folder contains the SA-1B pseudo-label generation and visualization tools used by Stage 3.

Main scripts:
- generate.py: runs mask crop labeling and writes raw records and enhanced annotation JSON files
- visualize_sa1b.py: renders per-mask visualization images from enhanced JSON, raw JSONL, or annotation files

## 1) Prerequisites

From repository root, make sure the environment has:
- Python dependencies from this project
- pycocotools (required by both generator and visualizer)
- pillow, numpy
- transformers (required only for local_transformers backend)

Recommended setup from repo root:

~~~bash
pip install -e .
~~~

## 2) Data Preparation

The generator expects a reorganized SA-1B layout:

~~~text
data/sa-1b-1p_reorg/
  images/
    train/
      sa_XXXXXX.jpg
    val/
      sa_XXXXXX.jpg
  annotations/
    train/
      sa_XXXXXX.json
    val/
      sa_XXXXXX.json
~~~

Where each annotation JSON has:
- top-level image object with file_name, width, height, image_id
- annotations list with id, bbox, segmentation, area, predicted_iou, stability_score, point_coords

### How to prepare this layout

1. Download SA-1B subset archives (see README_dataset.md).
2. Extract and reorganize into train and val.

This repository includes data/reorg_sa1b.py for this purpose.
Note: this script uses constants inside main() (source_dir, output_dir, val_ratio, move_files, etc.).
Update those constants, then run:

~~~bash
python data/reorg_sa1b.py
~~~

Typical values in that script:
- source_dir = sa-1b-1p
- output_dir = sa-1b-1p_reorg

## 3) Generator Usage

Run from repo root.

### Minimal example (local transformers backend)

~~~bash
python data_engine/generate.py \
  --sa1b-root data/sa-1b-1p_reorg \
  --split train \
  --inference-backend local_transformers \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --limit-images 10
~~~

### Stub backend (pipeline smoke test only)

~~~bash
python data_engine/generate.py \
  --sa1b-root data/sa-1b-1p_reorg \
  --split train \
  --inference-backend stub \
  --limit-images 10
~~~

### Useful quality and control flags

- --start-index: start from a specific annotation index
- --limit-images: process only N images
- --max-masks-per-image: cap masks per image
- --min-area (default 4000)
- --min-predicted-iou (default 0.93)
- --min-stability-score (default 0.95)
- --crop-box-source mask|bbox (default mask)
- --overwrite: regenerate even if mask_id already exists in raw JSONL
- --num-mask-sample-points (default 10)
- --no-write-enhanced-annotations: disable enhanced JSON writing
- --text-rewrite-model (default Qwen/Qwen3.5-2B)
- --rewrite-max-tokens (default 96)
- --disable-label-rewrite

Notes:
- The quality thresholds are hard-clamped to minimum values of 4000 / 0.93 / 0.95. Lower CLI values are ignored.
- Label rewrite uses two separate text-model calls:
  - label_5: noun phrase with at most 5 words
  - label_2: noun phrase with exactly 2 words

### Generator outputs

Given --output-root data/sa1b_stage3_pseudo and --split train, outputs are:

1. Raw JSONL:

~~~text
data/sa1b_stage3_pseudo/raw/train.jsonl
~~~

2. Crop images used for prompting:

~~~text
data/sa1b_stage3_pseudo/prompt_renders/train/<image_id>/<mask_id>_crop.jpg
~~~

3. Enhanced annotation JSON files beside source annotations:

~~~text
data/sa-1b-1p_reorg/annotations/train/sa_xxxxxx_enhanced.json
~~~

Enhanced files contain only accepted masks. Each accepted annotation keeps original fields and adds:
- label_10
- label_5
- label_2
- mask_sample_points_xy

An *_enhanced.json file is written for each processed source annotation JSON.
If no masks pass gating for that image, the enhanced file is still written with an empty annotations list.

## 4) Visualizer Usage

Run from repo root.

### A) Visualize preprocessed enhanced JSON (recommended after generation)

~~~bash
python data_engine/visualize_sa1b.py \
  --sa1b-root data/sa-1b-1p_reorg \
  --split train \
  --pre_processed \
  --num-examples 10 \
  --output-dir output/data_engine_sa1b_examples_pre_processed
~~~

This mode reads only:

~~~text
data/sa-1b-1p_reorg/annotations/<split>/*_enhanced.json
~~~

### B) Visualize from raw JSONL records

~~~bash
python data_engine/visualize_sa1b.py \
  --sa1b-root data/sa-1b-1p_reorg \
  --split train \
  --raw-jsonl data/sa1b_stage3_pseudo/raw/train.jsonl \
  --num-examples 20 \
-  --min-area 4000 \
-  --min-predicted-iou 0.93 \
-  --min-stability-score 0.95 \
  --require-label \
  --output-dir output/data_engine_sa1b_examples
~~~

Useful options in raw-jsonl mode:
- --include-rejected
- --require-label

### C) Visualize directly from annotation files

~~~bash
python data_engine/visualize_sa1b.py \
  --sa1b-root data/sa-1b-1p_reorg \
  --split train \
  --annotation-source auto \
  --num-examples 10 \
  --output-dir output/data_engine_sa1b_examples
~~~

annotation-source behavior:
- auto: use *_enhanced.json if present, then *_text.json, otherwise base .json
- enhanced: use only *_enhanced.json
- text: use only *_text.json
- base: use only base .json

### Visualizer outputs

The visualizer writes one JPG per rendered mask into --output-dir.
 