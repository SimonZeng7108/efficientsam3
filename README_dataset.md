## Datasets and Download Scripts

This repository provides helper scripts to download all datasets used in EfficientSAM3. Each script places files under the `data/` directory and handles extraction/cleanup where appropriate.

### Prerequisites

- bash, wget, unzip
- Optional per-dataset tools:
  - gdown: `pip install gdown`
  - huggingface-cli: `pip install huggingface_hub`

Note: These datasets are large (hundreds of GBs for SA-1B/SA-V). Ensure you have sufficient storage and a stable connection.

### Quickstart

```bash
# COCO 2017
bash data/download_coco.sh

# DAVIS 2016/2017
bash data/download_davis.sh

# LVIS v1 annotations + COCO train/val images
bash data/download_lvis.sh

# SA-1B (from TSV list) — supports parallel downloads
bash data/download_sa1b.sh data/sa-1b.txt data/sa-1b 8

# SA-V (wrapper around the SA-1B downloader)
bash data/download_sa_v.sh data/sa-v.txt data/sa-v 8

# LVOS v2 (Google Drive; requires gdown)
bash data/download_lvos.sh

# MOSE 1/2 (Hugging Face; requires huggingface-cli)
bash data/download_mose.sh

# YouTube-VOS 2019 (Google Drive folder; requires gdown)
bash data/download_ytvos.sh

# Recap-DataComp-1B (Subset) — 1% of the dataset
python3 data/download_datacomp.py

# Recap-COCO-30K
bash data/download_recap_coco.sh
```

---

### COCO 2017 — `data/download_coco.sh`

Downloads and extracts COCO 2017 images and annotations into `data/coco`.

Important: The script currently has `wget` lines commented out and will unzip any existing `*.zip` in `data/coco/images`. Uncomment lines 9–12 in the script if you want it to download the zips automatically.

Outputs:
- `data/coco/images/{train2017,val2017,test2017,unlabeled2017}`
- `data/coco/annotations/*`

Example directory tree:
```text
data/coco/
  images/
    train2017/
    val2017/
    test2017/
    unlabeled2017/
  annotations/
    instances_train2017.json
    instances_val2017.json
    image_info_test2017.json
    image_info_unlabeled2017.json
    // additional stuff_* JSON files may be present
```

Docs: `https://cocodataset.org/#download`

### DAVIS 2016 / 2017 — `data/download_davis.sh`

Downloads DAVIS 2016 and 2017 (trainval + unsupervised) and extracts them into `data/davis`.

Outputs:
- `data/davis/2016/*`
- `data/davis/2017/{trainval,unsupervised}/*`

Example directory tree:
```text
data/davis/
  2016/
    DAVIS/
      Annotations/
      ImageSets/
      JPEGImages/
      // exact layout per official zip
  2017/
    trainval/
      DAVIS/
        Annotations/
        ImageSets/
        JPEGImages/
    unsupervised/
      DAVIS/
        Annotations_unsupervised/
        ImageSets/
        JPEGImages/
```

Docs:
- 2016: `https://davischallenge.org/davis2016/code.html`
- 2017: `https://davischallenge.org/davis2017/code.html`

### LVIS v1 (+ COCO images) — `data/download_lvis.sh`

Downloads LVIS v1 train/val annotations and the required COCO 2017 train/val images. Extracts under `data/lvis`.

Outputs:
- `data/lvis/annotations/*`
- `data/lvis/images/{train2017,val2017}`

Example directory tree:
```text
data/lvis/
  annotations/
    lvis_v1_train.json
    lvis_v1_val.json
  images/
    train2017/
    val2017/
```

Docs: `https://www.lvisdataset.org/dataset`

### SA-1B — `data/download_sa1b.sh`

Downloads SA-1B archives listed in a TSV with two columns: `file_name<TAB>cdn_link`.

Usage:
```bash
bash data/download_sa1b.sh [INPUT_TSV] [OUTPUT_DIR] [CONCURRENCY]
# defaults: INPUT_TSV=data/sa-1b.txt, OUTPUT_DIR=data/sa-1b, CONCURRENCY=1
```

Example (8 parallel downloads):
```bash
bash data/download_sa1b.sh data/sa-1b.txt data/sa-1b 8
```

Features:
- Resumes partial downloads (`wget -c`), retries, and timeouts
- Gracefully handles Ctrl-C and waits for child processes

Docs: `https://ai.meta.com/datasets/segment-anything/`

Outputs:
- `.tar` archives under `data/sa-1b/` (no automatic extraction)

Example listing and optional extraction helper:
```bash
# After download
ls -1 data/sa-1b | head

# Extract all .tar files into a sibling directory (optional)
mkdir -p data/sa-1b-extracted
find data/sa-1b -type f -name "*.tar" -print0 \
  | xargs -0 -I{} tar -xf {} -C data/sa-1b-extracted
```

### SA-V — `data/download_sa_v.sh`

Thin wrapper around the SA-1B downloader; points to `data/sa-v.txt` by default.

Usage:
```bash
bash data/download_sa_v.sh [INPUT_TSV] [OUTPUT_DIR] [CONCURRENCY]
# defaults: INPUT_TSV=data/sa-v.txt, OUTPUT_DIR=data/sa-v, CONCURRENCY=1
```

Example:
```bash
bash data/download_sa_v.sh data/sa-v.txt data/sa-v 8
```

Docs: `https://ai.meta.com/datasets/segment-anything-video/`

Outputs:
- `.tar` archives and checksum files under `data/sa-v/` (no automatic extraction)

Example directory tree and verification:
```text
data/sa-v/
  sav_000.tar
  sav_001.tar
  ...
  sav_md5sum.chk
  sav_sha256sum.chk
```

Verify checksums (optional):
```bash
pushd data/sa-v >/dev/null
md5sum -c sav_md5sum.chk || true       # prints mismatches; depends on upstream list
sha256sum -c sav_sha256sum.chk || true
popd >/dev/null
```

Extract helper (optional):
```bash
mkdir -p data/sa-v-extracted
find data/sa-v -type f -name "*.tar" -print0 \
  | xargs -0 -I{} tar -xf {} -C data/sa-v-extracted
```

### LVOS v2 — `data/download_lvos.sh`

Downloads LVOS v2 train/val from Google Drive using `gdown`. Accepts URLs via args or env vars `LVOS_TRAIN_URL` and `LVOS_VAL_URL`.

Example:
```bash
# Use defaults embedded in the script
bash data/download_lvos.sh

# Or override via args
bash data/download_lvos.sh "https://drive.google.com/file/d/<train_id>/view" \
                          "https://drive.google.com/file/d/<val_id>/view"
```

Outputs:
- `data/lvos/{train,val}/*` (any `*.zip` found will be extracted and removed)

Example directory tree:
```text
data/lvos/
  train/
    <unzipped contents>
  val/
    <unzipped contents>
```

### MOSE 1 / 2 — `data/download_mose.sh`

Downloads MOSE1 and MOSE2 dataset repos from Hugging Face using `huggingface-cli`.

Usage:
```bash
# Defaults: MOSE1 = FudanCVL/MOSE, MOSE2 = FudanCVL/MOSEv2
bash data/download_mose.sh

# Or specify via args
bash data/download_mose.sh FudanCVL/MOSE FudanCVL/MOSEv2

# Or full HF dataset URLs are also accepted
bash data/download_mose.sh https://huggingface.co/datasets/FudanCVL/MOSE \
                           https://huggingface.co/datasets/FudanCVL/MOSEv2
```

Outputs:
- `data/mose/mose1/*` and `data/mose/mose2/*`
- Automatically joins MOSEv2 multipart archives (e.g., `train.tar.gz.aa`, `train.tar.gz.ab`, `train.tar.gz.ac` -> `train.tar.gz`) before extraction.
- Recursively extracts nested archives (`*.zip`, `*.tar`, `*.tar.gz`, `*.tgz`, `*.tar.bz2`, `*.tar.xz`) and deletes archives after extraction.

Note: Some MOSE repositories require authentication; if you encounter permission errors, run:
```bash
huggingface-cli login
```

Example directory tree (varies per release):
```text
data/mose/
  mose1/
    train/
    val/
    # or equivalent folders from the HF dataset
  mose2/
    train/
    val/
```

### YouTube-VOS 2019 — `data/download_ytvos.sh`

Downloads the official YouTube-VOS 2019 Google Drive folder using `gdown --folder` and extracts any zips.

Usage:
```bash
# Use default folder URL embedded in the script
bash data/download_ytvos.sh

# Or override via arg or env YTVOS_URL
bash data/download_ytvos.sh "https://drive.google.com/drive/folders/<folder_id>"
```

Outputs:
- `data/ytvos/*` (any `*.zip` found will be extracted and removed)

Example directory tree:
```text
data/ytvos/
  train/
  valid/   # or val depending on release
  metas.json (or similar)
  # exact layout depends on the official folder contents
```

### Recap-DataComp-1B (Subset) — `data/download_datacomp.py`

Downloads a 1% subset of the Recap-DataComp-1B dataset from Hugging Face.

Outputs:
- `data/recap_subset/data/*.parquet`

Usage:
```bash
python3 data/download_datacomp.py
```

Docs: `https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B`

### Recap-COCO-30K — `data/download_recap_coco.sh`

Downloads the Recap-COCO-30K dataset (parquet format).

Outputs:
- `data/recap_coco/new_data.parquet`

Usage:
```bash
bash data/download_recap_coco.sh
```

Docs: `https://huggingface.co/datasets/UCSC-VLAA/Recap-COCO-30K`

---

### TSV Lists

- `data/sa-1b.txt`: SA-1B archive list (file name and CDN link)
- `data/sa-v.txt`: SA-V archive list (file name and CDN link)

Checksum files provided by SA-V are included alongside the TSV (e.g., `sav_md5sum.chk`, `sav_sha256sum.chk`).

---

### Licensing

Each dataset has its own license and terms. Please review and comply with the original dataset licenses and usage policies before downloading and using the data.