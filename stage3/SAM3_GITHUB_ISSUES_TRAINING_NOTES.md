# SAM3 upstream GitHub issues — training setup notes

Short summaries and practical tips from [facebookresearch/sam3](https://github.com/facebookresearch/sam3) issues [#251](https://github.com/facebookresearch/sam3/issues/251), [#270](https://github.com/facebookresearch/sam3/issues/270), and [#163](https://github.com/facebookresearch/sam3/issues/163), plus how they relate to **EfficientSAM3 Stage 3** in this repository.

---

## Issue [#251](https://github.com/facebookresearch/sam3/issues/251) — Resolution, `min_size`, and GPU memory

**Question:** The default training setup uses `resolution` 1008 and `min_size` 480. Users want smaller values to save VRAM but report the transform pipeline breaking after changes.

**Community takeaways (not an official maintainer checklist):**

- Several users report that **changing `resolution` away from the defaults is fragile**: one comment attributes failures to **RoPE-related assumptions** in the model (e.g. attention modules using fixed `feat_sizes` tuned for a particular feature-grid geometry — see `sam3/sam/transformer.py` `RoPEAttention` defaults such as `feat_sizes=(64, 64)` with a comment referencing 1024-scale inputs).
- **Trying “multiples of 14” or 32** did not reliably fix the problem for everyone.
- **Memory vs. batch size:** One user saw **fewer OOMs with a larger batch size**, which turned out to be a **red herring**: with **`drop_last: true`**, a **small `num_images`** (or small effective dataset) plus a **large batch** can yield an **empty train dataloader** — training appears to “run” without doing useful work. Fix: increase data, lower batch size, or set `drop_last: false` where appropriate.
- **Practical implication:** Treat **1008 + `min_size: 480` + `PadToSizeAPI` to `resolution`** as the **known-good path** unless you are prepared to **audit RoPE / feature map sizes** for your backbone and resolution.

**VRAM levers that do not require changing resolution (aligned with this repo’s Stage 3 docs):**

- Lower **per-GPU batch size** and/or use **gradient accumulation** (where the trainer supports it).
- Reduce **`max_find_queries_per_img`**, **`max_ann_per_img`**, and (for geometry) **`max_geo_queries`** — decoder attention cost grows with query count.
- Keep **`enable_segmentation: false`** for the default detection-only Stage 3 profile; segmentation + mask losses materially increase activations and peak memory ([README_stage3.md](../README_stage3.md)).

---

## Issue [#270](https://github.com/facebookresearch/sam3/issues/270) — “No `.pt` after training” and loading checkpoints for inference

**Symptom:** Training finishes without errors but **no checkpoint files** appear on disk.

**Root cause (community):** In some example configs (e.g. Roboflow fine-tuning YAMLs), **`skip_saving_ckpts: true`** disables checkpoint writes even though training runs normally. Setting **`skip_saving_ckpts: false`** restores saving.

**Loading fine-tuned weights for inference:**

- Prefer **`build_sam3_image_model(..., checkpoint_path=..., load_from_HF=False)`** when the file is a **full** state dict compatible with the builder.
- Trainer checkpoints are often stored as **`checkpoint["model"]`** (nested dict). If keys do not match, compare **checkpoint keys vs. official `sam3.pt` keys** (export both to JSON) to spot **prefix mismatches** (e.g. `detector.` stripping/renaming in `model_builder.py` — discussed in the thread).
- **Partial / mismatched checkpoints:** Community workarounds include **`load_state_dict(..., strict=False)`**, **prefix rewriting** (`detector.`), or **copying missing modules** (e.g. tracker / interaction weights) from the official checkpoint. These are **brittle**; match **training config** and **inference `build_*` flags** to the same architecture.

**EfficientSAM3 Stage 3 in *this* repo:** Training intentionally saves **encoder-only** weights in `checkpoint.pt` and documents **`merge_stage3_checkpoint_for_eval.py`** to rebuild a **full** `.pth` for `build_efficientsam3_image_model`. That workflow avoids the “partial dict vs. strict load” confusion for the default Stage 3 setup ([README_stage3.md](../README_stage3.md) — Checkpoint Format, Step 5).

---

## Issue [#163](https://github.com/facebookresearch/sam3/issues/163) — Custom image/video datasets and format

**Image data (grounding / detection-style training):**

- **COCO-like JSON** with `images`, `annotations`, `categories`.
- Annotations should include a **text phrase** field such as **`noun_phrase`** per annotation (in addition to `bbox`, optional `segmentation`, etc.).
- **Negative prompts:** phrases present as queries but **without** matching masks are supported in SA-Co-style setups.

**Video data:**

- **YTVIS-like** JSON: `videos` (with `file_names` per frame), `annotations` with **per-frame** `segmentations` / `bboxes`, **`noun_phrase`**, etc.
- **Extract frames** first; **frame filenames must match** annotation `file_names`.

**Code pointers (upstream):** `sam3_image_dataset.py`, `sam3_video_dataset.py`; entrypoint `sam3/train/train.py` with configs under `configs/`.

**Stage 3 in this repo:** Default training uses **`stage3.data.mixed_text_mask_dataset.Stage3MixedTextMaskDataset`** with **`json_category`** and similar **COCO-style** sources (COCO, LVIS, ODINW, RF100-VL, RefCOCO). Adding a **custom** split means supplying **compatible JSON + directory layout** and registering a **new source** in the Hydra config — same *spirit* as [#163](https://github.com/facebookresearch/sam3/issues/163), but not the upstream `Sam3ImageDataset` path unless you switch configs.

**Other notes from the thread:**

- **Fine-tuning cost:** Author ballpark — **~18 GB** with **batch size 1** and **resolution 1008** for full fine-tuning in one scenario; your mileage varies with segmentation, queries, and model variant.
- **Tracker:** Training for **tracker** modules is **not** a focus of the open fine-tuning recipe; detector/backbone fine-tuning is what people actually run.

---

## Checklist: Stage 3 configs and code in *this* repository

| Topic | Upstream issue lesson | Status in this repo |
|--------|------------------------|---------------------|
| **Checkpoint saving** | Set `skip_saving_ckpts: false` if you expect `checkpoint.pt` | **`trainer.skip_saving_ckpts: false`** in `sam3/sam3/train/configs/stage3/mixed/stage3_mixed_local_train.yaml`. All listed Stage 3 mixed configs **`defaults: - stage3_mixed_local_train`**, so they inherit this. |
| **Inference from trainer output** | Full-model key layout vs. `checkpoint["model"]` | Documented **partial save** + **`merge_stage3_checkpoint_for_eval.py`** in [README_stage3.md](../README_stage3.md). |
| **Resolution / `min_size`** | Changing 1008 / 480 is risky (RoPE / grid assumptions) | Stage 3 base config uses **`resolution: 1008`**, **`min_size: 480`**, **`PadToSizeAPI`** to 1008 — **no change recommended** unless you invest in model-side verification. |
| **Empty dataloader** | `drop_last` + tiny dataset + large batch | Stage 3 **`drop_last: true`** on train; **smoketest** uses small **`max_samples`** — keep **batch size ≤ effective dataset size** per rank (smoketest uses small batch). |
| **Custom data** | COCO-like + `noun_phrase` | Extend **`stage3_mixed_train.train_sources`** / dataset loaders consistently with existing `json_category` entries. |

**Conclusion:** For the **current Stage 3 YAMLs** in `sam3/sam3/train/configs/stage3/mixed/`, **no additional modifications are required** solely to satisfy the lessons from these three issues: checkpoint saving is already enabled on the shared base config, resolution matches the known-good upstream recipe, and checkpoint usage is documented for EfficientSAM3’s encoder-only save format.

If you **fork a new config** without inheriting `stage3_mixed_local_train`, explicitly set **`trainer.skip_saving_ckpts: false`** and keep **`resolution` / transform consistency** with the working pipeline unless you validate the full model forward path at a new size.
