# Stage 3 Plan

Last updated: 2026-05-04 UTC

Investigation update: 2026-05-01 18:25 UTC. The completed SA-Co Gold runs are not valid segmentation fine-tuning evidence. The training config disabled segmentation (`enable_segmentation: false`), the collator omitted segmentation masks (`with_seg_masks: false`), and the configured loss had no mask loss (`loss_fn_semantic_seg: null`, only box/class losses). The eval then built a segmentation-enabled model and loaded the training checkpoint with 28 missing `segmentation_head.pixel_decoder.*` keys, so the segmentation head was not loaded from the fine-tuned checkpoint. This explains near-zero mask CGF1 even though jobs completed cleanly.

## Current Direction

**Active scaling run (May 2026):** The best validated line below
(image encoder + projector + FPN, no KD, text-only segmentation) is
being scaled from SA-Co Gold (~14k images, 5 epochs) to **half** of
`data/sa-1b-5p-sacap/anno.parquet` (~25.8k SA-1B images with SACap
per-segment captions) for **30 epochs**, using the same TinyViT-11M /
MobileCLIP-S0 / ctx16 / 1008^2 / batch-1 stack and the same Stage 1
merged checkpoint. New artifacts:

- Hydra: `sam3/sam3/train/configs/stage3/mixed/stage3_mixed_sacap_sa1b_tvm_mcs0_seg_img_fpn.yaml`
  (full) and `stage3_mixed_sacap_sa1b_tvm_mcs0_seg_diag.yaml` (gate).
- SLURM: `scripts/train_eval_stage3_sacap_sa1b_seg_1node.sh`
  (`LINE=img_fpn|diag`, `MODE=train|eval`, `EVAL_TARGET=gold`).
- Data source: `kind: sacap_sa1b` in
  `stage3/data/mixed_text_mask_dataset.py`. Lazy per-image annotation
  loading + a single `os.listdir` per image split keeps Lustre
  metadata cost flat at startup (per Isambard-AI guidance).
- Submitted chain: diag `4445809` → train `4445810` → eval `4445811`
  (eval reuses the SA-Co Gold 5% val protocol so numbers are
  directly comparable to the `4.58` baseline below).

Fine-tune `/home/b5cz/simonz.b5cz/program/stage2/efficientsam3/output/sam3_stage1_image_5p/efficient_sam3_tv_m_mobileclip_s0_ctx16_5p_highmse.pt` on the SA-Co Gold dataset with a deterministic 95/5 train/eval split.

**Note:** Early SA-Co Gold retries (batch/GPU `4`) are superseded for segmentation by the corrected Hydra recipe below (batch/GPU `1`, full mask supervision). Use the **best validated recipe** section for current training.

## Best validated result (May 2026)

This is the **strongest comparable** line we ran: **supervised SA-Co Gold segmentation** with **TinyViT-11M image encoder + projector + FPN** trained together, **text-only prompts** (no box prompts), **no teacher KD**, **5** data epochs, **batch size 1 per GPU** on **4** GPUs, **1008** square inputs, deterministic **95% train / 5% val** split (`split_seed=123`, `val_holdout_frac=0.05`).

### Numbers (5% val split, all seven subsets, confidence 0.5)

Metrics are **CGF1, IL_MCC, pmF1** per subset (evaluator summary). **Mean CGF1** is the unweighted average of the seven subset CGF1 scores (~**4.58**).

| Subset | CGF1 | IL_MCC | pmF1 |
| --- | ---: | ---: | ---: |
| `metaclip_nps` | 5.41 | 0.23 | 23.70 |
| `sa1b_nps` | 11.53 | 0.38 | 30.51 |
| `crowded` | 2.10 | 0.21 | 10.12 |
| `fg_food` | 4.58 | 0.20 | 22.59 |
| `fg_sports_equipment` | 4.63 | 0.19 | 24.75 |
| `attributes` | 3.66 | 0.15 | 23.91 |
| `wiki_common` | 0.15 | 0.04 | 4.01 |

- **Eval artifact:** `/home/b5cz/simonz.b5cz/program/stage2/efficientsam3/output/efficientsam3_tv_m_mcs0_ctx16_sa1b_only_seg_img_fpn_gold_val_fixed2/results_summary.txt`
- **Fine-tuned trainer checkpoint:** `/home/b5cz/simonz.b5cz/program/stage2/efficientsam3/output/stage3/sa1b_only_gold_tvm_mcs0_ctx16_seg_img_fpn/checkpoints/checkpoint.pt`
- **Base (Stage 1) checkpoint loaded into the trainer:** `/home/b5cz/simonz.b5cz/program/stage2/efficientsam3/output/sam3_stage1_image_5p/efficient_sam3_tv_m_mobileclip_s0_ctx16_5p_highmse.pt`

For comparison on the **same** val protocol: **image-only** mean CGF1 ~**4.14**; **FPN-only** ~**2.91**; **KD FPN** ~**4.39**; **KD trunk** ~**4.35** (see `results_summary.txt` under `output/efficientsam3_tv_m_mcs0_ctx16_sa1b_only_seg_*_gold_val_full2` for KD runs). Teacher KD did **not** beat this best line on mean CGF1.

### How it was configured (Hydra)

- **Entry config:** `sam3/sam3/train/configs/stage3/mixed/stage3_mixed_sa1b_only_gold_tvm_mcs0_seg_img_fpn.yaml`  
  - Sets `train_vision_encoder: true`, `train_vision_fpn: true`, `train_text_encoder: false`.
- **Base config (inherits):** `stage3_mixed_sa1b_only_gold_tvm_mcs0_seg_base.yaml` → `stage3_mixed_local_train` (standard SAM3 trainer stack).
- **Data:** `kind: sa1b_only_gold`, root `${paths.sa1b_only_gold_root}` = `.../data/sa-co/sa-co-gold/all`, all seven subsets, `annotators: all`, `require_masks: true`, **`use_prompt_boxes: false`**, deterministic split `val_holdout_frac: 0.05`, `split_seed: 123` (train `val_is_holdout: false`, val `val_is_holdout: true`).
- **Segmentation:** `scratch.enable_segmentation: true`, collator `with_seg_masks: true`, dataset `load_segmentation: true`, `Masks` loss (`loss_mask: 200`, `loss_dice: 10`) plus box / focal classification losses as in base YAML.
- **Model:** `backbone_type: tinyvit`, `model_name: 11m`, `text_encoder_type: MobileCLIP-S0`, `text_encoder_context_length: 16`, `text_encoder_pos_embed_table_size: 16`, `interpolate_pos_embed: false`, `checkpoint_path` = merged Stage 1 TinyViT checkpoint above, **`freeze_non_encoder_parameters: true`**, **`freeze_vision_batchnorm: true`**, **`keep_frozen_modules_eval: true`** (text / geometry / transformer / scoring frozen; image trunk + FPN trainable).
- **Optimization / schedule:** inherited local train defaults with `scratch.max_data_epochs: 5`, `scratch.train_batch_size: 1`, `scratch.val_batch_size: 1`, `scratch.resolution: 1008`, `scratch.lr_scale: 0.1` (applied within the composed config’s LR rules).

### How to train (SLURM, 1 node, 4 GPUs)

From repo root `/home/b5cz/simonz.b5cz/program/stage2/efficientsam3`, use `scripts/train_eval_stage3_sa1b_only_gold_seg_1node.sh` (already sets `module load cuda/12.6`, conda `efficientsam3`, `PYTHONPATH=sam3:.`, `TMPDIR=/tmp`, unsets `PYTHONNOUSERSITE`).

**Train:**

```bash
cd /home/b5cz/simonz.b5cz/program/stage2/efficientsam3
sbatch --job-name=es3_sa1b_only_img_fpn_train \
  --export=ALL,LINE=img_fpn,MODE=train,BATCH_SIZE=1,EPOCHS=5 \
  scripts/train_eval_stage3_sa1b_only_gold_seg_1node.sh
```

Hydra overrides are appended by the script as `paths.experiment_log_dir=...`, `scratch.train_batch_size=${BATCH_SIZE}`, `scratch.max_data_epochs=${EPOCHS}`. To change experiment dir, either edit the script `case` block for `img_fpn` or pass extra overrides via `EXTRA_OVERRIDES` if you extend the script.

**Evaluate on the held-out 5% val** (matches training split; evaluator loads **base** then **overlays** fine-tuned weights; strict key match):

```bash
sbatch --job-name=es3_sa1b_only_img_fpn_eval \
  --export=ALL,LINE=img_fpn,MODE=eval,SPLIT=val,EVAL_SUFFIX=_rerun \
  scripts/train_eval_stage3_sa1b_only_gold_seg_1node.sh
```

Set `CKPT=/path/to/checkpoint.pt` in the environment if not using the default `${EXP_DIR}/checkpoints/checkpoint.pt`. The eval invocation uses `--split val --split-fraction 0.05 --split-seed 123`, `--resolution 1008`, `--vision-backbone-type tinyvit --vision-model-name 11m`, `--base-checkpoint` = same merged Stage 1 path.

**Optional diagnostic gate** (short run before full train): `LINE=diag` uses `stage3_mixed_sa1b_only_gold_tvm_mcs0_seg_diag.yaml` (1 epoch, capped samples); submit full `img_fpn` only after the diagnostic exits `0`.

### Reproduce without SLURM (single node, four processes)

Same env as the script, then from repo root:

```bash
export PYTHONPATH="sam3:."
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500   # pick a free port
srun -n 4 --ntasks-per-node=4 --gpus-per-task=1 python stage3/train_stage3_srun.py \
  -c configs/stage3/mixed/stage3_mixed_sa1b_only_gold_tvm_mcs0_seg_img_fpn.yaml \
  paths.experiment_log_dir=/path/to/your_exp_dir \
  scratch.train_batch_size=1 \
  scratch.max_data_epochs=5
```

Match `paths.experiment_log_dir` to where you want checkpoints and logs.

---

Training uses 1 node and 4 GPUs. Early SA-Co Gold retries used per-GPU batch size `4` (see tables below); **corrected segmentation** jobs used **batch/GPU `1`** (above).

## Completed SA-Co Gold Jobs

| Job | Line | Train job | Eval job | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| J19/J20 | image encoder/projector only | `4429716` | `4429717` | completed | Train elapsed `01:46:56`; eval elapsed `00:05:57`; exit `0:0`. Mean 5% Gold CGF1 about `0.06`. |
| J21/J22 | FPN only | `4429718` | `4429719` | completed | Train elapsed `01:36:09`; eval elapsed `00:05:43`; exit `0:0`. Mean 5% Gold CGF1 about `0.06`. |
| J23/J24 | image encoder/projector + FPN | `4429720` | `4429721` | completed | Train elapsed `01:46:32`; eval elapsed `00:06:08`; exit `0:0`. Mean 5% Gold CGF1 about `0.03`. |

## Memory and Batch Size

Per-GPU batch size `4` was stable for all final training jobs.

| Line | Typical/peak memory from logs |
| --- | --- |
| image encoder/projector only | about `18 GB` average, `73 GB` peak |
| FPN only | about `12 GB` average, `67 GB` peak |
| image encoder/projector + FPN | about `18 GB` average, `73 GB` peak |

No batch-size reduction is needed for stability. The jobs are already complete, so changing batch size now would only matter for a new longer retry.

## 5% SA-Co Gold Evaluation Results

Each subset is reported as `CGF1, IL_MCC, pmF1`.

Image encoder/projector only (`4429717`):

| Subset | Result |
| --- | --- |
| `metaclip_nps` | `0.03, 0.23, 0.12` |
| `sa1b_nps` | `0.09, 0.25, 0.38` |
| `crowded` | `0.00, 0.11, 0.03` |
| `fg_food` | `0.03, 0.22, 0.15` |
| `fg_sports_equipment` | `0.03, 0.28, 0.09` |
| `attributes` | `0.24, 0.25, 0.94` |
| `wiki_common` | `0.00, 0.05, 0.07` |

FPN only (`4429719`):

| Subset | Result |
| --- | --- |
| `metaclip_nps` | `0.06, 0.13, 0.48` |
| `sa1b_nps` | `0.05, 0.17, 0.28` |
| `crowded` | `0.01, 0.06, 0.09` |
| `fg_food` | `0.08, 0.24, 0.32` |
| `fg_sports_equipment` | `0.02, 0.26, 0.09` |
| `attributes` | `0.17, 0.17, 1.00` |
| `wiki_common` | `0.00, 0.04, 0.08` |

Image encoder/projector + FPN (`4429721`):

| Subset | Result |
| --- | --- |
| `metaclip_nps` | `0.05, 0.18, 0.29` |
| `sa1b_nps` | `0.05, 0.23, 0.21` |
| `crowded` | `0.00, 0.06, 0.02` |
| `fg_food` | `0.01, 0.24, 0.04` |
| `fg_sports_equipment` | `0.00, 0.31, 0.00` |
| `attributes` | `0.08, 0.20, 0.39` |
| `wiki_common` | `0.00, 0.07, 0.02` |

## Interpretation

All final train/eval chains ran cleanly after the dataloader and transform fixes, but the training/eval setup was wrong for segmentation evaluation. The runs optimized box/class grounding while evaluation measured segmentation masks with an incompletely loaded segmentation head.

Required next fix: run a small compute-node diagnostic and then resubmit corrected SA-Co Gold jobs with segmentation enabled, `with_seg_masks: true`, a `Masks` loss, and checkpoint handoff that preserves or reloads the base segmentation-head weights before applying fine-tuned weights.

## Corrected Segmentation Training Submission

Submitted 2026-05-01 18:45 UTC, then resubmitted after fixing `TMPDIR`. These jobs replace the invalid detection-only SA-Co Gold runs above.

Fixes applied:

- Restored the deleted Stage 3 source files required by the training configs.
- Added a SA-Co Gold source adapter that reads `images[*].text_input`, uses `annotations[*].segmentation`, converts SA-Co normalized boxes to pixel `xyxy`, and resolves `sa1b-images` vs `metaclip-images`.
- Enabled segmentation training: `enable_segmentation: true`, `with_seg_masks: true`, and `sam3.train.loss.loss_fns.Masks`.
- Kept training text-only (`use_prompt_boxes: false`) to match the SA-Co Gold eval path.
- Added FPN unfreeze support via `train_vision_fpn`.
- Changed eval to load the full base checkpoint first and overlay the fine-tuned trainer checkpoint, preventing missing `segmentation_head.pixel_decoder.*` weights.
- Added deterministic eval split support matching the train/val hash split.

| Job | Purpose | Dependency | Status at submission |
| --- | --- | --- | --- |
| `4437283` | short segmentation diagnostic, image encoder only, 512 SA-Co samples, 1 epoch | none | pending |
| `4437284` | full corrected image encoder/projector-only segmentation training, 5 epochs, batch/GPU 1 | afterok `4437283` | pending |
| `4437285` | 5% SA-Co Gold eval for `4437284` | afterok `4437284` | pending |
| `4437286` | full corrected FPN-only segmentation training, 5 epochs, batch/GPU 1 | afterok `4437283` | pending |
| `4437287` | 5% SA-Co Gold eval for `4437286` | afterok `4437286` | pending |
| `4437288` | full corrected image encoder/projector + FPN segmentation training, 5 epochs, batch/GPU 1 | afterok `4437283` | pending |
| `4437289` | 5% SA-Co Gold eval for `4437288` | afterok `4437288` | pending |

The first diagnostic (`4437283`) failed before setup because SLURM exported a non-existent `/local/user/...` `TMPDIR`, and dependent jobs `4437284`-`4437289` were cancelled. The second diagnostic (`4437291`) still inherited the bad node-local temp path inside `srun` tasks, so dependent jobs `4437292`-`4437297` were also cancelled. The run script now forces `TMPDIR`, `TEMP`, and `TMP` to `/tmp` before launching Python.

Active resubmission:

| Job | Purpose | Dependency | Status at submission |
| --- | --- | --- | --- |
| `4437291` | short segmentation diagnostic, image encoder only, 512 SA-Co samples, 1 epoch | none | pending |
| `4437292` | full corrected image encoder/projector-only segmentation training, 5 epochs, batch/GPU 1 | afterok `4437291` | pending |
| `4437293` | 5% SA-Co Gold eval for `4437292` | afterok `4437292` | pending |
| `4437294` | full corrected FPN-only segmentation training, 5 epochs, batch/GPU 1 | afterok `4437291` | pending |
| `4437295` | 5% SA-Co Gold eval for `4437294` | afterok `4437294` | pending |
| `4437296` | full corrected image encoder/projector + FPN segmentation training, 5 epochs, batch/GPU 1 | afterok `4437291` | pending |
| `4437297` | 5% SA-Co Gold eval for `4437296` | afterok `4437296` | pending |

Active third resubmission:

| Job | Purpose | Dependency | Status at submission |
| --- | --- | --- | --- |
| `4437299` | short segmentation diagnostic, image encoder only, 512 SA-Co samples, 1 epoch | none | pending |
| `4437300` | full corrected image encoder/projector-only segmentation training, 5 epochs, batch/GPU 1 | afterok `4437299` | pending |
| `4437301` | 5% SA-Co Gold eval for `4437300` | afterok `4437300` | pending |
| `4437302` | full corrected FPN-only segmentation training, 5 epochs, batch/GPU 1 | afterok `4437299` | pending |
| `4437303` | 5% SA-Co Gold eval for `4437302` | afterok `4437302` | pending |
| `4437304` | full corrected image encoder/projector + FPN segmentation training, 5 epochs, batch/GPU 1 | afterok `4437299` | pending |
| `4437305` | 5% SA-Co Gold eval for `4437304` | afterok `4437304` | pending |

Progress note 2026-05-01 18:39 UTC: diagnostic job `4437299` is running on `nid010957`, has passed model/loss/optimizer setup, and logged the first training iteration (`Train Epoch: [0][0/128]`, `train_all_loss` about `1.86e+02`). This confirms the previous temp-dir failure is fixed and the corrected segmentation training path is executing on GPU.

Progress note 2026-05-01 19:10 UTC: diagnostic job `4437299` completed successfully in `00:06:03` with exit `0:0`. The diagnostic logged mask-loss terms (`loss_mask`, `loss_dice`, and o2m mask/dice losses), confirming segmentation GT is now reaching the loss. Full image-only training job `4437300` is running on `nid010346`; latest log is epoch 0 at about `3180/18652` iterations after about `26` minutes, with average loss about `1.56e+02` and memory peak `39 GB`. Eval job `4437301` is waiting on `4437300`. FPN-only train `4437302` and image+FPN train `4437304` are pending due `QOSMaxJobsPerUserLimit`; their eval jobs remain dependency-pending.

Progress note 2026-05-02 08:45 UTC: image-only segmentation training `4437300` completed in `08:54:52` with exit `0:0`. The final epoch kept mask losses active (`loss_mask`, `loss_dice`, and o2m mask/dice terms) and peaked at about `39 GB`, so batch/GPU `1` remains stable. Image-only eval `4437301` also completed in `00:24:14` with exit `0:0`, but returned zero predictions and `0.0` CGF1 for every subset. The eval log shows a new checkpoint-loading mismatch: the base checkpoint load reported `420` missing and `362` unexpected keys, and the fine-tuned overlay reported `420` missing and `300` unexpected keys, with unexpected TinyViT keys such as `backbone.vision_backbone.trunk.model.backbone...`. This means the corrected training path ran, but the current eval model builder is still not instantiating/loading the same EfficientSAM3 TinyViT/MobileCLIP-S0 architecture used in training. FPN-only training `4437302` is running on `nid010159`; latest log is epoch 2 at about `17140/18652` iterations, elapsed about `04:42`, average loss about `1.42e+02`, and peak memory about `37 GB`. FPN eval `4437303` is dependency-pending. Image+FPN train `4437304` remains pending due `QOSMaxJobsPerUserLimit`; eval `4437305` is dependency-pending.

Fix note 2026-05-02 08:47 UTC: patched `sam3/scripts/eval/gold/eval_efficientsam3_all_subsets.py` to instantiate `build_efficientsam3_image_model` instead of `build_sam3_image_model`. The eval script now takes `--vision-backbone-type` and `--vision-model-name`, and the SLURM wrapper passes `tinyvit`/`11m` plus `MobileCLIP-S0`, matching the Stage 3 training configs. The fine-tuned checkpoint overlay now raises an error if there are any missing or unexpected keys, because a fine-tuned model must have exactly the same structure before and after training. The script compiles cleanly. Submitted fixed image-only eval smoke `4438776` (`MAX_IMAGES_PER_SUBSET=50`, output suffix `_fixed_smoke`) and dependent full image-only re-eval `4438777` (output suffix `_fixed`). Both are pending due `QOSMaxJobsPerUserLimit`. Pending FPN eval `4437303` and image+FPN eval `4437305` will also use the corrected eval script when their dependencies finish.

Progress note 2026-05-02 19:06 UTC: FPN-only training `4437302` completed in `07:58:49` with exit `0:0`, final average loss about `1.43e+02`, active mask/dice losses, and peak memory about `37 GB`. Its dependent eval `4437303` started with the corrected EfficientSAM3 builder, loaded the FPN checkpoint with `missing=0 unexpected=0`, and produced nonzero predictions (`1316` for `metaclip_nps`, `527` for `sa1b_nps` before failure). It failed after about five minutes due a distributed eval bug, not a model-loading bug: all `srun` tasks fell back to rank 0/world size 1 and concurrently wrote/read the same `rank_0.json`, causing `json.decoder.JSONDecodeError: Extra data`. Patched `setup_distributed()` in the eval script to map `SLURM_PROCID`, `SLURM_LOCALID`, and `SLURM_NTASKS` to `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`; syntax/lints pass. Submitted replacement fixed FPN eval smoke `4440914` and dependent full FPN eval `4440915`, both pending due `QOSMaxJobsPerUserLimit`. Image+FPN training `4437304` is running on `nid010431`, currently epoch 3 around `17960/18652`, elapsed about `07:00`, average loss about `1.38e+02`, peak memory about `39 GB`. Fixed image-only eval smoke/full (`4438776`/`4438777`) are still pending and will use the distributed-rank fix when they start.

Progress note 2026-05-02 19:10 UTC: no eval jobs have started yet. `4437304` image encoder/projector + FPN training is still running on `nid010431`, elapsed about `07:03`, at epoch `3` near `18650/18652` with synchronized meters reporting `Trainer/where=0.79999`, average loss `1.38e+02`, active mask/dice loss terms, and peak memory about `39 GB`. Trainer estimates about `01:43` remaining. `4437305` remains dependency-pending on `4437304`; fixed image-only evals `4438776`/`4438777` and fixed FPN evals `4440914`/`4440915` remain pending under `QOSMaxJobsPerUserLimit` or their smoke/full dependencies.

Progress note 2026-05-02 20:18 UTC: `4437304` image encoder/projector + FPN training is still running on `nid010431`, elapsed about `08:11`. It is in final epoch `4`, around `12400/18652` iterations, with average loss about `1.35e+02`, active mask/dice loss terms, and peak memory still about `39 GB`; batch/GPU `1` remains stable. No eval jobs have started yet: `4437305` is dependency-pending on `4437304`, fixed image-only evals `4438776`/`4438777` are pending under QOS/dependency, and fixed FPN evals `4440914`/`4440915` are also pending under QOS/dependency. No new failures since the eval distributed-rank fix.

Progress note 2026-05-03 08:21 UTC: `4437304` image encoder/projector + FPN training completed successfully in `08:44:50` with exit `0:0`. Final average loss was about `1.36e+02`, mask/dice losses were active (`loss_mask`, `loss_dice`, and o2m mask/dice terms), and peak memory remained about `39 GB`, so the batch/GPU `1` setting is stable for all three corrected training lines. The eval jobs that started after training failed before model loading because distributed rendezvous was incomplete: after adding SLURM rank mapping, `torch.distributed` also required `MASTER_ADDR` and `MASTER_PORT`. Failed evals were `4437305` (image+FPN), `4438776` (image-only smoke), and `4440914` (FPN-only smoke); dependent full evals `4438777` and `4440915` were left with unsatisfied dependencies and were cancelled. Patched `setup_distributed()` to set single-node `MASTER_ADDR=127.0.0.1` and a deterministic `MASTER_PORT` from `SLURM_JOB_ID`; syntax/lints pass. Submitted replacement eval chains: image-only smoke/full `4442380`/`4442381`, FPN-only smoke/full `4442382`/`4442383`, and image+FPN smoke/full `4442384`/`4442385`. Smoke jobs are pending with no stated blocker at submission; full evals depend on their corresponding smoke jobs.

Progress note 2026-05-03 08:24 UTC: replacement image-only smoke eval `4442380` completed successfully in `00:01:01` with exit `0:0`. It confirms the full eval stack now works: distributed ranks are distinct, the EfficientSAM3 checkpoint overlay reports `missing=0 unexpected=0`, predictions are nonzero, and the smoke summary was written. Smoke metrics on 50 images/subset: `metaclip_nps` CGF1 `7.36`, `sa1b_nps` `12.15`, `crowded` `6.04`, `fg_food` `0.83`, `fg_sports_equipment` `-0.35`, `attributes` `5.56`, `wiki_common` `0.0`. Full image-only eval `4442381` is now dependency-satisfied but pending under `QOSMaxJobsPerUserLimit`. FPN smoke `4442382` is running on `nid010120`; FPN full `4442383`, image+FPN smoke `4442384`, and image+FPN full `4442385` remain pending.

The full jobs are intentionally held behind the diagnostic so they do not start if mask loading, mask loss, or segmentation-head checkpoint handoff fails.

## Teacher Feature KD Ablation Direction

Goal: test whether frozen SAM3 teacher image features regularize Stage 3 supervised SA-Co Gold fine-tuning. Both ablations keep the current best supervised setup: train image encoder/projector + FPN, freeze text encoder/geometry/transformer/scoring, use SA-Co Gold 95/5 split, and keep box/class/presence + mask/dice ground-truth losses. The teacher checkpoint is `/home/b5cz/simonz.b5cz/program/stage2/efficientsam3/sam3_checkpoints/sam3.pt`.

Implemented changes 2026-05-03:

- Added `stage3.teacher_feature_kd.TeacherFeatureDistillation`.
- Added post-FPN KD config `stage3_mixed_sa1b_only_gold_tvm_mcs0_seg_img_fpn_kd_fpn.yaml`.
- Added pre-FPN trunk KD config `stage3_mixed_sa1b_only_gold_tvm_mcs0_seg_img_fpn_kd_trunk.yaml`.
- Exposed `trunk_features` from the vision neck into `backbone_out` for the pre-FPN ablation.
- KD uses cosine feature loss with weight `0.05`; the teacher is built lazily per GPU and is not saved in the loss state dict.
- Future Stage 3 checkpoints now also preserve base-only non-image-model tensors under `preserved_base_model` so tracker/video keys from the base package can be exported later without breaking current strict image-model loading.

Batch-size/memory policy: use batch/GPU `1` for all KD jobs first. Teacher forward roughly doubles image-encoder memory, so full jobs are gated behind 512-sample diagnostic jobs. If either diagnostic OOMs or is unstable, do not run its dependent full/eval jobs; reduce memory before retrying.

KD ablation jobs submitted 2026-05-03:

| Job | Purpose | Dependency | Status at submission |
| --- | --- | --- | --- |
| `4442458` | post-FPN teacher KD diagnostic, 512 train samples, 1 epoch, batch/GPU 1 | none | pending |
| `4442459` | post-FPN teacher KD full train, 5 epochs, batch/GPU 1 | afterok `4442458` | pending |
| `4442460` | post-FPN teacher KD full eval on 5% SA-Co Gold val | afterok `4442459` | pending |
| `4442461` | pre-FPN trunk teacher KD diagnostic, 512 train samples, 1 epoch, batch/GPU 1 | none | pending |
| `4442462` | pre-FPN trunk teacher KD full train, 5 epochs, batch/GPU 1 | afterok `4442461` | pending |
| `4442463` | pre-FPN trunk teacher KD full eval on 5% SA-Co Gold val | afterok `4442462` | pending |

Progress note 2026-05-03 09:33 UTC: first KD diagnostics failed before a memory result. `4442458` (post-FPN KD) failed after `00:01:46`, and `4442461` (trunk KD) failed after `00:01:22`; both hit `KeyError: 'encoder_hidden_states'` inside `TeacherFeatureDistillation` when the loss wrapper called the KD loss on the one-to-many auxiliary output, which does not carry the full encoder/backbone output. This is a KD loss bookkeeping bug, not an OOM. Patched the KD loss to return a safe zero tensor for auxiliary/o2m calls without feature state; syntax/lints pass. Cancelled dependency-dead jobs `4442459`, `4442460`, `4442462`, `4442463`. Submitted fixed chains: post-FPN KD diagnostic/full/eval `4442504`/`4442505`/`4442506`, and trunk KD diagnostic/full/eval `4442507`/`4442508`/`4442509`. The fixed diagnostics are pending with batch/GPU `1`.

Progress note 2026-05-03 10:48 UTC: fixed post-FPN KD diagnostic `4442504` completed successfully in `00:06:23` with exit `0:0`. It logged active teacher KD terms (`loss_teacher_backbone_fpn`, aux/o2m KD terms zero as intended), active SA-Co Gold mask/dice losses, and peak memory about `42 GB`. The dependent full post-FPN KD training `4442505` is running on `nid011061`, elapsed about `01:07`, at epoch `0` around `9410/18652`, average loss about `1.47e+02`, and peak memory still about `42 GB`. This is higher than non-KD training but stable so far at batch/GPU `1`. Post-FPN KD eval `4442506` is dependency-pending. Trunk KD diagnostic `4442507` has not started yet due `QOSMaxJobsPerUserLimit`; trunk full/eval `4442508`/`4442509` remain dependency-pending.

Progress note 2026-05-03 15:37 UTC: full post-FPN KD training `4442505` is still `RUNNING` on `nid011061`, elapsed about `05:55`. It finished epoch `2` (`18652/18652` steps), synchronized meters for that epoch reported `Losses/train_all_loss` about `134.87`, active mask/dice losses, and `Losses/train_all_loss_teacher_backbone_fpn` about `0.0209` (aux KD still `0` as intended). Training has started epoch `3` (`~100/18652` in the log tail). Peak per-step memory still hits about `42 GB`, so batch/GPU `1` remains the safe setting; do not increase batch size until this run completes without OOM. Post-FPN KD eval `4442506` remains dependency-pending. Trunk KD diagnostic `4442507` is still pending with reason `QOSMaxJobsPerUserLimit` because `4442505` occupies the concurrent-job slot; trunk full/eval `4442508`/`4442509` remain dependency-pending.

Progress note 2026-05-03 18:56 UTC: full post-FPN KD training `4442505` is still `RUNNING` on `nid011061`, elapsed about `09:14`. Latest log is final epoch `4` around `13270/18652` steps, wall time about `09h13m`, running-average `train_all_loss` about `1.35e+02` to `1.36e+02`, and per-step memory still spikes to about `42 GB` (no OOM so far). This is the last training epoch for the `5`-epoch config, so eval `4442506` should start soon after completion. Trunk KD diagnostic `4442507` remains `QOSMaxJobsPerUserLimit`-pending until `4442505` releases the slot; trunk full/eval `4442508`/`4442509` remain dependency-pending.

Progress note 2026-05-03 19:47 UTC: post-FPN KD full training `4442505` completed successfully in `09:47:25` with exit `0:0`. Final epoch `4` synchronized meters reported `Losses/train_all_loss` about `135.95`, `Losses/train_all_loss_teacher_backbone_fpn` about `0.0211`, and active mask/dice losses. Post-FPN KD eval `4442506` completed in `00:03:23` with exit `0:0`; checkpoint overlay logged `missing=0 unexpected=0`. SA-Co Gold val CGF1 summary (subset: CGF1, IL_MCC, pmF1): `metaclip_nps: 4.86,0.21,22.97`; `sa1b_nps: 11.63,0.36,32.35`; `crowded: 1.91,0.2,9.71`; `fg_food: 3.85,0.17,22.09`; `fg_sports_equipment: 5.25,0.21,24.96`; `attributes: 3.05,0.13,22.87`; `wiki_common: 0.15,0.04,3.97`. Results written to `output/efficientsam3_tv_m_mcs0_ctx16_sa1b_only_seg_img_fpn_kd_fpn_gold_val_full2/results_summary.txt`. Trunk KD diagnostic `4442507` completed in `00:06:25` with exit `0:0`; diagnostic meters included `loss_teacher_trunk_features` about `0.0225` on the main pass (aux trunk KD `0` as intended). Trunk full training `4442508` and eval `4442509` completed later the same night; see the `2026-05-04 05:36 UTC` note below for final trunk metrics and val CGF1.

Progress note 2026-05-04 05:36 UTC: full trunk KD training `4442508` completed successfully in `09:55:23` with exit `0:0`. Final epoch `4` synchronized meters reported `Losses/train_all_loss` about `136.05`, `Losses/train_all_loss_teacher_trunk_features` about `0.0233`, and active mask/dice losses. Trunk KD eval `4442509` completed in `00:03:14` with exit `0:0`; checkpoint overlay still `missing=0 unexpected=0`. SA-Co Gold val CGF1 summary (subset: CGF1, IL_MCC, pmF1): `metaclip_nps: 4.96,0.21,23.52`; `sa1b_nps: 11.16,0.35,31.62`; `crowded: 1.71,0.18,9.4`; `fg_food: 4.12,0.19,21.94`; `fg_sports_equipment: 5.23,0.21,24.86`; `attributes: 3.14,0.13,23.84`; `wiki_common: 0.14,0.04,4.0`. Results written to `output/efficientsam3_tv_m_mcs0_ctx16_sa1b_only_seg_img_fpn_kd_trunk_gold_val_full2/results_summary.txt`. The fixed KD ablation chain `4442504`-`4442509` is now fully complete; both KD variants trained five epochs at batch/GPU `1` without OOM.
