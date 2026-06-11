# SAM3.1 Text Encoder Benchmark Results

This file summarizes text-encoder benchmark results in the same style as `BENCHMARK_README.md`, including the latest corrected SA-Co Gold evaluation run.

## Unified Configuration + Average Results (All Current Runs)
Each row below is an averaged result across the 7 SA-Co Gold subsets.
Best row for each (Text Encoder, Context Length) group is marked as `BEST` in the `Status` column.

| Experiment | Backbone | Text Encoder | Context Length | Train Mixture | Consistency Loss | Stability Settings | Avg CG_F1 | Avg IL_MCC | Avg pmF1 | Status |
|-----------|----------|--------------|----------------|---------------|------------------|--------------------|-----------|------------|----------|--------|
| SAM3.1 baseline | SAM3.1 multiplex | CLIP ViT-L/14 | - | Original SAM3.1 | - | - | 53.95 | 0.81 | 66.15 | valid |
| EfficientSAM3 ctx16 | EfficientSAM3 | MobileCLIP-S0 | 16 | 5-dataset | 0.05 (default) | standard | 54.26 | 0.82 | 66.32 | **BEST** |
| EfficientSAM3 ctx16 | EfficientSAM3 | MobileCLIP-S1 | 16 | 5-dataset | 0.05 (default) | standard | 54.27 | 0.82 | 66.39 | **BEST** |
| EfficientSAM3 ctx16 | EfficientSAM3 | MobileCLIP2-L | 16 | 5-dataset | 0.05 (default) | standard | 54.03 | 0.81 | 66.12 | **BEST** |
| EfficientSAM3 ctx32 | EfficientSAM3 | MobileCLIP-S0 | 32 | 5-dataset | 0.05 (default) | standard | 54.18 | 0.82 | 66.25 | valid |
| EfficientSAM3 ctx32 | EfficientSAM3 | MobileCLIP-S1 | 32 | 5-dataset | 0.01 | stable training config | 54.00 | 0.82 | 66.06 | **BEST (tie)** |
| EfficientSAM3 ctx32 | EfficientSAM3 | MobileCLIP2-L | 32 | 5-dataset | 0.05 (default) | standard | 54.17 | 0.82 | 66.25 | **BEST** |
| EfficientSAM3 ctx16 cons01 | EfficientSAM3 | MobileCLIP-S0 | 16 | 5-dataset | 0.10 | standard | 54.05 | 0.82 | 66.10 | valid |
| EfficientSAM3 ctx16 cons01 | EfficientSAM3 | MobileCLIP-S1 | 16 | 5-dataset | 0.10 | standard | 54.19 | 0.82 | 66.27 | valid |
| EfficientSAM3 ctx16 cons01 | EfficientSAM3 | MobileCLIP2-L | 16 | 5-dataset | 0.10 | standard | 53.99 | 0.82 | 66.17 | valid |
| EfficientSAM3 ctx32 cons01 | EfficientSAM3 | MobileCLIP-S0 | 32 | 5-dataset | 0.10 | standard | 54.28 | 0.82 | 66.32 | **BEST** |
| EfficientSAM3 ctx32 cons01 | EfficientSAM3 | MobileCLIP2-L | 32 | 5-dataset | 0.10 | standard | 0.00 | 0.00 | 0.00 | invalid (all-zero eval) |
| EfficientSAM3 ctx32 stable rerun | EfficientSAM3 | MobileCLIP-S1 | 32 | 5-dataset | 0.01 | stable training config | 54.00 | 0.82 | 66.06 | **BEST (tie)** |

## Latest Corrected SA-Co Gold Results (ctx16, 5-dataset)

### EfficientSAM3 (MobileCLIP-S0, ctx16)
| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 47.38 | 0.81 | 58.84 |
| sa1b_nps | 53.35 | 0.85 | 62.49 |
| crowded | 61.29 | 0.90 | 68.08 |
| fg_food | 53.08 | 0.79 | 66.90 |
| fg_sports_equipment | 65.86 | 0.89 | 74.15 |
| attributes | 56.12 | 0.77 | 72.71 |
| wiki_common | 42.71 | 0.70 | 61.07 |
| **Average** | **54.26** | **0.82** | **66.32** |

### EfficientSAM3 (MobileCLIP-S1, ctx16)
| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 47.21 | 0.80 | 58.69 |
| sa1b_nps | 53.56 | 0.86 | 62.57 |
| crowded | 60.77 | 0.90 | 67.75 |
| fg_food | 52.77 | 0.79 | 66.73 |
| fg_sports_equipment | 66.00 | 0.89 | 74.19 |
| attributes | 56.25 | 0.77 | 73.00 |
| wiki_common | 43.31 | 0.70 | 61.79 |
| **Average** | **54.27** | **0.82** | **66.39** |

### EfficientSAM3 (MobileCLIP2-L, ctx16)
| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 47.10 | 0.80 | 58.60 |
| sa1b_nps | 53.37 | 0.85 | 62.46 |
| crowded | 60.94 | 0.90 | 67.82 |
| fg_food | 52.17 | 0.79 | 65.92 |
| fg_sports_equipment | 65.49 | 0.89 | 73.88 |
| attributes | 56.00 | 0.77 | 72.76 |
| wiki_common | 43.16 | 0.70 | 61.42 |
| **Average** | **54.03** | **0.81** | **66.12** |

### Summary Comparison (Latest Corrected ctx16)
| Model | Text Encoder | Avg CG_F1 | Avg IL_MCC | Avg pmF1 | Relative cgF1 |
|-------|--------------|-----------|------------|----------|---------------|
| SAM3 | CLIP ViT-L/14 | **54.04** | **0.82** | **66.11** | 100.0% |
| EfficientSAM3 | MobileCLIP-S1 | **54.27** | **0.82** | **66.39** | **100.4%** |
| EfficientSAM3 | MobileCLIP-S0 | **54.26** | **0.82** | **66.32** | **100.4%** |
| EfficientSAM3 | MobileCLIP2-L | 54.03 | 0.81 | 66.12 | 100.0% |

## Original SAM3.1 Baseline (sam3.1_multiplex.pt)

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 47.02 | 0.80 | 58.57 |
| sa1b_nps | 53.54 | 0.86 | 62.54 |
| crowded | 60.79 | 0.90 | 67.66 |
| fg_food | 53.11 | 0.79 | 67.51 |
| fg_sports_equipment | 66.03 | 0.89 | 74.16 |
| attributes | 55.25 | 0.76 | 72.29 |
| wiki_common | 41.94 | 0.70 | 60.33 |
| **Average** | **53.95** | **0.81** | **66.15** |

## Latest SA-Co Gold Results (ctx32, 5-dataset)

### EfficientSAM3 (MobileCLIP-S0, ctx32)
| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 47.41 | 0.81 | 58.79 |
| sa1b_nps | 53.56 | 0.86 | 62.48 |
| crowded | 60.90 | 0.90 | 67.73 |
| fg_food | 52.67 | 0.79 | 66.68 |
| fg_sports_equipment | 66.03 | 0.89 | 74.41 |
| attributes | 56.18 | 0.77 | 73.02 |
| wiki_common | 42.48 | 0.70 | 60.65 |
| **Average** | **54.18** | **0.82** | **66.25** |

### EfficientSAM3 (MobileCLIP-S1, ctx32)
| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 47.16 | 0.80 | 58.71 |
| sa1b_nps | 53.59 | 0.86 | 62.55 |
| crowded | 60.83 | 0.90 | 67.78 |
| fg_food | 52.01 | 0.79 | 65.81 |
| fg_sports_equipment | 65.54 | 0.89 | 73.96 |
| attributes | 56.02 | 0.77 | 72.70 |
| wiki_common | 42.83 | 0.70 | 60.88 |
| **Average** | **54.00** | **0.82** | **66.06** |

### EfficientSAM3 (MobileCLIP2-L, ctx32)
| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 47.22 | 0.80 | 58.75 |
| sa1b_nps | 53.51 | 0.86 | 62.55 |
| crowded | 60.78 | 0.90 | 67.65 |
| fg_food | 52.21 | 0.79 | 66.12 |
| fg_sports_equipment | 66.35 | 0.89 | 74.39 |
| attributes | 55.84 | 0.77 | 72.80 |
| wiki_common | 43.27 | 0.70 | 61.52 |
| **Average** | **54.17** | **0.82** | **66.25** |

### Summary Comparison (Latest ctx32)
| Model | Text Encoder | Avg CG_F1 | Avg IL_MCC | Avg pmF1 |
|-------|--------------|-----------|------------|----------|
| EfficientSAM3 | MobileCLIP-S0 | **54.18** | **0.82** | **66.25** |
| EfficientSAM3 | MobileCLIP-S1 | 54.00 | 0.82 | 66.06 |
| EfficientSAM3 | MobileCLIP2-L | 54.17 | 0.82 | 66.25 |

## SA-Co Gold Results (CONSISTENCY_LOSS=0.1 + Stable Rerun)

### Summary Comparison (Average Rows Only)
| Model | Text Encoder | Context Length | Config | Avg CG_F1 | Avg IL_MCC | Avg pmF1 | Note |
|-------|--------------|----------------|--------|-----------|------------|----------|------|
| EfficientSAM3 | MobileCLIP-S0 | 16 | `CONSISTENCY_LOSS=0.1` | 54.05 | 0.82 | 66.10 | valid |
| EfficientSAM3 | MobileCLIP-S1 | 16 | `CONSISTENCY_LOSS=0.1` | 54.19 | 0.82 | 66.27 | valid |
| EfficientSAM3 | MobileCLIP2-L | 16 | `CONSISTENCY_LOSS=0.1` | 53.99 | 0.82 | 66.17 | valid |
| EfficientSAM3 | MobileCLIP-S0 | 32 | `CONSISTENCY_LOSS=0.1` | **54.28** | **0.82** | **66.32** | valid |
| EfficientSAM3 | MobileCLIP2-L | 32 | `CONSISTENCY_LOSS=0.1` | 0.00 | 0.00 | 0.00 | invalid (all-zero eval) |
| EfficientSAM3 | MobileCLIP-S1 | 32 | stable rerun (`CONSISTENCY_LOSS=0.01`) | 54.00 | 0.82 | 66.06 | valid |

---

## Historical Results (from BENCHMARK_README)
These are previous tracked results kept for comparison.

### SAM3 (CLIP ViT-L/14 Text Encoder)
| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 47.27 | 0.81 | 58.61 |
| sa1b_nps | 53.72 | 0.86 | 62.57 |
| crowded | 61.07 | 0.90 | 67.75 |
| fg_food | 53.39 | 0.79 | 67.29 |
| fg_sports_equipment | 65.53 | 0.89 | 73.75 |
| attributes | 54.96 | 0.76 | 72.04 |
| wiki_common | 42.29 | 0.70 | 60.78 |
| **Average** | **54.04** | **0.82** | **66.11** |

### EfficientSAM3 (Context Length 16, prior reported)
| Model | Avg CG_F1 | Avg IL_MCC | Avg pmF1 |
|-------|-----------|------------|----------|
| MobileCLIP-S0 | 54.30 | 0.81 | 66.86 |
| MobileCLIP-S1 | 54.14 | 0.81 | 66.20 |
| MobileCLIP2-L | 53.95 | 0.81 | 65.87 |

### EfficientSAM3 (Context Length 32)
| Model | Avg CG_F1 | Avg IL_MCC | Avg pmF1 |
|-------|-----------|------------|----------|
| MobileCLIP-S0 | 54.18 | 0.82 | 66.25 |
| MobileCLIP-S1 | 54.00 | 0.82 | 66.06 |
| MobileCLIP2-L | 54.17 | 0.82 | 66.25 |

### EfficientSAM3 (3-dataset mixture)
| Model | Avg CG_F1 | Avg IL_MCC | Avg pmF1 |
|-------|-----------|------------|----------|
| MobileCLIP-S0 | 38.24 | 0.65 | 57.45 |
| MobileCLIP-S1 | 38.58 | 0.65 | 57.73 |
| MobileCLIP-S2 | 38.58 | 0.65 | 57.73 |

---

## COCO Validation (Image Encoder Stage-1, Original SAM3.1 Text Path)

### Models Used

| Variant | Backbone | Size | Checkpoint Source |
|---------|----------|------|-------------------|
| rv_s | repvit | s | stage1 image-only merged checkpoint |
| rv_m | repvit | m | stage1 image-only merged checkpoint |
| rv_l | repvit | l | stage1 image-only merged checkpoint (epoch41 rerun path) |
| tv_s | tinyvit | s | stage1 image-only merged checkpoint |
| tv_m | tinyvit | m | stage1 image-only merged checkpoint |
| tv_l | tinyvit | l | stage1 image-only merged checkpoint |
| ev_s | efficientvit | s | stage1 image-only merged checkpoint |
| ev_m | efficientvit | m | stage1 image-only merged checkpoint |
| ev_l | efficientvit | l | stage1 image-only merged checkpoint |

### Results: Image Encoder Stage-1 + Original SAM3.1 Text Path

| Model | Backbone | COCO Job | mIoU | Time (s) | Note |
|-------|----------|----------|------|----------|------|
| rv_s | RepViT-S | 3717494 | 0.6731 | 450.64 | completed |
| rv_m | RepViT-M | 3717498 | 0.6782 | 412.52 | completed |
| rv_l | RepViT-L | 3760059 | 0.6927 | 579.84 | completed via e41 rerun path |
| tv_s | TinyViT-S | 3717506 | 0.6825 | 464.91 | completed |
| tv_m | TinyViT-M | 3660911 | 0.6906 | 450.93 | completed |
| tv_l | TinyViT-L | 3717510 | 0.6967 | 393.54 | completed |
| ev_s | EfficientViT-S | 3717514 | 0.6476 | 433.84 | completed |
| ev_m | EfficientViT-M | 3717518 | 0.6721 | 414.70 | completed |
| ev_l | EfficientViT-L | 3717522 | 0.6889 | 487.47 | completed |

### Models Used (Both Student Encoders Merge)

All rows below are produced by `convert_both_encoders_weights_stage1.py` using MobileCLIP-S0 ctx16 text encoder + each image encoder variant:

| Backbone | Size | Checkpoint |
|----------|------|------------|
| repvit | s | output/efficient_sam3_repvit_s_mobileclip_s0_ctx16_5dataset.pt |
| repvit | m | output/efficient_sam3_repvit_m_mobileclip_s0_ctx16_5dataset.pt |
| repvit | l | output/efficient_sam3_repvit_l_mobileclip_s0_ctx16_5dataset.pt |
| tinyvit | s | output/efficient_sam3_tinyvit_s_mobileclip_s0_ctx16_5dataset.pt |
| tinyvit | m | output/efficient_sam3_tinyvit_m_mobileclip_s0_ctx16_5dataset.pt |
| tinyvit | l | output/efficient_sam3_tinyvit_l_mobileclip_s0_ctx16_5dataset.pt |
| efficientvit | s | output/efficient_sam3_efficientvit_s_mobileclip_s0_ctx16_5dataset.pt |
| efficientvit | m | output/efficient_sam3_efficientvit_m_mobileclip_s0_ctx16_5dataset.pt |
| efficientvit | l | output/efficient_sam3_efficientvit_l_mobileclip_s0_ctx16_5dataset.pt |

### Results: Both Student Text + Image Encoders (MobileCLIP-S0 ctx16)

| Model | Backbone | COCO Job | mIoU | Time (s) | Comparison vs image-only |
|-------|----------|----------|------|----------|--------------------------|
| rv_s both s0 ctx16 | RepViT-S | 3763474 | 0.6731 | 511.10 | matches rv_s |
| rv_m both s0 ctx16 | RepViT-M | 3763473 | 0.6782 | 483.94 | matches rv_m |
| rv_l both s0 ctx16 | RepViT-L | 3763472 | 0.6927 | 588.56 | matches rv_l |
| tv_s both s0 ctx16 | TinyViT-S | 3763477 | 0.6825 | 450.82 | matches tv_s |
| tv_m both s0 ctx16 | TinyViT-M | 3763476 | 0.6906 | 434.13 | matches tv_m |
| tv_l both s0 ctx16 | TinyViT-L | 3763475 | 0.6967 | 452.12 | matches tv_l |
| ev_s both s0 ctx16 | EfficientViT-S | 3763471 | 0.6476 | 439.01 | matches ev_s |
| ev_m both s0 ctx16 | EfficientViT-M | 3763470 | 0.6721 | 464.81 | matches ev_m |
| ev_l both s0 ctx16 | EfficientViT-L | 3763469 | 0.6889 | 467.30 | matches ev_l |

---

## Notes
1. The latest corrected run fixed evaluator failures caused by array/list empty-check behavior in `sam3/sam3/eval/cgf1_eval.py`.
2. Context-length aligned inference settings (`--context-length 16 --pos-embed-table-size 16`) are required for fair ctx16 comparison.
3. For full ablation tables (36 experiments), refer to `BENCHMARK_README.md`.

