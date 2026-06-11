
---

## Text Encoder Positional Embedding Ablation (36 Experiments)

These results compare `interp` vs `fixed` positional-embedding strategies during training, and `slice` vs `interp` positional handling during evaluation on SA-Co Gold.

### Summary Comparison (All 36 Experiments)

| Experiment | Text Encoder | Train Ctx | Train Strategy | Eval Strategy | Avg CG_F1 | Avg IL_MCC | Avg pmF1 |
|------------|--------------|-----------|----------------|---------------|-----------|------------|----------|
| `s0_ctx16_interp @ ctx16_slice` | MobileCLIP-S0 | 16 | interp (77-table) | ctx16 slice | 22.44 | 0.52 | 41.78 |
| `s0_ctx16_interp @ ctx16_interp` | MobileCLIP-S0 | 16 | interp (77-table) | ctx16 interp | 52.39 | 0.82 | 63.87 |
| `s0_ctx16_fixed @ ctx16_slice` | MobileCLIP-S0 | 16 | fixed (matching-table) | ctx16 slice | 52.39 | 0.82 | 63.77 |
| `s0_ctx16_fixed @ ctx16_interp` | MobileCLIP-S0 | 16 | fixed (matching-table) | ctx16 interp | 52.39 | 0.82 | 63.77 |
| `s1_ctx16_interp @ ctx16_slice` | MobileCLIP-S1 | 16 | interp (77-table) | ctx16 slice | 5.36 | 0.24 | 19.63 |
| `s1_ctx16_interp @ ctx16_interp` | MobileCLIP-S1 | 16 | interp (77-table) | ctx16 interp | 52.41 | 0.82 | 63.84 |
| `s1_ctx16_fixed @ ctx16_slice` | MobileCLIP-S1 | 16 | fixed (matching-table) | ctx16 slice | 52.18 | 0.82 | 63.64 |
| `s1_ctx16_fixed @ ctx16_interp` | MobileCLIP-S1 | 16 | fixed (matching-table) | ctx16 interp | 52.18 | 0.82 | 63.64 |
| `l_ctx16_interp @ ctx16_slice` | MobileCLIP2-L | 16 | interp (77-table) | ctx16 slice | 37.49 | 0.69 | 53.97 |
| `l_ctx16_interp @ ctx16_interp` | MobileCLIP2-L | 16 | interp (77-table) | ctx16 interp | 52.18 | 0.82 | 63.59 |
| `l_ctx16_fixed @ ctx16_slice` | MobileCLIP2-L | 16 | fixed (matching-table) | ctx16 slice | 52.18 | 0.82 | 63.60 |
| `l_ctx16_fixed @ ctx16_interp` | MobileCLIP2-L | 16 | fixed (matching-table) | ctx16 interp | 52.18 | 0.82 | 63.60 |
| `s0_ctx32_interp @ ctx16_slice` | MobileCLIP-S0 | 32 | interp (77-table) | ctx16 slice | 47.88 | 0.78 | 61.27 |
| `s0_ctx32_interp @ ctx16_interp` | MobileCLIP-S0 | 32 | interp (77-table) | ctx16 interp | 50.70 | 0.80 | 63.06 |
| `s0_ctx32_interp @ ctx32_slice` | MobileCLIP-S0 | 32 | interp (77-table) | ctx32 slice | 47.73 | 0.78 | 61.17 |
| `s0_ctx32_interp @ ctx32_interp` | MobileCLIP-S0 | 32 | interp (77-table) | ctx32 interp | 52.31 | 0.82 | 63.72 |
| `s0_ctx32_fixed @ ctx16_slice` | MobileCLIP-S0 | 32 | fixed (matching-table) | ctx16 slice | 50.59 | 0.80 | 62.83 |
| `s0_ctx32_fixed @ ctx16_interp` | MobileCLIP-S0 | 32 | fixed (matching-table) | ctx16 interp | 51.63 | 0.81 | 63.59 |
| `s0_ctx32_fixed @ ctx32_slice` | MobileCLIP-S0 | 32 | fixed (matching-table) | ctx32 slice | 52.40 | 0.82 | 63.90 |
| `s0_ctx32_fixed @ ctx32_interp` | MobileCLIP-S0 | 32 | fixed (matching-table) | ctx32 interp | 52.40 | 0.82 | 63.90 |
| `s1_ctx32_interp @ ctx16_slice` | MobileCLIP-S1 | 32 | interp (77-table) | ctx16 slice | 18.61 | 0.46 | 38.37 |
| `s1_ctx32_interp @ ctx16_interp` | MobileCLIP-S1 | 32 | interp (77-table) | ctx16 interp | 44.33 | 0.75 | 59.00 |
| `s1_ctx32_interp @ ctx32_slice` | MobileCLIP-S1 | 32 | interp (77-table) | ctx32 slice | 18.71 | 0.47 | 38.22 |
| `s1_ctx32_interp @ ctx32_interp` | MobileCLIP-S1 | 32 | interp (77-table) | ctx32 interp | 52.50 | 0.82 | 63.93 |
| `s1_ctx32_fixed @ ctx16_slice` | MobileCLIP-S1 | 32 | fixed (matching-table) | ctx16 slice | 51.07 | 0.81 | 63.03 |
| `s1_ctx32_fixed @ ctx16_interp` | MobileCLIP-S1 | 32 | fixed (matching-table) | ctx16 interp | 46.45 | 0.76 | 61.04 |
| `s1_ctx32_fixed @ ctx32_slice` | MobileCLIP-S1 | 32 | fixed (matching-table) | ctx32 slice | 52.15 | 0.82 | 63.57 |
| `s1_ctx32_fixed @ ctx32_interp` | MobileCLIP-S1 | 32 | fixed (matching-table) | ctx32 interp | 52.15 | 0.82 | 63.57 |
| `l_ctx32_interp @ ctx16_slice` | MobileCLIP2-L | 32 | interp (77-table) | ctx16 slice | 28.27 | 0.59 | 46.59 |
| `l_ctx32_interp @ ctx16_interp` | MobileCLIP2-L | 32 | interp (77-table) | ctx16 interp | 44.71 | 0.75 | 59.48 |
| `l_ctx32_interp @ ctx32_slice` | MobileCLIP2-L | 32 | interp (77-table) | ctx32 slice | 29.15 | 0.60 | 47.55 |
| `l_ctx32_interp @ ctx32_interp` | MobileCLIP2-L | 32 | interp (77-table) | ctx32 interp | 52.06 | 0.82 | 63.61 |
| `l_ctx32_fixed @ ctx16_slice` | MobileCLIP2-L | 32 | fixed (matching-table) | ctx16 slice | 50.15 | 0.80 | 62.54 |
| `l_ctx32_fixed @ ctx16_interp` | MobileCLIP2-L | 32 | fixed (matching-table) | ctx16 interp | 45.10 | 0.75 | 59.90 |
| `l_ctx32_fixed @ ctx32_slice` | MobileCLIP2-L | 32 | fixed (matching-table) | ctx32 slice | 52.15 | 0.82 | 63.67 |
| `l_ctx32_fixed @ ctx32_interp` | MobileCLIP2-L | 32 | fixed (matching-table) | ctx32 interp | 52.15 | 0.82 | 63.67 |

### Detailed Per-Experiment Results

#### `s0_ctx16_interp @ ctx16_slice`

- Text Encoder: `MobileCLIP-S0`
- Train Context: `16`
- Train Strategy: `interp (77-table)`
- Eval Strategy: `ctx16 slice`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 21.46 | 0.55 | 38.69 |
| sa1b_nps | 28.02 | 0.61 | 46.03 |
| crowded | 27.61 | 0.60 | 45.75 |
| fg_food | 13.97 | 0.43 | 32.70 |
| fg_sports_equipment | 23.28 | 0.53 | 43.88 |
| attributes | 31.93 | 0.59 | 53.93 |
| wiki_common | 10.78 | 0.34 | 31.51 |
| **Average** | **22.44** | **0.52** | **41.78** |

#### `s0_ctx16_interp @ ctx16_interp`

- Text Encoder: `MobileCLIP-S0`
- Train Context: `16`
- Train Strategy: `interp (77-table)`
- Eval Strategy: `ctx16 interp`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 45.89 | 0.81 | 56.98 |
| sa1b_nps | 53.06 | 0.86 | 61.88 |
| crowded | 60.33 | 0.90 | 66.84 |
| fg_food | 51.36 | 0.79 | 64.68 |
| fg_sports_equipment | 64.03 | 0.89 | 72.07 |
| attributes | 51.11 | 0.77 | 66.50 |
| wiki_common | 40.96 | 0.70 | 58.12 |
| **Average** | **52.39** | **0.82** | **63.87** |

#### `s0_ctx16_fixed @ ctx16_slice`

- Text Encoder: `MobileCLIP-S0`
- Train Context: `16`
- Train Strategy: `fixed (matching-table)`
- Eval Strategy: `ctx16 slice`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 46.40 | 0.81 | 57.35 |
| sa1b_nps | 53.13 | 0.86 | 61.91 |
| crowded | 60.32 | 0.90 | 66.83 |
| fg_food | 51.76 | 0.80 | 65.09 |
| fg_sports_equipment | 64.02 | 0.89 | 71.88 |
| attributes | 50.25 | 0.77 | 65.46 |
| wiki_common | 40.88 | 0.71 | 57.86 |
| **Average** | **52.39** | **0.82** | **63.77** |

#### `s0_ctx16_fixed @ ctx16_interp`

- Text Encoder: `MobileCLIP-S0`
- Train Context: `16`
- Train Strategy: `fixed (matching-table)`
- Eval Strategy: `ctx16 interp`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 46.40 | 0.81 | 57.35 |
| sa1b_nps | 53.12 | 0.86 | 61.91 |
| crowded | 60.32 | 0.90 | 66.83 |
| fg_food | 51.76 | 0.80 | 65.09 |
| fg_sports_equipment | 64.02 | 0.89 | 71.88 |
| attributes | 50.25 | 0.77 | 65.46 |
| wiki_common | 40.88 | 0.71 | 57.86 |
| **Average** | **52.39** | **0.82** | **63.77** |

#### `s1_ctx16_interp @ ctx16_slice`

- Text Encoder: `MobileCLIP-S1`
- Train Context: `16`
- Train Strategy: `interp (77-table)`
- Eval Strategy: `ctx16 slice`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 6.42 | 0.30 | 21.41 |
| sa1b_nps | 9.52 | 0.36 | 26.34 |
| crowded | 7.70 | 0.31 | 24.93 |
| fg_food | 1.65 | 0.15 | 10.84 |
| fg_sports_equipment | 3.74 | 0.19 | 19.78 |
| attributes | 8.06 | 0.30 | 26.60 |
| wiki_common | 0.42 | 0.06 | 7.54 |
| **Average** | **5.36** | **0.24** | **19.63** |

#### `s1_ctx16_interp @ ctx16_interp`

- Text Encoder: `MobileCLIP-S1`
- Train Context: `16`
- Train Strategy: `interp (77-table)`
- Eval Strategy: `ctx16 interp`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 46.12 | 0.81 | 57.07 |
| sa1b_nps | 53.33 | 0.86 | 61.99 |
| crowded | 59.97 | 0.90 | 66.55 |
| fg_food | 50.80 | 0.79 | 64.22 |
| fg_sports_equipment | 64.41 | 0.89 | 72.05 |
| attributes | 50.85 | 0.77 | 66.41 |
| wiki_common | 41.42 | 0.71 | 58.56 |
| **Average** | **52.41** | **0.82** | **63.84** |

#### `s1_ctx16_fixed @ ctx16_slice`

- Text Encoder: `MobileCLIP-S1`
- Train Context: `16`
- Train Strategy: `fixed (matching-table)`
- Eval Strategy: `ctx16 slice`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 45.93 | 0.81 | 56.93 |
| sa1b_nps | 53.15 | 0.86 | 61.87 |
| crowded | 59.78 | 0.90 | 66.39 |
| fg_food | 50.45 | 0.79 | 63.87 |
| fg_sports_equipment | 64.16 | 0.89 | 72.02 |
| attributes | 50.86 | 0.77 | 66.20 |
| wiki_common | 40.90 | 0.70 | 58.17 |
| **Average** | **52.18** | **0.82** | **63.64** |

#### `s1_ctx16_fixed @ ctx16_interp`

- Text Encoder: `MobileCLIP-S1`
- Train Context: `16`
- Train Strategy: `fixed (matching-table)`
- Eval Strategy: `ctx16 interp`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 45.93 | 0.81 | 56.93 |
| sa1b_nps | 53.15 | 0.86 | 61.87 |
| crowded | 59.78 | 0.90 | 66.39 |
| fg_food | 50.45 | 0.79 | 63.87 |
| fg_sports_equipment | 64.16 | 0.89 | 72.02 |
| attributes | 50.86 | 0.77 | 66.20 |
| wiki_common | 40.90 | 0.70 | 58.17 |
| **Average** | **52.18** | **0.82** | **63.64** |

#### `l_ctx16_interp @ ctx16_slice`

- Text Encoder: `MobileCLIP2-L`
- Train Context: `16`
- Train Strategy: `interp (77-table)`
- Eval Strategy: `ctx16 slice`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 34.37 | 0.70 | 49.13 |
| sa1b_nps | 42.27 | 0.75 | 56.49 |
| crowded | 45.57 | 0.77 | 59.01 |
| fg_food | 34.17 | 0.67 | 51.37 |
| fg_sports_equipment | 43.59 | 0.73 | 60.08 |
| attributes | 41.03 | 0.70 | 58.72 |
| wiki_common | 21.46 | 0.50 | 43.01 |
| **Average** | **37.49** | **0.69** | **53.97** |

#### `l_ctx16_interp @ ctx16_interp`

- Text Encoder: `MobileCLIP2-L`
- Train Context: `16`
- Train Strategy: `interp (77-table)`
- Eval Strategy: `ctx16 interp`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 46.18 | 0.81 | 57.11 |
| sa1b_nps | 53.29 | 0.86 | 61.93 |
| crowded | 59.85 | 0.90 | 66.44 |
| fg_food | 50.85 | 0.79 | 64.15 |
| fg_sports_equipment | 64.07 | 0.89 | 71.81 |
| attributes | 50.61 | 0.77 | 66.07 |
| wiki_common | 40.39 | 0.70 | 57.62 |
| **Average** | **52.18** | **0.82** | **63.59** |

#### `l_ctx16_fixed @ ctx16_slice`

- Text Encoder: `MobileCLIP2-L`
- Train Context: `16`
- Train Strategy: `fixed (matching-table)`
- Eval Strategy: `ctx16 slice`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 46.02 | 0.81 | 56.98 |
| sa1b_nps | 53.07 | 0.86 | 61.83 |
| crowded | 59.89 | 0.90 | 66.46 |
| fg_food | 51.35 | 0.80 | 64.46 |
| fg_sports_equipment | 63.55 | 0.89 | 71.62 |
| attributes | 50.74 | 0.77 | 66.05 |
| wiki_common | 40.62 | 0.70 | 57.81 |
| **Average** | **52.18** | **0.82** | **63.60** |

#### `l_ctx16_fixed @ ctx16_interp`

- Text Encoder: `MobileCLIP2-L`
- Train Context: `16`
- Train Strategy: `fixed (matching-table)`
- Eval Strategy: `ctx16 interp`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 46.02 | 0.81 | 56.98 |
| sa1b_nps | 53.07 | 0.86 | 61.83 |
| crowded | 59.89 | 0.90 | 66.46 |
| fg_food | 51.35 | 0.80 | 64.46 |
| fg_sports_equipment | 63.55 | 0.89 | 71.62 |
| attributes | 50.74 | 0.77 | 66.05 |
| wiki_common | 40.62 | 0.70 | 57.81 |
| **Average** | **52.18** | **0.82** | **63.60** |

#### `s0_ctx32_interp @ ctx16_slice`

- Text Encoder: `MobileCLIP-S0`
- Train Context: `32`
- Train Strategy: `interp (77-table)`
- Eval Strategy: `ctx16 slice`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 44.00 | 0.78 | 56.20 |
| sa1b_nps | 49.96 | 0.83 | 60.43 |
| crowded | 55.79 | 0.86 | 64.62 |
| fg_food | 46.85 | 0.77 | 61.22 |
| fg_sports_equipment | 54.08 | 0.82 | 65.99 |
| attributes | 51.22 | 0.77 | 66.73 |
| wiki_common | 33.25 | 0.62 | 53.70 |
| **Average** | **47.88** | **0.78** | **61.27** |

#### `s0_ctx32_interp @ ctx16_interp`

- Text Encoder: `MobileCLIP-S0`
- Train Context: `32`
- Train Strategy: `interp (77-table)`
- Eval Strategy: `ctx16 interp`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 44.98 | 0.80 | 56.31 |
| sa1b_nps | 52.69 | 0.85 | 62.12 |
| crowded | 58.30 | 0.89 | 65.59 |
| fg_food | 49.38 | 0.78 | 62.91 |
| fg_sports_equipment | 59.70 | 0.85 | 70.15 |
| attributes | 51.28 | 0.76 | 67.23 |
| wiki_common | 38.60 | 0.68 | 57.10 |
| **Average** | **50.70** | **0.80** | **63.06** |

#### `s0_ctx32_interp @ ctx32_slice`

- Text Encoder: `MobileCLIP-S0`
- Train Context: `32`
- Train Strategy: `interp (77-table)`
- Eval Strategy: `ctx32 slice`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 43.90 | 0.78 | 56.27 |
| sa1b_nps | 50.59 | 0.83 | 60.89 |
| crowded | 55.64 | 0.86 | 64.44 |
| fg_food | 47.21 | 0.77 | 61.35 |
| fg_sports_equipment | 53.91 | 0.82 | 66.03 |
| attributes | 50.96 | 0.77 | 66.54 |
| wiki_common | 31.91 | 0.61 | 52.69 |
| **Average** | **47.73** | **0.78** | **61.17** |

#### `s0_ctx32_interp @ ctx32_interp`

- Text Encoder: `MobileCLIP-S0`
- Train Context: `32`
- Train Strategy: `interp (77-table)`
- Eval Strategy: `ctx32 interp`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 46.23 | 0.81 | 56.97 |
| sa1b_nps | 53.12 | 0.86 | 61.85 |
| crowded | 60.00 | 0.90 | 66.44 |
| fg_food | 51.01 | 0.79 | 64.33 |
| fg_sports_equipment | 64.22 | 0.89 | 72.28 |
| attributes | 51.01 | 0.77 | 66.32 |
| wiki_common | 40.60 | 0.70 | 57.82 |
| **Average** | **52.31** | **0.82** | **63.72** |

#### `s0_ctx32_fixed @ ctx16_slice`

- Text Encoder: `MobileCLIP-S0`
- Train Context: `32`
- Train Strategy: `fixed (matching-table)`
- Eval Strategy: `ctx16 slice`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 45.17 | 0.80 | 56.51 |
| sa1b_nps | 52.18 | 0.84 | 61.79 |
| crowded | 58.96 | 0.89 | 66.19 |
| fg_food | 50.06 | 0.78 | 64.31 |
| fg_sports_equipment | 63.41 | 0.88 | 72.42 |
| attributes | 46.98 | 0.74 | 63.46 |
| wiki_common | 37.36 | 0.68 | 55.16 |
| **Average** | **50.59** | **0.80** | **62.83** |

#### `s0_ctx32_fixed @ ctx16_interp`

- Text Encoder: `MobileCLIP-S0`
- Train Context: `32`
- Train Strategy: `fixed (matching-table)`
- Eval Strategy: `ctx16 interp`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 45.49 | 0.80 | 56.56 |
| sa1b_nps | 52.46 | 0.85 | 61.76 |
| crowded | 59.31 | 0.89 | 66.39 |
| fg_food | 49.85 | 0.78 | 63.98 |
| fg_sports_equipment | 63.20 | 0.88 | 72.15 |
| attributes | 51.84 | 0.77 | 67.06 |
| wiki_common | 39.29 | 0.69 | 57.25 |
| **Average** | **51.63** | **0.81** | **63.59** |

#### `s0_ctx32_fixed @ ctx32_slice`

- Text Encoder: `MobileCLIP-S0`
- Train Context: `32`
- Train Strategy: `fixed (matching-table)`
- Eval Strategy: `ctx32 slice`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 46.25 | 0.81 | 57.17 |
| sa1b_nps | 53.07 | 0.86 | 61.83 |
| crowded | 59.77 | 0.90 | 66.52 |
| fg_food | 51.49 | 0.79 | 64.81 |
| fg_sports_equipment | 64.33 | 0.89 | 72.29 |
| attributes | 50.81 | 0.77 | 66.27 |
| wiki_common | 41.10 | 0.70 | 58.39 |
| **Average** | **52.40** | **0.82** | **63.90** |

#### `s0_ctx32_fixed @ ctx32_interp`

- Text Encoder: `MobileCLIP-S0`
- Train Context: `32`
- Train Strategy: `fixed (matching-table)`
- Eval Strategy: `ctx32 interp`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 46.25 | 0.81 | 57.17 |
| sa1b_nps | 53.07 | 0.86 | 61.83 |
| crowded | 59.77 | 0.90 | 66.52 |
| fg_food | 51.49 | 0.79 | 64.81 |
| fg_sports_equipment | 64.33 | 0.89 | 72.29 |
| attributes | 50.81 | 0.77 | 66.27 |
| wiki_common | 41.10 | 0.70 | 58.39 |
| **Average** | **52.40** | **0.82** | **63.90** |

#### `s1_ctx32_interp @ ctx16_slice`

- Text Encoder: `MobileCLIP-S1`
- Train Context: `32`
- Train Strategy: `interp (77-table)`
- Eval Strategy: `ctx16 slice`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 20.23 | 0.54 | 37.77 |
| sa1b_nps | 24.97 | 0.56 | 44.47 |
| crowded | 23.97 | 0.56 | 43.00 |
| fg_food | 9.94 | 0.36 | 27.69 |
| fg_sports_equipment | 14.71 | 0.40 | 37.08 |
| attributes | 29.81 | 0.58 | 51.38 |
| wiki_common | 6.63 | 0.24 | 27.23 |
| **Average** | **18.61** | **0.46** | **38.37** |

#### `s1_ctx32_interp @ ctx16_interp`

- Text Encoder: `MobileCLIP-S1`
- Train Context: `32`
- Train Strategy: `interp (77-table)`
- Eval Strategy: `ctx16 interp`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 40.01 | 0.74 | 53.71 |
| sa1b_nps | 47.60 | 0.79 | 60.50 |
| crowded | 51.05 | 0.82 | 62.46 |
| fg_food | 42.73 | 0.74 | 57.59 |
| fg_sports_equipment | 54.53 | 0.82 | 66.64 |
| attributes | 45.88 | 0.73 | 63.25 |
| wiki_common | 28.52 | 0.58 | 48.87 |
| **Average** | **44.33** | **0.75** | **59.00** |

#### `s1_ctx32_interp @ ctx32_slice`

- Text Encoder: `MobileCLIP-S1`
- Train Context: `32`
- Train Strategy: `interp (77-table)`
- Eval Strategy: `ctx32 slice`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 19.72 | 0.53 | 37.12 |
| sa1b_nps | 24.26 | 0.57 | 42.84 |
| crowded | 25.12 | 0.57 | 44.22 |
| fg_food | 10.56 | 0.37 | 28.40 |
| fg_sports_equipment | 14.09 | 0.39 | 36.00 |
| attributes | 30.23 | 0.58 | 51.89 |
| wiki_common | 6.97 | 0.26 | 27.06 |
| **Average** | **18.71** | **0.47** | **38.22** |

#### `s1_ctx32_interp @ ctx32_interp`

- Text Encoder: `MobileCLIP-S1`
- Train Context: `32`
- Train Strategy: `interp (77-table)`
- Eval Strategy: `ctx32 interp`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 45.96 | 0.81 | 56.95 |
| sa1b_nps | 53.34 | 0.86 | 62.00 |
| crowded | 59.81 | 0.90 | 66.47 |
| fg_food | 51.71 | 0.80 | 64.74 |
| fg_sports_equipment | 64.34 | 0.89 | 72.32 |
| attributes | 50.90 | 0.77 | 66.21 |
| wiki_common | 41.44 | 0.70 | 58.81 |
| **Average** | **52.50** | **0.82** | **63.93** |

#### `s1_ctx32_fixed @ ctx16_slice`

- Text Encoder: `MobileCLIP-S1`
- Train Context: `32`
- Train Strategy: `fixed (matching-table)`
- Eval Strategy: `ctx16 slice`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 45.57 | 0.80 | 56.75 |
| sa1b_nps | 52.97 | 0.86 | 61.82 |
| crowded | 59.32 | 0.90 | 66.18 |
| fg_food | 48.56 | 0.77 | 62.73 |
| fg_sports_equipment | 63.55 | 0.89 | 71.67 |
| attributes | 47.80 | 0.74 | 64.65 |
| wiki_common | 39.71 | 0.69 | 57.42 |
| **Average** | **51.07** | **0.81** | **63.03** |

#### `s1_ctx32_fixed @ ctx16_interp`

- Text Encoder: `MobileCLIP-S1`
- Train Context: `32`
- Train Strategy: `fixed (matching-table)`
- Eval Strategy: `ctx16 interp`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 39.62 | 0.74 | 53.23 |
| sa1b_nps | 47.12 | 0.78 | 60.23 |
| crowded | 52.79 | 0.83 | 63.80 |
| fg_food | 47.30 | 0.76 | 62.42 |
| fg_sports_equipment | 56.20 | 0.84 | 67.17 |
| attributes | 48.69 | 0.74 | 65.99 |
| wiki_common | 33.46 | 0.61 | 54.42 |
| **Average** | **46.45** | **0.76** | **61.04** |

#### `s1_ctx32_fixed @ ctx32_slice`

- Text Encoder: `MobileCLIP-S1`
- Train Context: `32`
- Train Strategy: `fixed (matching-table)`
- Eval Strategy: `ctx32 slice`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 45.91 | 0.81 | 56.92 |
| sa1b_nps | 53.18 | 0.86 | 61.91 |
| crowded | 59.81 | 0.90 | 66.39 |
| fg_food | 50.73 | 0.79 | 63.84 |
| fg_sports_equipment | 64.31 | 0.89 | 72.13 |
| attributes | 50.77 | 0.77 | 66.00 |
| wiki_common | 40.35 | 0.70 | 57.83 |
| **Average** | **52.15** | **0.82** | **63.57** |

#### `s1_ctx32_fixed @ ctx32_interp`

- Text Encoder: `MobileCLIP-S1`
- Train Context: `32`
- Train Strategy: `fixed (matching-table)`
- Eval Strategy: `ctx32 interp`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 45.91 | 0.81 | 56.92 |
| sa1b_nps | 53.18 | 0.86 | 61.91 |
| crowded | 59.81 | 0.90 | 66.39 |
| fg_food | 50.73 | 0.79 | 63.84 |
| fg_sports_equipment | 64.31 | 0.89 | 72.13 |
| attributes | 50.77 | 0.77 | 66.00 |
| wiki_common | 40.35 | 0.70 | 57.83 |
| **Average** | **52.15** | **0.82** | **63.57** |

#### `l_ctx32_interp @ ctx16_slice`

- Text Encoder: `MobileCLIP2-L`
- Train Context: `32`
- Train Strategy: `interp (77-table)`
- Eval Strategy: `ctx16 slice`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 26.78 | 0.62 | 43.37 |
| sa1b_nps | 36.31 | 0.69 | 52.28 |
| crowded | 35.98 | 0.69 | 52.41 |
| fg_food | 20.30 | 0.54 | 37.65 |
| fg_sports_equipment | 27.89 | 0.57 | 48.93 |
| attributes | 36.49 | 0.65 | 56.05 |
| wiki_common | 14.15 | 0.40 | 35.46 |
| **Average** | **28.27** | **0.59** | **46.59** |

#### `l_ctx32_interp @ ctx16_interp`

- Text Encoder: `MobileCLIP2-L`
- Train Context: `32`
- Train Strategy: `interp (77-table)`
- Eval Strategy: `ctx16 interp`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 39.72 | 0.75 | 53.19 |
| sa1b_nps | 46.35 | 0.78 | 59.12 |
| crowded | 51.83 | 0.82 | 63.25 |
| fg_food | 42.87 | 0.74 | 57.86 |
| fg_sports_equipment | 54.58 | 0.81 | 67.07 |
| attributes | 47.27 | 0.73 | 65.07 |
| wiki_common | 30.33 | 0.60 | 50.81 |
| **Average** | **44.71** | **0.75** | **59.48** |

#### `l_ctx32_interp @ ctx32_slice`

- Text Encoder: `MobileCLIP2-L`
- Train Context: `32`
- Train Strategy: `interp (77-table)`
- Eval Strategy: `ctx32 slice`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 27.11 | 0.62 | 43.73 |
| sa1b_nps | 37.20 | 0.71 | 52.74 |
| crowded | 37.15 | 0.69 | 53.68 |
| fg_food | 21.45 | 0.55 | 39.23 |
| fg_sports_equipment | 28.06 | 0.57 | 49.23 |
| attributes | 38.45 | 0.66 | 58.14 |
| wiki_common | 14.62 | 0.41 | 36.09 |
| **Average** | **29.15** | **0.60** | **47.55** |

#### `l_ctx32_interp @ ctx32_interp`

- Text Encoder: `MobileCLIP2-L`
- Train Context: `32`
- Train Strategy: `interp (77-table)`
- Eval Strategy: `ctx32 interp`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 46.00 | 0.81 | 57.08 |
| sa1b_nps | 53.08 | 0.86 | 61.90 |
| crowded | 59.52 | 0.90 | 66.35 |
| fg_food | 50.68 | 0.79 | 64.13 |
| fg_sports_equipment | 63.83 | 0.89 | 71.70 |
| attributes | 50.60 | 0.77 | 65.99 |
| wiki_common | 40.69 | 0.70 | 58.09 |
| **Average** | **52.06** | **0.82** | **63.61** |

#### `l_ctx32_fixed @ ctx16_slice`

- Text Encoder: `MobileCLIP2-L`
- Train Context: `32`
- Train Strategy: `fixed (matching-table)`
- Eval Strategy: `ctx16 slice`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 45.31 | 0.80 | 56.62 |
| sa1b_nps | 52.38 | 0.85 | 61.80 |
| crowded | 59.05 | 0.90 | 65.90 |
| fg_food | 49.10 | 0.78 | 63.33 |
| fg_sports_equipment | 61.61 | 0.87 | 70.65 |
| attributes | 46.49 | 0.72 | 64.16 |
| wiki_common | 37.13 | 0.67 | 55.34 |
| **Average** | **50.15** | **0.80** | **62.54** |

#### `l_ctx32_fixed @ ctx16_interp`

- Text Encoder: `MobileCLIP2-L`
- Train Context: `32`
- Train Strategy: `fixed (matching-table)`
- Eval Strategy: `ctx16 interp`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 40.41 | 0.75 | 53.84 |
| sa1b_nps | 46.88 | 0.78 | 60.09 |
| crowded | 51.46 | 0.82 | 62.75 |
| fg_food | 44.08 | 0.75 | 59.03 |
| fg_sports_equipment | 52.43 | 0.80 | 65.91 |
| attributes | 48.86 | 0.74 | 65.84 |
| wiki_common | 31.59 | 0.61 | 51.87 |
| **Average** | **45.10** | **0.75** | **59.90** |

#### `l_ctx32_fixed @ ctx32_slice`

- Text Encoder: `MobileCLIP2-L`
- Train Context: `32`
- Train Strategy: `fixed (matching-table)`
- Eval Strategy: `ctx32 slice`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 46.01 | 0.81 | 57.01 |
| sa1b_nps | 53.26 | 0.86 | 61.96 |
| crowded | 59.90 | 0.90 | 66.52 |
| fg_food | 51.16 | 0.79 | 64.79 |
| fg_sports_equipment | 63.82 | 0.89 | 71.61 |
| attributes | 50.58 | 0.77 | 66.10 |
| wiki_common | 40.34 | 0.70 | 57.71 |
| **Average** | **52.15** | **0.82** | **63.67** |

#### `l_ctx32_fixed @ ctx32_interp`

- Text Encoder: `MobileCLIP2-L`
- Train Context: `32`
- Train Strategy: `fixed (matching-table)`
- Eval Strategy: `ctx32 interp`

| Subset Name | CG_F1 | IL_MCC | pmF1 |
|-------------|-------|--------|------|
| metaclip_nps | 46.01 | 0.81 | 57.01 |
| sa1b_nps | 53.26 | 0.86 | 61.96 |
| crowded | 59.90 | 0.90 | 66.52 |
| fg_food | 51.16 | 0.79 | 64.79 |
| fg_sports_equipment | 63.82 | 0.89 | 71.61 |
| attributes | 50.58 | 0.77 | 66.10 |
| wiki_common | 40.34 | 0.70 | 57.71 |
| **Average** | **52.15** | **0.82** | **63.67** |
