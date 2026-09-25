# Budget-Matched PGD vs TEMP-DRIFT-v2 Comparison (Seed-42)

**Evidence basis:** Seed-42 only, 100 common clean-correct samples.
**Frozen models:** SNN, QSNN-v3. No retraining. No five-seed campaign.

## Budget Definitions

**Query-cost model:** backward pass = 0.5 forward equivalents
**PGD original:** 20 steps = 21 forwards + 20 backward = 31 forward-equiv
**TEMP original:** 1600 candidates = 1600 forwards

| Condition | PGD steps | PGD cost | TEMP initial | TEMP gen x per | TEMP cost |
|-----------|-----------|----------|--------------|----------------|-----------|
| A: Original fixed-budget | 20 | 31.0 | 600 | 4x250 | 1600 |
| B1: Forward-count matched (TEMP=PGD forwards only) | 20 | 31.0 | 21 | 0x0 | 21 |
| B2: Query-cost matched (alpha=0.5) | 20 | 31.0 | 31 | 0x0 | 31 |
| B3: Query-cost matched with evolution (15+16) | 20 | 31.0 | 15 | 1x16 | 31 |
| C: Wall-clock matched for SNN (TEMP init=200) | 20 | 31.0 | 200 | 0x0 | 200 |
| C: Wall-clock matched for QSNN (TEMP init=200) | 20 | 31.0 | 200 | 0x0 | 200 |

## Wall-Clock Calibration

### SNN
- PGD mean runtime: 0.596s
- TEMP matched initial: 200 (0.053s)
- TEMP runtime profile:
  - init=10: 0.038s
  - init=20: 0.022s
  - init=30: 0.025s
  - init=50: 0.014s
  - init=80: 0.026s
  - init=100: 0.035s
  - init=150: 0.051s
  - init=200: 0.053s

### QSNN
- PGD mean runtime: 2.441s
- TEMP matched initial: 200 (0.125s)
- TEMP runtime profile:
  - init=10: 0.033s
  - init=20: 0.031s
  - init=30: 0.029s
  - init=50: 0.048s
  - init=80: 0.066s
  - init=100: 0.068s
  - init=150: 0.097s
  - init=200: 0.125s

## ASR Summary by Condition

| Model | Condition | ε | PGD ASR | TEMP ASR | Δ (T-P) | McNemar p | PGD IDs | TEMP IDs |
|-------|-----------|---|---------|----------|---------|-----------|---------|----------|

## Detailed Paired Comparisons

## Distortion Metrics by Condition (successful attacks only)

| Model | Condition | ε | Attack | Mean L1 | Max Linf | Mean |dt|/dur | Mean bin change |
|-------|-----------|---|--------|---------|----------|-------------|-----------------|

## Caveats

- All results are Seed-42 evidence only; not multi-seed robustness evidence.
- With very few successes (0-5 per cell), McNemar tests have very low power.
- Do NOT claim one attack is stronger unless the relevant matched comparison supports it.
- The five-seed attack campaign has NOT been started.
