# Phase 15.2 Paired Robustness Analysis

## Clean Membership

- Common clean-correct (25): `38, 127, 57, 93, 42, 22, 20, 147, 84, 107, 141, 104, 7, 49, 14, 69, 63, 138, 10, 140, 134, 132, 18, 116, 28`
- Baseline-only correct (1): `56`
- Defense-only correct (0): none
- Both wrong (4): `51, 58, 77, 75`

Original Iris dataset indices are used as stable sample IDs.

## Lost Clean Sample

Sample 56 has true label 1. Baseline predicts 1 with confidence 0.4470; the defense
predicts 2 with confidence 0.4422. Its decision margins are only 0.00453 and 0.00101,
respectively, so it is clearly near the class 1/2 boundary. The clean measured-feature
representation changes by L2 distance 0.04545.

Baseline logits: `[-0.8929, 0.5043, 0.4941]`

Defense logits: `[-0.8689, 0.4624, 0.4647]`

## Paired Attack Summary

All ASRs below use only the 25 common clean-correct samples.

| Attack | Epsilon | Tau | Base fail | Defense fail | Both robust | Both fail | Rescued | Broken | Net | Delta ASR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 1% | 10% | 0 | 0 | 25 | 0 | 0 | 0 | 0 | 0.00 |
| Random | 2% | 10% | 0 | 0 | 25 | 0 | 0 | 0 | 0 | 0.00 |
| Random | 5% | 10% | 0 | 0 | 25 | 0 | 0 | 0 | 0 | 0.00 |
| Random | 10% | 10% | 1 | 1 | 24 | 1 | 0 | 0 | 0 | 0.00 |
| Classical PGD | 1% | 10% | 2 | 2 | 23 | 2 | 0 | 0 | 0 | 0.00 |
| Classical PGD | 2% | 10% | 2 | 2 | 23 | 2 | 0 | 0 | 0 | 0.00 |
| Classical PGD | 5% | 10% | 4 | 4 | 21 | 4 | 0 | 0 | 0 | 0.00 |
| Classical PGD | 10% | 10% | 8 | 7 | 17 | 7 | 1 | 0 | +1 | -0.04 |
| Reference TEMP-DRIFT | 1% | 10% | 0 | 0 | 25 | 0 | 0 | 0 | 0 | 0.00 |
| Reference TEMP-DRIFT | 2% | 10% | 0 | 0 | 25 | 0 | 0 | 0 | 0 | 0.00 |
| Reference TEMP-DRIFT | 5% | 10% | 1 | 1 | 24 | 1 | 0 | 0 | 0 | 0.00 |
| Reference TEMP-DRIFT | 10% | 10% | 3 | 2 | 22 | 2 | 1 | 0 | +1 | -0.04 |
| Gradient TEMP-DRIFT | 1% | 1/5/10% | 2 | 1 | 23 | 1 | 1 | 0 | +1 | -0.04 |
| Gradient TEMP-DRIFT | 2% | 1% | 1 | 1 | 24 | 1 | 0 | 0 | 0 | 0.00 |
| Gradient TEMP-DRIFT | 2% | 5/10% | 2 | 2 | 23 | 2 | 0 | 0 | 0 | 0.00 |
| Gradient TEMP-DRIFT | 5% | 1% | 1 | 1 | 24 | 1 | 0 | 0 | 0 | 0.00 |
| Gradient TEMP-DRIFT | 5% | 5/10% | 2 | 2 | 23 | 2 | 0 | 0 | 0 | 0.00 |
| Gradient TEMP-DRIFT | 10% | 1% | 2 | 1 | 23 | 1 | 1 | 0 | +1 | -0.04 |
| Gradient TEMP-DRIFT | 10% | 5% | 3 | 2 | 22 | 2 | 1 | 0 | +1 | -0.04 |
| Gradient TEMP-DRIFT | 10% | 10% | 4 | 3 | 21 | 3 | 1 | 0 | +1 | -0.04 |

## Priority Findings

Classical PGD at 5% has `both robust=21`, `both fail=4`, `rescued=0`, and `broken=0`.
The apparent unpaired improvement from 5/26 to 4/25 was entirely the lost clean sample.

Classical PGD at 10% has `both robust=17`, `both fail=7`, `rescued=1`, and `broken=0`.
The paired ASR changes from 8/25 to 7/25, a net gain of one sample.

Gradient TEMP-DRIFT at 10% retains a one-sample net gain for every tau. No paired
configuration has a net gain larger than one sample.

## Drift on Rescued Samples

The rescued 10% gradient TEMP-DRIFT sample shows reductions in all three quantities:

- feature drift: 0.19279 to 0.18738
- logit drift: 0.42983 to 0.41743
- prediction JS: 0.007062 to 0.006688

The rescued 10% PGD sample has slightly lower logit drift, but feature drift and JS are
slightly higher. The drift evidence is therefore not consistent across attack types.

## Statistics

For every configuration with a rescue, `b=1` and `c=0`. The two-sided exact
McNemar/binomial p-value is 1.0. The discordant sample count is far too small for a
statistical significance claim.

## Transfer Diagnostic

Not implemented. White-box paired evaluation remains the primary analysis; no attack
definition was changed.

## Assessment

There are no samples made less robust on the common set, but gains occur only as
isolated one-sample rescues. The 5% PGD improvement disappears after denominator
control, while 10% PGD and several TEMP-DRIFT settings retain net gain +1. This is a
weak, mixed robustness signal, not sufficient evidence for a defense claim.
