# 1. Interpreter

Used `C:\Users\jafari.h\Desktop\ai_project\.venv\Scripts\python.exe`. The supplied
path without the separator before `.venv` does not exist. No environment was changed.

# 2. Files Added

Added `experiments/iris/phase172.py`, the frozen diagnosis runner, focused tests, all
required trajectory/fragility/margin/centroid/gradient/checkpoint artifacts, a validation
invariant artifact, and this report.

# 3. Files Modified

Added an optional observation callback to the existing trainer; it does not affect loss,
optimization, or checkpoint selection. Updated documentation, manifest, and phase runner.
The defense, hyperparameters, QSNN architecture, and Phase 14 attacks are unchanged.

# 4. Regression Status

Phase 17.2 gate: 6 passed. Full regression suite: 65 passed in 5.18 seconds. Stable-ID,
trajectory-alignment, validation-only, frozen-config, and unchanged-Phase-14 checks pass.

# 5. Frozen Models and Seeds

Compared baseline against PGD-1 adversarial training with `epsilon=0.02T`, `alpha=0.02T`,
clean start, `lambda_adv=0.5`, `lambda_margin=0.5`, target 0.1, no JS/fidelity, for seeds
42, 777, and 2026 on fixed split 42. Exact configurations were reconstructed for all 80 epochs.

# 6. Sample 119 Baseline Fragility

Sample 119 is intrinsically fragile. Baseline seed 42 crosses to class 1 at epoch 28 and
recovers at 75. Seed 777 approaches the boundary smoothly and crosses immediately after
its selected epoch: selected margin 0.00496 at epoch 63, crossing at 64. Seed 2026 crosses
at 17, recovers at 38, and crosses again after its epoch-71 selection. Its selected margin
is 0.00103. Thus baseline correctness depends on checkpoint timing.

# 7. Sample 119 Training Trajectory

Defense crossings are progressive. Seed 42 crosses at epoch 5, recovers at 13, crosses
to class 1 at 22, and recovers at 64; selected epoch 80 is correct with margin 0.0202.
Seed 777 crosses at 65 and never recovers, reaching margin -0.0960 at selected epoch 80.
Seed 2026 crosses at 15 and never recovers; selected epoch 39 has margin -0.0409.

# 8. Boundary-Crossing Epochs

| Seed | Baseline first crossing | Baseline recovery | Defense first crossing | Defense final recovery |
|---:|---:|---|---:|---|
| 42 | 28 | 75 | 5 | 64 |
| 777 | 64 | none | 65 | none |
| 2026 | 17 | 38, then recrosses | 15 | none |

The trajectories show gradual margin shrinkage rather than a single abrupt update.

# 9. Checkpoint-Selection Diagnostic

At seed 777, defense epochs 50-64 have the same 0.9333 validation accuracy as the selected
epoch and preserve sample 119; global validation-loss tie-breaking continues to epoch 80
after the crossing. Checkpoint selection therefore contributes. At seed 2026, no epoch
with the selected 0.9000 accuracy also preserves sample 119, so checkpoint selection is
not the primary cause there. The global metric can hide a class-2 exchange at seed 777.

# 10. Class 1/2 Fragile Samples

Across final checkpoints, boundary IDs are:

```text
|margin| < 0.05: 70, 78, 85, 119, 142
|margin| < 0.10: 52, 61, 70, 78, 85, 119, 137, 142, 148
|margin| < 0.20: 52, 61, 70, 78, 80, 81, 85, 89, 90, 99, 100, 115, 119, 122, 129, 130, 137, 142, 148
```

Sample 119 is the only repeated defense flip in the frozen comparisons. Full STABLE,
LOW_MARGIN_STABLE, DEFENSE_MARGIN_DEGRADED, DEFENSE_FLIPPED, and SEED_SENSITIVE labels
are in the fragile-sample artifacts.

# 11. Class 1/2 Margin Distributions

Seed 42 defense reduces class-2 mean margin from 0.2379 to 0.1807 but narrows its spread.
Seed 777 class-2 mean margin rises from 0.5405 to 0.6161 despite sample 119 flipping,
showing a localized tail failure. Seed 2026 compresses class-1 mean margin from 0.1502 to
0.0562 and class-2 from 0.2898 to 0.1098; negative class-2 margins increase from one to
two. Compression is broad only for seed 2026.

# 12. Feature-Centroid Separation

Class-1/2 centroid distance changes from 0.3346 to 0.3861 at seed 42 and 0.5089 to 0.5295
at seed 777. It falls from 0.4770 to 0.3412 at seed 2026. Adversarial training therefore
does not globally reduce separation across every seed, but causes substantial compression
in the failing seed 2026.

# 13. Sample 119 Feature Position

Sample-119 distances to class-1/class-2 centroids are 0.3053/0.2262 baseline and
0.3115/0.1828 defense at seed 42. At seed 777 they become 0.3658/0.3176 and
0.3607/0.3533, nearly equidistant. At seed 2026 they change from 0.3486/0.2265 to
0.2768/0.2818, moving slightly closer to class 1 than class 2. The classifier flip agrees
with this model-dependent feature displacement.

# 14. Class-Wise Adversarial Pressure

Timing displacement is essentially identical across classes (about 4.0 L2 units). Class-2
timing-gradient norms exceed class 1 at seeds 42/777 but not 2026. Adversarial-minus-clean
CE pressure is largest for class 2 only at seed 777. There is no consistent three-seed
evidence that class 2 receives uniquely stronger attack pressure.

# 15. Gradient-Conflict Analysis

For sample 119, clean/adversarial CE cosine similarity is 0.9991, 0.9993, and 0.9994.
Clean/margin cosine is 0.9778, 0.9954, and 0.9217. Adversarial/margin cosine is 0.9714,
0.9957, and 0.9161. The losses strongly reinforce rather than oppose one another. Similar
alignment appears for low-margin control 142. Direct objective-gradient conflict is not
supported.

# 16. Stable vs Fragile Control Samples

Controls were selected by seed-42 baseline margin before comparison: stable class-2 ID
122 has maximum margin; low-margin preserved class-2 ID 142 has the smallest positive
margin excluding 119; ID 119 is the target. Stable ID 122 has zero active margin gradient.
IDs 142 and 119 have similar CE-gradient alignment and large active margin gradients,
showing that low margin, not unique gradient opposition, distinguishes the fragile cases.

# 17. Root-Cause Assessment

The dominant explanation is mixed. Sample 119 is a pre-existing sample-specific boundary
case. Seed 777 failure is amplified by global-loss checkpoint tie-breaking. Seed 2026
shows genuine class-1/2 feature and margin compression. Objective conflict is not supported,
and stronger class-2 attack pressure is not consistent. The defense exposes and sometimes
amplifies an unstable class boundary rather than creating one universal failure mechanism.

# 18. Test-Access Status

`test_set_accessed=false` and `test_loader_invoked=false`. Only train/validation data were
loaded. No final-test files were created.

# 19. Scientific Conclusion

1. Sample 119 flips because its baseline margin is nearly zero and defense features become equidistant or class-1-nearer in failing seeds.
2. It is already fragile in every baseline trajectory.
3. Defense crossings are progressive, not abrupt.
4. Persistent final flips occur in seeds 777 and 2026, while seed 42 recovers.
5. Separation decreases globally only in seed 2026.
6. Class-2 margins are compressed at seeds 42/2026, but increase on average at 777 despite the tail flip.
7. Several samples are low-margin, but 119 is the repeated defense flip.
8. Checkpoint selection contributes at seed 777, not seed 2026.
9. Clean and adversarial CE gradients do not conflict for sample 119.
10. Margin loss reinforces both CE gradients; it does not create an opposing direction.
11. The issue is primarily sample-specific with seed-dependent checkpoint and boundary-compression components.

This diagnosis does not support a robustness claim.

# 20. Recommended Next Phase

Do not redesign the whole defense around sample 119. First test a validation-only robust
checkpoint-selection diagnostic that includes minimum class-wise accuracy or boundary
stability, because it directly addresses seed 777 without changing training. Treat seed
2026 compression separately; any later objective redesign must be justified across more
than this single fragile sample.
