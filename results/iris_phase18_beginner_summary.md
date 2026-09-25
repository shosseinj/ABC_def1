# Phase 18 Beginner Summary

## What did we check?

We traced raw Iris measurements through training-fitted normalization, continuous TTFS timing, quantum measured features, classifier logits, and prediction. Phase 18 produced the frozen diagnostics without retraining or held-out test access.

## Where does the problem start?

Some difficulty exists in the raw data because classes 1 and 2 naturally overlap. Raw nearest-centroid and training-fitted linear validation accuracy were both 0.90. The evidence does not establish one unique causal bottleneck, so the qualified result is `MIXED_CAUSE`.

## Is sample 119 bad data?

No. It is a valid canonical Iris row, `[6.0, 2.2, 5.0, 1.5]`, with class 2. It has an unusual mixed profile that resembles class 1 in several local diagnostics, making it naturally ambiguous rather than corrupted or mislabeled.

## Is TTFS losing information?

No additional loss was detected for this continuous TTFS encoder. Normalized and TTFS features had the same 0.90 linear accuracy and separation ratio, with no exact or near cross-class collisions at the fixed tolerance.

## Is the quantum layer causing the issue?

Not consistently. Quantum-feature linear accuracy ranged from 0.85 to 1.00. One checkpoint showed modest compression, but this did not reproduce across seeds.

## Is the classifier causing the issue?

The classifier head is a plausible contributor. Quantum features were often linearly separable even when the frozen model had small margins or errors, but Phase 18 does not prove causation.

## Why is seed 2026 worse?

Its quantum features retained strong class information while its frozen head produced weaker margins and more errors, especially for the defended checkpoint. Training randomness is plausible, but three checkpoint seeds on one split cannot establish why.

## What should we change next?

Phase 19 should preregister one classifier/head diagnostic across additional split seeds, fitted only on training representations and evaluated with paired validation IDs before any held-out test access.

These findings use only 20 class-1/class-2 validation samples on one split. They support neither a causal claim nor a robustness claim.
