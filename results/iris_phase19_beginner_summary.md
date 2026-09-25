# Phase 19 Beginner Summary

## What is the classifier head?

It is the final classical layer that converts four measured quantum features into three class scores. `HEAD_A_CURRENT` is the warm, co-adapted deployment baseline. `HEAD_B_LINEAR_REINIT` is the fair reinitialized linear control for comparing formulations.

## Why freeze the quantum part?

Freezing it isolates changes caused by the head. All quantum feature-extractor hashes were identical before and after head training, so this isolation succeeded technically.

## Why use multiple splits?

One split can be misleading. Five stratified split seeds tested sensitivity to validation membership. Split seed, not individual samples or the 15 model cells, was the replication unit.

## Which head was most stable?

HEAD_A had the smallest overall split-to-split accuracy SD, 0.0236. Among candidates, `HEAD_E_COSINE_MARGIN` had SD 0.0256 and the highest mean accuracy, 0.9400, but it still failed the clean deployment gate against A.

## Did classes 1 and 2 improve?

Compared fairly with HEAD_B, HEAD_E improved class-2 accuracy by about 0.087 across split means. Its class-1 gain was small and uncertain, so both classes did not improve reliably together.

## Did sample 119 remain difficult?

Yes. In post-hoc descriptive analysis it was often misclassified or low-margin when present. It was excluded from selection and every gate, so this does not define the chosen result.

## Did attack resistance improve?

Unknown. No candidate passed clean eligibility, so no candidate was attacked. Only HEAD_A was characterized, and candidate robustness is not estimable.

## Did results repeat across splits?

HEAD_E showed favorable clean results relative to the fair HEAD_B control, especially for class 2. However, the deployment result also repeated: no candidate met the full gate. Winner count was zero and no end-to-end run occurred.

## What should happen next?

Preregister either a new representation-level hypothesis or a confirmation using new untouched development splits. Do not tune these heads or inspect the held-out test set.

Phase 19 makes `CLASSIFIER_HEAD_PARTIAL` plausible, but it does not prove classifier causality, formulation superiority, or robustness.
