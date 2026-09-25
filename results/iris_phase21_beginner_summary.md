# Phase 21 Beginner Summary

## What is a quantum representation?

It is the four measured values produced by the quantum circuit and passed to the final linear classifier. They summarize how the circuit represents each Iris sample.

## What does representation drift mean?

It is the distance between a sample's clean and attacked quantum features. Larger distance means the timing perturbation changed the learned representation more.

## Did attacked samples move more than robust samples?

Only slightly on average. The successful-minus-robust quantum L2 contrast was `+0.004901`, but its descriptive 95% interval was `[-0.008996, 0.018799]`.

## Was that consistent across splits?

No. The five split contrasts were negative, positive, negative, positive, and negative. Only two of five splits showed the required direction.

## Did movement track margin degradation?

Not under the required split-level comparison. Successful-minus-robust margin-drop contrasts were negative in all five splits. A positive pooled correlation was descriptive and used non-independent observations, so it could not rescue the gate.

## Did we build a representation-stability defense?

No. Stage A failed, so Stage B and its lambda grid were correctly blocked.

## Did a defense reduce movement or preserve separation and accuracy?

Not measured because no defense was trained. Anti-collapse geometry, clean preservation, and candidate attack tests were therefore not estimable.

## Did attack success decrease?

Not measured. No candidate, paired robustness comparison, or TEMP-DRIFT transfer evaluation exists.

## What should happen next?

Do not continue tuning this representation-stability loss. Representation instability was not reproducibly supported by the predefined gate. Revisit architecture or encoding assumptions only if new evidence motivates a specific hypothesis, and keep the final test set untouched.

Root classification: `REPRESENTATION_INSTABILITY_NOT_SUPPORTED`.
