# Phase 20 Beginner Summary

## Why did we choose HEAD_E?

It was the strongest alternative classifier family in Phase 19. Phase 20 changed only this final cosine-margin decision layer while keeping the quantum feature extractor frozen. HEAD_A remained the warm, co-adapted deployment reference.

## What are scale and margin?

Scale controls how large the cosine class scores are. Margin makes training demand extra separation for the correct class. Training used `s*(cosine-m)` for the true class; inference used `s*cosine`.

## Why only nine settings?

The protocol fixed scales `5, 10, 15` and margins `0.05, 0.10, 0.15`, giving exactly nine combinations. The grid was not extended after results were observed.

## Did clean accuracy improve?

Not reproducibly enough to pass the gate. The largest observed mean gain was about 0.67 percentage points, but its descriptive interval included zero and split outcomes were mixed.

## Did class 2 improve?

Some configurations increased mean class-2 accuracy by roughly 1.3 to 2.0 percentage points, but directions varied across splits and intervals included zero. Class-2 improvement was therefore not stable.

## Did results repeat across new splits?

No. The five new split seeds were the replication units. Minimum-class accuracy was non-worse in only one of five splits for the best-observed configuration, below the required three.

## Was attack testing allowed?

No. No configuration passed the clean gate, so random timing and Classical PGD remained blocked.

## Did attack robustness improve?

Unknown. No attacks ran, so no paired robustness outcomes exist.

## Is there a classifier candidate worth keeping?

No. The selected configuration and Phase 21 candidate are both null. Sample 119 was not specially used for tuning or selection, and no hidden or final-test feature rows were accessed.

The narrow conclusion is that this fixed nine-setting grid was insufficient. It does not prove that all calibration methods are ineffective.
