# INVALID: Historical Debug Artifact

These N-MNIST attack results are retained for provenance only and must not be
used as scientific evidence. The attack-path event-to-frame helper used
polarity-interleaved channel indexing, while the frozen training/canonical
representation is `[time, polarity, y, x]` with
`polarity * 34 * 34 + y * 34 + x`. The epsilon-zero control therefore changed
the model input despite unchanged timestamps.

Superseded by the explicitly versioned preprocessing-corrected protocol under
`results/nmnist_attack_protocol_v2_canonical_seed42/`.
