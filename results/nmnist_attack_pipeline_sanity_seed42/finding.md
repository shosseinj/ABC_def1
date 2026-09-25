# Seed-42 Pipeline Sanity Finding

The sanity check fails. At epsilon zero, attack success is 89% for both SNN attack paths and 97% for both QSNN attack paths. Timestamps and temporal bins are unchanged, but the attack-path tensor differs from the canonical clean tensor by a mean of about 3,593 frame elements per sample.

Root cause: the attack helper `frames_torch()` computes a flattened event channel as `2 * (y * 34 + x) + p`, which interleaves polarity within each pixel. The frozen training representation `events_to_frames()` produces the tensor layout `[time, polarity, y, x]`, whose flattened channel is `p * 34 * 34 + y * 34 + x`. Therefore the attack path feeds a polarity-interleaved tensor while clean selection/training uses polarity-major frames. This is a preprocessing mismatch, not a genuine extreme low-epsilon vulnerability.

The requested checks also show that zero-epsilon success does not arise from identical tensors: no final tensor is identical to the canonical clean tensor, so the identical-input implication is not violated. However, that does not rescue the protocol because the epsilon-zero input is already the wrong input representation.

The frozen attack algorithms and models were not modified. The five-seed attack campaign must not start until a separately versioned attack representation is made exactly identical to `events_to_frames()`, then the common-set calibration and all downstream attack results are rerun. Existing attack outputs remain historical/debug artifacts.
