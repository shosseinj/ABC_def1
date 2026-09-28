# Baselines for the paper

**Compact SNN (local custom model).** A 25,482-parameter two-stage Conv/LIF classifier averages per-time logits and is the principal classical model used alongside QSNN-v3 in the timestamp attack study. In the later controlled study it is trained afresh under the same recipe as the three comparison architectures. Sources: `models/nmnist_snn.py:23-55`, `configs/nmnist_controlled_four_models_lr1e4.json`.

**QSNN-v3 (local hybrid model).** A 32-feature Conv/LIF frontend feeds an eight-qubit simulated variational circuit with a 256-probability readout and no classical bypass. Its five-seed results are validation-only and its seed-42 PGD/TEMP-DRIFT study uses 100 shared clean-correct validation samples. Sources: `models/nmnist_hybrid_qsnn.py:59-279`, `results/nmnist_hybrid_qsnn_seed42/nmnist_qsnn_v3_multiseed/summary.json`.

**ConvNet.** The local controlled baseline instantiates the paper repository's SimpleNet/ConvNet source and retrains it on the same N-MNIST split and preprocessing as the local SNN. It is architecturally different, and its measured ASR is a local result rather than the paper's published number. Sources: `experiments/nmnist/reference_models/simplenet.py`, `experiments/nmnist/controlled_models.py`, `Reports/nmnist_controlled_four_models_lr1e4_comparison.md`.

**Spiking ResNet18.** The local controlled baseline uses the paper repository's spiking ResNet source with locally selected validation checkpoint. It shares the data and attack protocol with the other three local models, but architecture and capacity remain model-specific. Sources: `experiments/nmnist/reference_models/resnet.py`, `experiments/nmnist/controlled_models.py`, `Reports/nmnist_controlled_four_models_lr1e4_comparison.md`.

**VGGSNN.** The local controlled baseline uses the paper repository's VGG-style SNN source and the shared local training recipe. The measured local ASR should not be presented as a direct reproduction of published VGGSNN checkpoints. Sources: `experiments/nmnist/reference_models/vgg.py`, `experiments/nmnist/controlled_models.py`, `Reports/nmnist_controlled_four_models_lr1e4_comparison.md`.

**Timestamp PGD.** This white-box baseline changes raw event timestamps with 20 projected sign-gradient steps and true-label cross entropy. It is compared to TEMP-DRIFT-v2 on the same frozen 100 validation samples, with explicit access and runtime matching. Sources: `scripts/run_nmnist_common_attack_protocol_seed42.py:55-80`, `scripts/run_nmnist_budget_matched_comparison_v2_seed42.py:686-695`.

**TEMP-DRIFT-v2.** This derivative-free timestamp search preserves coordinates, polarity and event count while obeying the same per-event epsilon bound as PGD. The final N-MNIST comparison reports small, statistically nonsignificant ASR differences. Sources: `scripts/run_nmnist_attack_protocol_v3_auditable_seed42.py:238-253`, `results/nmnist_budget_matched_comparison_v2_seed42/report.md`.

**Grid PIL-PGD.** This attack optimizes a soft retiming distribution but evaluates strict, collision-free grid-cell packet moves under `B_inf`, `B1`, or `B0` limits. Its controlled four-model results are a separate experiment from the raw-timestamp PGD/TEMP study. Sources: `attacks/pil_pgd.py:1-14,266-331`, `scripts/run_nmnist_controlled_attack.py:25-133`.
