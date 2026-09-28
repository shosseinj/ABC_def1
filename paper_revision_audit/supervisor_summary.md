# Supervisor summary

## 1. Model changes
The final QSNN-v3 is a 40,762-parameter Conv/LIF-to-eight-qubit simulated circuit with learned 32→8 projection, two re-upload blocks, ring CNOTs, learned measurement and a 256-probability classifier; the controlled architecture study instead uses the 25,482-parameter compact SNN and three locally retrained reference architectures. These are different experiments. Sources: `models/nmnist_hybrid_qsnn.py:59-279`, `models/nmnist_snn.py:23-55`, `Reports/nmnist_controlled_four_models_lr1e4_comparison.md`.

## 2. Attack changes
The QSNN study uses raw-timestamp PGD versus derivative-free TEMP-DRIFT-v2 under duration-scaled epsilon on 100 shared validation samples. The controlled four-model study uses ten-bin packet-preserving PIL-PGD under grid budgets on 1,000 shared official-test samples per representation. Sources: `scripts/run_nmnist_attack_protocol_v3_auditable_seed42.py:238-253`, `scripts/run_nmnist_controlled_attack.py:25-133`.

## 3. Defense changes
Quantum/timing consistency and margin losses exist in source, but final QSNN-v3 training records no defense; no defended N-MNIST robustness improvement is verified. Sources: `defenses/quantum_temp.py:6-128`, `scripts/run_nmnist_qsnn_v3_multiseed.py:99-101`.

## 4. Mathematical formulas
`mathematical_formulation.tex` specifies LIF, the circuit outline, timestamp feasible set, packet identity and three grid budgets. `evaluation_metrics.tex` specifies accuracy, macro-F1, clean-correct ASR, paired discordance, distortion and query accounting.

## 5. Hardware/software tables
Current environment: RTX 4090, Python 3.10.4, PyTorch 2.5.1+cu124, toolkit 12.4, cuDNN 9.1.0; driver version, RAM and historical run-time environment are NOT VERIFIED. See `hardware_software_versions.md`.

## 6. Seeds
QSNN-v3 clean validation used 42/123/777/2026/6543; timestamp attacks and four-model controlled robustness use seed 42 only. Sources: `scripts/run_nmnist_qsnn_v3_multiseed.py:52-100`, `configs/nmnist_controlled_four_models_lr1e4.json`.

## 7. Training/validation/testing
QSNN-v3 was validation-only; the four-model study used common 55,000/5,000 training/validation and all 10,000 official-test examples for clean accuracy. Attack manifests differ and are frozen within their own studies. See `seeds_and_dataset_protocol.md`.

## 8. Metrics
ASR is misclassification among frozen clean-correct examples; numerator and denominator must travel with each result. The saved timestamp comparison includes paired outcomes and audit; the grid comparison records all realized budgets and independent reconstruction. See `evaluation_metrics.tex`.

## 9. Baselines (at most two sentences each)
ConvNet is the paper-source SimpleNet retrained locally under the common recipe. Spiking ResNet18 is the paper-source residual SNN retrained locally. VGGSNN is the paper-source VGG-style SNN retrained locally. Timestamp PGD is a 20-step white-box event-time attack. TEMP-DRIFT-v2 is a derivative-free event-time search. Grid PIL-PGD is a separate packet-retiming attack. See `baseline_descriptions.md`.

## 10. Outstanding issues
The four-model seed-42 controlled result shows 100%/99.5% Binary ASR for the custom SNN at `B_inf=1`/`B0=200`, versus 0–1.9% for the other three. This gap is real in the saved artifacts but attack-strength parity and multi-seed generality are NOT VERIFIED; the paper/source B0 penalty discrepancy, absent QSNN-v3 official-test score and lack of final defended-model evaluation must be disclosed. See `verified_experimental_results.md` and `missing_information_and_inconsistencies.md`.
