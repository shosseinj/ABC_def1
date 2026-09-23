import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

PHASE_TESTS = {
    1: "tests/test_phase01_iris.py",
    2: "tests/test_phase02_ttfs.py",
    3: "tests/test_phase03_quantum_encoding.py",
    4: "tests/test_phase04_qsnn.py",
    5: "tests/test_phase05_training.py",
    6: "tests/test_phase06_evaluation.py",
    7: "tests/test_phase07_depth_ablation.py",
    8: "tests/test_phase08_random_jitter.py",
    9: "tests/test_phase09_quantum_metrics.py",
    10:"tests/test_phase10_stealth.py",
    11:"tests/test_phase11_temp_drift.py",
    12: "tests/test_phase12_attack_evaluation.py",
    13: "tests/test_phase13_attack_sweep.py",
    14: "tests/test_phase14_classical_timing.py",
    14.5: "tests/test_phase145_temp_drift_gradient.py",
    15: "tests/test_phase15_quantum_temp.py",
    15.1: "tests/test_phase151_quantum_temp_ablation.py",
    15.2: "tests/test_phase152_paired_analysis.py",
    15.3: "tests/test_phase153_multiseed.py",
    16: "tests/test_phase16_decision_losses.py",
    17: "tests/test_phase17_adversarial_training.py",
    17.1: "tests/test_phase171_calibration.py",
    17.2: "tests/test_phase172_diagnosis.py",
    17.3: "tests/test_phase173_checkpoint_selection.py",
    18: "tests/test_phase18_representation_diagnosis.py",
    19: "tests/test_phase19_frozen_heads.py",
    20: "tests/test_phase20_calibration.py",
    21: "tests/test_phase21_representation_consistency.py",
}

NEXT = {1:2,2:3,3:4,4:5,5:6,6:7,7:8,8:9,9:10,10:11,11:12,12:13,13:14,14:14.5,14.5:15,15:15.1,15.1:15.2,15.2:15.3,15.3:16,16:17,17:17.1,17.1:17.2,17.2:17.3,17.3:18,18:19,19:20,20:21}

def run_phase(phase):
    phase_label = f"{phase:g}" if isinstance(phase, float) else str(phase)
    if phase not in PHASE_TESTS:
        print(f"Phase {phase_label} has no unit-test gate in this starter.")
        return 2
    test=PHASE_TESTS[phase]
    print(f"Running PHASE {phase_label}: {test}\n")
    proc=subprocess.run([sys.executable,"-m","pytest",str(ROOT / test),"-q"], cwd=ROOT)
    if proc.returncode==0:
        print(f"\nPHASE {phase_label} PASSED")
        if phase in NEXT:
            print(f"You can proceed to PHASE {NEXT[phase]}.")
    else:
        print(f"\nPHASE {phase_label} FAILED")
        print("Do not proceed. Fix failing tests first.")
    return proc.returncode

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase",type=float)
    args=ap.parse_args()
    if args.phase is not None:
        raise SystemExit(run_phase(args.phase))
    for ph in PHASE_TESTS:
        code=run_phase(ph)
        if code!=0:
            raise SystemExit(code)

if __name__=="__main__":
    main()
