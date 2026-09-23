from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from encoding.quantum import angle_encode
from attacks.random_jitter import random_timing_jitter
from attacks.temp_drift import temp_drift_reference
from metrics.quantum_drift import trace_distance_from_bloch, fidelity_from_angles
from metrics.spike_statistics import classical_mismatch


def main():
    clean = np.array([20., 40., 60., 80.])
    T = 100.
    rand, _ = random_timing_jitter(clean, epsilon=5, T=T, seed=42)
    td, _ = temp_drift_reference(
        clean, epsilon=5, tau=0.10, T=T, steps=50, candidates=32
    )

    for name, adv in [("random", rand), ("temp_drift_reference", td)]:
        qc = angle_encode(clean, T)
        qa = angle_encode(adv, T)
        print("\n", name)
        print("clean:", clean)
        print("adv  :", np.round(adv, 3))
        print("trace_distance:", trace_distance_from_bloch(qc, qa))
        print("fidelity:", fidelity_from_angles(qc, qa))
        print("classical:", classical_mismatch(clean, adv, T=T))


if __name__ == "__main__":
    main()
