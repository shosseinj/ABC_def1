from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from experiments.iris.data import load_iris_splits
from encoding.ttfs import ttfs_encode
from encoding.quantum import angle_encode


def main():
    _, _, X_test, _, _, y_test, _ = load_iris_splits()
    x = X_test[0]
    t = ttfs_encode(x, T=100)
    theta = angle_encode(t, T=100)

    print("Normalized Iris features:", np.round(x, 4))
    print("TTFS spike times:", np.round(t, 4))
    print("Quantum angles:", np.round(theta, 4))
    print("Target class:", int(y_test[0]))


if __name__ == "__main__":
    main()
