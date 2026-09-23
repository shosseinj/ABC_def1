from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.iris.data import load_iris_splits
from scripts.run_quantum_temp_phase151 import train_validation_candidate


def main():
    config = json.loads((ROOT / "configs" / "iris.json").read_text(encoding="utf-8"))
    config.update({
        "quantum_temp_enabled": True,
        "quantum_temp_epsilon": 0.02,
        "lambda_q": 0.1,
        "lambda_pred": 0.0,
        "consistency_enabled": False,
        "normalize_quantum_loss": True,
        "quantum_loss_ema_decay": 0.9,
    })
    _, Xval, _, _, yval, _, _ = load_iris_splits(
        seed=config["seed"], test_size=config["test_size"], val_size=config["val_size"]
    )
    result = train_validation_candidate("phase151_q_normalized_0p1", config, Xval, yval)
    payload = {
        "variant": "normalized_quantum_loss_validation_only",
        "normalization": "L_quantum / stopgrad(EMA_0.9(L_quantum) + numerical floor)",
        "excluded_from_completed_grid_selection": True,
        "test_evaluated": False,
        "result": result,
    }
    (ROOT / "results" / "iris_quantum_temp_normalized_check.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(
        f"normalized val={result['clean_validation_accuracy']:.3f} "
        f"R_q={result['r_q']:.3f} PGD_ASR={result['mean_pgd_asr']:.3f}"
    )


if __name__ == "__main__":
    main()
