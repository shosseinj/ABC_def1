"""Run the frozen clean-only 12-checkpoint campaign in protocol order."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe")
SEEDS = (42, 123, 777)


def main() -> None:
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"Use {PYTHON}")
    # Seed 42 is completed for all conditions before the remaining seeds.
    conditions = [(dataset, representation, seed)
                  for seed in SEEDS
                  for dataset, representation in (
                      ("dvs_gesture", "binary"),
                      ("dvs_gesture", "integer"),
                      ("cifar10_dvs", "binary"),
                      ("cifar10_dvs", "integer"))]
    log_dir = ROOT / "Reports/logs/clean_improved_final"
    log_dir.mkdir(parents=True, exist_ok=True)
    for dataset, representation, seed in conditions:
        command = [str(PYTHON), "scripts/train_clean_improved.py", "--mode", "final",
                   "--dataset", dataset, "--representation", representation,
                   "--seed", str(seed)]
        log_path = log_dir / f"{dataset}_{representation}_seed{seed}.log"
        print(f"START | {dataset} | {representation} | {seed}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           check=True)
        print(f"DONE | {dataset} | {representation} | {seed}", flush=True)
    subprocess.run([str(PYTHON), "scripts/train_clean_improved.py", "--mode",
                    "aggregate"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
