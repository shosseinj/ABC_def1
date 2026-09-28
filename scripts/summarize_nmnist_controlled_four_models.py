"""Publish a controlled local N-MNIST comparison only after every cell audits PASS."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe")
OUT = ROOT / "Reports/nmnist_controlled_four_models_lr1e4_comparison.md"
NAMES = ("custom", "convnet", "resnet18", "vggsnn")
BUDGETS = {"binary": [("B_inf", 1), ("B_inf", 2), ("B_inf", 3), ("B1", 500), ("B1", 750), ("B1", 1000), ("B0", 200), ("B0", 300), ("B0", 400)],
           "integer": [("B_inf", 1), ("B_inf", 2), ("B_inf", 3), ("B1", 500), ("B1", 750), ("B1", 1000), ("B1", 1500), ("B0", 200), ("B0", 300), ("B0", 400), ("B0", 600)]}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_attack(run_id: str, manifest_hash: str) -> float:
    root = ROOT / "Reports/results/nmnist_controlled_four_models_lr1e4/attacks"
    state = ROOT / "Reports/checkpoints/nmnist_controlled_four_models_lr1e4"
    metadata_path = root / f"{run_id}.json"
    artifact_path = root / f"{run_id}.npz"
    audit_path = root / f"{run_id}.audit.json"
    marker = json.loads((state / f"{run_id}.complete.json").read_text(encoding="utf-8"))
    for path, field in ((metadata_path, "metadata_sha256"), (artifact_path, "artifact_sha256"), (audit_path, "audit_sha256")):
        if sha256(path) != marker[field]:
            raise RuntimeError(f"{run_id}: completion hash mismatch for {field}")
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (marker["status"] != audit["status"] or audit["status"] != "PASS" or audit["errors"]
            or meta["manifest_sha256"] != manifest_hash or audit["clean_correct_denominator"] != 1000):
        raise RuntimeError(f"{run_id}: invalid independent audit or manifest")
    return float(audit["asr_percent"])


def main() -> None:
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"use required interpreter {PYTHON}")
    lines = ["# Controlled local four-model N-MNIST comparison", "",
             "All values below are new local measurements from the same seed-42 data split, training recipe, equal-event-count T=10 representation, validation-selected checkpoints, and the first 1,000 official test examples jointly classified correctly by all four models. Every attack cell passed independent raw-data reconstruction and prediction audit. The paper's published numbers are not included in this controlled table.", ""]
    for rep in ("binary", "integer"):
        manifest_path = ROOT / f"Reports/results/nmnist_controlled_four_models_lr1e4/{rep}_joint_clean_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "FROZEN" or len(manifest["sample_ids"]) != 1000:
            raise RuntimeError(f"{rep}: invalid common manifest")
        manifest_hash = sha256(manifest_path)
        full = all((ROOT / f"Reports/checkpoints/nmnist_controlled_four_models_lr1e4/controlled_{rep}_{name}_{kind}_{budget}_seed42.complete.json").exists()
                   for name in NAMES for kind, budget in BUDGETS[rep])
        budgets = BUDGETS[rep] if full else [("B_inf", 1), ("B0", 200)]
        headers = [f"{kind}={budget} ASR (%)" for kind, budget in budgets]
        lines += [f"## {rep.capitalize()}", "",
                  "| Model | Clean accuracy (%) | " + " | ".join(headers) + " |",
                  "|---|---:|" + "---:|" * len(headers)]
        for name in NAMES:
            source = ROOT / manifest["models"][name]["result_path"]
            if sha256(source) != manifest["models"][name]["result_sha256"]:
                raise RuntimeError(f"{rep}/{name}: clean result hash mismatch")
            result = json.loads(source.read_text(encoding="utf-8"))
            values = [checked_attack(f"controlled_{rep}_{name}_{kind}_{budget}_seed42", manifest_hash)
                      for kind, budget in budgets]
            lines.append(f"| {name} | {100*result['test']['accuracy']:.2f} | " + " | ".join(f"{value:.1f}" for value in values) + " |")
        lines += ["", f"Joint clean-correct pool: {manifest['eligible_intersection_size']} samples; first 1,000 frozen. Manifest SHA-256: `{manifest_hash}`.", ""]
    lines += ["The four architectures differ by design; the dataset, train/validation/test split, preprocessing, seed, optimizer settings, checkpoint rule, attacked sample IDs, attack implementation, and budgets are held constant. Lower ASR means greater robustness under this local attack. This is one seed; it does not establish multi-seed uncertainty. The official source hard-codes the B0 penalty at 5, whereas the paper's general text states 10. This source/paper ambiguity is shared across all local models and is not resolved by these measurements.", "",
              "Evidence: `Reports/results/nmnist_controlled_four_models_lr1e4/`, `Reports/checkpoints/nmnist_controlled_four_models_lr1e4/`, and `Reports/logs/nmnist_controlled_four_models_lr1e4/`.", ""]
    temp = OUT.with_name(OUT.name + f".{os.getpid()}.tmp")
    temp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temp, OUT)
    print(f"CONTROLLED REPORT COMPLETE {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
