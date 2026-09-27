"""Refresh only the local rows of the N-MNIST paper-reference tables."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe")
README = ROOT / "readme_jafar.md"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_rows(representation: str, budgets: list[tuple[str, int]]) -> tuple[float, list[float]]:
    suffix = "_integer" if representation == "Integer" else ""
    root = ROOT / f"Reports/results/nmnist_seed42_paper_aligned{suffix}"
    state = ROOT / f"Reports/checkpoints/nmnist_seed42_paper_aligned{suffix}"
    clean = json.loads((root / "seed42_clean_result.json").read_text(encoding="utf-8"))
    if not clean.get("clean_gate_pass") or clean.get("seed") != 42 or clean["test"]["samples"] != 10000:
        raise RuntimeError(f"{representation}: invalid clean result")
    for name in ("checkpoint", "predictions"):
        if digest(ROOT / clean[f"{name}_path"]) != clean[f"{name}_sha256"]:
            raise RuntimeError(f"{representation}: {name} hash mismatch")
    values = []
    manifests = set()
    checkpoints = set()
    for kind, budget in budgets:
        run_id = f"seed42_paper_aligned{suffix}_{kind}_{budget}"
        base = root / "attacks" / run_id
        meta_path = base.with_suffix(".json")
        artifact_path = base.with_suffix(".npz")
        audit_path = base.with_suffix(".audit.json")
        marker = json.loads((state / f"{run_id}.complete.json").read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if marker.get("status") != audit.get("status") or marker.get("status") != "PASS":
            raise RuntimeError(f"{run_id}: completion/audit status mismatch")
        if marker.get("run_id") != audit.get("run_id") or marker["run_id"] != run_id:
            raise RuntimeError(f"{run_id}: run ID mismatch")
        for path, expected in (
            (meta_path, marker["metadata_sha256"]),
            (artifact_path, marker["artifact_sha256"]),
            (audit_path, marker["audit_sha256"]),
        ):
            if digest(path) != expected:
                raise RuntimeError(f"{run_id}: hash mismatch: {path}")
        if meta["budget_type"] != kind or int(meta["requested_budget"]) != budget:
            raise RuntimeError(f"{run_id}: budget metadata mismatch")
        if meta["representation"] != representation or meta["checkpoint_sha256"] != clean["checkpoint_sha256"]:
            raise RuntimeError(f"{run_id}: representation/checkpoint mismatch")
        if audit["clean_correct_denominator"] != 1000 or audit.get("errors"):
            raise RuntimeError(f"{run_id}: invalid audit denominator/errors")
        numerator = int(audit["successful_attacks"])
        if not (0 <= numerator <= 1000) or abs(audit["asr_percent"] - numerator / 10) > 1e-9:
            raise RuntimeError(f"{run_id}: ASR/numerator mismatch")
        manifests.add(meta["manifest_sha256"])
        checkpoints.add(meta["checkpoint_sha256"])
        values.append(numerator / 10)
    if len(manifests) != 1 or len(checkpoints) != 1:
        raise RuntimeError(f"{representation}: runs do not share a frozen manifest/checkpoint")
    return clean["test"]["accuracy"] * 100, values


def formatted(value: float) -> str:
    return f"{value:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify artifacts and show rows without editing")
    args = parser.parse_args()
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"use required interpreter {PYTHON}")
    binary = [("B_inf", x) for x in (1, 2, 3)] + [("B1", x) for x in (500, 750, 1000)] + [("B0", x) for x in (200, 300, 400)]
    integer = [("B_inf", x) for x in (1, 2, 3)] + [("B1", x) for x in (500, 750, 1000, 1500)] + [("B0", x) for x in (200, 300, 400, 600)]
    rows = {}
    for representation, budgets in (("Binary", binary), ("Integer", integer)):
        accuracy, asrs = checked_rows(representation, budgets)
        label = f"Custom SNN (ours; seed 42; {representation}; exploratory / training unmatched)"
        row = "|  | **" + label + "** | **" + f"{accuracy:.2f}" + "** | " + " | ".join(f"**{formatted(value)}**" for value in asrs) + " |"
        rows[representation] = row
        print(f"{representation}: {row}")
    original = README.read_text(encoding="utf-8")
    updated = original
    for representation, row in rows.items():
        table_number = 1 if representation == "Binary" else 2
        start = updated.index(f"#### Table {table_number} ")
        end = updated.find("\n#### ", start + 1)
        if end < 0:
            raise RuntimeError(f"Table {table_number}: next heading missing")
        section = updated[start:end]
        pattern = re.compile(r"^\|\s*\|\s*\*\*[^\n|]*(?:ours|Ours)[^\n|]*(?:paper-aligned attack evaluation|exploratory / training unmatched)[^\n]*\|$", re.MULTILINE)
        replaced, count = pattern.subn(lambda _: row, section)
        if count != 1:
            raise RuntimeError(f"Table {table_number}: expected one current local row, found {count}")
        updated = updated[:start] + replaced + updated[end:]
    if args.check:
        print("CHECK ONLY: tables were not edited")
    elif updated != original:
        temporary = README.with_name(README.name + f".{os.getpid()}.tmp")
        temporary.write_text(updated, encoding="utf-8")
        os.replace(temporary, README)
        print("Updated readme_jafar.md from hash-verified audited artifacts")
    else:
        print("readme_jafar.md already matches hash-verified audited artifacts")


if __name__ == "__main__":
    main()
