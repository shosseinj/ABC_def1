"""Refresh the B_inf=3 mixed-prefix provenance and its independent audit."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe")
STEM = "seed42_paper_aligned_B_inf_3"
META = ROOT / "Reports/results/nmnist_seed42_paper_aligned/attacks" / f"{STEM}.json"
AUDIT = META.with_suffix(".audit.json")
MARKER = ROOT / "Reports/checkpoints/nmnist_seed42_paper_aligned" / f"{STEM}.complete.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"required interpreter: {PYTHON}")
    meta = json.loads(META.read_text(encoding="utf-8"))
    if meta["run_id"] != STEM or meta["requested_budget"] != 3:
        raise RuntimeError("unexpected artifact")
    adapter = ROOT / meta["adapter_path"]
    meta["adapter_sha256"] = digest(adapter)
    meta["resumed_prefix_samples"] = 768
    meta["prefix_projection"] = "upstream per-sample strict greedy projection, hybrid batched relaxed projection"
    meta["suffix_projection"] = "CUDA per-event-line strict greedy projection, hybrid batched relaxed projection"
    meta["suffix_samples"] = 232
    meta["implementation_note"] = (
        "First 768 records use the prior strict greedy adapter; final 232 use the CUDA "
        "event-line greedy adapter. A 36-case projector equivalence test and four-sample "
        "end-to-end attack equivalence test passed. The independent auditor verifies all "
        "1,000 serialized records. The two adapters were not compared on every record."
    )
    atomic_json(META, meta)
    command = [str(PYTHON), "-u", "scripts/audit_nmnist_seed42_paper_aligned_attacks.py",
               str(META.relative_to(ROOT)), "--output", str(AUDIT.relative_to(ROOT))]
    subprocess.run(command, cwd=ROOT, check=True)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit["status"] != "PASS":
        raise RuntimeError("independent audit did not pass")
    atomic_json(MARKER, {"status": "PASS", "run_id": STEM,
                         "metadata_sha256": digest(META),
                         "artifact_sha256": digest(ROOT / meta["artifact_path"]),
                         "audit_sha256": digest(AUDIT)})
    print("MIXED-PREFIX PROVENANCE REPAIRED; INDEPENDENT AUDIT PASS", flush=True)


if __name__ == "__main__":
    main()
