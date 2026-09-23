"""Calibration sweep on the existing Seed-42 common attack manifest.

This file imports the frozen attack functions unchanged. It does not retrain or
alter either model and writes to a new calibration directory.
"""
import csv, json, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from scripts.run_nmnist_common_attack_protocol_seed42 import (
    SEED, SPLIT, SNN_CKPT, QSNN_CKPT, OUT, EPS, load_models, pgd,
    temp_drift, project, sha256,
)
from experiments.nmnist.snn_baseline import events_to_frames

CAL = ROOT / "results/nmnist_attack_calibration_seed42"
CAL_EPS = (0.0025, 0.005, 0.01, 0.02, 0.05, 0.10)

def temporal_bins(events, timestamps):
    t0 = float(events["t"][0]); duration = max(float(events["t"][-1])-t0+1, 1.0)
    return np.minimum(((np.asarray(timestamps)-t0)*10//duration).astype(int), 9)

def main():
    manifest = json.loads((OUT / "common_clean_correct_manifest.json").read_text())
    data = __import__("tonic.datasets", fromlist=["NMNIST"]).NMNIST(save_to=str(ROOT/"data/nmnist"), train=True)
    models = load_models(); rows=[]
    # Repeat only the non-deterministic derivative-free QSNN search at the two
    # disputed epsilons; the main sweep retains one reproducible seed per cell.
    repeats = {("QSNN", "TEMP-DRIFT", 0.05): 3, ("QSNN", "TEMP-DRIFT", 0.10): 3}
    for pos, item in enumerate(manifest["samples"]):
        sid, label = int(item["sample_id"]), int(item["label"])
        events, actual_label = data[sid]
        if int(actual_label) != label: raise RuntimeError("Manifest label mismatch")
        clean_t=np.asarray(events["t"], dtype=np.float64); duration=clean_t[-1]-clean_t[0]+1
        clean_bins=temporal_bins(events, clean_t)
        for model_name, model in models.items():
            for ef in CAL_EPS:
                for attack in ("PGD", "TEMP-DRIFT"):
                    nrun=repeats.get((model_name,attack,ef),1)
                    for trial in range(nrun):
                        eps=float(ef*duration); rng=np.random.default_rng(SEED+pos*1009+int(ef*10000)+trial*1000003)
                        started=time.perf_counter()
                        attacked=pgd(model,events,label,eps) if attack=="PGD" else temp_drift(model,events,label,eps,rng)
                        attacked=project(attacked,clean_t,eps)
                        with torch.no_grad():
                            out=model(__import__("scripts.run_nmnist_common_attack_protocol_seed42",fromlist=["frames_torch"]).frames_torch(events,torch.tensor(attacked,device="cuda",dtype=torch.float32)))
                            pred=int(out.argmax(1).item())
                        delta=np.abs(attacked-clean_t); moved=temporal_bins(events,attacked)!=clean_bins
                        rows.append({"sample_id":sid,"label":label,"model":model_name,"attack":attack,"epsilon_fraction":ef,"trial":trial,"event_count":len(events),"mean_abs_dt":float(delta.mean()),"median_abs_dt":float(np.median(delta)),"max_abs_dt":float(delta.max()),"mean_normalized_abs_dt":float((delta/duration).mean()),"median_normalized_abs_dt":float(np.median(delta/duration)),"max_normalized_abs_dt":float((delta/duration).max()),"fraction_events_bin_changed":float(moved.mean()),"events_bin_changed":int(moved.sum()),"attack_success":bool(pred!=label),"attacked_prediction":pred,"feasible":bool(np.all(np.diff(attacked)>=0) and np.all(delta<=eps+1e-5)),"runtime_seconds":time.perf_counter()-started,"attacked_timestamps":attacked.tolist()})
                        print(f"{pos+1}/100 {model_name} {attack} {ef:.4f} trial={trial} success={pred!=label}",flush=True)
    CAL.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0])
    with (CAL/"per_sample_results.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    # JSON retains full timestamp vectors; CSV remains convenient for tabular analysis.
    (CAL/"per_sample_results.json").write_text(json.dumps(rows,indent=2)+"\n")
    summary=[]
    for model in ("SNN","QSNN"):
      for attack in ("PGD","TEMP-DRIFT"):
       for ef in CAL_EPS:
        sel=[r for r in rows if r["model"]==model and r["attack"]==attack and r["epsilon_fraction"]==ef and r["trial"]==0]
        summary.append({"model":model,"attack":attack,"epsilon_fraction":ef,"n":len(sel),"asr":float(np.mean([r["attack_success"] for r in sel])),"mean_abs_dt":float(np.mean([r["mean_abs_dt"] for r in sel])),"median_abs_dt":float(np.median([r["median_abs_dt"] for r in sel])),"max_abs_dt":float(max(r["max_abs_dt"] for r in sel)),"mean_normalized_abs_dt":float(np.mean([r["mean_normalized_abs_dt"] for r in sel])),"median_normalized_abs_dt":float(np.median([r["median_normalized_abs_dt"] for r in sel])),"max_normalized_abs_dt":float(max(r["max_normalized_abs_dt"] for r in sel)),"mean_fraction_events_bin_changed":float(np.mean([r["fraction_events_bin_changed"] for r in sel])),"median_fraction_events_bin_changed":float(np.median([r["fraction_events_bin_changed"] for r in sel])),"max_fraction_events_bin_changed":float(max(r["fraction_events_bin_changed"] for r in sel)),"feasibility_rate":float(np.mean([r["feasible"] for r in sel]))})
    repeats_summary=[]
    for ef in (0.05,0.10):
      for model in ("QSNN",):
       sel=[r for r in rows if r["model"]==model and r["attack"]=="TEMP-DRIFT" and r["epsilon_fraction"]==ef]
       repeats_summary.append({"model":model,"attack":"TEMP-DRIFT","epsilon_fraction":ef,"trial_asr":[float(np.mean([r["attack_success"] for r in sel if r["trial"]==t])) for t in range(3)]})
    artifact={"calibration_epsilons":CAL_EPS,"base_manifest":str((OUT/"common_clean_correct_manifest.json").relative_to(ROOT)),"split_sha256":sha256(SPLIT),"snn_checkpoint_sha256":sha256(SNN_CKPT),"qsnn_checkpoint_sha256":sha256(QSNN_CKPT),"attack_functions_unchanged":True,"summary":summary,"repeated_qsnn_temp_drift":repeats_summary,"note":"Single Seed-42 calibration only; no five-seed attack campaign."}
    (CAL/"summary.json").write_text(json.dumps(artifact,indent=2)+"\n")

if __name__=="__main__": main()
