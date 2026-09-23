"""Final low-epsilon calibration; attack implementations are imported unchanged."""
import csv, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.run_nmnist_common_attack_protocol_seed42 import (SEED,OUT,SPLIT,SNN_CKPT,QSNN_CKPT,load_models,frames_torch,pgd,project,sha256)
from scripts.run_nmnist_temp_drift_v2_seed42 import temp_drift_v2,bins

EPS=(0.0001,0.00025,0.0005,0.001,0.0025)
DEST=ROOT/"results/nmnist_low_epsilon_calibration_seed42"
PGD_STEPS=20
# PGD code performs one hard forward at each of 21 iterates, one gradient
# forward for each of 20 updates, and one backward pass for each update.
PGD_ACCOUNTING={"candidate_evaluations":21,"model_forward_evaluations":41,"backward_gradient_evaluations":20}

def main():
 from tonic.datasets import NMNIST
 manifest=json.loads((OUT/"common_clean_correct_manifest.json").read_text()); data=NMNIST(save_to=str(ROOT/"data/nmnist"),train=True); models=load_models(); rows=[]
 for pos,item in enumerate(manifest["samples"]):
  sid,label=int(item["sample_id"]),int(item["label"]); events,actual=data[sid]
  if int(actual)!=label: raise RuntimeError("Manifest mismatch")
  clean=np.asarray(events["t"],dtype=np.float64); duration=clean[-1]-clean[0]+1; clean_bins=bins(events,clean)
  for name,model in models.items():
   for ef in EPS:
    for attack in ("PGD","TEMP-DRIFT-v2"):
     epsilon=ef*duration; rng=np.random.default_rng(SEED+pos*1009+int(ef*1000000)); start=time.perf_counter()
     if attack=="PGD": attacked=pgd(model,events,label,epsilon); accounting=PGD_ACCOUNTING
     else: attacked,objective,prediction,queries=temp_drift_v2(model,events,label,epsilon,rng); accounting={"candidate_evaluations":queries,"model_forward_evaluations":queries,"backward_gradient_evaluations":0}
     attacked=project(attacked,clean,epsilon)
     with torch.no_grad(): output=model(frames_torch(events,torch.tensor(attacked,device="cuda",dtype=torch.float32))); prediction=int(output.argmax(1).item())
     delta=np.abs(attacked-clean); changed=bins(events,attacked)!=clean_bins
     rows.append({"sample_id":sid,"label":label,"model":name,"attack":attack,"epsilon_fraction":ef,"attack_success":prediction!=label,"attacked_prediction":prediction,"mean_normalized_abs_dt":float(np.mean(delta/duration)),"median_normalized_abs_dt":float(np.median(delta/duration)),"max_normalized_abs_dt":float(np.max(delta/duration)),"fraction_events_bin_changed":float(changed.mean()),"feasible":bool(np.all(np.diff(attacked)>=0) and np.all(delta<=epsilon+1e-5)),"runtime_seconds":time.perf_counter()-start,**accounting})
     print(f"{pos+1}/100 {name} {attack} {ef:.4%} success={prediction!=label}",flush=True)
 DEST.mkdir(parents=True,exist_ok=True)
 with (DEST/"per_sample_results.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 summary=[]
 for name in ("SNN","QSNN"):
  for attack in ("PGD","TEMP-DRIFT-v2"):
   for ef in EPS:
    s=[r for r in rows if r["model"]==name and r["attack"]==attack and r["epsilon_fraction"]==ef]
    summary.append({"model":name,"attack":attack,"epsilon_fraction":ef,"n":len(s),"asr":float(np.mean([r["attack_success"] for r in s])),"normalized_abs_dt":{"mean":float(np.mean([r["mean_normalized_abs_dt"] for r in s])),"median":float(np.median([r["median_normalized_abs_dt"] for r in s])),"max":float(max(r["max_normalized_abs_dt"] for r in s))},"fraction_events_bin_changed":{"mean":float(np.mean([r["fraction_events_bin_changed"] for r in s])),"median":float(np.median([r["fraction_events_bin_changed"] for r in s])),"max":float(max(r["fraction_events_bin_changed"] for r in s))},"runtime_seconds":{"total":float(sum(r["runtime_seconds"] for r in s)),"mean":float(np.mean([r["runtime_seconds"] for r in s]))},"evaluation_accounting":{k:s[0][k] for k in ("candidate_evaluations","model_forward_evaluations","backward_gradient_evaluations")},"feasibility_rate":float(np.mean([r["feasible"] for r in s]))})
 artifact={"epsilons":EPS,"summary":summary,"manifest":str((OUT/"common_clean_correct_manifest.json").relative_to(ROOT)),"manifest_sha256":sha256(OUT/"common_clean_correct_manifest.json"),"split_sha256":sha256(SPLIT),"snn_checkpoint_sha256":sha256(SNN_CKPT),"qsnn_checkpoint_sha256":sha256(QSNN_CKPT),"attacks_unchanged":True,"five_seed_campaign_started":False}
 (DEST/"summary.json").write_text(json.dumps(artifact,indent=2)+"\n")

if __name__=="__main__": main()
