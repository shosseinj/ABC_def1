"""Corrected TEMP-DRIFT-v2 on the frozen Seed-42 common sample manifest.

Version 2 minimizes true-class margin z_y - max_{j != y}(z_j). Lower values
mean less true-class separation; negative values are misclassified. Historical
TEMP-DRIFT outputs are never read as completion markers or overwritten.
"""
import csv, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.run_nmnist_common_attack_protocol_seed42 import (
    SEED, OUT, SPLIT, SNN_CKPT, QSNN_CKPT, load_models, frames_torch, project, sha256,
)

VERSION="TEMP-DRIFT-v2-margin"
EPS=(0.0025,0.005,0.01,0.02,0.05,0.10)
INITIAL,GENERATIONS,PER_GENERATION=600,4,250
QUERIES=INITIAL+GENERATIONS*PER_GENERATION
DEST=ROOT/"results/nmnist_temp_drift_v2_seed42"

def true_class_margin(logits,label):
    """Return z_y - max competitor; minimizing is an untargeted objective."""
    competitors=logits.clone(); competitors[:,int(label)]=-torch.inf
    return logits[:,int(label)]-competitors.max(1).values

def objective_values(logits,label): return true_class_margin(logits,label)

@torch.no_grad()
def evaluate_candidates(model,events,candidates,label,chunk=64):
    values=[]; predictions=[]
    for start in range(0,len(candidates),chunk):
        timestamps=torch.as_tensor(candidates[start:start+chunk],device="cuda",dtype=torch.float32)
        output=model(frames_torch(events,timestamps))
        values.extend(objective_values(output,label).cpu().tolist())
        predictions.extend(output.argmax(1).cpu().tolist())
    return np.asarray(values),np.asarray(predictions)

def temp_drift_v2(model,events,label,epsilon,rng):
    clean=np.asarray(events["t"],dtype=np.float64); n=len(clean)
    pool=np.asarray([project(clean+rng.uniform(-epsilon,epsilon,n),clean,epsilon) for _ in range(INITIAL)])
    values,predictions=evaluate_candidates(model,events,pool,label)
    for scale in (0.50,0.30,0.18,0.10):
        # Lowest margin is best; keep successful low-margin candidates naturally.
        elite_indices=np.argsort(values,kind="stable")[:12]
        elite=pool[elite_indices]
        children=[]
        for i in range(PER_GENERATION):
            parent=elite[i%len(elite)]
            proposal=parent+0.5*(elite[rng.integers(12)]-elite[rng.integers(12)])
            proposal+=rng.normal(0,epsilon*scale,n)
            children.append(project(proposal,clean,epsilon))
        children=np.asarray(children); child_values,child_predictions=evaluate_candidates(model,events,children,label)
        pool=np.vstack((elite,children)); values=np.concatenate((values[elite_indices],child_values)); predictions=np.concatenate((predictions[elite_indices],child_predictions))
    best=int(np.argmin(values))
    return pool[best],float(values[best]),int(predictions[best]),QUERIES

def bins(events,timestamps):
    t0=float(events["t"][0]); duration=max(float(events["t"][-1])-t0+1,1.0)
    return np.minimum(((np.asarray(timestamps)-t0)*10//duration).astype(int),9)

def main():
    manifest=json.loads((OUT/"common_clean_correct_manifest.json").read_text())
    from tonic.datasets import NMNIST
    dataset=NMNIST(save_to=str(ROOT/"data/nmnist"),train=True); models=load_models(); rows=[]
    for pos,item in enumerate(manifest["samples"]):
      sid,label=int(item["sample_id"]),int(item["label"]); events,actual=dataset[sid]
      if int(actual)!=label: raise RuntimeError("Manifest label mismatch")
      clean=np.asarray(events["t"],dtype=np.float64); duration=clean[-1]-clean[0]+1; clean_bins=bins(events,clean)
      for model_name,model in models.items():
       with torch.no_grad():
        clean_logits=model(frames_torch(events,torch.tensor(clean,device="cuda",dtype=torch.float32)))
        clean_objective=float(objective_values(clean_logits,label).item())
       for ef in EPS:
        trials=range(3) if ef in (0.05,0.10) else range(1)
        for trial in trials:
         rng=np.random.default_rng(SEED+pos*1009+int(ef*10000)+trial*1000003); epsilon=ef*duration; started=time.perf_counter()
         attacked,objective,prediction,queries=temp_drift_v2(model,events,label,epsilon,rng); runtime=time.perf_counter()-started
         attacked=project(attacked,clean,epsilon); delta=np.abs(attacked-clean); changed=bins(events,attacked)!=clean_bins
         rows.append({"protocol_version":VERSION,"sample_id":sid,"label":label,"model":model_name,"epsilon_fraction":ef,"trial":trial,"event_count":len(events),"clean_objective_margin":clean_objective,"attacked_objective_margin":objective,"objective_improvement_clean_minus_attacked":clean_objective-objective,"attack_success":prediction!=label,"attacked_prediction":prediction,"query_count":queries,"runtime_seconds":runtime,"mean_normalized_abs_dt":float(np.mean(delta/duration)),"median_normalized_abs_dt":float(np.median(delta/duration)),"max_normalized_abs_dt":float(np.max(delta/duration)),"fraction_events_bin_changed":float(np.mean(changed)),"events_bin_changed":int(changed.sum()),"feasible":bool(np.all(np.diff(attacked)>=0) and np.all(delta<=epsilon+1e-5)),"attacked_timestamps":attacked.tolist()})
         print(f"{pos+1}/100 {model_name} eps={ef:.4f} trial={trial} success={prediction!=label} margin={objective:.6f}",flush=True)
    DEST.mkdir(parents=True,exist_ok=True)
    with (DEST/"per_sample_results.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (DEST/"per_sample_results.json").write_text(json.dumps(rows,indent=2)+"\n")
    summary=[]
    for model in ("SNN","QSNN"):
     for ef in EPS:
      selected=[r for r in rows if r["model"]==model and r["epsilon_fraction"]==ef and r["trial"]==0]
      summary.append({"model":model,"epsilon_fraction":ef,"n":len(selected),"asr":float(np.mean([r["attack_success"] for r in selected])),"normalized_abs_dt":{"mean":float(np.mean([r["mean_normalized_abs_dt"] for r in selected])),"median":float(np.median([r["median_normalized_abs_dt"] for r in selected])),"max":float(max(r["max_normalized_abs_dt"] for r in selected))},"fraction_events_bin_changed":{"mean":float(np.mean([r["fraction_events_bin_changed"] for r in selected])),"median":float(np.median([r["fraction_events_bin_changed"] for r in selected])),"max":float(max(r["fraction_events_bin_changed"] for r in selected))},"query_count_per_sample":QUERIES,"runtime_seconds":{"total":float(sum(r["runtime_seconds"] for r in selected)),"mean_per_sample":float(np.mean([r["runtime_seconds"] for r in selected]))},"feasibility_rate":float(np.mean([r["feasible"] for r in selected])),"objective_improved_rate":float(np.mean([r["attacked_objective_margin"]<=r["clean_objective_margin"] for r in selected]))})
    repeats=[]
    for model in ("SNN","QSNN"):
     for ef in (0.05,0.10):
      cell=[r for r in rows if r["model"]==model and r["epsilon_fraction"]==ef]
      repeats.append({"model":model,"epsilon_fraction":ef,"trial_asr":[float(np.mean([r["attack_success"] for r in cell if r["trial"]==t])) for t in range(3)]})
    artifact={"protocol_version":VERSION,"objective":"minimize z_y - max_{j != y}(z_j); lower is more adversarial and negative implies misclassification","candidate_budget_per_sample":QUERIES,"epsilons":EPS,"base_manifest":str((OUT/"common_clean_correct_manifest.json").relative_to(ROOT)),"split_sha256":sha256(SPLIT),"snn_checkpoint_sha256":sha256(SNN_CKPT),"qsnn_checkpoint_sha256":sha256(QSNN_CKPT),"summary":summary,"reproducibility":repeats,"five_seed_campaign_started":False}
    (DEST/"summary.json").write_text(json.dumps(artifact,indent=2)+"\n")

if __name__=="__main__": main()
