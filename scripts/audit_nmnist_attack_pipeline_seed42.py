"""Fail-closed tensor-level audit of the frozen Seed-42 attack pipeline."""
import csv,json,sys,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from experiments.nmnist.snn_baseline import events_to_frames
from scripts.run_nmnist_common_attack_protocol_seed42 import SEED,OUT,load_models,frames_torch,pgd,project
from scripts.run_nmnist_temp_drift_v2_seed42 import temp_drift_v2,bins

DEST=ROOT/"results/nmnist_attack_pipeline_sanity_seed42"
EPS=(0.0,0.0001)

def tensor_metrics(clean,adv):
 d=(adv-clean).float(); flat=d.flatten()
 return {"changed_frame_elements":int((flat!=0).sum()),"frame_l0":int((flat!=0).sum()),"frame_l1":float(flat.abs().sum()),"frame_l2":float(torch.linalg.vector_norm(flat,2)),"frame_linf":float(flat.abs().max())}

def main():
 from tonic.datasets import NMNIST
 manifest=json.loads((OUT/"common_clean_correct_manifest.json").read_text()); data=NMNIST(save_to=str(ROOT/"data/nmnist"),train=True); models=load_models(); rows=[]; traces=[]
 for pos,item in enumerate(manifest["samples"]):
  sid,label=int(item["sample_id"]),int(item["label"]); events,actual=data[sid]
  if int(actual)!=label: raise RuntimeError("Label/index mismatch")
  clean_t=np.asarray(events["t"],dtype=np.float64); duration=clean_t[-1]-clean_t[0]+1; clean_bins=bins(events,clean_t)
  canonical=torch.from_numpy(events_to_frames(events,10)).float()[None].cuda()
  attack_clean=frames_torch(events,torch.tensor(clean_t,device="cuda",dtype=torch.float32))
  for model_name,model in models.items():
   with torch.no_grad(): clean_pred=int(model(canonical).argmax(1)); attack_clean_pred=int(model(attack_clean).argmax(1))
   for ef in EPS:
    epsilon=ef*duration
    for attack in ("PGD","TEMP-DRIFT-v2"):
     rng=np.random.default_rng(SEED+pos*1009+int(ef*1000000)); start=time.perf_counter()
     if attack=="PGD": attacked=pgd(model,events,label,epsilon)
     else: attacked,_,_,_=temp_drift_v2(model,events,label,epsilon,rng)
     attacked=project(attacked,clean_t,epsilon); adv=frames_torch(events,torch.tensor(attacked,device="cuda",dtype=torch.float32))
     with torch.no_grad(): adv_pred=int(model(adv).argmax(1))
     changed_t=attacked!=clean_t; changed_bins=bins(events,attacked)!=clean_bins; metrics=tensor_metrics(canonical,adv); identical=bool(torch.equal(canonical,adv))
     row={"sample_id":sid,"validation_position":pos,"label":label,"model":model_name,"attack":attack,"epsilon_fraction":ef,"event_count":len(events),"timestamps_changed":int(changed_t.sum()),"fraction_timestamps_changed":float(changed_t.mean()),"events_changing_bin":int(changed_bins.sum()),"fraction_events_changing_bin":float(changed_bins.mean()),**metrics,"clean_prediction":clean_pred,"attack_path_clean_prediction":attack_clean_pred,"adversarial_prediction":adv_pred,"attack_success":adv_pred!=label,"final_tensor_identical_to_canonical_clean":identical,"runtime_seconds":time.perf_counter()-start}
     rows.append(row)
     if ef==0.0001 and row["attack_success"] and len([t for t in traces if t["model"]==model_name and t["attack"]==attack])<3:
      moved=np.flatnonzero(changed_bins)
      traces.append({"sample_id":sid,"label":label,"model":model_name,"attack":attack,"clean_prediction":clean_pred,"attack_path_clean_prediction":attack_clean_pred,"adversarial_prediction":adv_pred,"first_changed_event_indices":moved[:10].tolist(),"first_clean_timestamps":clean_t[moved[:10]].tolist(),"first_attacked_timestamps":attacked[moved[:10]].tolist(),"first_clean_bins":clean_bins[moved[:10]].tolist(),"first_attacked_bins":bins(events,attacked)[moved[:10]].tolist(),"canonical_vs_attack_clean":tensor_metrics(canonical,attack_clean),"canonical_vs_adversarial":metrics})
     print(f"{pos+1}/100 {model_name} {attack} eps={ef:.4%} success={row['attack_success']} identical={identical}",flush=True)
 DEST.mkdir(parents=True,exist_ok=True)
 with (DEST/"per_sample_results.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 controls=[]
 for model in ("SNN","QSNN"):
  for attack in ("PGD","TEMP-DRIFT-v2"):
   for ef in EPS:
    s=[r for r in rows if r["model"]==model and r["attack"]==attack and r["epsilon_fraction"]==ef]
    controls.append({"model":model,"attack":attack,"epsilon_fraction":ef,"n":len(s),"asr":float(np.mean([r["attack_success"] for r in s])),"identical_tensor_count":sum(r["final_tensor_identical_to_canonical_clean"] for r in s),"identical_tensor_successes":sum(r["final_tensor_identical_to_canonical_clean"] and r["attack_success"] for r in s),"mean_timestamps_changed":float(np.mean([r["timestamps_changed"] for r in s])),"mean_events_changing_bin":float(np.mean([r["events_changing_bin"] for r in s])),"mean_changed_frame_elements":float(np.mean([r["changed_frame_elements"] for r in s])),"success_bin_changes":[r["events_changing_bin"] for r in s if r["attack_success"]],"failure_bin_changes":[r["events_changing_bin"] for r in s if not r["attack_success"]]})
 artifact={"controls":controls,"traces":traces,"zero_asr_pass":all(x["asr"]==0 for x in controls if x["epsilon_fraction"]==0),"no_identical_input_success_pass":all(x["identical_tensor_successes"]==0 for x in controls),"five_seed_campaign_started":False}
 (DEST/"audit.json").write_text(json.dumps(artifact,indent=2)+"\n")

if __name__=="__main__":main()
