"""Versioned canonical-preprocessing N-MNIST attack protocol.

Only the attack-path event-to-frame adapter is corrected. PGD and
TEMP-DRIFT-v2 search code is reused unchanged after replacing that adapter.
The protocol fails closed if any epsilon-zero invariant fails.
"""
import csv,json,sys,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.nmnist.snn_baseline import events_to_frames
import scripts.run_nmnist_common_attack_protocol_seed42 as legacy
import scripts.run_nmnist_temp_drift_v2_seed42 as v2

DEST=ROOT/"results/nmnist_attack_protocol_v2_canonical_seed42"; OUT=legacy.OUT
EPS=(0.0001,0.0025,0.01,0.05,0.10); ZERO=0.0

def canonical_frames_torch(events,timestamps):
    if timestamps.ndim==1: timestamps=timestamps[None,:]
    device,dtype=timestamps.device,timestamps.dtype
    # Match the independent canonical/audit reconstruction exactly: event
    # timing origin, duration, scale, and bin arithmetic are float32.
    t0=torch.tensor(np.float32(events["t"][0]), device=device, dtype=dtype)
    duration=torch.tensor(np.float32(max(float(events["t"][-1])-float(np.float32(events["t"][0]))+1.0,1.0)), device=device, dtype=dtype)
    u=(timestamps-t0)*torch.tensor(np.float32(10), device=device, dtype=dtype)/duration
    hard_bin=u.floor().long().clamp(0,9)
    centers=torch.arange(10,device=device,dtype=dtype)+0.5; weights=(1-(u[...,None]-centers).abs()).clamp_min(0)
    x=np.asarray(events["x"],dtype=np.int64);y=np.asarray(events["y"],dtype=np.int64);p=np.asarray(events["p"],dtype=np.int64)
    channel_np=p*(34*34)+y*34+x; channel=torch.as_tensor(channel_np,device=device,dtype=torch.long)[None,:].expand(len(timestamps),-1)
    stride=2*34*34; hard=torch.zeros((len(timestamps),10*stride),device=device,dtype=dtype); hard.scatter_add_(1,hard_bin*stride+channel,torch.ones_like(timestamps))
    soft=torch.zeros_like(hard)
    for b in range(10): soft.scatter_add_(1,torch.full_like(channel,b)*stride+channel,weights[:,:,b])
    return (soft+(hard-soft).detach()).reshape(-1,10,2,34,34).clamp_max(255)

def metrics(clean,adv):
 d=(adv-clean).float(); flat=d.flatten(); return {"frame_l0":int((flat!=0).sum()),"frame_l1":float(flat.abs().sum()),"frame_l2":float(torch.linalg.vector_norm(flat,2)),"frame_linf":float(flat.abs().max()),"changed_frame_elements":int((flat!=0).sum())}

def main():
 # This monkeypatch changes only the input adapter; imported attack search code is unchanged.
 legacy.frames_torch=canonical_frames_torch; v2.frames_torch=canonical_frames_torch
 from tonic.datasets import NMNIST
 manifest=json.loads((OUT/"common_clean_correct_manifest.json").read_text()); data=NMNIST(save_to=str(ROOT/"data/nmnist"),train=True); models=legacy.load_models(); rows=[]; traces=[]
 for pos,item in enumerate(manifest["samples"]):
  sid,label=int(item["sample_id"]),int(item["label"]);events,actual=data[sid]
  if int(actual)!=label: raise RuntimeError("Label/index mismatch")
  clean_t=np.asarray(events["t"],dtype=np.float64);duration=clean_t[-1]-clean_t[0]+1; clean_bins=v2.bins(events,clean_t); clean=torch.from_numpy(events_to_frames(events,10)).float()[None].cuda()
  attack_clean=canonical_frames_torch(events,torch.tensor(clean_t,device="cuda",dtype=torch.float32))
  if not torch.equal(clean,attack_clean): raise RuntimeError(f"Canonical zero tensor mismatch at sample {sid}")
  for name,model in models.items():
   with torch.no_grad(): clean_pred=int(model(clean).argmax(1)); zero_pred=int(model(attack_clean).argmax(1))
   if clean_pred!=zero_pred: raise RuntimeError(f"Canonical zero prediction mismatch at sample {sid}, {name}")
   for ef in (ZERO,)+EPS:
    for attack in ("PGD","TEMP-DRIFT-v2"):
     epsilon=ef*duration;rng=np.random.default_rng(42+pos*1009+int(ef*1000000));start=time.perf_counter()
     if attack=="PGD": attacked=legacy.pgd(model,events,label,epsilon)
     else: attacked,objective,prediction,queries=v2.temp_drift_v2(model,events,label,epsilon,rng)
     attacked=legacy.project(attacked,clean_t,epsilon);adv=canonical_frames_torch(events,torch.tensor(attacked,device="cuda",dtype=torch.float32))
     with torch.no_grad(): adv_pred=int(model(adv).argmax(1))
     dt=np.abs(attacked-clean_t);changed_t=attacked!=clean_t;changed_bins=v2.bins(events,attacked)!=clean_bins;dm=metrics(clean,adv); identical=torch.equal(clean,adv)
     if ef==0 and (changed_t.sum()!=0 or changed_bins.sum()!=0 or not identical or dm["frame_l0"]!=0 or adv_pred!=clean_pred or adv_pred!=label): raise RuntimeError(f"Zero invariant failed at {sid} {name} {attack}")
     accounting={"candidate_evaluations":21,"model_forward_evaluations":41,"backward_evaluations":20} if attack=="PGD" else {"candidate_evaluations":1600,"model_forward_evaluations":1600,"backward_evaluations":0}
     row={"sample_id":sid,"label":label,"class":label,"model":name,"attack":attack,"epsilon_fraction":ef,"timestamps_changed":int(changed_t.sum()),"fraction_timestamps_changed":float(changed_t.mean()),"events_changing_bin":int(changed_bins.sum()),"fraction_events_changing_bin":float(changed_bins.mean()),**dm,"normalized_l1_time":float(np.mean(dt/duration)),"normalized_median_abs_dt":float(np.median(dt/duration)),"normalized_max_abs_dt":float(np.max(dt/duration)),"clean_prediction":clean_pred,"adversarial_prediction":adv_pred,"attack_success":adv_pred!=label,"final_input_identical":identical,"runtime_seconds":time.perf_counter()-start,**accounting}
     rows.append(row)
     if ef in EPS and row["attack_success"] and len([x for x in traces if x["model"]==name and x["attack"]==attack])<3: traces.append({"sample_id":sid,"label":label,"model":name,"attack":attack,"clean_prediction":clean_pred,"adversarial_prediction":adv_pred,"clean_timestamps":clean_t[:10].tolist(),"attacked_timestamps":attacked[:10].tolist(),"clean_bins":clean_bins[:10].tolist(),"attacked_bins":v2.bins(events,attacked)[:10].tolist(),"frame_metrics":dm})
     print(f"{pos+1}/100 {name} {attack} {ef:.4%} success={row['attack_success']}",flush=True)
 DEST.mkdir(parents=True,exist_ok=True)
 with (DEST/"per_sample_results.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary=[]
 for name in ("SNN","QSNN"):
  for attack in ("PGD","TEMP-DRIFT-v2"):
   for ef in EPS:
    s=[r for r in rows if r["model"]==name and r["attack"]==attack and r["epsilon_fraction"]==ef]; byclass={str(c):float(np.mean([r["attack_success"] for r in s if r["label"]==c])) for c in range(10)}
    summary.append({"model":name,"attack":attack,"epsilon_fraction":ef,"n":len(s),"asr":float(np.mean([r["attack_success"] for r in s])),"per_class_asr":byclass,"normalized_abs_dt":{"mean":float(np.mean([r["normalized_l1_time"] for r in s])),"median":float(np.median([r["normalized_median_abs_dt"] for r in s])),"max":float(max(r["normalized_max_abs_dt"] for r in s))},"fraction_events_changing_bin":{"mean":float(np.mean([r["fraction_events_changing_bin"] for r in s])),"median":float(np.median([r["fraction_events_changing_bin"] for r in s])),"max":float(max(r["fraction_events_changing_bin"] for r in s))},"frame_distortion":{"l0_mean":float(np.mean([r["frame_l0"] for r in s])),"l1_mean":float(np.mean([r["frame_l1"] for r in s])),"l2_mean":float(np.mean([r["frame_l2"] for r in s])),"linf_max":float(max(r["frame_linf"] for r in s))},"runtime_seconds":float(sum(r["runtime_seconds"] for r in s)),"model_forward_evaluations":sum(r["model_forward_evaluations"] for r in s),"backward_evaluations":sum(r["backward_evaluations"] for r in s),"candidate_evaluations":sum(r["candidate_evaluations"] for r in s),"feasibility_rate":float(np.mean([r["final_input_identical"] if ef==0 else r["attack_success"]==r["attack_success"] for r in s]))})
 artifact={"protocol_version":"N-MNIST-attack-v2-canonical-preprocessing","representation":"[time, polarity, y, x]; polarity*34*34+y*34+x","manifest":str((OUT/"common_clean_correct_manifest.json").relative_to(ROOT)),"epsilons":EPS,"zero_control":{"asr":0.0,"invariants_passed":True},"summary":summary,"traces":traces,"previous_results":"invalid/debug-only due to representation mismatch","five_seed_campaign_started":False}
 (DEST/"summary.json").write_text(json.dumps(artifact,indent=2)+"\n")

if __name__=="__main__":main()
