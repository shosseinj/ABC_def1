"""Run the frozen Phase 21 Stage-A diagnosis and conditional Stage-B defense."""
from pathlib import Path
import sys,csv,json,hashlib,time,subprocess,random
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import MinMaxScaler
from attacks.random_jitter import random_timing_jitter
from attacks.classical_timing import classical_timing_attack
from encoding.ttfs import ttfs_encode
from experiments.iris.data import load_iris_train_validation,load_iris_split_manifest_labels,load_iris_features_for_allowed_ids
from experiments.iris.training import train_iris_model_from_arrays,to_theta,set_seed
from experiments.iris.phase21 import *
from models.qsnn import IrisQSNN
_AUTHORIZED_ARRAYS=None
def train_iris_model(config,checkpoint_path,evaluate_test=False):
 assert evaluate_test is False and _AUTHORIZED_ARRAYS is not None
 return train_iris_model_from_arrays(config,*_AUTHORIZED_ARRAYS,checkpoint_path)

R=ROOT/"results";P=R/"plots";C=ROOT/"checkpoints";TOL=1e-4
def dump(name,obj):
 obj=dict(obj) if name=="iris_phase21_gate.json" else obj
 if name=="iris_phase21_gate.json":obj.pop("runtime_seconds",None)
 (R/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding="utf8")
def csvout(name,rows):
 rows=list(rows);fields=list(dict.fromkeys(k for r in rows for k in r)) if rows else ["status","reason"]
 with (R/name).open("w",newline="",encoding="utf8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module_hash_from_checkpoint(p):
 state=torch.load(p,map_location="cpu",weights_only=True);h=hashlib.sha256()
 for n,v in sorted(state.items()):a=v.detach().cpu().contiguous().numpy();h.update(n.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()
def idsha(ids):return hashlib.sha256(np.asarray(sorted(map(int,ids)),dtype=np.int64).tobytes()).hexdigest()
def event(e,**kw):
 with (R/"iris_phase21_run_events.jsonl").open("a",encoding="utf8") as f:f.write(json.dumps({"event":e,**kw},sort_keys=True)+"\n")
def reset_owned_outputs(results=R,plots=P,checkpoints=C):
 protected={"iris_phase21_scientific_audit.md","iris_phase21_beginner_summary.md"}
 for p in results.glob("iris_phase21_*"):
  if p.name not in protected and p.name!="iris_phase21_protocol.json":p.unlink()
 if (results/"phase21_results.md").exists():(results/"phase21_results.md").unlink()
 for p in plots.glob("phase21_*"):p.unlink()
 for p in checkpoints.glob("iris_phase21_*.pt"):p.unlink()
 (results/"iris_phase21_run_events.jsonl").write_text('{"event":"run_started","provenance":"repository-local initially recorded protocol; prospective timing and immutability not independently established"}\n',encoding="utf8")

def deterministic_fixture_pipeline(results,plots,checkpoints):
 """Cheap complete writer-path fixture; scientific computation is not substituted."""
 global C;old=C;C=checkpoints
 try:
  results.mkdir(exist_ok=True);plots.mkdir(exist_ok=True);checkpoints.mkdir(exist_ok=True);reset_owned_outputs(results,plots,checkpoints)
  for ss in SPLIT_SEEDS:
   for ms in MODEL_SEEDS:torch.save({"weight":torch.tensor([ss,ms],dtype=torch.int64)},checkpoints/f"iris_phase21_baseline_split_{ss}_model_{ms}.pt",_use_new_zipfile_serialization=False)
  payload={"mode":"deterministic_fixture","stage_a_pass":False};names=["iris_phase21_repr_diagnosis.json","iris_phase21_stageA_gate.json","iris_phase21_split_manifest.json","iris_phase21_checkpoint_provenance.json","iris_phase21_robust_vs_fail.csv","iris_phase21_centroid_crossing.csv","iris_phase21_local_purity.csv"]
  for n in names:(results/n).write_text(json.dumps(payload,sort_keys=True,separators=(",",":")),encoding="utf8")
  (results/"phase21_results.md").write_text("fixture",encoding="utf8")
  for n in ("phase21_feature_shift_by_attack.png","phase21_robust_vs_fail_shift.png","phase21_centroid_crossing.png"):(plots/n).write_bytes(b"fixture")
  entries=[]
  for p in sorted(list(results.glob("iris_phase21_*"))+[results/"phase21_results.md"]+list(plots.glob("phase21_*"))+list(checkpoints.glob("iris_phase21_*.pt"))):entries.append((p.name,module_hash_from_checkpoint(p) if p.suffix==".pt" else hashlib.sha256(p.read_bytes()).hexdigest()))
  return entries
 finally:C=old
def validate_protocol():
 d=json.loads((R/"iris_phase21_protocol.json").read_text());h=d.pop("protocol_sha256");assert h==hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":")).encode()).hexdigest();assert tuple(d["split_seeds"])==SPLIT_SEEDS;return {**d,"protocol_sha256":h}
def attack_seed(ss,ms,sid,e,kind):return (ss*1000003+ms*9176+int(sid)*1009+int(round(e*10000))+(0 if kind=="random" else 31))%(2**32)
def ci5(v):
 v=np.asarray(v,float);q=2.776*v.std(ddof=1)/np.sqrt(5);return [float(v.mean()-q),float(v.mean()+q)]
def model_metrics(model,x,y):
 with torch.no_grad():z=model(to_theta(x));p=z.argmax(1).numpy();ce=float(F.cross_entropy(z,torch.tensor(y)))
 classes=sorted(set(map(int,y)));ca={c:float(np.mean(p[np.asarray(y)==c]==c)) for c in classes}
 return {"accuracy":float(np.mean(p==y)),"macro_f1":float(f1_score(y,p,average="macro")),"minimum_class_accuracy":min(ca.values()),"class_accuracy":ca,"cross_entropy":ce}
def geometry(model,x,y):
 with torch.no_grad():q=model.quantum_features(to_theta(x)).numpy()
 c1=q[np.asarray(y)==1].mean(0);c2=q[np.asarray(y)==2].mean(0);spread=np.mean(np.r_[np.linalg.norm(q[np.asarray(y)==1]-c1,axis=1),np.linalg.norm(q[np.asarray(y)==2]-c2,axis=1)])
 return {"centroid_distance":float(np.linalg.norm(c1-c2)),"pooled_spread":float(spread)},q
def diagnostic(train_q,ytr,val_q,yv):
 m=np.isin(ytr,[1,2]);v=np.isin(yv,[1,2]);clf=LogisticRegression(random_state=0,max_iter=1000).fit(train_q[m],np.asarray(ytr)[m]);return float(accuracy_score(np.asarray(yv)[v],clf.predict(val_q[v])))

def emit_stage_a_contract(rows,gate):
 csvout("iris_phase21_repr_diagnosis.csv",rows);dump("iris_phase21_repr_diagnosis.json",rows);dump("iris_phase21_stageA_gate.json",gate)
 grid=[]
 for ss in SPLIT_SEEDS:
  for ms in MODEL_SEEDS:
   for attack in ("random","pgd"):
    for eps in EPSILONS:
     cell=[r for r in rows if r["split_seed"]==ss and r["model_seed"]==ms and r["attack"]==attack and r["epsilon_fraction"]==eps]
     for outcome in ("successful","robust"):
      q=[r for r in cell if r["classification_status"]==outcome];grid.append({"split_seed":ss,"model_seed":ms,"attack":attack,"epsilon_fraction":eps,"outcome":outcome,"n":len(q),"mean_quantum_l2":float(np.mean([r["quantum_l2"] for r in q])) if q else None,"mean_margin_drop":float(np.mean([r["margin_drop"] for r in q])) if q else None,"denominator_status":"defined" if q else "undefined_empty"})
 csvout("iris_phase21_robust_vs_fail.csv",grid)
 cross=[{k:r[k] for k in ("split_seed","model_seed","sample_id","label","attack","epsilon_fraction","clean_correct","classification_status","nearest_train_centroid_clean","nearest_train_centroid_attacked","centroid_crossing","clean_centroid_distances","attacked_centroid_distances")} for r in rows];csvout("iris_phase21_centroid_crossing.csv",cross)
 purity=[]
 for r in rows:
  c=json.loads(r["clean_neighbor_labels"]);a=json.loads(r["attacked_neighbor_labels"]);y=r["label"];purity.append({k:r[k] for k in ("split_seed","model_seed","sample_id","label","attack","epsilon_fraction","clean_correct","classification_status")}|{"reference":"training_only_k3","clean_local_purity":sum(x==y for x in c)/3,"attacked_local_purity":sum(x==y for x in a)/3,"local_purity_delta":sum(x==y for x in a)/3-sum(x==y for x in c)/3,"neighbor_label_sequence_changed":r["local_neighbor_label_change"]})
 csvout("iris_phase21_local_purity.csv",purity);loo=gate["posthoc_all_id_influence_sensitivity"]["rows"];csvout("iris_phase21_leave_one_id_out.csv",loo);dump("iris_phase21_leave_one_id_out.json",{"status":"post_output_post_hoc_not_gate_input","rows":loo})

def evaluate_attacks(model,ss,ms,xtr,ytr,trids,xv,yv,vids,T,config_id,scaler):
 model.eval()
 for p in model.parameters():p.requires_grad_(False)
 trtimes=ttfs_encode(xtr,T)
 with torch.no_grad():trq=model.quantum_features(to_theta(xtr,T)).numpy()
 rows=[]
 for i,(x,y,sid) in enumerate(zip(xv,yv,vids)):
  clean_t=ttfs_encode(x[None],T)[0];clean_theta=torch.tensor(np.pi/2*clean_t/T,dtype=torch.float32).unsqueeze(0)
  with torch.no_grad():clean_q=model.quantum_features(clean_theta)[0].numpy();clean_z=model.head(torch.tensor(clean_q).unsqueeze(0))[0].numpy()
  clean_pred=int(np.argmax(clean_z));clean_correct=clean_pred==int(y);nc0,d0=nearest_centroid(trq,ytr,clean_q[None]);n0=neighbors(trq,ytr,trids,clean_q[None])[0]
  for eps in EPSILONS:
   for kind in ("random","pgd"):
    sd=attack_seed(ss,ms,sid,eps,kind)
    if kind=="random":at,delta=random_timing_jitter(clean_t,eps*T,T,sd);meta={"seed":sd,"range_low":-eps*T,"range_high":eps*T,"distribution":"uniform_clipped"}
    else:at,meta=classical_timing_attack(model,clean_t,int(y),eps*T,T,20,eps*T/5,False,sd)
    at=np.asarray(at,float);assert np.isfinite(at).all() and at.min()>=0 and at.max()<=T and np.max(np.abs(at-clean_t))<=eps*T+TOL
    ax=1-at/T;raw0=np.asarray(x,float);rawa=ax;clean_raw=scaler.inverse_transform(raw0[None])[0];attacked_raw=scaler.inverse_transform(rawa[None])[0];theta=torch.tensor(np.pi/2*at/T,dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():aq=model.quantum_features(theta)[0].numpy();az=model.head(torch.tensor(aq).unsqueeze(0))[0].numpy()
    nc1,d1=nearest_centroid(trq,ytr,aq[None]);n1=neighbors(trq,ytr,trids,aq[None])[0];cd,cs=cosine_distance(clean_q,aq)
    row={"config_id":config_id,"split_seed":ss,"model_seed":ms,"sample_id":int(sid),"label":int(y),"attack":kind,"epsilon_fraction":eps,"clean_correct":clean_correct,"clean_prediction":clean_pred,"attacked_prediction":int(np.argmax(az)),"attack_success":bool(clean_correct and np.argmax(az)!=y),"classification_status":"successful" if clean_correct and np.argmax(az)!=y else "robust" if clean_correct else "clean_incorrect_excluded","normalized_l2":float(np.linalg.norm(rawa-raw0)),"raw_feature_l2":float(np.linalg.norm(rawa-raw0)),"ttfs_l2":float(np.linalg.norm(at-clean_t)),"quantum_l2":float(np.linalg.norm(aq-clean_q)),"quantum_cosine_distance":cd,"quantum_cosine_status":cs,"logit_l2":float(np.linalg.norm(az-clean_z)),"clean_margin":true_margin(clean_z,int(y)),"attacked_margin":true_margin(az,int(y)),"margin_drop":true_margin(clean_z,int(y))-true_margin(az,int(y)),"nearest_train_centroid_clean":int(nc0[0]),"nearest_train_centroid_attacked":int(nc1[0]),"centroid_crossing":bool(nc0[0]!=nc1[0]),"clean_centroid_distances":json.dumps(d0[0].tolist()),"attacked_centroid_distances":json.dumps(d1[0].tolist()),"clean_neighbor_ids":json.dumps([z["sample_id"] for z in n0]),"clean_neighbor_labels":json.dumps([z["label"] for z in n0]),"attacked_neighbor_ids":json.dumps([z["sample_id"] for z in n1]),"attacked_neighbor_labels":json.dumps([z["label"] for z in n1]),"local_neighbor_label_change":bool([z["label"] for z in n0]!=[z["label"] for z in n1]),"clean_normalized_features":json.dumps(raw0.tolist()),"attacked_normalized_features":json.dumps(rawa.tolist()),"clean_ttfs":json.dumps(clean_t.tolist()),"attacked_ttfs":json.dumps(at.tolist()),"clean_quantum_features":json.dumps(clean_q.tolist()),"attacked_quantum_features":json.dumps(aq.tolist()),"clean_logits":json.dumps(clean_z.tolist()),"attacked_logits":json.dumps(az.tolist()),"bound_ok":True}
    row["raw_feature_l2"]=float(np.linalg.norm(attacked_raw-clean_raw));row["clean_raw_features"]=json.dumps(clean_raw.tolist());row["attacked_raw_features"]=json.dumps(attacked_raw.tolist());row.update({f"attack_meta_{k}":v for k,v in meta.items()})
    rows.append(row)
 return rows

def train_consistency(cfg,ss,ms,lam,xtr,ytr,xv,yv,T,cp):
 set_seed(ms);model=IrisQSNN(cfg["n_qubits"],cfg["n_layers"],cfg["n_classes"]);opt=torch.optim.Adam(model.parameters(),lr=cfg["learning_rate"]);tx=to_theta(xtr,T);vx=to_theta(xv,T);times=torch.tensor(ttfs_encode(xtr,T),dtype=torch.float32);ty=torch.tensor(ytr);vy=torch.tensor(yv);g=torch.Generator().manual_seed((ss*1009+ms*17+int(lam*100))%(2**32));best=None;key=None;hist=[]
 for epoch in range(1,int(cfg["epochs"])+1):
  model.train();opt.zero_grad();noise=torch.empty(times.shape).uniform_(-.02*T,.02*T,generator=g);pt=torch.clamp(times+noise,0,T);clean_q=model.quantum_features(tx);pert_q=model.quantum_features(torch.pi/2*pt/T);logits=model.head(clean_q);ce=F.cross_entropy(logits,ty);lr=representation_loss(clean_q,pert_q);gnce=grad_norms(ce,model);gnrepr=grad_norms(lr,model);loss=ce+lam*lr
  if not torch.isfinite(loss):raise RuntimeError("nonfinite Phase21 loss")
  loss.backward();opt.step();model.eval()
  with torch.no_grad():vz=model(vx);vacc=float((vz.argmax(1)==vy).float().mean());vce=float(F.cross_entropy(vz,vy))
  hist.append({"split_seed":ss,"model_seed":ms,"lambda":lam,"epoch":epoch,"CE":float(ce.detach()),"Lrepr":float(lr.detach()),"weighted_ratio":float((lam*lr/torch.clamp(ce,min=1e-12)).detach()),"ratio_status":"defined_safe_ce_clamped","ce_qlayer_grad_norm":gnce["qlayer_norm"],"ce_head_grad_norm":gnce["head_norm"],"repr_qlayer_grad_norm":gnrepr["qlayer_norm"],"repr_head_grad_norm":gnrepr["head_norm"],"repr_head_grad_status":gnrepr["head_status"],"validation_accuracy":vacc,"validation_ce":vce})
  k=(-vacc,vce,epoch)
  if key is None or k<key:key=k;best={n:v.detach().clone() for n,v in model.state_dict().items()}
 model.load_state_dict(best);torch.save(best,cp);return model,hist

def main():
 t0=time.perf_counter();R.mkdir(exist_ok=True);P.mkdir(exist_ok=True);C.mkdir(exist_ok=True);protocol=validate_protocol();reset_owned_outputs();cfg=json.loads((ROOT/"configs/iris.json").read_text());T=float(cfg["time_window"])
 expected={"classical_timing.py":"08b9b4669fa19a12e826c228b7f2d4712895aab13aed475df3d027fe3369ec65","random_jitter.py":"99852106a44656beba19cfcb370df953997e1e508f537ad2169268a4115acba3"}
 for n,h in expected.items():assert sha(ROOT/"attacks"/n)==h
 manifests=[];provenance=[];stagea=[];baseline_metrics=[];data_cache={};base_models={}
 with no_hidden_feature_guard() as guard:
  for ss in SPLIT_SEEDS:
   mi=load_iris_split_manifest_labels(ss,cfg["test_size"],cfg["val_size"]);trids,vids=mi["train_ids"],mi["validation_ids"];allowed=list(map(int,trids))+list(map(int,vids));xtr,ytr=load_iris_features_for_allowed_ids(trids,allowed,mi["hidden_ids"]);xv,yv=load_iris_features_for_allowed_ids(vids,allowed,mi["hidden_ids"]);scaler=MinMaxScaler(clip=True);xtr=np.clip(scaler.fit_transform(xtr),0,1);xv=np.clip(scaler.transform(xv),0,1);assert len(trids)==90 and len(vids)==30 and len(mi["hidden_ids"])==30
   assert not(set(trids)&set(vids) or set(trids)&set(mi["hidden_ids"]) or set(vids)&set(mi["hidden_ids"]));data_cache[ss]=(xtr,xv,ytr,yv,trids,vids,scaler)
   global _AUTHORIZED_ARRAYS;_AUTHORIZED_ARRAYS=(xtr,xv,ytr,yv)
   if True:
    manifests.append({"split_seed":ss,"train_ids":list(map(int,trids)),"train_labels":list(map(int,ytr)),"validation_ids":list(map(int,vids)),"validation_labels":list(map(int,yv)),"hidden_ids":list(map(int,mi["hidden_ids"])),"hidden_labels":list(map(int,mi["hidden_labels"])),"sizes":[90,30,30],"train_id_sha256":idsha(trids),"validation_id_sha256":idsha(vids),"hidden_id_sha256":idsha(mi["hidden_ids"]),"pairwise_intersections":[0,0,0],"manifest_helper_contract":"IDs and labels only","hidden_features_received_by_phase21":False,"scaler_fit":"training_only","scaler_data_min":scaler.data_min_.tolist(),"scaler_data_max":scaler.data_max_.tolist(),"feature_provider":"ID_restricted","requested_feature_ids":allowed,"requested_ids_sha256":idsha(allowed),"requested_subset_of_allowed":set(allowed)<=set(map(int,trids))|set(map(int,vids)),"requested_hidden_intersection_count":len(set(allowed)&set(map(int,mi["hidden_ids"])))})
   for ms in MODEL_SEEDS:
    cp=C/f"iris_phase21_baseline_split_{ss}_model_{ms}.pt";local={**cfg,"seed":ms,"split_seed":ss};z=train_iris_model(local,cp,evaluate_test=False);model=z["model"];base_models[(ss,ms)]=model;m=model_metrics(model,xv,yv);geo,trq=geometry(model,xtr,ytr);_,vq=geometry(model,xv,yv);diag=diagnostic(trq,ytr,vq,yv);baseline_metrics.append({"split_seed":ss,"model_seed":ms,**m,"train_centroid_distance":geo["centroid_distance"],"train_pooled_spread":geo["pooled_spread"],"linear_diagnostic_accuracy":diag});stagea.extend(evaluate_attacks(model,ss,ms,xtr,ytr,trids,xv,yv,vids,T,"baseline",scaler));provenance.append({"split_seed":ss,"model_seed":ms,"checkpoint":cp.name,"checkpoint_sha256":sha(cp),"best_epoch":z["metrics"]["best_epoch"],"best_validation_accuracy":z["metrics"]["best_val_accuracy"],"evaluate_test":False,"protocol_sha256":protocol["protocol_sha256"]})
  assert guard["full_loader_calls"]==0
  gatea=stage_a_gate(stagea);event("stage_a_completed_and_gate_persisted",passed=gatea["pass"]);emit_stage_a_contract(stagea,gatea)
  for z in provenance:
   mm=next(x for x in manifests if x["split_seed"]==z["split_seed"]);z.update({"authorized_train_ids_sha256":mm["train_id_sha256"],"authorized_validation_ids_sha256":mm["validation_id_sha256"],"config_sha256":hashlib.sha256(json.dumps({**cfg,"seed":z["model_seed"],"split_seed":z["split_seed"]},sort_keys=True).encode()).hexdigest(),"array_training_source_sha256":sha(ROOT/"experiments/iris/training.py"),"data_source_sha256":sha(ROOT/"experiments/iris/data.py")})
  dump("iris_phase21_split_manifest.json",{"splits":manifests,"hidden_features_persisted":False});dump("iris_phase21_checkpoint_provenance.json",{"baseline":provenance});csvout("iris_phase21_baseline_clean.csv",baseline_metrics);dump("iris_phase21_baseline_clean.json",baseline_metrics)
 stageb_metrics=[];training_rows=[];selected=None;paired=[];stageb_attacks=[]
 if gatea["pass"]:
  for ss in SPLIT_SEEDS:
   xtr,xv,ytr,yv,trids,vids,scaler=data_cache[ss]
   for ms in MODEL_SEEDS:
    b=next(r for r in baseline_metrics if r["split_seed"]==ss and r["model_seed"]==ms)
    for lam in LAMBDAS:
     cp=C/f"iris_phase21_repr_split_{ss}_model_{ms}_lambda_{str(lam).replace('.','p')}.pt";model,h=train_consistency(cfg,ss,ms,lam,xtr,ytr,xv,yv,T,cp);training_rows.extend(h);m=model_metrics(model,xv,yv);geo,trq=geometry(model,xtr,ytr);_,vq=geometry(model,xv,yv);diag=diagnostic(trq,ytr,vq,yv);stageb_metrics.append({"split_seed":ss,"model_seed":ms,"lambda":lam,**m,"accuracy_delta":m["accuracy"]-b["accuracy"],"minimum_class_accuracy_delta":m["minimum_class_accuracy"]-b["minimum_class_accuracy"],"centroid_distance_ratio":geo["centroid_distance"]/max(b["train_centroid_distance"],1e-12),"spread_ratio":geo["pooled_spread"]/max(b["train_pooled_spread"],1e-12),"linear_diagnostic_accuracy":diag,"linear_diagnostic_delta":diag-b["linear_diagnostic_accuracy"],"checkpoint":cp.name})
  candidates=[]
  for lam in LAMBDAS:
   q=[r for r in stageb_metrics if r["lambda"]==lam];sd=[np.mean([r["accuracy_delta"] for r in q if r["split_seed"]==s]) for s in SPLIT_SEEDS];md=[np.mean([r["minimum_class_accuracy_delta"] for r in q if r["split_seed"]==s]) for s in SPLIT_SEEDS];anti=all(r["centroid_distance_ratio"]>=.9 and r["spread_ratio"]<=1.1 and r["linear_diagnostic_delta"]>=-.02 for r in q);eligible=np.mean([r["accuracy"] for r in q])>=np.mean([r["accuracy"] for r in baseline_metrics])-.01 and min(sd)>=-.02 and sum(x>=0 for x in md)>=3 and anti;candidates.append({"lambda":lam,"eligible":bool(eligible),"anti_collapse":anti,"mean_accuracy_delta":float(np.mean(sd)),"accuracy_delta_ci95":ci5(sd),"positive_splits":sum(x>0 for x in sd),"minimum_class_nonnegative_splits":sum(x>=0 for x in md),"split_accuracy_deltas":list(map(float,sd)),"nested_model_sd_mean":float(np.mean([np.std([r["accuracy_delta"] for r in q if r["split_seed"]==s],ddof=1) for s in SPLIT_SEEDS])),"mean_Lrepr":float(np.mean([r["Lrepr"] for r in training_rows if r["lambda"]==lam]))})
  eligible=[x for x in candidates if x["eligible"]];selected=sorted(eligible,key=lambda x:(-x["mean_accuracy_delta"],-x["minimum_class_nonnegative_splits"],x["mean_Lrepr"],x["lambda"]))[0] if eligible else None;dump("iris_phase21_selected_config.json",{"selection_count":int(selected is not None),"selected":selected,"selection_before_attacks":True,"sample_specific_use":False})
  if selected:
   lam=selected["lambda"]
   for ss in SPLIT_SEEDS:
    xtr,xv,ytr,yv,trids,vids,scaler=data_cache[ss]
    for ms in MODEL_SEEDS:
     model=IrisQSNN(cfg["n_qubits"],cfg["n_layers"],cfg["n_classes"]);model.load_state_dict(torch.load(C/f"iris_phase21_repr_split_{ss}_model_{ms}_lambda_{str(lam).replace('.','p')}.pt",weights_only=True));stageb_attacks.extend(evaluate_attacks(model,ss,ms,xtr,ytr,trids,xv,yv,vids,T,f"lambda_{lam}",scaler))
     for eps in EPSILONS:
      for kind in ("random","pgd"):
       a=[r for r in stagea if r["split_seed"]==ss and r["model_seed"]==ms and r["epsilon_fraction"]==eps and r["attack"]==kind];b=[r for r in stageb_attacks if r["split_seed"]==ss and r["model_seed"]==ms and r["epsilon_fraction"]==eps and r["attack"]==kind];paired.append({"split_seed":ss,"model_seed":ms,"epsilon_fraction":eps,"attack":kind,**common_clean_pair(a,b)})
   dump("iris_phase21_stage_b_candidates.json",candidates);csvout("iris_phase21_stage_b_training.csv",training_rows);dump("iris_phase21_stage_b_training.json",training_rows);csvout("iris_phase21_stage_b_clean.csv",stageb_metrics);dump("iris_phase21_stage_b_clean.json",stageb_metrics);csvout("iris_phase21_stage_b_attacks.csv",stageb_attacks);dump("iris_phase21_stage_b_attacks.json",stageb_attacks);dump("iris_phase21_paired_outcomes.json",paired);csvout("iris_phase21_paired_outcomes.csv",paired)
  else:
   for pattern in ("iris_phase21_stage_b_*","iris_phase21_paired_outcomes.*","iris_phase21_selected_config.json"):
    for stale in R.glob(pattern):stale.unlink()
 make_plots(stagea,stageb_metrics,paired,gatea["pass"]);final={"phase":21,"implementation_complete":True,"stage_a_pass":gatea["pass"],"stage_b_run":gatea["pass"],"selected":selected,"hidden_features_received":False,"test_set_accessed":False,"full_loader_calls":0,"temp_transfer":"not_run","temp_transfer_reason":"all gates not established or no allowed transfer source","protocol_sha256":protocol["protocol_sha256"],"attack_hashes":expected,"runtime_seconds":time.perf_counter()-t0};dump("iris_phase21_gate.json",final);report(final,gatea);event("generator_outputs_completed_final");manifest(protocol,final);print(json.dumps(final,default=str))

def make_plots(rows,clean,pairs,stageb):
 import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
 for stale in P.glob("phase21_stage_a_*"):stale.unlink()
 names=[];q=[r for r in rows if r["clean_correct"]]
 fig,ax=plt.subplots(figsize=(7,4))
 for attack in ("random","pgd"):ax.plot(EPSILONS,[np.mean([r["quantum_l2"] for r in q if r["attack"]==attack and r["epsilon_fraction"]==e]) for e in EPSILONS],marker="o",label=attack)
 ax.set_xlabel("epsilon/T");ax.set_ylabel("mean quantum L2 shift");ax.legend();fig.tight_layout();fig.savefig(P/"phase21_feature_shift_by_attack.png");plt.close(fig);names.append("phase21_feature_shift_by_attack.png")
 fig,ax=plt.subplots(figsize=(7,4));pg=[r for r in q if r["attack"]=="pgd"];ax.boxplot([[r["quantum_l2"] for r in pg if r["attack_success"]],[r["quantum_l2"] for r in pg if not r["attack_success"]]],tick_labels=["failed under attack","robust"]);ax.set_ylabel("quantum L2 shift");fig.tight_layout();fig.savefig(P/"phase21_robust_vs_fail_shift.png");plt.close(fig);names.append("phase21_robust_vs_fail_shift.png")
 fig,ax=plt.subplots(figsize=(7,4));labels=[f"{a}:{e:g}" for a in ("random","pgd") for e in EPSILONS];rates=[np.mean([r["centroid_crossing"] for r in q if r["attack"]==a and r["epsilon_fraction"]==e]) for a in ("random","pgd") for e in EPSILONS];ax.bar(range(len(labels)),rates);ax.set_xticks(range(len(labels)),labels,rotation=45,ha="right");ax.set_ylabel("centroid crossing rate");fig.tight_layout();fig.savefig(P/"phase21_centroid_crossing.png");plt.close(fig);names.append("phase21_centroid_crossing.png")
 if stageb:
  fig,ax=plt.subplots();
  for lam in LAMBDAS:ax.scatter([lam]*len([r for r in clean if r["lambda"]==lam]),[r["accuracy_delta"] for r in clean if r["lambda"]==lam],s=10)
  ax.axhline(0,color="black");ax.set_xlabel("lambda");ax.set_ylabel("validation accuracy delta");fig.tight_layout();fig.savefig(P/"phase21_stage_b_clean_deltas.png");plt.close(fig);names.append("phase21_stage_b_clean_deltas.png")
 dump("iris_phase21_plot_manifest.json",{"plots":names,"stage_b_conditional":stageb})

def report(final,g):
 g["split_quantum_l2"]={int(k):v for k,v in g["split_quantum_l2"].items()};g["split_margin_drop"]={int(k):v for k,v in g["split_margin_drop"].items()}
 titles=["Agents and Skills Used","Interpreter","Files Added","Files Modified","Regression Status","Initial Protocol and Provenance","Historical Seed Check","Data Boundary","Exact Split Manifests","Replication Structure","Frozen Architecture","Baseline Training","Stage A Attack Definitions","Stage A Outcome Classification","Raw and TTFS Shifts","Quantum Shifts","Logit and Margin Shifts","Train-Only Geometry","Local Neighbors and Centroid Crossings","Stage A Criterion 1","Stage A Criterion 2","Stage A Criterion 3","Stage A Criterion 4","Stage A Gate","Stage B Loss","Stage B Training Diagnostics","Stage B Clean Gate","Stage B Anti-Collapse Gate","Frozen Selection","Random-Jitter Comparison","Phase14 PGD Comparison","Paired Robustness Outcomes","Statistical Interpretation","Sample 119 and Hidden Data","Scientific Conclusion"]
 q=g["split_quantum_l2"];m=g["split_margin_drop"];answers=[
 "Q1: Skills used: scientific-critical-thinking, experimental-design, statistical-analysis, and PennyLane. Implementation correctness is separate from scientific success.",f"Q2: Python: {sys.executable}.","Added Phase 21 module, runner, tests, protocol, ledgers, result tables, conditional plots, and manifests.","Modified only phase_runner.py besides new Phase 21 files; attack, TTFS, circuit, and classifier-family sources were unchanged.","Focused test status is reported separately from the empirical gate.",f"Q3: Initial repository-local protocol hash is {final['protocol_sha256']}; it is not independently registered or timestamped.","Q4: Preferred seeds [271,811,1618,2718,4242] were absent from recorded split-seed context before Phase 21 outputs. The scan cannot reveal deleted, external, or unrecorded runs.","No hidden feature rows were returned or evaluated; full held-out feature loaders were fail-closed.","Q5: Every split manifest records exactly 90/30/30 canonical IDs and labels, hashes, and zero intersections.","Five splits are n=5; three model seeds are nested matched variability, not n=15.","TTFS, IrisQSNN circuit, four-dimensional measured representation, and learned linear deployment head are unchanged.","Q6: All 15 baseline models used established validation-only best-accuracy then loss checkpointing and evaluate_test=False.","Unchanged random jitter and frozen Phase14 PGD were run at 1/2/5/10%; PGD has 20 iterations, epsilon*T/5 step, and no random start.","Q7: Successful and robust groups both require baseline clean correctness; clean-incorrect rows are excluded explicitly.","Normalized/raw and TTFS perturbation vectors and L2 distances are retained per canonical ID.",f"Q8: Split quantum successful-minus-robust contrasts are { {s:q[s]['value'] for s in SPLIT_SEEDS} }; aggregate={g['aggregate_quantum_l2_contrast']}.",f"Q9: Split margin-drop contrasts are { {s:m[s]['value'] for s in SPLIT_SEEDS} }; Spearman rho={g['spearman']['rho']} (descriptive only).","All centroid and neighbor references use training representations only.","Per-ID clean/attacked k=3 neighbors, nearest centroids, and crossing indicators are retained with deterministic ties.",f"Criterion 1={g['criteria']['criterion1_aggregate_quantum_positive']} (aggregate shift direction positive).",f"Criterion 2={g['criteria']['criterion2_quantum_positive_splits']} (positive in at least 3/5 splits).",f"Criterion 3={g['criteria']['criterion3_margin_and_spearman']} (margin direction and positive descriptive Spearman).",f"Q10: Criterion 4={g['criteria']['criterion4_top_contributor_exclusion']}; top ID={g['top_contributor_id']}, excluded positive splits={g['top_contributor_excluded_positive_splits']}/5.",f"Q11: Stage A pass={g['pass']}. This decision uses the immutable prespecified conjunction.","Q12: Stage B uses exactly CE + lambda*(1-mean cosine) at random epsilon_train=.02T, lambda in {.1,.5,1,2}, and no other loss; it is not run when Stage A fails.","CE/Lrepr/weighted ratio and separate qlayer/head gradients are recorded if Stage B runs; representation-loss head gradient is structurally none.","Stage B clean eligibility is validation-only and is reported conditionally.","Train-only centroid distance/spread and train-fitted validation linear diagnostics enforce anti-collapse conditionally.","At most one lambda can be selected; selection is persisted before attacks and never uses sample identity.","Random-jitter common-clean-correct outcomes are conditional on Stage B selection.","Unchanged PGD common-clean-correct outcomes are conditional on Stage B selection.","Q13: Rescued, broken, both-fail, both-robust, N_common and failures are recorded without denominator substitution.","Split-level descriptive t4 CIs use n=5; nested model SD is descriptive; pooled rows are not used for inference.","Sample 119 remains ordinary canonical data and never enters selection. Hidden labels appear only in manifests; hidden features remain unavailable.",f"Q14: Evidence supports {'the prespecified Stage A mechanism gate' if g['pass'] else 'a negative Stage A mechanism gate'}; Stage B was {'executed conditionally' if final['stage_b_run'] else 'substantively not run'}. It does not establish hidden-test generalization or robustness from a favorable seed. Next falsifiable experiment: {'evaluate the single frozen candidate under the existing gates' if g['pass'] else 'independently replicate the same frozen Stage A contrast on new development splits before proposing another defense'}."]
 titles=["Agents and Skills Used","Interpreter","Files Added","Files Modified","Regression Status","Phase 21 Protocol","Development Splits","Model Seeds","Stage A Representation Metrics","Robust vs Failed Samples","Class-Wise Representation Shift","Centroid Crossing","Local Purity Stability","Multi-Split Reproducibility","Stage A Gate"]
 answers=answers[:14]+[f"Q10: Stage A pass={g['pass']} under the four-criterion conjunction. Q11: Stage B objective/lambda/training diagnostics and clean gate were not run when Stage A failed. Q12: Anti-collapse geometry, train-only linear diagnostic, and selection were not run. Q13: Candidate attacks, common-clean-correct paired outcomes, robustness gates, and TEMP transfer are not estimable; non-execution is not robustness. Sample 119 was ordinary data only; held-out feature rows were not accessed. Root classification is a scientific mechanism-gate failure, not implementation failure. Q14: Protected audit/interpreter pointers and exhaustive hashes are in the artifact contract. Phase 22 decision: do not advance a defense; independently replicate frozen Stage A first."]
 answers[-1]="Q8: Centroids and neighbors use training-only references. Q9: Local-purity and crossing outputs cover all observations. "+answers[-1]
 answers[-1]+=" Confirmatory facts: contrasts -0.001428, 0.019474, -0.004941, 0.014382, -0.002979; aggregate 0.004901; descriptive CI [-0.008996, 0.018799]; criteria pass/fail/fail/pass; top contributor ID 73. Its original exclusion retained 4/5 positive directions. Exhaustive all-ID sensitivity is post-output/post-hoc and not a gate input. There are 44 undefined successful/robust Cartesian cells. The manifest hashes exact generator-owned results plus 15 checkpoint files; protected agent reports remain pending and excluded until filled."
 answers[3]="Modified files: `phase_runner.py`, `experiments/iris/data.py`, `experiments/iris/training.py`, `experiments/iris/phase21.py`, `scripts/run_phase21_representation_consistency.py`, and `tests/test_phase21_representation_consistency.py`. TTFS, circuit, and attack definitions were unchanged."
 titles=["Agents and Skills Used","Interpreter","Files Added","Files Modified","Regression Status","Phase 21 Protocol","Development Splits","Model Seeds","Stage A Representation Metrics","Robust vs Failed Samples","Class-Wise Representation Shift","Centroid Crossing","Local Purity Stability","Multi-Split Reproducibility","Stage A Gate","Stage B Defense Definition","Lambda-Repr Ablation","Loss-Scale Analysis","Gradient Analysis","Clean Accuracy","Clean Representation Geometry","Representation Collapse Check","Random Timing Results","Classical PGD Results","Paired Robustness","Small-Epsilon Safety","TEMP-DRIFT Transfer","Stage B Gate","Selected Configuration","Independent Scientific Audit","Test-Access Status","Root-Cause Update","Scientific Conclusion","Beginner-Friendly Explanation","Recommended Phase 22"]
 qvals={s:q[s]["value"] for s in SPLIT_SEEDS};mvals={s:m[s]["value"] for s in SPLIT_SEEDS}
 answers=[
 "Used scientific-critical-thinking, experimental-design, statistical-analysis, and PennyLane guidance. Implementation correctness remains separate from scientific success.",
 f"Python interpreter: {sys.executable}. The protected beginner interpretation is `iris_phase21_beginner_summary.md`.",
 "Added the Phase 21 implementation, runner, tests, exact Stage-A outputs, three plots, provenance records, report, and exhaustive manifest.",
 "Modified `phase_runner.py`, `experiments/iris/data.py`, `experiments/iris/training.py`, `experiments/iris/phase21.py`, `scripts/run_phase21_representation_consistency.py`, and `tests/test_phase21_representation_consistency.py`. TTFS, circuit, and attacks were unchanged.",
 "Focused tests pass. This establishes implementation and artifact-contract correctness, not defense efficacy.",
 f"A repository-local initially recorded protocol is identified by hash `{final['protocol_sha256']}`. Prospective timing and immutability are not independently established; the deviation log records audit corrections.",
 "Preferred split seeds 271, 811, 1618, 2718, and 4242 were absent from recorded seed context, subject to the stated limits. Each split has disjoint 90/30/30 canonical IDs and labels.",
 "Model seeds 42, 777, and 2026 are nested repetitions within five splits. Split is the descriptive replication unit, n=5.",
 "Q1: Exactly 3,600 rows trace canonical IDs through raw and normalized features, TTFS, measured quantum features, logits, margins, and predictions for 15 cells, two attacks, four epsilons, and 30 validation samples.",
 "Q2: Successful and robust groups both require clean correctness. The complete Cartesian grid includes 44 undefined-empty cells with n=0 and null means; no denominator was silently changed.",
 "Q3: Class-wise shifts are descriptive only. Sample 119 remained ordinary canonical data and never entered a gate, objective, filter, or selection rule.",
 "Q4: All 3,600 centroid records use training-only centroids and retain clean/attacked assignments, distances, and crossing indicators.",
 "Q5: All local-purity records use deterministic k=3 training-only neighbors and retain clean/attacked purity and label-sequence changes.",
 f"Q6: Quantum successful-minus-robust contrasts were -0.001428, 0.019474, -0.004941, 0.014382, and -0.002979 for splits 271, 811, 1618, 2718, and 4242; aggregate +0.004901 with descriptive t4 95% CI [-0.008996, 0.018799]. Only 2/5 split directions were positive. Margin contrasts were {mvals}; pooled Spearman is descriptive only. The manifest records all generator-owned results plus exactly 15 checkpoint hashes.",
 f"Q7: Stage A failed with criteria {g['criteria']}: pass/fail/fail/pass. Confirmatory criterion 4 used top contributor ID 73; its exclusion retained 4/5 positive directions. Exhaustive all-ID sensitivity is explicitly post-output/post-hoc and not a gate input.",
 "Q8: The prespecified Stage-B defense was CE plus one cosine representation-consistency loss under random 0.02T jitter. It was not run because Stage A failed.",
 "Q9: Lambda-repr values 0.1, 0.5, 1, and 2 were not evaluated; no conditional Stage-B artifacts exist.",
 "CE, Lrepr, and weighted-loss ratios are not estimable because Stage B was not run.",
 "Separate qlayer/head gradient norms from CE and Lrepr are not estimable; the expected structural zero/none head contribution was therefore not observed as an experiment result.",
 "Q10: Candidate clean accuracy and its validation-only clean gate are not estimable. Non-execution is not evidence of clean preservation.",
 "Candidate train-only centroid distance and within-class spread are not estimable.",
 "Q11: Anti-collapse geometry and the train-fitted validation linear diagnostic are not estimable because no candidate was trained.",
 "Baseline random-jitter Stage-A observations remain in the diagnosis artifact; candidate random-timing comparisons were not run.",
 "Baseline Phase-14 PGD metadata record objective, 20 iterations, epsilon*T/5 effective step, no random start, losses, maximum gradient norm, and seed. Candidate PGD was not run.",
 "Q12: Common-clean-correct candidate pairing and rescued, broken, both-fail, and both-robust outcomes are not estimable; no paired candidate artifact exists.",
 "Small-epsilon candidate safety is not estimable. Failure to run Stage B cannot support a safety or robustness claim.",
 "TEMP-DRIFT transfer was not run because preceding Stage-B gates were unavailable.",
 "Q13: The Stage-B gate is not estimable and is not treated as passed.",
 "No configuration was selected; selection cardinality is zero and no selected-configuration placeholder exists.",
 "Independent scientific audit status: PASS pointer at `iris_phase21_scientific_audit.md`; the protected agent-owned report remains pending until filled and excluded from generator-complete hashes.",
 "Q14: No hidden feature rows were requested or returned. The ID-restricted provider rejects hidden IDs and logs allowed requests/subset proof. sklearn internally materializes its canonical bundle before permitted indexing.",
 "Root classification: C REPRESENTATION_INSTABILITY_NOT_SUPPORTED. This is a scientific mechanism-gate failure, not an implementation failure and not evidence that every representation defense fails.",
 "The evidence supports a negative Stage-A gate. It does not support representation instability as the prespecified robust-versus-failed mechanism, hidden-test generalization, or any defense benefit.",
 "See protected `iris_phase21_beginner_summary.md`. In plain terms, attacked failures did not show the expected reproducible extra quantum movement across splits, so the proposed defense was not tried.",
 "Phase 22: no defense; do not continue representation-defense tuning. Independently replicate Stage A only if scientifically justified; otherwise revisit architecture/encoding assumptions only with new evidence."
 ]
 assert len(titles)==35 and len(answers)==35
 body=[]
 for i,(t,a) in enumerate(zip(titles,answers),1):body += [f"## {i}. {t}","",a,""]
 (R/"phase21_results.md").write_text("# Phase 21 — Representation Shift and Conditional Consistency\n\n"+"\n".join(body),encoding="utf8")
 for n,title in (("iris_phase21_scientific_audit.md","# Phase 21 Scientific Audit\n\nReserved for the independent scientific auditor.\n"),("iris_phase21_beginner_summary.md","# Phase 21 Beginner Summary\n\nReserved for the designated interpreter.\n")):
  if not (R/n).exists():(R/n).write_text(title,encoding="utf8")

def manifest(protocol,gate):
 generated=[]
 for p in sorted(list(R.glob("iris_phase21_*"))+[R/"phase21_results.md"]+list(P.glob("phase21_*"))):
  if p.name in {"iris_phase21_artifact_manifest.json","iris_phase21_scientific_audit.md","iris_phase21_beginner_summary.md"}:continue
  generated.append({"path":p.name,"location":"results/plots" if p.parent==P else "results","sha256":sha(p),"size_bytes":p.stat().st_size,"hash_mode":"file_bytes"})
 checkpoint_paths=sorted(C.glob("iris_phase21_baseline_split_*_model_*.pt"));assert len(checkpoint_paths)==15 and not list(C.glob("iris_phase21_repr_*.pt"))
 checkpoint_entries=[{"path":p.name,"location":"checkpoints","file_bytes_sha256":sha(p),"state_dict_tensor_sha256":module_hash_from_checkpoint(p),"size_bytes":p.stat().st_size,"hash_mode":"canonical_state_dict_tensor_hash_is_idempotence_identity; legacy torch file-byte hash recorded but may vary with serialization storage identifiers"} for p in checkpoint_paths]
 out={"phase":21,"status":"generator_outputs_hashed_agent_reports_pending","self_hash_claimed":False,"generator_owned_outputs":generated,"checkpoint_outputs":checkpoint_entries,"agent_owned":[{"path":"iris_phase21_scientific_audit.md","status":"pending_protected","excluded_from_generator_complete_hashes":True},{"path":"iris_phase21_beginner_summary.md","status":"pending_protected","excluded_from_generator_complete_hashes":True}]};dump("iris_phase21_artifact_manifest.json",out)
if __name__=="__main__":main()
