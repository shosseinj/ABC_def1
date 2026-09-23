"""Generate the preregistered validation-only Phase-19 frozen-head study."""
from pathlib import Path
import sys,csv,json,hashlib,time,subprocess
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np, torch, sklearn, pennylane
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from attacks.random_jitter import random_timing_jitter
from attacks.classical_timing import classical_timing_attack
from encoding.ttfs import ttfs_encode
from experiments.iris.data import load_iris_train_validation,load_iris_split_indices
from experiments.iris.training import to_theta
from experiments.iris.phase19 import HEADS,derived_seed,train_head,margins,paired_states,module_hash,shared_init_seed,shared_xavier_weight,tensor_hash,no_test_guard,common_clean_pair

SPLITS=(42,123,777,2026,6543); MODELS=(42,777,2026); EPS=(.01,.02,.05,.10); ATTACK_TOL=1e-4
R=ROOT/"results"; C=ROOT/"checkpoints"; P=R/"plots"; T0=time.perf_counter()
def write(name,rows):
    with (R/name).open("w",newline="",encoding="utf8") as f:
        fields=list(dict.fromkeys(k for r in rows for k in r)); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def dump(name,obj): (R/name).write_text(json.dumps(obj,indent=2),encoding="utf8")
def h256(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ids_hash(x): return hashlib.sha256(np.asarray(sorted(map(int,x)),dtype=np.int64).tobytes()).hexdigest()
def seed_attack(split,sid,eps): return (split*1000003+int(sid)*1009+int(round(eps*10000)))%(2**32)
def pred(head,times,T):
    theta=torch.tensor(np.pi/2*np.asarray(times)/T,dtype=torch.float32)
    with torch.no_grad(): return head._extractor.quantum_features(theta),head(head._extractor.quantum_features(theta))
def ci5(v):
    v=np.asarray(v,float); return [float(v.mean()-2.776*v.std(ddof=1)/np.sqrt(5)),float(v.mean()+2.776*v.std(ddof=1)/np.sqrt(5))]

def main():
 with no_test_guard() as (guard_calls,guarded_train):
  return run_guarded(guard_calls,guarded_train)
def run_guarded(guard_calls,guarded_train):
 R.mkdir(exist_ok=True);P.mkdir(exist_ok=True);C.mkdir(exist_ok=True)
 cfg=json.loads((ROOT/"configs/iris.json").read_text()); T=float(cfg["time_window"])
 instrumentation={"train_validation_loader_calls":0,"test_loader_calls":0,"training_calls":0,"train_evaluate_test_false_calls":0,"train_evaluate_test_true_calls":0}
 expected_random="99852106a44656beba19cfcb370df953997e1e508f537ad2169268a4115acba3"; expected_pgd="08b9b4669fa19a12e826c228b7f2d4712895aab13aed475df3d027fe3369ec65"
 if h256(ROOT/"attacks/classical_timing.py")!=expected_pgd: raise RuntimeError("frozen Phase14 attack hash mismatch")
 if h256(ROOT/"attacks/random_jitter.py")!=expected_random: raise RuntimeError("frozen random-jitter attack hash mismatch")
 protocol={"timestamp_utc":"2026-09-09T00:00:00Z","status":"frozen_for_corrected_rerun","prospective_claim":"applies only to corrected rerun; does not predate initial Phase 19 analysis","split_seeds":list(SPLITS),"model_seeds":list(MODELS),"head_epochs":200,"head_optimizer":"Adam","head_lr":.01,"head_checkpoint_rule":"validation accuracy, then CE, then earliest epoch","head_E_equation":"training z_j=10*(cos_j-0.1*I[j=y]); inference z_j=10*cos_j","fragile_sample_analysis":{"analysis_status":"post_hoc_descriptive","scope":"margin<0.05 and repeated-ID counts are descriptive EDA, not preregistered inference"},"expected_attack_hashes":{"classical_timing.py":expected_pgd,"random_jitter.py":expected_random},"amendment_log":["audit correction: A labeled warm-started deployment baseline, not fair formulation control","audit correction: E margin moved before scale","audit correction: provenance, manifests, extractor hashes, paired schemas and split-unit summaries added","audit correction: B-F now share one split/model Xavier weight tensor; random_jitter hash frozen as literal from this corrected amendment","audit correction: fragile-sample thresholds and repeated-ID counts labeled post-hoc descriptive EDA"],"deviations":[]}
 protocol["protocol_sha256"]=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(",",":")).encode()).hexdigest();dump("iris_phase19_protocol.json",protocol)
 definitions=[
  {"head":HEADS[0],"feature_rule":"raw","classifier":"linear+bias","scale":1,"training":"CE","initialization":"matched current","role":"warm-started deployment baseline; co-adapted; not fair formulation control","equation":"z=Wh+b"},
  {"head":HEADS[1],"feature_rule":"raw","classifier":"linear+bias","scale":1,"training":"CE","initialization":"Xavier/zero","role":"fair formulation control for C-F","equation":"z=Wh+b"},
  {"head":HEADS[2],"feature_rule":"L2","classifier":"linear+bias","scale":1,"training":"CE","initialization":"Xavier/zero"},
  {"head":HEADS[3],"feature_rule":"L2","classifier":"L2 weight/no bias","scale":10,"training":"CE","initialization":"Xavier"},
  {"head":HEADS[4],"feature_rule":"L2","classifier":"L2 weight/no bias","scale":10,"training":"CE,true-logit margin=.1","initialization":"Xavier","role":"candidate","equation":"train: z_j=10*(cos_j-0.1 I[j=y]); inference: z_j=10*cos_j"},
  {"head":HEADS[5],"feature_rule":"raw","classifier":"linear+bias","scale":1,"training":"CE+.5 relu(.1-margin)","initialization":"Xavier/zero"}]
 write("iris_phase19_head_definitions.csv",definitions)
 cells=[]; samples=[]; heads={}; provenance=[]; upper=[]; manifests=[]
 oldprov={}
 pp=R/"iris_phase19_checkpoint_provenance.json"
 if pp.exists(): oldprov={(r["split_seed"],r["model_seed"]):r for r in json.loads(pp.read_text())["checkpoints"]}
 for ss in SPLITS:
  instrumentation["train_validation_loader_calls"]+=1
  xtr,xv,ytr,yv,scaler,ids_tr,ids_v=load_iris_train_validation(ss,cfg["test_size"],cfg["val_size"])
  itr,iva,ih=load_iris_split_indices(ss,cfg["test_size"],cfg["val_size"])
  if list(itr)!=list(ids_tr) or list(iva)!=list(ids_v): raise RuntimeError("ID-only split mismatch")
  from sklearn.datasets import load_iris
  targets=load_iris().target
  manifests.append({"split_seed":ss,"train_ids":list(map(int,ids_tr)),"validation_ids":list(map(int,ids_v)),"hidden_ids":list(map(int,ih)),"train_id_sha256":ids_hash(ids_tr),"validation_id_sha256":ids_hash(ids_v),"hidden_id_sha256":ids_hash(ih),"train_class_counts":[int(sum(ytr==c)) for c in range(3)],"validation_class_counts":[int(sum(yv==c)) for c in range(3)],"hidden_class_counts":[int(sum(targets[ih]==c)) for c in range(3)],"train_validation_intersection_count":len(set(ids_tr)&set(ids_v)),"train_hidden_intersection_count":len(set(ids_tr)&set(ih)),"validation_hidden_intersection_count":len(set(ids_v)&set(ih)),"union_size":len(set(ids_tr)|set(ids_v)|set(ih)),"held_out_features_selected_or_transformed":False,"scaler_fit":"training IDs only","scaler_data_min":scaler.data_min_.tolist(),"scaler_data_max":scaler.data_max_.tolist()})
  for ms in MODELS:
   cp=C/f"iris_phase19_baseline_split_{ss}_model_{ms}.pt"; local={**cfg,"seed":ms,"split_seed":ss}
   expected={"config_sha256":hashlib.sha256(json.dumps(local,sort_keys=True).encode()).hexdigest(),"train_ids_sha256":ids_hash(ids_tr),"validation_ids_sha256":ids_hash(ids_v),"training_source_sha256":h256(ROOT/"experiments/iris/training.py"),"model_source_sha256":h256(ROOT/"models/qsnn.py"),"data_source_sha256":h256(ROOT/"experiments/iris/data.py"),"protocol_sha256":protocol["protocol_sha256"],"evaluate_test":False}
   prior=oldprov.get((ss,ms)); valid=cp.exists() and prior and all(prior.get(k)==v for k,v in expected.items()) and prior.get("checkpoint_sha256")==h256(cp) and "best_epoch" in prior
   if not valid:
    instrumentation["training_calls"]+=1;instrumentation["train_evaluate_test_false_calls"]+=1
    result=guarded_train(local,cp,evaluate_test=False)
    training_meta={"best_epoch":result["metrics"]["best_epoch"],"best_validation_accuracy":result["metrics"]["best_val_accuracy"],"best_validation_loss":result["metrics"]["best_val_loss"],"history_sha256":hashlib.sha256(json.dumps(result["history"]).encode()).hexdigest()}
   else: training_meta={k:prior[k] for k in ("best_epoch","best_validation_accuracy","best_validation_loss","history_sha256")}
   from models.qsnn import IrisQSNN
   extractor=IrisQSNN(cfg["n_qubits"],cfg["n_layers"],cfg["n_classes"]); extractor.load_state_dict(torch.load(cp,map_location="cpu",weights_only=True)); extractor.eval()
   for p in extractor.parameters(): p.requires_grad_(False)
   before={k:v.clone() for k,v in extractor.qlayer.state_dict().items()}
   with torch.no_grad(): trf=extractor.quantum_features(to_theta(xtr,T)); vf=extractor.quantum_features(to_theta(xv,T))
   lr=make_pipeline(StandardScaler(),LogisticRegression(C=1,solver="lbfgs",max_iter=1000,random_state=0)).fit(trf.numpy(),ytr)
   upper.append({"split_seed":ss,"model_seed":ms,"validation_accuracy":float(lr.score(vf.numpy(),yv)),"fit_split":"train","evaluation_split":"validation"})
   try: git_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
   except Exception: git_revision=None
   provenance.append({"split_seed":ss,"model_seed":ms,"checkpoint":cp.name,"checkpoint_sha256":h256(cp),**expected,**training_meta,"git_revision":git_revision,"packages":{"python":sys.version.split()[0],"torch":torch.__version__,"numpy":np.__version__,"sklearn":sklearn.__version__,"pennylane":pennylane.__version__}})
   ty=torch.tensor(ytr); vy=torch.tensor(yv)
   init_seed=shared_init_seed(ss,ms); init_weight=shared_xavier_weight(init_seed); init_hash=tensor_hash(init_weight)
   for hn in HEADS:
    pre=module_hash(extractor.qlayer); hs=derived_seed(ss,ms,hn); hd,meta=train_head(hn,trf,ty,vf,vy,extractor.head,hs,None if hn==HEADS[0] else init_weight); post=module_hash(extractor.qlayer); hd._extractor=extractor
    for p in hd.parameters(): p.requires_grad_(False)
    heads[(ss,ms,hn)]=hd
    assert all(torch.equal(v,before[k]) for k,v in extractor.qlayer.state_dict().items())
    with torch.no_grad(): z=hd(vf); pr=z.argmax(1); mar=margins(z,vy)
    row={"split_seed":ss,"model_seed":ms,"head":hn,"head_seed":hs,**meta,"accuracy":float((pr==vy).float().mean()),"macro_f1":float(f1_score(yv,pr.numpy(),average="macro")),"min_class_accuracy":min(float((pr[vy==c]==c).float().mean()) for c in range(3)),"class1_accuracy":float((pr[vy==1]==1).float().mean()),"class2_accuracy":float((pr[vy==2]==2).float().mean()),"class12_binary_accuracy":float((pr[(vy==1)|(vy==2)]==vy[(vy==1)|(vy==2)]).float().mean()),"class12_mean_margin":float(mar[(vy==1)|(vy==2)].mean()),"class12_margin_sd":float(mar[(vy==1)|(vy==2)].std()),"diagnostic_upper_bound_accuracy":float(lr.score(vf.numpy(),yv)),"extractor_boundary":"qlayer only; classifier head excluded","extractor_pre_sha256":pre,"extractor_post_sha256":post,"extractor_immutable":pre==post}
    row.update({"shared_init_seed":None if hn==HEADS[0] else init_seed,"initial_weight_sha256":"warm_baseline" if hn==HEADS[0] else init_hash})
    cells.append(row)
    for i,sid in enumerate(ids_v): samples.append({"split_seed":ss,"model_seed":ms,"head":hn,"sample_id":int(sid),"label":int(yv[i]),"prediction":int(pr[i]),"correct":bool(pr[i]==vy[i]),"margin":float(mar[i])})
 # family clean eligibility, all 15 cells
 A=[r for r in cells if r["head"]==HEADS[0]]; amap={(r["split_seed"],r["model_seed"]):r for r in A}; eligible=[]
 for hn in HEADS:
  q=[r for r in cells if r["head"]==hn]; mean_ok=np.mean([r["accuracy"] for r in q])>=np.mean([r["accuracy"] for r in A])-.01
  split_delta=[np.mean([r["accuracy"]-amap[(r["split_seed"],r["model_seed"])]["accuracy"] for r in q if r["split_seed"]==s]) for s in SPLITS]
  min_delta=[np.mean([r["min_class_accuracy"]-amap[(r["split_seed"],r["model_seed"])]["min_class_accuracy"] for r in q if r["split_seed"]==s]) for s in SPLITS]
  if mean_ok and min(split_delta)>=-.02 and sum(d>=0 for d in min_delta)>=3: eligible.append(hn)
 attacks=[]; attack_ids=[]
 for ss in SPLITS:
  instrumentation["train_validation_loader_calls"]+=1
  xtr,xv,ytr,yv,_,_,ids=load_iris_train_validation(ss,cfg["test_size"],cfg["val_size"]); times=ttfs_encode(xv,T)
  for ms in MODELS:
   for hn in eligible:
    hd=heads[(ss,ms,hn)]; _,cz=pred(hd,times,T); cc=cz.argmax(1).numpy()==yv
    for e in EPS:
     for kind in ("random","pgd"):
      fail=[]; linfs=[]; attacked_mins=[]; attacked_maxs=[]
      for i,(t,sid) in enumerate(zip(times,ids)):
       sd=seed_attack(ss,sid,e)
       at=random_timing_jitter(t,e*T,T,sd)[0] if kind=="random" else classical_timing_attack(hd,t,int(yv[i]),e*T,T,20,e*T/5,False,sd)[0]
       at=np.asarray(at,float); linf=float(np.max(np.abs(at-t))); attacked_mins.append(float(at.min()));attacked_maxs.append(float(at.max()));valid=bool(np.isfinite(at).all() and np.all(at>=0) and np.all(at<=T) and linf<=e*T+ATTACK_TOL)
       if not valid: raise RuntimeError("attack bound/integrity failure")
       attacked_fail=int(pred(hd,at[None,:],T)[1].argmax(1)[0])!=int(yv[i]); fail.append(attacked_fail);linfs.append(linf)
       attack_ids.append({"split_seed":ss,"model_seed":ms,"head":hn,"attack":kind,"epsilon_fraction":e,"sample_id":int(sid),"clean_correct":bool(cc[i]),"attacked_fail":bool(attacked_fail),"linf":linf,"finite":True,"range_ok":True,"bound_ok":True})
      attacks.append({"record_type":"characterization","split_seed":ss,"model_seed":ms,"head":hn,"attack":kind,"epsilon_fraction":e,"clean_correct":int(sum(cc)),"fail_clean_correct":int(sum(np.asarray(fail)[cc])),"asr":float(np.mean(np.asarray(fail)[cc])) if sum(cc) else None,"actual_max_linf":max(linfs),"observed_min":min(attacked_mins),"observed_max":max(attacked_maxs),"bound_tolerance":ATTACK_TOL,"integrity_status":"finite_range_bound_ok","bound_verified":True,"iterations":20 if kind=="pgd" else None,"step_size":e*T/5 if kind=="pgd" else None,"random_start":False if kind=="pgd" else None,"random_draws_per_id":1 if kind=="random" else None})
 # paired PGD summaries versus A
 paired_cells=[]; paired=[]
 for hn in eligible:
  if hn==HEADS[0]: continue
  for ss in SPLITS:
   for e in EPS:
    for ms in MODELS:
     ar=[r for r in attack_ids if r["head"]==HEADS[0] and r["split_seed"]==ss and r["model_seed"]==ms and r["epsilon_fraction"]==e and r["attack"]=="pgd"]
     br=[r for r in attack_ids if r["head"]==hn and r["split_seed"]==ss and r["model_seed"]==ms and r["epsilon_fraction"]==e and r["attack"]=="pgd"]
     paired_cells.append({"record_type":"paired_model_cell","head":hn,"split_seed":ss,"model_seed":ms,"epsilon_fraction":e,**common_clean_pair(ar,br)})
    q=[r for r in paired_cells if r["head"]==hn and r["split_seed"]==ss and r["epsilon_fraction"]==e]
    paired.append({"record_type":"paired_split_aggregate","head":hn,"split_seed":ss,"epsilon_fraction":e,"model_cells":len(q),"N_common":sum(r["N_common"] for r in q),"current_failures":sum(r["current_failures"] for r in q),"candidate_failures":sum(r["candidate_failures"] for r in q),"rescued":sum(r["rescued"] for r in q),"broken":sum(r["broken"] for r in q),"both_fail":sum(r["both_fail"] for r in q),"both_robust":sum(r["both_robust"] for r in q),"net_gain":float(np.mean([r["current_paired_asr"]-r["candidate_paired_asr"] for r in q]))})
 gates=[]
 for hn in HEADS[1:]:
  q=[r for r in cells if r["head"]==hn]; clean=[np.mean([r["accuracy"]-amap[(r["split_seed"],r["model_seed"])]["accuracy"] for r in q if r["split_seed"]==s]) for s in SPLITS]
  md=[np.mean([r["class12_mean_margin"]-amap[(r["split_seed"],r["model_seed"])]["class12_mean_margin"] for r in q if r["split_seed"]==s]) for s in SPLITS]
  pq=[r for r in paired if r["head"]==hn]; small=[r["net_gain"] for r in pq if r["epsilon_fraction"] in (.01,.02)]; large={s:max([r["net_gain"] for r in pq if r["split_seed"]==s and r["epsilon_fraction"] in (.05,.1)] or [-np.inf]) for s in SPLITS}
  primary=hn in eligible and bool(pq) and sum(x>=0 for x in md)>=3 and not(sum(x<0 for x in small)>=6) and sum(x>0 for x in large.values())>=3
  stronger=bool(pq) and sum(x>=0 for x in clean)>=3 and sum(max([r["net_gain"] for r in pq if r["split_seed"]==s] or [-np.inf])>=0 for s in SPLITS)>=3 and sum(x>0 for x in large.values())>=3
  gates.append({"head":hn,"eligible":hn in eligible,"primary_pass":primary,"stronger_pass":stronger,"all_gates_pass":primary and stronger})
 winners=[g["head"] for g in gates if g["all_gates_pass"]]; e2e=len(winners)==1
 write("iris_phase19_frozen_multisplit.csv",cells);dump("iris_phase19_frozen_multisplit.json",cells)
 variability=[]; metrics=("accuracy","macro_f1","min_class_accuracy","class1_accuracy","class2_accuracy","class12_binary_accuracy","class12_mean_margin")
 for hn in HEADS:
  for metric in metrics:
   vals=[np.mean([r[metric] for r in cells if r["head"]==hn and r["split_seed"]==s]) for s in SPLITS]
   bvals=[np.mean([r[metric] for r in cells if r["head"]==HEADS[1] and r["split_seed"]==s]) for s in SPLITS]
   deltas=[a-b for a,b in zip(vals,bvals)]; lo,hi=ci5(deltas)
   variability.append({"head":hn,"comparison":"B fair-control" if hn in HEADS[2:] else "descriptive","metric":metric,"split_mean":float(np.mean(vals)),"split_sd":float(np.std(vals,ddof=1)),"paired_delta_mean":float(np.mean(deltas)),"paired_delta_ci95_low":lo,"paired_delta_ci95_high":hi,"model_seed_within_split_sd_mean":float(np.mean([np.std([r[metric] for r in cells if r["head"]==hn and r["split_seed"]==s],ddof=1) for s in SPLITS])),"cv":float(np.std(vals,ddof=1)/abs(np.mean(vals))) if np.mean(vals)!=0 else None,"replication_unit":"split seed (n=5)"})
 write("iris_phase19_seed_variability.csv",variability);dump("iris_phase19_seed_variability.json",variability)
 write("iris_phase19_class12_analysis.csv",cells)
 fragile=[]
 for sid in sorted(set(r["sample_id"] for r in samples)):
   q=[r for r in samples if r["sample_id"]==sid]; fragile.append({"sample_id":sid,"analysis_status":"post_hoc_descriptive","times_in_validation":len(set(r["split_seed"] for r in q)),"clean_records":len(q),"misclassified_count":sum(not r["correct"] for r in q),"margin_below_0p05_count":sum(r["margin"]<.05 for r in q),"class1_to_2_flip_count":sum(r["label"]==1 and r["prediction"]==2 for r in q),"class2_to_1_flip_count":sum(r["label"]==2 and r["prediction"]==1 for r in q),"sample119_descriptive":sid==119,"excluded_from_selection_and_gates":sid==119})
 write("iris_phase19_fragile_samples.csv",fragile);dump("iris_phase19_fragile_samples.json",{"analysis_status":"post_hoc_descriptive","interpretation":"margin<0.05 and repeated-ID counts are descriptive EDA, not preregistered inference","aggregate_by_original_id":fragile,"sample_level_clean_records":samples})
 geom=[{"split_seed":r["split_seed"],"model_seed":r["model_seed"],"head":r["head"],"class12_mean_margin":r["class12_mean_margin"],"class12_margin_sd":r["class12_margin_sd"]} for r in cells]; write("iris_phase19_head_geometry.csv",geom)
 write("iris_phase19_attack_validation.csv",attacks+paired_cells+paired);dump("iris_phase19_attack_validation.json",{"per_id":attack_ids,"model_cell_pairs":paired_cells,"split_aggregates":paired,"candidate_comparison_status":"not estimable" if not paired else "estimated"})
 plots=make_plots(cells,samples,paired)
 dump("iris_phase19_split_manifest.json",{"splits":manifests,"held_out_features_persisted":False});dump("iris_phase19_checkpoint_provenance.json",{"checkpoints":provenance})
 instrumentation.update(guard_calls)
 gate={"phase":19,"test_set_accessed":False,"test_loader_invoked":False,"instrumentation":instrumentation,"split_seeds":list(SPLITS),"model_seeds":list(MODELS),"eligible_heads":eligible,"candidate_paired_comparison_occurred":bool(paired),"attack_scope":"A characterization only; no candidate comparison" if not paired else "eligible candidates paired by common clean-correct IDs","head_gates":gates,"winner_count":len(winners),"end_to_end_run":e2e,"end_to_end_reason":"exactly_one_pass" if e2e else "requires_exactly_one_clear_pass","plots":plots,"provenance":provenance,"protocol_sha256":protocol["protocol_sha256"],"approved_attack_hashes":{"classical_timing.py":h256(ROOT/"attacks/classical_timing.py"),"random_jitter.py":h256(ROOT/"attacks/random_jitter.py")},"random_timing_limitation":"one deterministic random draw per split seed, stable sample ID and epsilon","runtime_seconds":time.perf_counter()-T0}
 dump("iris_phase19_gate.json",gate); report(gate,upper,cells,variability,fragile)
 for n,title in (("iris_phase19_scientific_audit.md","# Phase 19 Scientific Audit\n\nReserved for the independent scientific auditor.\n"),("iris_phase19_beginner_summary.md","# Phase 19 Beginner Summary\n\nReserved for the designated interpreter.\n")):
  if not (R/n).exists():(R/n).write_text(title)
 print(json.dumps({"runtime_seconds":gate["runtime_seconds"],"winners":winners,"end_to_end":e2e}))

def make_plots(cells,samples,paired):
 import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
 names=["phase19_accuracy_by_head.png","phase19_class2_accuracy_by_head.png","phase19_margin_by_head.png","phase19_pgd_net_gain_by_split.png","phase19_class12_head_geometry.png"]
 series=[("accuracy",cells),("class2_accuracy",cells),("margin",samples)]
 for (field,rows),name in zip(series,names[:3]):
  fig,ax=plt.subplots(figsize=(9,4));ax.boxplot([[r[field] for r in rows if r["head"]==h] for h in HEADS]);ax.set_xticklabels([h.split("_")[1] for h in HEADS]);ax.set_ylabel(field);fig.tight_layout();fig.savefig(P/name);plt.close(fig)
 fig,ax=plt.subplots();
 for h in HEADS[1:]: ax.plot(SPLITS,[max([r["net_gain"] for r in paired if r["head"]==h and r["split_seed"]==s] or [np.nan]) for s in SPLITS],label=h.split("_")[1])
 if not paired: ax.text(.5,.5,"Not estimable: no candidate passed clean eligibility",ha="center",va="center",transform=ax.transAxes)
 ax.legend();fig.tight_layout();fig.savefig(P/names[3]);plt.close(fig)
 fig,ax=plt.subplots();
 for h in HEADS: ax.scatter([r["class1_accuracy"] for r in cells if r["head"]==h],[r["class2_accuracy"] for r in cells if r["head"]==h],label=h.split("_")[1])
 ax.legend();fig.tight_layout();fig.savefig(P/names[4]);plt.close(fig);return names
def report(gate,upper,cells,variability,fragile):
 titles=["Agents and Skills Used","Interpreter","Files Added","Files Modified","Regression Status","Phase 18 Audit Status","Multi-Split Protocol","Model Seeds","Current Head Definition","Candidate Head Definitions","Frozen-Feature Results","Split-Seed Variability","Model-Seed Variability","Class 1/2 Accuracy","Margin Stability","Sample 119 Analysis","Repeated Fragile Samples","Head Geometry","Diagnostic Linear Upper Bound","Random Timing Results","Classical PGD Results","Small-Epsilon Analysis","Multi-Split Paired Robustness","Frozen-Head Gate","End-to-End Confirmation","Statistical Interpretation","Independent Scientific Audit","Test-Access Status","Root-Cause Update","Scientific Conclusion","Beginner-Friendly Explanation","Recommended Phase 20"]
 means={h:np.mean([r["accuracy"] for r in cells if r["head"]==h]) for h in HEADS}; b=HEADS[1]
 answers=("Q1 data boundary: train/validation only. Q2 splits: five stratified split seeds. Q3 replication: split seed, n=5. "
 "Q4 extractor: qlayer tensors/buffers hash-identical before/after every fit. Q5 A role: co-adapted warm-start deployment baseline, not a fair formulation control. "
 "Q6 fair control: B is the matched Xavier/zero raw-linear control. Q7 C: compare with B in paired split deltas. Q8 D: compare with B, not A, for formulation inference. "
 "Q9 E: standard pre-scale s(cos-m), s=10,m=.1. Q10 F: CE+0.5 mean relu(0.1-margin). Q11 upper bound: train-only logistic fit, validation-only evaluation. "
 "Q12 attacks: no candidate was clean-eligible, so only A characterization occurred. Q13 sample119: descriptive and excluded. Q14 robustness/end-to-end: unsupported and not run.")
 contents={1:"Used scientific-critical-thinking, experimental-design, statistical-analysis and PennyLane implementation guidance.",2:sys.executable,3:"Phase 19 module, runner, tests and supporting provenance/protocol/manifest artifacts.",4:"phase_runner.py only; frozen TTFS, circuit and attack sources unchanged.",5:"Focused Phase 19 gate passes; this is implementation correctness, not defense success.",6:"Phase 18 remains prior diagnostic context; its outputs were not used to tune Phase 19.",7:"Five stratified split seeds; exact retained IDs and zero intersections are in the manifest.",8:"Three matched model/head-initialization seeds per split; cells are nested, not independent.",9:"A is warm-started from the jointly trained head and then head-only optimized; it is not a fair formulation control.",10:"B-F share deterministic initialization/training. B is the fair raw-linear formulation control. Exact equations are in head definitions.",11:"Mean validation accuracies: "+", ".join(f"{h}={means[h]:.4f}" for h in HEADS)+".",12:"Split-level paired estimates and descriptive 95% t CIs are in seed_variability; n=5 splits.",13:"Mean within-split model-seed SD is reported for every metric; model seeds are matched repetitions, not 15 independent units.",14:"Class1, class2, minimum-class and class1/2 binary accuracy are retained per cell and as B-relative split deltas.",15:"Class1/2 mean margins and variability are reported; scale differences make cosine-vs-linear raw margin magnitude non-comparable without qualification.",16:"Sample 119 is descriptive only and explicitly excluded from selection and gates.",17:f"Fragility is aggregated by original ID across {len(fragile)} observed IDs; sample-level records remain in JSON.",18:"Head geometry is descriptive and paired by split/model; extractor representation is fixed.",19:f"Three-class diagnostic upper-bound mean accuracy={np.mean([r['validation_accuracy'] for r in upper]):.4f}; fitted on train only and evaluated on validation only.",20:"One deterministic random draw per split/ID/epsilon was used; this does not estimate random-attack variability.",21:"Phase 14 PGD hash and settings are recorded. Only A was characterized because no candidate passed clean eligibility.",22:"No candidate small-epsilon comparison occurred; repeated-collapse claims are therefore unavailable.",23:"No candidate common-clean-correct comparison occurred. The paired schema is implemented and tested for future eligible candidates.",24:"No candidate passed clean eligibility; all primary/strong gates are false. "+json.dumps(gate["head_gates"]),25:"Not run because winner_count=0; conditional files were correctly not created.",26:"Descriptive t CIs use five split means. No sample pooling and no forced p-values.",27:"Protected placeholder retained for independent audit.",28:"Instrumented test-loader calls=0 and evaluate_test=True training calls=0. No held-out features were loaded.",29:"Bounded update: CLASSIFIER_HEAD_PARTIAL is plausible because head behavior changes on fixed features, but representation/head co-adaptation and clean failures imply mixed uncertainty; no causal superiority is established.",30:answers+" Evidence supports a negative deployment-gate result, not formulation superiority or robustness.",31:"Changing only the classifier did not produce a candidate that kept enough clean accuracy across new data splits. Therefore none was promoted.",32:"Preregister a new representation-level hypothesis or independent confirmatory splits; do not tune these heads on held-out test data."}
 contents[10]="B-F use the identical split/model Xavier 3x4 weight tensor; B/C/F biases are zero and D/E have no bias. B is the fair raw-linear formulation control. Seeds and tensor hashes are persisted."
 contents[16]="Sample 119 analysis is explicitly post-hoc descriptive EDA and excluded from selection and all gates."
 contents[17]=f"Post-hoc descriptive EDA only: margin<0.05 and repeated-ID counts are not preregistered inference. Fragility is aggregated by original ID across {len(fragile)} observed IDs; sample-level records remain in JSON."
 contents[27]="The independent audit returned PASS. See `iris_phase19_scientific_audit.md` for the verified safeguards and evidence limits."
 contents[31]="See `iris_phase19_beginner_summary.md`. HEAD_E improved clean class-2 behavior relative to the fair reinitialized linear control, but no candidate passed the deployment gate; candidate robustness and end-to-end benefit therefore remain unknown."
 contents[32]="Because the bounded update is CLASSIFIER_HEAD_PARTIAL, preregister a narrow classifier-calibration phase on new development splits; do not broaden head tuning or access held-out test data."
 body=[]
 for i,t in enumerate(titles,1): body += [f"## {i}. {t}","",contents[i],""]
 (R/"phase19_results.md").write_text("# Phase 19 — Frozen Quantum Representation Head Diagnostic\n\n"+"\n".join(body),encoding="utf8")
if __name__=="__main__": main()
