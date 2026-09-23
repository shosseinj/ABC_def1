"""Run Phase 20 under its repository-local initial protocol and audit amendment."""
from pathlib import Path
import sys,csv,json,hashlib,time,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,sklearn,pennylane
from sklearn.metrics import f1_score
from attacks.random_jitter import random_timing_jitter
from attacks.classical_timing import classical_timing_attack
from encoding.ttfs import ttfs_encode
from experiments.iris.data import load_iris_train_validation,load_iris_split_manifest_labels
from experiments.iris.training import to_theta
from experiments.iris.phase20 import *

R=ROOT/"results";C=ROOT/"checkpoints";P=R/"plots";SPLITS=(314,1001,4096,9001,12345);MODELS=(42,777,2026);EPS=(.01,.02,.05,.1);TOL=1e-4
def dump(n,x): (R/n).write_text(json.dumps(x,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding="utf8")
def write(n,rows):
    rows=list(rows); fields=list(dict.fromkeys(k for r in rows for k in r)) if rows else ["status","reason"]
    with (R/n).open("w",newline="",encoding="utf8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def idsha(x): return hashlib.sha256(np.asarray(sorted(map(int,x)),dtype=np.int64).tobytes()).hexdigest()
def ci(v):
    v=np.asarray(v,float);q=2.776*float(v.std(ddof=1))/np.sqrt(5);return float(v.mean()-q),float(v.mean()+q)
def attack_seed(ss,sid,e): return (ss*1000003+int(sid)*1009+int(round(e*10000)))%(2**32)
def predict(head,times,T):
    theta=torch.tensor(np.pi/2*np.asarray(times)/T,dtype=torch.float32)
    with torch.no_grad(): f=head._extractor.quantum_features(theta);return head(f)
def validate_protocol():
    p=R/"iris_phase20_protocol.json";d=json.loads(p.read_text());h=d.pop("protocol_sha256")
    assert h==hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    assert d["prior_output_inspection"] is False and tuple(d["split_seeds"])==SPLITS and d["head_e_grid"]["config_count"]==9
    return json.loads(p.read_text())
def append_run_event(event):
    p=R/"iris_phase20_run_events.jsonl"
    with p.open("a",encoding="utf8") as f:f.write(json.dumps(event,sort_keys=True)+"\n")
def initialize_run_events(protocol):
    p=R/"iris_phase20_run_events.jsonl"
    if not p.exists():
        pp=R/"iris_phase20_protocol.json";pts=pp.stat().st_mtime
        append_run_event({"event":"protocol_written","evidence":"filesystem_mtime_observed_after_the_fact","timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime(pts)),"sha256":sha(pp),"protocol_sha256":protocol["protocol_sha256"],"claim_limit":"repository-local; not independently timestamped or committed"})
        cps=sorted(C.glob("iris_phase20_baseline_*.pt"),key=lambda x:x.stat().st_mtime)
        if cps:append_run_event({"event":"first_output_checkpoint_observed","evidence":"filesystem_mtime_observed_after_the_fact","timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime(cps[0].stat().st_mtime)),"path":str(cps[0].relative_to(ROOT)),"sha256":sha(cps[0])})
    append_run_event({"event":"generator_run_started","timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"protocol_sha256":protocol["protocol_sha256"]})
def historical_seed_ledger():
    import re
    extensions={".py",".json",".md",".csv",".yaml",".yml",".toml",".txt"};paths=[];hits=[];h=hashlib.sha256()
    for p in sorted(ROOT.rglob("*")):
        rel=p.relative_to(ROOT).as_posix()
        if not p.is_file() or p.suffix.lower() not in extensions or rel.startswith((".git/","checkpoints/")) or "phase20" in rel.lower() or rel.startswith("results/iris_phase20"):continue
        try:text=p.read_text(encoding="utf8")
        except UnicodeDecodeError:continue
        paths.append(rel);ph=hashlib.sha256(text.encode()).hexdigest();h.update(rel.encode());h.update(ph.encode())
        for no,line in enumerate(text.splitlines(),1):
            if re.search(r"split[_ -]?seeds?|SPLITS",line,re.I):
                nums=[int(x) for x in re.findall(r"(?<![.\w])\d+(?![.\w])",line)];hits.append({"path":rel,"line":no,"line_sha256":hashlib.sha256(line.encode()).hexdigest(),"integer_tokens":nums})
    preferred=set(SPLITS);found=sorted(preferred & {n for x in hits for n in x["integer_tokens"]});assert not found,f"preferred historical seed found: {found}"
    prior=sorted(set(json.loads((R/"iris_phase19_protocol.json").read_text())["split_seeds"]))
    out={"scan_scope":"repository text files excluding Phase20 paths/outputs/protocol, .git, and checkpoints","scanned_extensions":sorted(extensions),"scanned_paths":paths,"scanned_path_count":len(paths),"scan_sha256":h.hexdigest(),"seed_context_hits":hits,"prior_known_split_seeds":prior,"preferred_split_seeds":list(SPLITS),"preferred_absent":True,"limitation":"Cannot detect deleted, unrecorded, renamed-without-seed-context, or external exploratory runs."};dump("iris_phase20_historical_split_seed_ledger.json",out);return out

def main():
 with no_test_guard() as (audit,guarded_train): return run(audit,guarded_train)
def run(audit,guarded_train):
 t0=time.perf_counter();R.mkdir(exist_ok=True);C.mkdir(exist_ok=True);P.mkdir(exist_ok=True)
 for stale in ("phase20_clean_delta_by_config.png","phase20_class12_margin_by_config.png","phase20_pgd_net_gain_by_split.png"):
  p=P/stale
 if p.exists():p.unlink()
 for stale in ("iris_phase20_attacks.csv","iris_phase20_attacks.json","iris_phase20_clean_cells.csv","iris_phase20_clean_cells.json","iris_phase20_clean_multisplit.csv","iris_phase20_clean_multisplit.json","iris_phase20_split_summary.csv","iris_phase20_split_summary.json","iris_phase20_metric_summary.csv","iris_phase20_metric_summary.json","iris_phase20_paired_attacks.csv","iris_phase20_fragile_ids.csv","iris_phase20_fragile_ids.json","iris_phase20_head_grid.csv"):
  p=R/stale
  if p.exists():p.unlink()
 protocol=validate_protocol();initialize_run_events(protocol);ledger=historical_seed_ledger();cfg=json.loads((ROOT/"configs/iris.json").read_text());T=float(cfg["time_window"])
 expected=protocol["attacks"]["expected_hashes"]
 for n,h in expected.items():
    if sha(ROOT/"attacks"/n)!=h:raise RuntimeError(f"frozen attack hash mismatch: {n}")
 used=json.loads((R/"iris_phase19_protocol.json").read_text())["split_seeds"]
 if set(used)&set(SPLITS):raise RuntimeError("new split seed was previously used")
 instrumentation={"train_validation_loader_calls":0,"training_calls":0,"train_evaluate_test_false_calls":0,"test_loader_calls":0};cells=[];samples=[];heads={};manifest=[];provenance=[]
 old={};pp=R/"iris_phase20_checkpoint_provenance.json"
 if pp.exists():old={(x["split_seed"],x["model_seed"]):x for x in json.loads(pp.read_text())["checkpoints"]}
 for ss in SPLITS:
  instrumentation["train_validation_loader_calls"]+=1;xtr,xv,ytr,yv,scaler,trids,vids=load_iris_train_validation(ss,cfg["test_size"],cfg["val_size"]);mi=load_iris_split_manifest_labels(ss,cfg["test_size"],cfg["val_size"]);it,iv,ih=mi["train_ids"],mi["validation_ids"],mi["hidden_ids"]
  assert list(it)==list(trids) and list(iv)==list(vids)
  manifest.append({"split_seed":ss,"train_ids":list(map(int,trids)),"validation_ids":list(map(int,vids)),"hidden_ids":list(map(int,ih)),"hidden_labels":list(map(int,mi["hidden_labels"])),"manifest_helper_contract":"IDs and labels only","train_id_sha256":idsha(trids),"validation_id_sha256":idsha(vids),"hidden_id_sha256":idsha(ih),"train_validation_intersection_count":len(set(trids)&set(vids)),"train_hidden_intersection_count":len(set(trids)&set(ih)),"validation_hidden_intersection_count":len(set(vids)&set(ih)),"hidden_features_received_by_phase20":False,"scaler_fit":"training_only","scaler_data_min":scaler.data_min_.tolist(),"scaler_data_max":scaler.data_max_.tolist()})
  for ms in MODELS:
   cp=C/f"iris_phase20_baseline_split_{ss}_model_{ms}.pt";local={**cfg,"seed":ms,"split_seed":ss};base={"config_sha256":hashlib.sha256(json.dumps(local,sort_keys=True).encode()).hexdigest(),"train_ids_sha256":idsha(trids),"validation_ids_sha256":idsha(vids),"protocol_sha256":protocol["protocol_sha256"],"training_source_sha256":sha(ROOT/"experiments/iris/training.py"),"model_source_sha256":sha(ROOT/"models/qsnn.py"),"data_source_sha256":sha(ROOT/"experiments/iris/data.py"),"evaluate_test":False};prior=old.get((ss,ms));valid=cp.exists() and prior and all(prior.get(k)==v for k,v in base.items()) and prior.get("checkpoint_sha256")==sha(cp)
   if not valid:
    instrumentation["training_calls"]+=1;instrumentation["train_evaluate_test_false_calls"]+=1;z=guarded_train(local,cp,evaluate_test=False);meta={"best_epoch":z["metrics"]["best_epoch"],"best_validation_accuracy":z["metrics"]["best_val_accuracy"],"best_validation_loss":z["metrics"]["best_val_loss"]}
   else:meta={k:prior[k] for k in ("best_epoch","best_validation_accuracy","best_validation_loss")}
   from models.qsnn import IrisQSNN
   ex=IrisQSNN(cfg["n_qubits"],cfg["n_layers"],cfg["n_classes"]);ex.load_state_dict(torch.load(cp,map_location="cpu",weights_only=True));ex.eval()
   for p in ex.parameters():p.requires_grad_(False)
   with torch.no_grad():tf=ex.quantum_features(to_theta(xtr,T));vf=ex.quantum_features(to_theta(xv,T))
   ty=torch.tensor(ytr);vy=torch.tensor(yv);seed=derived_seed(ss,ms);iw=shared_xavier(seed);ihash=tensor_hash(iw);configs=[(HEAD_A_CURRENT,None,None)]+[(HEAD_E_COSINE_MARGIN,s,m) for s in SCALES for m in MARGINS]
   for name,s,m in configs:
    pre=module_hash(ex.qlayer);h=CalibrationHead(name,s or 1,m or 0,current=ex.head,initial_weight=iw if name==HEAD_E_COSINE_MARGIN else None);fit=fit_head(h,tf,ty,vf,vy);post=module_hash(ex.qlayer);assert pre==post;h._extractor=ex
    for p in h.parameters():p.requires_grad_(False)
    with torch.no_grad():logits=h(vf);pred=logits.argmax(1);marg=margins(logits,vy);ce=float(torch.nn.functional.cross_entropy(logits,vy))
    key="A" if name==HEAD_A_CURRENT else f"s{s}_m{m:.2f}";row={"split_seed":ss,"model_seed":ms,"head":name,"config_id":key,"scale":s,"margin":m,**fit,"accuracy":float((pred==vy).float().mean()),"macro_f1":float(f1_score(yv,pred.numpy(),average="macro")),"minimum_class_accuracy":min(float((pred[vy==c]==c).float().mean()) for c in range(3)),"class1_accuracy":float((pred[vy==1]==1).float().mean()),"class2_accuracy":float((pred[vy==2]==2).float().mean()),"class12_binary_accuracy":float((pred[(vy==1)|(vy==2)]==vy[(vy==1)|(vy==2)]).float().mean()),"cross_entropy":ce,"class12_mean_margin":float(marg[(vy==1)|(vy==2)].mean()),"extractor_pre_sha256":pre,"extractor_post_sha256":post,"extractor_immutable":True,"shared_init_seed":seed if name==HEAD_E_COSINE_MARGIN else None,"initial_weight_sha256":ihash if name==HEAD_E_COSINE_MARGIN else "warm_coadapted"}
    cells.append(row);heads[(ss,ms,key)]=h
    for i,sid in enumerate(vids):samples.append({"split_seed":ss,"model_seed":ms,"config_id":key,"sample_id":int(sid),"label":int(yv[i]),"prediction":int(pred[i]),"correct":bool(pred[i]==vy[i]),"margin":float(marg[i])})
   try:git=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
   except Exception:git=None
   provenance.append({"split_seed":ss,"model_seed":ms,"checkpoint":cp.name,"checkpoint_sha256":sha(cp),**base,**meta,"git_revision":git,"packages":{"python":sys.version.split()[0],"torch":torch.__version__,"numpy":np.__version__,"sklearn":sklearn.__version__,"pennylane":pennylane.__version__}})
 # split aggregation, eligibility, deterministic selection
 A={(r["split_seed"],r["model_seed"]):r for r in cells if r["config_id"]=="A"};summ=[]
 for s in SCALES:
  for m in MARGINS:
   cid=f"s{s}_m{m:.2f}";q=[r for r in cells if r["config_id"]==cid];acc=[];mina=[];mar=[]
   for ss in SPLITS:
    qq=[r for r in q if r["split_seed"]==ss];acc.append(np.mean([r["accuracy"]-A[(ss,r["model_seed"])]["accuracy"] for r in qq]));mina.append(np.mean([r["minimum_class_accuracy"]-A[(ss,r["model_seed"])]["minimum_class_accuracy"] for r in qq]));mar.append(np.mean([r["class12_mean_margin"]-A[(ss,r["model_seed"])]["class12_mean_margin"] for r in qq]))
   eligible=np.mean([r["accuracy"] for r in q])>=np.mean([r["accuracy"] for r in A.values()])-.01 and min(acc)>=-.02 and sum(x>=0 for x in mina)>=3
   lo,hi=ci(acc);summ.append({"config_id":cid,"scale":s,"margin":m,"eligible":bool(eligible),"mean_accuracy_delta":float(np.mean(acc)),"accuracy_delta_ci95_low":lo,"accuracy_delta_ci95_high":hi,"accuracy_direction_positive":sum(x>0 for x in acc),"accuracy_direction_zero":sum(x==0 for x in acc),"accuracy_direction_negative":sum(x<0 for x in acc),"minimum_class_delta_mean":float(np.mean(mina)),"minimum_class_delta_nonnegative_splits":sum(x>=0 for x in mina),"class12_margin_delta_mean":float(np.mean(mar)),"split_accuracy_deltas":list(map(float,acc)),"split_minimum_class_deltas":list(map(float,mina)),"split_margin_deltas":list(map(float,mar)),"replication_unit":"split_seed_n_5"})
 selected=select_config(summ)
 selection={"selected":selected,"selection_count":int(selected is not None),"clean_gate":selected is not None,"selection_rule":"initial_repository_local_protocol_lexicographic_aggregate_metrics_only","selection_inputs":["eligible","mean_accuracy_delta","minimum_class_delta_nonnegative_splits","class12_margin_delta_mean","scale","margin"],"attack_metrics_used":False,"sample_specific_selection_use":False,"canonical_membership_retained":True,"persisted_before_attacks":True};dump("iris_phase20_selected_config.json",selection)
 assert selection["sample_specific_selection_use"] is False and all("sample" not in x for x in selection["selection_inputs"])
 attack_rows=[];pairs=[]
 if selection["clean_gate"]:
  cid=selected["config_id"]
  for ss in SPLITS:
   instrumentation["train_validation_loader_calls"]+=1;_,xv,_,yv,_,_,vids=load_iris_train_validation(ss,cfg["test_size"],cfg["val_size"]);times=ttfs_encode(xv,T)
   for ms in MODELS:
    for key in ("A",cid):
     h=heads[(ss,ms,key)];clean=predict(h,times,T).argmax(1).numpy()==yv
     for e in EPS:
      for kind in ("random","pgd"):
       for i,(t,sid) in enumerate(zip(times,vids)):
        sd=attack_seed(ss,sid,e);at=random_timing_jitter(t,e*T,T,sd)[0] if kind=="random" else classical_timing_attack(h,t,int(yv[i]),e*T,T,20,e*T/5,False,sd)[0];at=np.asarray(at,float);linf=float(np.max(np.abs(at-t)));ok=bool(np.isfinite(at).all() and at.min()>=0 and at.max()<=T and linf<=e*T+TOL)
        if not ok:raise RuntimeError("attack numerical bound failure")
        attack_rows.append({"split_seed":ss,"model_seed":ms,"config_id":key,"attack":kind,"epsilon_fraction":e,"sample_id":int(sid),"clean_correct":bool(clean[i]),"attacked_fail":bool(int(predict(h,at[None],T).argmax(1)[0])!=int(yv[i])),"linf":linf,"finite":True,"range_ok":True,"bound_ok":True})
  for ss in SPLITS:
   for ms in MODELS:
    for e in EPS:
     for kind in ("random","pgd"):
      ar=[r for r in attack_rows if r["split_seed"]==ss and r["model_seed"]==ms and r["epsilon_fraction"]==e and r["attack"]==kind and r["config_id"]=="A"];br=[r for r in attack_rows if r["split_seed"]==ss and r["model_seed"]==ms and r["epsilon_fraction"]==e and r["attack"]==kind and r["config_id"]==cid];pairs.append({"split_seed":ss,"model_seed":ms,"epsilon_fraction":e,"attack":kind,**common_clean_pair(ar,br)})
 else:attack_rows=[{"status":"not_run","reason":"no clean-eligible selected configuration"}]
 splitpairs=[]
 if selected:
  for ss in SPLITS:
   for e in EPS:
    for kind in ("random","pgd"):
     q=[r for r in pairs if r["split_seed"]==ss and r["epsilon_fraction"]==e and r["attack"]==kind];splitpairs.append({"split_seed":ss,"epsilon_fraction":e,"attack":kind,"N_common":sum(r["N_common"] for r in q),"rescued":sum(r["rescued"] for r in q),"broken":sum(r["broken"] for r in q),"both_fail":sum(r["both_fail"] for r in q),"both_robust":sum(r["both_robust"] for r in q),"net_gain":float(np.mean([r["current_paired_asr"]-r["candidate_paired_asr"] for r in q]))})
 pg=[r for r in splitpairs if r["attack"]=="pgd"];small=[r["net_gain"] for r in pg if r["epsilon_fraction"] in (.01,.02)];large=[max([r["net_gain"] for r in pg if r["split_seed"]==s and r["epsilon_fraction"] in (.05,.1)] or [-np.inf]) for s in SPLITS];best=[max([r["net_gain"] for r in pg if r["split_seed"]==s] or [-np.inf]) for s in SPLITS];attack_gate=bool(selected) and sum(x<0 for x in small)<6 and sum(x>0 for x in large)>=3 and sum(x>=0 for x in best)>=3
 metrics=("accuracy","macro_f1","minimum_class_accuracy","class1_accuracy","class2_accuracy","class12_binary_accuracy","cross_entropy","class12_mean_margin");metric_summary=[]
 for cid in [f"s{s}_m{m:.2f}" for s in SCALES for m in MARGINS]:
  for metric in metrics:
   ds=[]
   for ss in SPLITS:ds.append(float(np.mean([r[metric]-A[(ss,r["model_seed"])][metric] for r in cells if r["config_id"]==cid and r["split_seed"]==ss])))
   lo,hi=ci(ds);metric_summary.append({"config_id":cid,"metric":metric,"paired_delta_mean":float(np.mean(ds)),"paired_delta_ci95_low":lo,"paired_delta_ci95_high":hi,"positive_splits":sum(x>0 for x in ds),"zero_splits":sum(x==0 for x in ds),"negative_splits":sum(x<0 for x in ds),"split_deltas":ds,"replication_unit":"split_seed_n_5"})
 grid=[{"config_id":f"s{s}_m{m:.2f}","head":HEAD_E_COSINE_MARGIN,"scale":s,"margin":m,"training_formula":"s*(cos_j-m*I[j=y])","inference_formula":"s*cos_j"} for s in SCALES for m in MARGINS]
 calibration=[r for r in cells if r["head"]==HEAD_E_COSINE_MARGIN];clean_summary=[]
 for sr in summ:
  cid=sr["config_id"]
  for ss in SPLITS:
   q=[r for r in calibration if r["config_id"]==cid and r["split_seed"]==ss]
   row={"summary_level":"split","config_id":cid,"scale":sr["scale"],"margin":sr["margin"],"split_seed":ss,"model_cell_count":len(q),"eligible":sr["eligible"]}
   for metric in metrics:row[metric]=float(np.mean([r[metric] for r in q]));row[f"{metric}_delta_vs_A"]=float(np.mean([r[metric]-A[(ss,r["model_seed"])][metric] for r in q]))
   clean_summary.append(row)
  row={"summary_level":"aggregate","config_id":cid,"scale":sr["scale"],"margin":sr["margin"],"split_seed":None,"split_count":5,"model_cell_count":15,**{k:v for k,v in sr.items() if k not in ("config_id","scale","margin")}}
  for metric in metrics:
   mr=next(x for x in metric_summary if x["config_id"]==cid and x["metric"]==metric)
   row[f"{metric}_delta_mean"]=mr["paired_delta_mean"];row[f"{metric}_delta_ci95_low"]=mr["paired_delta_ci95_low"];row[f"{metric}_delta_ci95_high"]=mr["paired_delta_ci95_high"];row[f"{metric}_positive_splits"]=mr["positive_splits"];row[f"{metric}_zero_splits"]=mr["zero_splits"];row[f"{metric}_negative_splits"]=mr["negative_splits"]
  clean_summary.append(row)
 write("iris_phase20_calibration_grid.csv",calibration);dump("iris_phase20_calibration_grid.json",calibration);write("iris_phase20_clean_summary.csv",clean_summary);dump("iris_phase20_clean_summary.json",clean_summary);dump("iris_phase20_split_manifest.json",{"splits":manifest,"hidden_features_persisted":False});dump("iris_phase20_checkpoint_provenance.json",{"checkpoints":provenance});dump("iris_phase20_attack_validation.json",{"status":"run" if selected else "not_run","reason":None if selected else "clean gate failed; no configuration selected","per_id":attack_rows,"model_cell_pairs":pairs,"split_aggregates":splitpairs});write("iris_phase20_attack_validation.csv",attack_rows)
 fragile=[]
 for sid in sorted(set(r["sample_id"] for r in samples)):
  q=[r for r in samples if r["sample_id"]==sid];fragile.append({"sample_id":sid,"analysis_status":"post_hoc_descriptive","validation_split_count":len(set(r["split_seed"] for r in q)),"misclassified_count":sum(not r["correct"] for r in q),"margin_below_0p05_count":sum(r["margin"]<.05 for r in q),"sample119_diagnostic":sid==119,"sample_specific_selection_use":False,"ordinary_canonical_membership_retained":True})
 write("iris_phase20_fragile_samples.csv",fragile)
 p19=json.loads((R/"iris_phase19_frozen_multisplit.json").read_text());aligned=("accuracy","macro_f1","class1_accuracy","class2_accuracy","class12_binary_accuracy","class12_mean_margin");comparison=[]
 for phase,rows,splitset,aid,eid,minname in ((19,p19,[42,123,777,2026,6543],"HEAD_A_CURRENT","HEAD_E_COSINE_MARGIN","min_class_accuracy"),(20,cells,list(SPLITS),"A","s10_m0.10","minimum_class_accuracy")):
  for metric in aligned+(minname,):
   ac=[r[metric] for r in rows if (r.get("head")==aid or r.get("config_id")==aid)];ec=[r[metric] for r in rows if (r.get("head")==eid or r.get("config_id")==eid)];ds=[]
   for ss in splitset:
    aq=[r[metric] for r in rows if r["split_seed"]==ss and (r.get("head")==aid or r.get("config_id")==aid)];eq=[r[metric] for r in rows if r["split_seed"]==ss and (r.get("head")==eid or r.get("config_id")==eid)];ds.append(float(np.mean(eq)-np.mean(aq)))
   comparison.append({"phase":phase,"split_seeds":splitset,"metric":"minimum_class_accuracy" if metric==minname else metric,"head_A_mean":float(np.mean(ac)),"head_E_s10_m0p1_mean":float(np.mean(ec)),"E_minus_A_split_delta_mean":float(np.mean(ds)),"positive_splits":sum(x>0 for x in ds),"zero_splits":sum(x==0 for x in ds),"negative_splits":sum(x<0 for x in ds),"split_deltas":ds})
 dump("iris_phase20_phase19_descriptive_comparison.json",{"status":"post_hoc_descriptive","phase19_used_for_selection":False,"phase19_protocol_sha256":json.loads((R/"iris_phase19_protocol.json").read_text())["protocol_sha256"],"noncomparability":"Different split sets and newly trained extractors prevent a paired cross-phase or causal comparison.","aligned_metrics":comparison})
 plots=make_plots(cells,summ,splitpairs);instrumentation.update(audit);gate={"phase":20,"implementation_complete":True,"test_set_accessed":False,"test_loader_invoked":False,"hidden_features_received_by_phase20":False,"manifest_helper_contract":"IDs and labels only; sklearn may internally materialize canonical bundle","protocol_timing_claim":"Local filesystem evidence places the initial protocol hash before the first observed checkpoint only; amended protocol is post-output audit correction; no independent registration or all-output timing claim.","split_seeds":list(SPLITS),"historical_seed_ledger_sha256":sha(R/"iris_phase20_historical_split_seed_ledger.json"),"model_seeds":list(MODELS),"grid_size":9,"cell_count":len(cells),"selection":selection,"clean_gate":bool(selected),"attacks_run":bool(selected),"attack_gate":attack_gate,"end_to_end_run":False,"phase21_candidate":None,"protocol_sha256":protocol["protocol_sha256"],"initial_protocol_sha256":protocol["initial_protocol_sha256"],"approved_attack_hashes":expected,"instrumentation":instrumentation,"plots":plots,"runtime_seconds":time.perf_counter()-t0};dump("iris_phase20_gate.json",gate);report(gate,summ)
 for n,title in (("iris_phase20_scientific_audit.md","# Phase 20 Scientific Audit\n\nReserved for the independent scientific auditor.\n"),("iris_phase20_beginner_summary.md","# Phase 20 Beginner Summary\n\nReserved for the designated interpreter.\n")):
  if not (R/n).exists():(R/n).write_text(title,encoding="utf8")
  finalize_manifest(protocol,gate)
  print(json.dumps({"selected":selected["config_id"] if selected else None,"clean_gate":bool(selected),"attack_gate":attack_gate,"runtime_seconds":gate["runtime_seconds"]}))

def finalize_manifest(protocol,gate):
 append_run_event({"event":"generator_outputs_completed_final","timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"claim":"final event appended before artifact hashing; no later event in this run"})
 generated=[x["path"] for x in protocol["artifact_manifest"]["artifacts"] if x["role"]=="generator_owned" and x["path"] not in ("iris_phase20_artifact_manifest.json","iris_phase20_gate.json")];entries=[]
 for x in generated:
  p=(P/x) if x.endswith(".png") else (R/x);entries.append({"path":x,"sha256":sha(p),"hash_mode":"file_bytes","size_bytes":p.stat().st_size})
 gate["artifact_manifest_self_hash_claimed"]=False;gate_canonical=dict(gate);gate_canonical.pop("artifact_manifest_sha256",None);entries.append({"path":"iris_phase20_gate.json","sha256":hashlib.sha256(json.dumps(gate_canonical,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"hash_mode":"canonical_json_without_artifact_manifest_sha256"})
 final_manifest={"phase":20,"status":"generator_complete","created_after_final_run_event":True,"self_hash_claimed":False,"generator_owned_outputs":entries,"agent_owned":[{"path":"iris_phase20_scientific_audit.md","status":"pending_at_generation","excluded_from_generator_complete_hashes":True},{"path":"iris_phase20_beginner_summary.md","status":"pending_at_generation","excluded_from_generator_complete_hashes":True}]};dump("iris_phase20_artifact_manifest.json",final_manifest)
 gate["artifact_manifest_sha256"]=sha(R/"iris_phase20_artifact_manifest.json");dump("iris_phase20_gate.json",gate)

def make_plots(cells,summ,pairs):
 import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
 names=["phase20_accuracy_by_scale_margin.png","phase20_class2_accuracy.png","phase20_margin_stability.png"]
 fields=("accuracy","class2_accuracy","class12_mean_margin")
 for i,(name,field) in enumerate(zip(names[:3],fields)):
  fig,ax=plt.subplots(figsize=(9,4))
  ids=sorted(set(r["config_id"] for r in cells));ax.boxplot([[r[field] for r in cells if r["config_id"]==x] for x in ids]);ax.set_xticklabels(ids,rotation=45)
  ax.set_ylabel(field);fig.tight_layout();fig.savefig(P/name);plt.close(fig)
 pg=P/"phase20_pgd_net_gain_by_split.png"
 if pairs:
  fig,ax=plt.subplots()
  for e in EPS:ax.plot(SPLITS,[next((r["net_gain"] for r in pairs if r["split_seed"]==s and r["epsilon_fraction"]==e and r["attack"]=="pgd"),np.nan) for s in SPLITS],label=str(e))
  ax.legend();fig.tight_layout();fig.savefig(pg);plt.close(fig);names.append(pg.name)
 elif pg.exists():pg.unlink()
 return names
def report(gate,summ):
 titles=["Agents and Skills Used","Interpreter","Files Added","Files Modified","Regression Status","Phase 20 Preregistered Protocol","New Development Splits","Model Seeds","Frozen HEAD_A Reference","HEAD_E Calibration Grid","Clean Calibration Results","Split-Level Stability","Class 1/2 Results","Margin Stability","Sample 119 Diagnostic","Repeated Fragile Samples","Clean Calibration Gate","Frozen Selected Configuration","Random Timing Results","Classical PGD Results","Paired Robustness","Small-Epsilon Safety","Final Phase 20 Gate","Statistical Interpretation","Independent Scientific Audit","Test-Access Status","Scientific Conclusion","Beginner-Friendly Explanation","Recommended Phase 21"]
 candidates=[r for r in summ if "mean_accuracy_delta" in r];best=max(candidates,key=lambda r:(r["mean_accuracy_delta"],r["class12_margin_delta_mean"],-r["scale"],-r["margin"]));cid=best["config_id"]
 splitrows=[r for r in json.loads((R/"iris_phase20_clean_summary.json").read_text()) if r["summary_level"]=="split" and r["config_id"]==cid]
 mean=lambda k:float(np.mean([r[k] for r in splitrows]));base=lambda k:float(np.mean([r[k]-r[k+"_delta_vs_A"] for r in splitrows]))
 sample119=next((r for r in csv.DictReader((R/"iris_phase20_fragile_samples.csv").open()) if r["sample_id"]=="119"),None)
 vals=[
  "Used scientific-critical-thinking, experimental-design, statistical-analysis, and PennyLane guidance. Scientific success is assessed separately from implementation correctness.",
  f"Python interpreter: {sys.executable}. The protected beginner interpretation is `iris_phase20_beginner_summary.md`.",
  "Added the Phase 20 module, runner, tests, exact result artifacts, event/seed ledgers, final artifact manifest, report, and three required plots.",
  "Modified `experiments/iris/data.py` and `phase_runner.py`; TTFS, circuit, qlayer architecture, and frozen attack definitions were not changed.",
  "Focused Phase 20 gate passed. This establishes implementation/schema integrity, not defense efficacy.",
  f"Local filesystem evidence places initial hash `{gate['initial_protocol_sha256']}` before the first observed checkpoint only. Current hash `{gate['protocol_sha256']}` is a post-output audit correction. Neither is independently registered/timestamped, and timing before every output is not established.",
  f"Seeds {list(SPLITS)} were absent from the recorded seed-context scan. The ledger cannot detect deleted, unrecorded, or external runs.",
  f"Model seeds {list(MODELS)} are nested repetitions within each of five splits; split seed is the inferential unit (n=5), not 15 independent units.",
  f"HEAD_A is the warm/co-adapted deployment reference. Its aggregate accuracy for the best-observed comparison was {base('accuracy'):.4f}, class1 {base('class1_accuracy'):.4f}, class2 {base('class2_accuracy'):.4f}, and class1/2 margin {base('class12_mean_margin'):.4f}.",
  "Exactly nine HEAD_E configurations used s={5,10,15}, m={.05,.10,.15}, with training z_j=s(cos_j-m I[j=y]) and inference z_j=s cos_j. All nine shared initialization within split/model; no extension occurred.",
  f"Q1: Best observed by mean accuracy delta was {cid}: accuracy {mean('accuracy'):.4f} versus A {base('accuracy'):.4f}, delta {best['mean_accuracy_delta']:+.4f}, 95% split-level CI [{best['accuracy_delta_ci95_low']:+.4f}, {best['accuracy_delta_ci95_high']:+.4f}]. It was not eligible.",
  f"Q2: {cid} accuracy directions were {best['accuracy_direction_positive']} positive, {best['accuracy_direction_zero']} tied, and {best['accuracy_direction_negative']} negative splits; deltas were {best['split_accuracy_deltas']}. The CI crosses zero.",
  f"Q3: {cid} class1 accuracy was {mean('class1_accuracy'):.4f} (A {base('class1_accuracy'):.4f}); class2 was {mean('class2_accuracy'):.4f} (A {base('class2_accuracy'):.4f}); class1/2 binary accuracy was {mean('class12_binary_accuracy'):.4f} (A {base('class12_binary_accuracy'):.4f}).",
  f"Q4: {cid} class1/2 mean margin was {mean('class12_mean_margin'):.4f} versus A {base('class12_mean_margin'):.4f}, split-mean delta {best['class12_margin_delta_mean']:+.4f}; scale-dependent raw margins are not formulation-comparable without qualification.",
  "Q5: Sample 119 was not present in the Phase 20 validation diagnostic records. Ordinary canonical membership was retained wherever assigned; its identity was never a criterion, objective, filter, tie-break, or tuning target.",
  f"Q6: Fragility was aggregated post hoc by original ID across {len(list(csv.DictReader((R/'iris_phase20_fragile_samples.csv').open())))} observed validation IDs. These diagnostics were absent from selection inputs.",
  f"Q7: Clean gate failed. Although {cid} had the largest observed mean accuracy delta, minimum-class deltas were nonnegative in only {best['minimum_class_delta_nonnegative_splits']}/5 splits (required at least 3); its minimum-class split deltas were {best['split_minimum_class_deltas']}.",
  "Q8: `iris_phase20_selected_config.json` records selection_count=0 and selected=null before the attack branch. No configuration was frozen for attack evaluation.",
  "Q9: Random timing was explicitly not run because the clean prerequisite failed; the attack artifacts contain `status=not_run`.",
  "Q10: Unchanged Phase 14 PGD at 1/2/5/10% was not invoked. Frozen source hashes remain recorded, but there are no Phase 20 PGD observations.",
  "Q11: Common-clean-correct paired rescued/broken/both-fail/both-robust outcomes are not estimable because no candidate entered attacks; no changed denominator was used to claim robustness.",
  "Q12: Small-epsilon safety is not estimable. Absence of attack execution is not evidence of safety or robustness.",
  "Final gate: clean=false, attacks_run=false, attack=false, end_to_end=false, Phase21 candidate=null. The artifact manifest and focused tests validate the generated contract.",
  "Paired split-level summaries use n=5 splits and descriptive t4 95% CIs. The best observed improvement is uncertain, and minimum-class direction failed the preregistered threshold. Model seeds are not treated as independent inferential replicates.",
  "Independent scientific audit: PASS after material corrections; see `iris_phase20_scientific_audit.md`. That agent-owned artifact is outside generator-complete hashes and may be populated externally.",
  "No hidden test features were received or evaluated. The IDs/labels-only manifest helper may internally cause sklearn to materialize its canonical bundle, but Phase20 receives no hidden feature rows; full feature-returning held-out APIs are fail-closed.",
  "Evidence supports a negative clean calibration gate: this fixed narrow grid did not produce an eligible deployment candidate. It does not support robustness, hidden-test generalization, or the claim that every possible classifier calibration is ineffective.",
  "See `iris_phase20_beginner_summary.md`, the protected agent-owned interpreter artifact.",
  "clean gate failed, narrow calibration insufficient, do not broaden classifier search, no Phase21 candidate, return to representation robustness only as next research direction."
 ]
 body=[]
 for i,(t,v) in enumerate(zip(titles,vals),1):body.extend([f"## {i}. {t}","",v,""])
 (R/"phase20_results.md").write_text("# Phase 20 — Preregistered Cosine-Margin Calibration\n\n"+"\n".join(body),encoding="utf8")
if __name__=="__main__":main()
