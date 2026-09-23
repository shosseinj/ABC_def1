"""Phase 21 representation-shift diagnosis and conditional consistency defense."""
import hashlib
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn.functional as F

SPLIT_SEEDS=(271,811,1618,2718,4242)
MODEL_SEEDS=(42,777,2026)
LAMBDAS=(0.1,0.5,1.0,2.0)
EPSILONS=(0.01,0.02,0.05,0.10)

def safe_mean(values):
    values=[float(x) for x in values if x is not None and np.isfinite(x)]
    return (float(np.mean(values)),"defined",len(values)) if values else (None,"undefined_empty",0)

def true_margin(logits,label):
    z=np.asarray(logits,float); return float(z[label]-np.max(np.delete(z,label)))

def cosine_distance(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);d=np.linalg.norm(a)*np.linalg.norm(b)
    return (float(1-np.dot(a,b)/d),"defined") if d>0 else (None,"undefined_zero_norm")

def nearest_centroid(reference_x,reference_y,query):
    classes=sorted(set(map(int,reference_y)));c=np.stack([np.asarray(reference_x)[np.asarray(reference_y)==x].mean(0) for x in classes])
    d=np.linalg.norm(np.asarray(query)[:,None,:]-c[None,:,:],axis=2)
    return np.asarray(classes)[d.argmin(1)],d

def neighbors(reference_x,reference_y,reference_ids,query,k=3):
    out=[]
    for q in np.asarray(query):
        order=sorted(range(len(reference_x)),key=lambda i:(float(np.linalg.norm(q-reference_x[i])),int(reference_ids[i])))[:k]
        out.append([{"sample_id":int(reference_ids[i]),"label":int(reference_y[i]),"distance":float(np.linalg.norm(q-reference_x[i]))} for i in order])
    return out

def representation_loss(clean_features,perturbed_features):
    return (1-F.cosine_similarity(clean_features,perturbed_features,dim=1)).mean()

def grad_norms(loss,model):
    q=list(model.qlayer.parameters());h=list(model.head.parameters());params=q+h
    grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    def norm(gs):
        present=[g.reshape(-1) for g in gs if g is not None]
        return (float(torch.linalg.vector_norm(torch.cat(present)).detach()),"defined") if present else (None,"none_structural")
    qn,qs=norm(grads[:len(q)]);hn,hs=norm(grads[len(q):])
    return {"qlayer_norm":qn,"qlayer_status":qs,"head_norm":hn,"head_status":hs}

def grouped_contrast(rows,metric,split_seed,excluded_id=None):
    """Equal epsilon then available-model successful-minus-robust PGD contrast."""
    epsilon_values=[];detail=[]
    for eps in EPSILONS:
        models=[]
        for ms in MODEL_SEEDS:
            q=[r for r in rows if r["attack"]=="pgd" and r["split_seed"]==split_seed and r["model_seed"]==ms and r["epsilon_fraction"]==eps and r["clean_correct"] and r["sample_id"]!=excluded_id]
            s=[r[metric] for r in q if r["attack_success"] and r.get(metric) is not None];b=[r[metric] for r in q if not r["attack_success"] and r.get(metric) is not None]
            if s and b:models.append(float(np.mean(s)-np.mean(b)))
            detail.append({"epsilon_fraction":eps,"model_seed":ms,"successful_n":len(s),"robust_n":len(b),"status":"defined" if s and b else "undefined_missing_group"})
        if models:epsilon_values.append(float(np.mean(models)))
    value,status,n=safe_mean(epsilon_values)
    return {"value":value,"status":status,"defined_epsilon_count":n,"denominators":detail}

def spearman_descriptive(rows):
    from scipy.stats import spearmanr
    q=[r for r in rows if r["attack"]=="pgd" and r["clean_correct"] and r.get("quantum_l2") is not None and r.get("margin_drop") is not None]
    if len(q)<2:return {"rho":None,"status":"undefined_n_lt_2","n":len(q)}
    rho=float(spearmanr([r["quantum_l2"] for r in q],[r["margin_drop"] for r in q]).statistic)
    return {"rho":rho if np.isfinite(rho) else None,"status":"defined" if np.isfinite(rho) else "undefined_constant","n":len(q),"inference":"descriptive_only_nonindependent_observations"}

def stage_a_gate(rows):
    q={s:grouped_contrast(rows,"quantum_l2",s) for s in SPLIT_SEEDS};m={s:grouped_contrast(rows,"margin_drop",s) for s in SPLIT_SEEDS}
    qvals=[q[s]["value"] for s in SPLIT_SEEDS if q[s]["value"] is not None];agg,st,_=safe_mean(qvals);rho=spearman_descriptive(rows)
    ids=sorted(set(r["sample_id"] for r in rows));effects=[]
    for sid in ids:
        vals=[grouped_contrast(rows,"quantum_l2",s,sid)["value"] for s in SPLIT_SEEDS];a,_,_=safe_mean(vals)
        effects.append((abs(a-agg) if a is not None and agg is not None else -1,sid,a,vals))
    top=max(effects,key=lambda x:(x[0],-x[1])) if effects else (None,None,None,[])
    loo_rows=[{"excluded_sample_id":sid,"aggregate_quantum_l2_contrast":a,"positive_split_count":sum(v is not None and v>0 for v in vals),"all_required_directions_hold":sum(v is not None and v>0 for v in vals)>=3,**{f"split_{s}":v for s,v in zip(SPLIT_SEEDS,vals)}} for _,sid,a,vals in effects]
    loo_positive=sum(v is not None and v>0 for v in top[3])
    criteria={"criterion1_aggregate_quantum_positive":bool(agg is not None and agg>0),"criterion2_quantum_positive_splits":sum(q[s]["value"] is not None and q[s]["value"]>0 for s in SPLIT_SEEDS)>=3,"criterion3_margin_and_spearman":sum(m[s]["value"] is not None and m[s]["value"]>0 for s in SPLIT_SEEDS)>=3 and rho["rho"] is not None and rho["rho"]>0,"criterion4_top_contributor_exclusion":loo_positive>=3}
    qci=None
    if len(qvals)==5:
        half=2.776*float(np.std(qvals,ddof=1))/np.sqrt(5);qci=[float(agg-half),float(agg+half)]
    return {"pass":all(criteria.values()),"criteria":criteria,"posthoc_all_id_influence_sensitivity":{"status":"post_output_post_hoc_not_gate_input","rows":loo_rows,"all_exclusions_retain_direction":bool(loo_rows) and all(r["all_required_directions_hold"] for r in loo_rows)},"aggregate_quantum_l2_contrast":agg,"aggregate_quantum_l2_descriptive_t4_ci95":qci,"aggregate_status":st,"split_quantum_l2":q,"split_margin_drop":m,"spearman":rho,"top_contributor_id":top[1],"top_contributor_absolute_aggregate_change":top[0],"top_contributor_excluded_aggregate":top[2],"top_contributor_excluded_split_values":dict(zip(SPLIT_SEEDS,top[3])),"top_contributor_excluded_positive_splits":loo_positive,"implementation":"confirmatory top-contributor exclusion retained from initially recorded criterion; exhaustive all-ID sensitivity is post-output/post-hoc and not a gate input"}

def common_clean_pair(base,candidate):
    a={r["sample_id"]:r for r in base if r["clean_correct"]};b={r["sample_id"]:r for r in candidate if r["clean_correct"]};ids=sorted(set(a)&set(b));out={"rescued":0,"broken":0,"both_fail":0,"both_robust":0}
    for i in ids:
        x,y=a[i]["attack_success"],b[i]["attack_success"]
        out["both_fail" if x and y else "rescued" if x else "broken" if y else "both_robust"]+=1
    return {"N_common_clean_correct":len(ids),**out,"baseline_failures":sum(a[i]["attack_success"] for i in ids),"candidate_failures":sum(b[i]["attack_success"] for i in ids),"common_ids":ids,"status":"defined" if ids else "undefined_empty"}

@contextmanager
def no_hidden_feature_guard():
    import experiments.iris.data as data,experiments.iris.training as training
    old_data,old_training=data.load_iris_splits,training.load_iris_splits
    audit={"full_loader_calls":0}
    def deny(*a,**k):audit["full_loader_calls"]+=1;raise RuntimeError("Phase21 prohibits full held-out feature loaders")
    data.load_iris_splits=deny;training.load_iris_splits=deny
    try:yield audit
    finally:data.load_iris_splits=old_data;training.load_iris_splits=old_training
