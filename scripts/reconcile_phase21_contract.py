"""Reconcile completed Phase-21 Stage-A outputs to the exact public contract.

This performs no training, attack generation, selection, or gate recomputation.
It only losslessly expands/partitions the already generated 3,600 Stage-A rows,
removes conditional Stage-B placeholders, rewrites the report, and re-hashes files.
"""
from pathlib import Path
import csv,hashlib,json,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];R=ROOT/"results";P=R/"plots"
def dump(path,obj):path.write_text(json.dumps(obj,indent=2),encoding="utf8")
def write(path,rows):
    rows=list(rows);fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

source_path=R/"iris_phase21_stage_a_samples.json"
if not source_path.exists():source_path=R/"iris_phase21_repr_diagnosis.json"
gate_path=R/"iris_phase21_stage_a_gate.json"
if not gate_path.exists():gate_path=R/"iris_phase21_stageA_gate.json"
source=json.loads(source_path.read_text())
gate=json.loads(gate_path.read_text())
manifests={x["split_seed"]:x for x in json.loads((R/"iris_phase21_split_manifest.json").read_text())["splits"]}
assert len(source)==3600 and len({(r["split_seed"],r["model_seed"],r["sample_id"],r["attack"],r["epsilon_fraction"]) for r in source})==3600

# Add true pre-normalization feature coordinates from the persisted training-only
# scaler parameters. This is a deterministic coordinate conversion, not data access.
diagnosis=[]
for old in source:
    r=dict(old);m=manifests[r["split_seed"]];lo=np.asarray(m["scaler_data_min"],float);hi=np.asarray(m["scaler_data_max"],float)
    clean=np.asarray(json.loads(r["clean_normalized_features"]));attacked=np.asarray(json.loads(r["attacked_normalized_features"]));cr=lo+clean*(hi-lo);ar=lo+attacked*(hi-lo)
    r["clean_raw_features"]=json.dumps(cr.tolist());r["attacked_raw_features"]=json.dumps(ar.tolist());r["raw_feature_l2"]=float(np.linalg.norm(ar-cr));diagnosis.append(r)
write(R/"iris_phase21_repr_diagnosis.csv",diagnosis);dump(R/"iris_phase21_repr_diagnosis.json",diagnosis)

groups={}
for r in diagnosis:
    key=(r["split_seed"],r["model_seed"],r["attack"],r["epsilon_fraction"],r["classification_status"])
    groups.setdefault(key,[]).append(r)
robust=[]
for key,q in sorted(groups.items()):
    clean=[x for x in q if x["clean_correct"]];status=key[-1]
    robust.append({"split_seed":key[0],"model_seed":key[1],"attack":key[2],"epsilon_fraction":key[3],"outcome":status,"n":len(q),"clean_correct_required":status in ("successful","robust"),"mean_quantum_l2":float(np.mean([x["quantum_l2"] for x in q])),"mean_margin_drop":float(np.mean([x["margin_drop"] for x in q])),"denominator_status":"defined" if q else "undefined_empty"})
write(R/"iris_phase21_robust_vs_fail.csv",robust)

cross=[{k:r[k] for k in ("split_seed","model_seed","sample_id","label","attack","epsilon_fraction","clean_correct","classification_status","nearest_train_centroid_clean","nearest_train_centroid_attacked","centroid_crossing","clean_centroid_distances","attacked_centroid_distances")} for r in diagnosis]
write(R/"iris_phase21_centroid_crossing.csv",cross)
purity=[]
for r in diagnosis:
    c=json.loads(r["clean_neighbor_labels"]);a=json.loads(r["attacked_neighbor_labels"]);y=r["label"]
    purity.append({k:r[k] for k in ("split_seed","model_seed","sample_id","label","attack","epsilon_fraction","clean_correct","classification_status")}|{"reference":"training_only_k3","clean_local_purity":sum(x==y for x in c)/len(c),"attacked_local_purity":sum(x==y for x in a)/len(a),"local_purity_delta":sum(x==y for x in a)/len(a)-sum(x==y for x in c)/len(c),"neighbor_label_sequence_changed":r["local_neighbor_label_change"],"clean_neighbor_ids":r["clean_neighbor_ids"],"attacked_neighbor_ids":r["attacked_neighbor_ids"]})
write(R/"iris_phase21_local_purity.csv",purity)
dump(R/"iris_phase21_stageA_gate.json",gate)

# Exact contract: conditional Stage-B artifacts do not exist after a failed Stage A.
for pattern in ("iris_phase21_stage_b_*","iris_phase21_paired_outcomes.*","iris_phase21_selected_config.json"):
    for p in R.glob(pattern):p.unlink()
for p in (R/"iris_phase21_stage_a_samples.csv",R/"iris_phase21_stage_a_samples.json",R/"iris_phase21_stage_a_gate.json"):
    if p.exists():p.unlink()

q={int(k):v["value"] for k,v in gate["split_quantum_l2"].items()};m={int(k):v["value"] for k,v in gate["split_margin_drop"].items()};vals=np.asarray(list(q.values()));half=2.776*vals.std(ddof=1)/np.sqrt(5);ci=[float(vals.mean()-half),float(vals.mean()+half)]
titles=["Agents and Skills Used","Interpreter","Files Added","Files Modified","Regression Status","Phase 21 Protocol","Development Splits","Model Seeds","Stage A Representation Metrics","Robust vs Failed Samples","Class-Wise Representation Shift","Centroid Crossing","Local Purity Stability","Multi-Split Reproducibility","Stage A Gate","Stage B Objective","Lambda Ablation","Stage B Training Diagnostics","Stage B Clean Gate","Anti-Collapse Geometry","Train-Only Linear Diagnostic","Stage B Selection","Random Timing Comparison","Phase 14 PGD Comparison","Common-Clean-Correct Pairing","Rescued and Broken Outcomes","Small-Epsilon Safety","Robustness Gate","TEMP Transfer","Statistical Interpretation","Denominator and Missingness Status","Sample 119","Held-Out Data Boundary","Artifact and Provenance Audit","Scientific Conclusion"]
texts=[
"Q1: Used scientific-critical-thinking, experimental-design, statistical-analysis, and PennyLane guidance. Implementation success and scientific success are reported separately.",
f"Q2: Python interpreter provenance is recorded by the generator report; reconciliation used the repository interpreter.",
"Added the exact Stage-A diagnosis, robust-versus-failed, centroid-crossing, local-purity, StageA-gate, report, and exhaustive manifest contract files.",
"Q3: Updated the Phase-21 contract exporter/tests only. TTFS, circuit, deployment head family, training rule, and attacks were not altered.",
"Focused Phase-21 tests pass. This verifies implementation/schema behavior, not defense efficacy.",
"Q4: The immutable repository-local protocol was persisted before Phase-21 outputs. Its provenance is not independent registration and cannot exclude deleted, external, or unrecorded work.",
"Seeds 271, 811, 1618, 2718, and 4242 were absent from recorded split-seed context before output inspection. Each exact manifest contains 90 train, 30 validation, and 30 hidden IDs/labels with disjoint membership.",
"Q5: Model seeds 42, 777, and 2026 are nested matched repetitions within each split. The scientific replication unit is split, n=5.",
f"Q6: `iris_phase21_repr_diagnosis.csv/json` contains exactly {len(diagnosis)} rows: 15 cells x 2 attacks x 4 epsilons x 30 validation samples, with raw, normalized, TTFS, measured quantum, logit, margin, prediction, geometry, and canonical-ID fields.",
"Q7: Successful means clean-correct then attacked-incorrect; robust means clean-correct then attacked-correct. Clean-incorrect rows are explicitly excluded rather than moved into either denominator. Group counts and means are in `iris_phase21_robust_vs_fail.csv`.",
"Class labels are retained per row for descriptive stratification only. No class-specific result, including sample 119, entered the gate or selected a defense.",
"Q8: `iris_phase21_centroid_crossing.csv` contains all 3,600 clean/attacked nearest-centroid states and crossing indicators using training-only centroids.",
"Q9: `iris_phase21_local_purity.csv` contains all 3,600 clean/attacked k=3 training-neighbor purities, deltas, IDs, and label-sequence change indicators.",
f"Split quantum successful-minus-robust contrasts were {q}. Their equal-split mean was {gate['aggregate_quantum_l2_contrast']:.9f}, descriptive t4 95% CI [{ci[0]:.9f}, {ci[1]:.9f}]. Only 2/5 directions were positive.",
f"Q10: Stage A failed. Criterion values were {gate['criteria']}; all four were required without post-result modification.",
"Q11: The prespecified Stage-B objective was CE + lambda times cosine representation inconsistency under random 0.02T timing jitter, with no other loss. It was not executed because Stage A failed.",
"The fixed lambda set {0.1, 0.5, 1, 2} was not evaluated. No lambda result files exist.",
"CE, Lrepr, weighted ratios, and separated qlayer/head gradients are substantively not available because Stage B was not run.",
"The validation-only clean gate is substantively not estimable; absence of a Stage-B run is not a clean-performance result.",
"Q12: Train-only centroid/spread anti-collapse checks are substantively not estimable for a candidate.",
"The train-fitted, validation-evaluated linear diagnostic is substantively not estimable for a candidate.",
"No candidate was selected. Selection cardinality is zero and no Stage-B selection artifact is emitted after the failed prerequisite.",
"Candidate random-jitter comparison was not run. Baseline Stage-A random-jitter observations remain in the diagnosis artifact.",
"Candidate Phase-14 PGD comparison was not run. Baseline Stage-A PGD used the unchanged frozen implementation at 1/2/5/10%.",
"Q13: Candidate common-clean-correct pairing is not estimable. No denominator was changed to manufacture a comparison.",
"Rescued, broken, both-fail, and both-robust candidate outcomes are not estimable and no misleading placeholder result artifact is retained.",
"Small-epsilon candidate safety is not estimable; non-execution is not evidence of safety.",
"The candidate robustness gate is not estimable and no robustness claim is made.",
"TEMP transfer was not run because the prerequisite gates were not satisfied.",
f"Split is n=5. The quantum-shift CI crosses zero. The pooled Spearman rho={gate['spearman']['rho']:.6f} over n={gate['spearman']['n']} correlated rows is descriptive only, not pooled inference.",
"All empty successful/robust cells carry explicit undefined-missing-group status in the gate; they were never imputed as zero.",
"Sample 119 remains ordinary canonical data wherever assigned and was never an objective, filter, criterion, tie-break, or selection input.",
"No hidden feature rows were received or evaluated. Hidden IDs and labels were retained only in the manifest; full feature-returning held-out loaders were fail-closed.",
"Q14: The final exhaustive manifest hashes all generator-owned Phase-21 outputs after reconciliation and excludes protected agent-owned audit/interpreter reports from generator-complete hash claims.",
f"Evidence supports a negative Stage-A gate: aggregate quantum shift was positive, but direction replicated in only 2/5 splits and all five margin-drop contrasts were negative ({m}). Therefore there is no defense result. Next falsifiable experiment: independently repeat the frozen Stage-A protocol on new development splits before proposing another representation defense."
]
titles=titles[:15]
texts=texts[:14]+[f"Q10: Stage A failed with criteria {gate['criteria']}; all four were required without post-result modification. The split quantum contrast was positive in only 2/5 splits and its descriptive t4 CI crossed zero. Q11: Stage B's prespecified CE-plus-cosine objective, fixed lambda ablation, loss/gradient monitoring, and clean gate were not run. Q12: Anti-collapse geometry, the train-only linear diagnostic, and at-most-one selection were not run. Q13: Candidate random-jitter/Phase14-PGD attacks, common-clean-correct rescued/broken/both-fail/both-robust outcomes, small-epsilon and robustness gates, and TEMP transfer are not estimable; no Stage-B artifacts exist and non-execution is not robustness. Split is n=5; nested model seeds and pooled Spearman rho={gate['spearman']['rho']:.6f} are descriptive only; undefined denominators were not set to zero. Sample 119 was ordinary data and never a criterion. No held-out feature rows were accessed; full loaders were fail-closed. Root classification: negative Stage-A scientific mechanism gate, not implementation failure and not proof all defenses fail. Q14: See protected `iris_phase21_scientific_audit.md` and `iris_phase21_beginner_summary.md`; the exhaustive manifest hashes generator outputs. Phase 22 decision: do not advance a defense; first independently replicate the frozen Stage-A protocol on new development splits."]
assert len(titles)==len(texts)==15
body=[]
for i,(t,x) in enumerate(zip(titles,texts),1):body.extend([f"## {i}. {t}","",x,""])
(R/"phase21_results.md").write_text("# Phase 21 — Representation Diagnosis\n\n"+"\n".join(body),encoding="utf8")

with (R/"iris_phase21_run_events.jsonl").open("a",encoding="utf8") as f:f.write(json.dumps({"event":"exact_contract_reconciled","timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"scientific_results_changed":False,"stage_b_artifacts_removed":True},sort_keys=True)+"\n")

generated=[]
for p in sorted(list(R.glob("iris_phase21_*"))+[R/"phase21_results.md"]+list(P.glob("phase21_*"))):
    if p.name in {"iris_phase21_artifact_manifest.json","iris_phase21_scientific_audit.md","iris_phase21_beginner_summary.md"}:continue
    generated.append({"path":p.name,"location":"results/plots" if p.parent==P else "results","sha256":sha(p),"size_bytes":p.stat().st_size,"hash_mode":"file_bytes"})
manifest={"phase":21,"status":"exact_contract_generator_complete","self_hash_claimed":False,"stage_b_executed":False,"conditional_stage_b_artifacts_present":False,"generator_owned_outputs":generated,"agent_owned":[{"path":"iris_phase21_scientific_audit.md","status":"protected","excluded_from_generator_complete_hashes":True},{"path":"iris_phase21_beginner_summary.md","status":"protected","excluded_from_generator_complete_hashes":True}]}
dump(R/"iris_phase21_artifact_manifest.json",manifest)
print(json.dumps({"diagnosis_rows":len(diagnosis),"stage_a_pass":gate["pass"],"artifact_count":len(generated)}))
