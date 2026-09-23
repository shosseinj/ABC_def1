from pathlib import Path
import hashlib,inspect,json
import numpy as np,torch
from experiments.iris.phase21 import *
ROOT=Path(__file__).resolve().parents[1]

def test_frozen_constants_and_cosine_loss():
    assert SPLIT_SEEDS==(271,811,1618,2718,4242) and MODEL_SEEDS==(42,777,2026)
    assert LAMBDAS==(.1,.5,1.,2.) and EPSILONS==(.01,.02,.05,.1)
    assert torch.isclose(representation_loss(torch.tensor([[1.,0]]),torch.tensor([[1.,0]])),torch.tensor(0.))

def test_protocol_hash_and_boundaries():
    p=json.loads((ROOT/"results/iris_phase21_protocol.json").read_text());h=p.pop("protocol_sha256")
    assert h==hashlib.sha256(json.dumps(p,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    assert p["prior_phase21_output_inspection"]=="not_independently_established" and p["data"]["sizes"]=={"train":90,"validation":30,"hidden":30}
    src=(ROOT/"scripts/run_phase21_representation_consistency.py").read_text()
    assert "load_iris_splits" not in src and "evaluate_test=False" in src
    assert "classical_timing_attack(model,clean_t,int(y),eps*T,T,20,eps*T/5,False" in src

def test_id_restricted_provider_rejects_hidden():
    import pytest
    from experiments.iris.data import load_iris_split_manifest_labels,load_iris_features_for_allowed_ids
    m=load_iris_split_manifest_labels(271,.2,.2);allowed=list(m["train_ids"])+list(m["validation_ids"])
    x,y=load_iris_features_for_allowed_ids(m["validation_ids"][:2],allowed,m["hidden_ids"]);assert x.shape==(2,4)
    with pytest.raises(PermissionError):load_iris_features_for_allowed_ids([m["hidden_ids"][0]],allowed,m["hidden_ids"])

def test_safe_denominators_pairing_and_gate():
    assert safe_mean([])[1]=="undefined_empty"
    a=[{"sample_id":1,"clean_correct":True,"attack_success":True},{"sample_id":2,"clean_correct":True,"attack_success":False}]
    b=[{"sample_id":1,"clean_correct":True,"attack_success":False},{"sample_id":2,"clean_correct":False,"attack_success":False}]
    q=common_clean_pair(a,b);assert q["N_common_clean_correct"]==1 and q["rescued"]==1
    assert "excluded_id" in inspect.signature(grouped_contrast).parameters

def test_independent_shift_centroid_neighbor_numerics():
    assert np.isclose(np.linalg.norm(np.array([3.,4.])-np.array([0.,0.])),5)
    pred,d=nearest_centroid(np.array([[0.],[2.]]),np.array([1,2]),np.array([[1.8]]));assert pred[0]==2
    n=neighbors(np.array([[0.],[2.],[2.]]),np.array([1,2,1]),np.array([3,2,1]),np.array([[2.]]));assert [x["sample_id"] for x in n[0]]==[1,2,3]

def test_primary_runner_direct_final_contract_and_cleanup():
    src=(ROOT/"scripts/run_phase21_representation_consistency.py").read_text()
    for name in ("iris_phase21_repr_diagnosis.csv","iris_phase21_robust_vs_fail.csv","iris_phase21_centroid_crossing.csv","iris_phase21_local_purity.csv","iris_phase21_stageA_gate.json","phase21_feature_shift_by_attack.png","phase21_robust_vs_fail_shift.png","phase21_centroid_crossing.png"):
        assert name in src
    assert "reconcile_phase21_contract" not in src
    assert "for stale in R.glob(pattern):stale.unlink()" in src
    assert "emit_stage_a_contract(stagea,gatea)" in src

def test_gradient_partition_head_is_structurally_none():
    class M(torch.nn.Module):
        def __init__(self):super().__init__();self.qlayer=torch.nn.Linear(2,2);self.head=torch.nn.Linear(2,2)
    m=M();x=torch.ones(2,2);q=m.qlayer(x);loss=representation_loss(q,q+.1)
    g=grad_norms(loss,m);assert g["qlayer_status"]=="defined" and g["head_status"]=="none_structural"

def test_registration_and_generated_contract_if_present():
    from phase_runner import PHASE_TESTS
    assert PHASE_TESTS[21].endswith("test_phase21_representation_consistency.py")
    gate=ROOT/"results/iris_phase21_gate.json"
    if not gate.exists():return
    g=json.loads(gate.read_text());assert not g["test_set_accessed"] and not g["hidden_features_received"] and g["full_loader_calls"]==0
    manifests=json.loads((ROOT/"results/iris_phase21_split_manifest.json").read_text())["splits"]
    assert all(x["sizes"]==[90,30,30] and x["pairwise_intersections"]==[0,0,0] for x in manifests)

def test_exact_failed_stage_a_public_contract():
    import csv
    required=["iris_phase21_repr_diagnosis.csv","iris_phase21_repr_diagnosis.json","iris_phase21_robust_vs_fail.csv","iris_phase21_centroid_crossing.csv","iris_phase21_local_purity.csv","iris_phase21_stageA_gate.json","iris_phase21_scientific_audit.md","iris_phase21_beginner_summary.md","phase21_results.md"]
    assert all((ROOT/"results"/x).is_file() for x in required)
    rows=json.loads((ROOT/"results/iris_phase21_repr_diagnosis.json").read_text())
    assert len(rows)==3600 and len({(r["split_seed"],r["model_seed"],r["sample_id"],r["attack"],r["epsilon_fraction"]) for r in rows})==3600
    fields=set(rows[0]);assert {"clean_raw_features","attacked_raw_features","clean_normalized_features","attacked_normalized_features","clean_ttfs","attacked_ttfs","clean_quantum_features","attacked_quantum_features","clean_logits","attacked_logits","quantum_l2","margin_drop","centroid_crossing","sample_id"}<=fields
    assert sum(1 for _ in csv.DictReader((ROOT/"results/iris_phase21_centroid_crossing.csv").open()))==3600
    assert sum(1 for _ in csv.DictReader((ROOT/"results/iris_phase21_local_purity.csv").open()))==3600
    gate=json.loads((ROOT/"results/iris_phase21_stageA_gate.json").read_text());assert gate["pass"] is False
    sensitivity=gate["posthoc_all_id_influence_sensitivity"];assert sensitivity["status"]=="post_output_post_hoc_not_gate_input" and len(sensitivity["rows"])==len({r["sample_id"] for r in rows})
    assert gate["criteria"]["criterion4_top_contributor_exclusion"] and "criterion4_all_single_id_exclusions" not in gate["criteria"]
    grid=list(csv.DictReader((ROOT/"results/iris_phase21_robust_vs_fail.csv").open()));assert len(grid)==5*3*2*4*2
    assert all((r["n"]!="0") or (r["mean_quantum_l2"]=="" and r["denominator_status"]=="undefined_empty") for r in grid)
    pgd=[r for r in rows if r["attack"]=="pgd"];assert all(r["attack_meta_objective"]=="untargeted_cross_entropy" and r["attack_meta_iterations"]==20 and not r["attack_meta_random_start"] and np.isclose(r["attack_meta_step_size"],r["epsilon_fraction"]*100/5) and "attack_meta_max_gradient_norm" in r for r in pgd)
    rnd=[r for r in rows if r["attack"]=="random"];assert all("attack_meta_seed" in r and np.isclose(r["attack_meta_range_high"],r["epsilon_fraction"]*100) for r in rnd)
    assert not any((ROOT/"results").glob("iris_phase21_stage_b_*")) and not (ROOT/"results/iris_phase21_selected_config.json").exists()

def test_exact_report_headings_questions_and_exhaustive_manifest():
    titles=["Agents and Skills Used","Interpreter","Files Added","Files Modified","Regression Status","Phase 21 Protocol","Development Splits","Model Seeds","Stage A Representation Metrics","Robust vs Failed Samples","Class-Wise Representation Shift","Centroid Crossing","Local Purity Stability","Multi-Split Reproducibility","Stage A Gate","Stage B Defense Definition","Lambda-Repr Ablation","Loss-Scale Analysis","Gradient Analysis","Clean Accuracy","Clean Representation Geometry","Representation Collapse Check","Random Timing Results","Classical PGD Results","Paired Robustness","Small-Epsilon Safety","TEMP-DRIFT Transfer","Stage B Gate","Selected Configuration","Independent Scientific Audit","Test-Access Status","Root-Cause Update","Scientific Conclusion","Beginner-Friendly Explanation","Recommended Phase 22"]
    report=(ROOT/"results/phase21_results.md").read_text();assert [x for x in report.splitlines() if x.startswith("## ")]==[f"## {i}. {x}" for i,x in enumerate(titles,1)]
    assert all(f"Q{i}:" in report for i in range(1,15))
    assert not any(x.startswith("## 36.") for x in report.splitlines())
    manifest=json.loads((ROOT/"results/iris_phase21_artifact_manifest.json").read_text());declared={x["path"] for x in manifest["generator_owned_outputs"]}|{x["path"] for x in manifest["agent_owned"]}|{"iris_phase21_artifact_manifest.json"}
    actual={p.name for p in (ROOT/"results").glob("iris_phase21_*")}|{p.name for p in (ROOT/"results/plots").glob("phase21_*")}|{"phase21_results.md"}
    assert actual==declared
    for x in manifest["generator_owned_outputs"]:
        p=(ROOT/x["location"]/x["path"]);assert hashlib.sha256(p.read_bytes()).hexdigest()==x["sha256"]
    assert {p.name for p in (ROOT/"results/plots").glob("phase21_*")}=={"phase21_feature_shift_by_attack.png","phase21_robust_vs_fail_shift.png","phase21_centroid_crossing.png"}
    assert not any((ROOT/"results/plots").glob("phase21_stage_a_*"))

def test_cleanup_writer_is_idempotent_and_preserves_agents(tmp_path):
    import scripts.run_phase21_representation_consistency as runner
    r=tmp_path/"results";p=r/"plots";c=tmp_path/"checkpoints";r.mkdir();p.mkdir();c.mkdir();(r/"iris_phase21_scientific_audit.md").write_text("owned");(r/"iris_phase21_junk.json").write_text("x");(p/"phase21_old.png").write_bytes(b"x");(c/"iris_phase21_stale.pt").write_bytes(b"x")
    runner.reset_owned_outputs(r,p,c);first=(r/"iris_phase21_run_events.jsonl").read_bytes();runner.reset_owned_outputs(r,p,c)
    assert first==(r/"iris_phase21_run_events.jsonl").read_bytes() and (r/"iris_phase21_scientific_audit.md").read_text()=="owned" and not (r/"iris_phase21_junk.json").exists() and not list(p.iterdir())

def test_complete_fixture_pipeline_two_run_names_and_hashes(tmp_path):
    import scripts.run_phase21_representation_consistency as runner
    r=tmp_path/"results";p=r/"plots";c=tmp_path/"checkpoints"
    first=runner.deterministic_fixture_pipeline(r,p,c);second=runner.deterministic_fixture_pipeline(r,p,c)
    assert first==second and len(list(c.glob("iris_phase21_*.pt")))==15

def test_array_training_api_and_report_audit_facts():
    import experiments.iris.training as tr
    src=inspect.getsource(tr.train_iris_model_from_arrays);assert "load_iris" not in src and "(-vacc,vce,epoch)" in src
    runner=(ROOT/"scripts/run_phase21_representation_consistency.py").read_text();assert "_AUTHORIZED_ARRAYS" in runner and "train_iris_model_from_arrays" in runner
    report=(ROOT/"results/phase21_results.md").read_text()
    for fact in ("-0.001428","0.019474","-0.004941","0.014382","-0.002979","0.004901","-0.008996","0.018799","ID 73","44 undefined","15 checkpoint","pending"):
        assert fact in report
