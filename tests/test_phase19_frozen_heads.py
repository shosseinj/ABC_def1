from pathlib import Path
import inspect,json,torch
from experiments.iris.phase19 import HEADS,DiagnosticHead,derived_seed,train_head,module_hash,paired_states,common_clean_pair,shared_init_seed,shared_xavier_weight,tensor_hash,no_test_guard
ROOT=Path(__file__).resolve().parents[1]
def test_exact_heads_and_deterministic_seeds():
    assert len(HEADS)==6 and HEADS[-1]=="HEAD_F_LINEAR_MARGIN"
    assert derived_seed(42,777,HEADS[1])==derived_seed(42,777,HEADS[1])
def test_standard_cosine_margin_is_applied_before_scale_and_pairing():
    h=DiagnosticHead("HEAD_E_COSINE_MARGIN"); h.weight.data.copy_(torch.eye(3,4))
    x=torch.tensor([[1.,0,0,0]]); y=torch.tensor([0])
    assert torch.allclose(h(x),torch.tensor([[10.,0,0]]))
    assert torch.allclose(h(x,y,True),torch.tensor([[9.,0,0]]))
    assert paired_states([1,1,0,0],[0,1,1,0])=={"rescued":1,"broken":1,"both_fail":1,"both_robust":1}
    assert module_hash(h)==module_hash(h)
    assert derived_seed(42,777,HEADS[1])!=derived_seed(42,777,HEADS[2])
def test_head_forms_and_protocol():
    x=torch.randn(8,4); y=torch.tensor([0,1,2,0,1,2,0,1])
    for n in HEADS: assert DiagnosticHead(n)(x).shape==(8,3)
    src=inspect.getsource(train_head); assert "range(1,201)" in src and "lr=.01" in src
def test_shared_initial_tensor_and_unequal_common_clean_pair():
    w=shared_xavier_weight(shared_init_seed(42,777))
    heads=[DiagnosticHead(n,initial_weight=w) for n in HEADS[1:]]
    assert len({tensor_hash(h.weight) for h in heads})==1
    assert all(torch.equal(h.bias,torch.zeros(3)) for h in (heads[0],heads[1],heads[4]))
    a=[{"sample_id":1,"clean_correct":True,"attacked_fail":True},{"sample_id":2,"clean_correct":True,"attacked_fail":False},{"sample_id":3,"clean_correct":False,"attacked_fail":True}]
    b=[{"sample_id":1,"clean_correct":True,"attacked_fail":False},{"sample_id":2,"clean_correct":False,"attacked_fail":False},{"sample_id":3,"clean_correct":True,"attacked_fail":True}]
    out=common_clean_pair(a,b); assert out["N_common"]==1 and out["rescued"]==1 and out["net_gain"]==1
def test_no_test_guard_deliberately_rejects_and_restores():
    import pytest
    import experiments.iris.data as data
    original=data.load_iris_splits
    with no_test_guard() as (audit,guarded_train):
        with pytest.raises(RuntimeError): data.load_iris_splits()
        with pytest.raises(RuntimeError): guarded_train({},evaluate_test=True)
        assert audit=={"prohibited_loader_calls":1,"evaluate_test_true_calls":1}
    assert data.load_iris_splits is original
def test_phase19_registered_and_no_test_loader():
    from phase_runner import PHASE_TESTS
    assert PHASE_TESTS[19].endswith("test_phase19_frozen_heads.py")
    src=(ROOT/"scripts/run_phase19_frozen_heads.py").read_text()
    assert "load_iris_splits" not in src and "X_test" not in src
    assert 'expected_random="99852106a44656beba19cfcb370df953997e1e508f537ad2169268a4115acba3"' in src
    assert 'h256(ROOT/"attacks/random_jitter.py")!=expected_random' in src
def test_generated_contract_if_present():
    gate=ROOT/"results/iris_phase19_gate.json"
    if not gate.exists(): return
    g=json.loads(gate.read_text()); assert not g["test_set_accessed"]
    assert g["split_seeds"]==[42,123,777,2026,6543] and g["model_seeds"]==[42,777,2026]
    assert g["instrumentation"]["test_loader_calls"]==0
    assert g["instrumentation"]["train_evaluate_test_true_calls"]==0
    assert g["candidate_paired_comparison_occurred"] is False
    assert (ROOT/"results/iris_phase19_split_manifest.json").is_file()
    assert (ROOT/"results/iris_phase19_checkpoint_provenance.json").is_file()
    assert (ROOT/"results/iris_phase19_protocol.json").is_file()
    protocol=json.loads((ROOT/"results/iris_phase19_protocol.json").read_text())
    expected="99852106a44656beba19cfcb370df953997e1e508f537ad2169268a4115acba3"
    assert protocol["expected_attack_hashes"]["random_jitter.py"]==expected
    assert f'expected_random="{expected}"' in (ROOT/"scripts/run_phase19_frozen_heads.py").read_text()
    protocol=json.loads((ROOT/"results/iris_phase19_protocol.json").read_text())
    assert protocol["expected_attack_hashes"]["random_jitter.py"]=="99852106a44656beba19cfcb370df953997e1e508f537ad2169268a4115acba3"
def test_manifest_provenance_hashes_fragility_and_report():
    import csv
    from experiments.iris.data import load_iris_train_validation
    manifest=json.loads((ROOT/"results/iris_phase19_split_manifest.json").read_text())["splits"]
    assert len(manifest)==5
    for row in manifest:
        _,_,ytr,yva,_,tr,va=load_iris_train_validation(row["split_seed"],.2,.2)
        assert row["train_ids"]==list(map(int,tr)) and row["validation_ids"]==list(map(int,va))
        assert row["train_class_counts"]==[30,30,30] and row["validation_class_counts"]==[10,10,10]
        assert row["train_validation_intersection_count"]==0
        assert row["hidden_class_counts"]==[10,10,10] and row["union_size"]==150
        assert row["train_hidden_intersection_count"]==row["validation_hidden_intersection_count"]==0
        assert row["held_out_features_selected_or_transformed"] is False
    provenance=json.loads((ROOT/"results/iris_phase19_checkpoint_provenance.json").read_text())["checkpoints"]
    assert len(provenance)==15 and all(not r["evaluate_test"] for r in provenance)
    with (ROOT/"results/iris_phase19_frozen_multisplit.csv").open(newline="") as f: rows=list(csv.DictReader(f))
    assert len(rows)==90 and all(r["extractor_pre_sha256"]==r["extractor_post_sha256"] for r in rows)
    with (ROOT/"results/iris_phase19_fragile_samples.csv").open(newline="") as f: fragile=list(csv.DictReader(f))
    assert all(r["analysis_status"]=="post_hoc_descriptive" for r in fragile)
    assert next(r for r in fragile if r["sample_id"]=="119")["excluded_from_selection_and_gates"]=="True"
    assert all(r["analysis_status"]=="post_hoc_descriptive" for r in fragile)
    fragile_json=json.loads((ROOT/"results/iris_phase19_fragile_samples.json").read_text())
    assert fragile_json["analysis_status"]=="post_hoc_descriptive" and "not preregistered inference" in fragile_json["interpretation"]
    fragile_json=json.loads((ROOT/"results/iris_phase19_fragile_samples.json").read_text())
    assert fragile_json["analysis_status"]=="post_hoc_descriptive"
    assert "not preregistered inference" in fragile_json["interpretation"]
    report=(ROOT/"results/phase19_results.md").read_text()
    titles=["Agents and Skills Used","Interpreter","Files Added","Files Modified","Regression Status","Phase 18 Audit Status","Multi-Split Protocol","Model Seeds","Current Head Definition","Candidate Head Definitions","Frozen-Feature Results","Split-Seed Variability","Model-Seed Variability","Class 1/2 Accuracy","Margin Stability","Sample 119 Analysis","Repeated Fragile Samples","Head Geometry","Diagnostic Linear Upper Bound","Random Timing Results","Classical PGD Results","Small-Epsilon Analysis","Multi-Split Paired Robustness","Frozen-Head Gate","End-to-End Confirmation","Statistical Interpretation","Independent Scientific Audit","Test-Access Status","Root-Cause Update","Scientific Conclusion","Beginner-Friendly Explanation","Recommended Phase 20"]
    assert all(f"## {i}. {t}" in report for i,t in enumerate(titles,1))
    section17=report.split("## 17. Repeated Fragile Samples",1)[1].split("## 18.",1)[0]
    assert "Post-hoc descriptive EDA" in section17 and "not preregistered inference" in section17
    section17=report.split("## 17. Repeated Fragile Samples",1)[1].split("## 18.",1)[0]
    assert "Post-hoc descriptive EDA" in section17 and "not preregistered inference" in section17
