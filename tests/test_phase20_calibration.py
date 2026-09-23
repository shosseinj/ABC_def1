from pathlib import Path
import json,inspect,torch
from experiments.iris.phase20 import *
ROOT=Path(__file__).resolve().parents[1]

def test_exact_grid_formula_and_shared_init():
    assert SCALES==(5,10,15) and MARGINS==(.05,.10,.15)
    w=torch.eye(3,4);h=CalibrationHead(HEAD_E_COSINE_MARGIN,5,.1,initial_weight=w);x=torch.tensor([[1.,0,0,0]]);y=torch.tensor([0])
    assert torch.allclose(h(x),torch.tensor([[5.,0,0]]))
    assert torch.allclose(h(x,y,True),torch.tensor([[4.5,0,0]]))
    assert tensor_hash(shared_xavier(derived_seed(314,42)))==tensor_hash(shared_xavier(derived_seed(314,42)))

def test_training_protocol_and_pairing():
    src=inspect.getsource(fit_head);assert "range(1,201)" in src and "lr=.01" in src and "(-acc,ce,epoch)" in src
    a=[{"sample_id":1,"clean_correct":True,"attacked_fail":True},{"sample_id":2,"clean_correct":True,"attacked_fail":False}]
    b=[{"sample_id":1,"clean_correct":True,"attacked_fail":False},{"sample_id":2,"clean_correct":False,"attacked_fail":False}]
    q=common_clean_pair(a,b);assert q["N_common"]==1 and q["rescued"]==1

def test_protocol_hash_and_boundaries():
    p=json.loads((ROOT/"results/iris_phase20_protocol.json").read_text());h=p.pop("protocol_sha256")
    import hashlib
    assert h==hashlib.sha256(json.dumps(p,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    assert p["split_seeds"]==[314,1001,4096,9001,12345] and p["head_e_grid"]["config_count"]==9
    src=(ROOT/"scripts/run_phase20_calibration.py").read_text();assert "from sklearn.datasets import load_iris" not in src and "X_test" not in src
    assert "load_iris_split_manifest_labels" in src

def test_guard_and_registration():
    import pytest,experiments.iris.data as data
    original=data.load_iris_splits
    with no_test_guard() as (audit,train):
        with pytest.raises(RuntimeError):data.load_iris_splits()
        with pytest.raises(RuntimeError):train({},evaluate_test=True)
    assert data.load_iris_splits is original and audit["prohibited_loader_calls"]==1
    from phase_runner import PHASE_TESTS
    assert PHASE_TESTS[20].endswith("test_phase20_calibration.py")

def test_generated_gate_if_present():
    p=ROOT/"results/iris_phase20_gate.json"
    if not p.exists():return
    g=json.loads(p.read_text());assert g["test_set_accessed"] is False and g["grid_size"]==9 and g["cell_count"]==150 and not g["end_to_end_run"]
    assert g["hidden_features_received_by_phase20"] is False
    assert g["selection"]["selection_count"]<=1 and not g["selection"]["attack_metrics_used"] and not g["selection"]["sample_specific_selection_use"]
    rows=json.loads((ROOT/"results/iris_phase20_calibration_grid.json").read_text());assert all(r["extractor_pre_sha256"]==r["extractor_post_sha256"] for r in rows)
    report=(ROOT/"results/phase20_results.md").read_text();titles=["Agents and Skills Used","Interpreter","Files Added","Files Modified","Regression Status","Phase 20 Preregistered Protocol","New Development Splits","Model Seeds","Frozen HEAD_A Reference","HEAD_E Calibration Grid","Clean Calibration Results","Split-Level Stability","Class 1/2 Results","Margin Stability","Sample 119 Diagnostic","Repeated Fragile Samples","Clean Calibration Gate","Frozen Selected Configuration","Random Timing Results","Classical PGD Results","Paired Robustness","Small-Epsilon Safety","Final Phase 20 Gate","Statistical Interpretation","Independent Scientific Audit","Test-Access Status","Scientific Conclusion","Beginner-Friendly Explanation","Recommended Phase 21"]
    assert [x for x in report.splitlines() if x.startswith("## ")]==[f"## {i}. {t}" for i,t in enumerate(titles,1)]

def test_exact_artifact_contract_and_row_coverage():
    import csv
    required=["iris_phase20_protocol.json","iris_phase20_calibration_grid.csv","iris_phase20_calibration_grid.json","iris_phase20_clean_summary.csv","iris_phase20_clean_summary.json","iris_phase20_fragile_samples.csv","iris_phase20_selected_config.json","iris_phase20_attack_validation.csv","iris_phase20_attack_validation.json","iris_phase20_gate.json","iris_phase20_scientific_audit.md","iris_phase20_beginner_summary.md","phase20_results.md"]
    assert all((ROOT/"results"/n).is_file() for n in required)
    grid=json.loads((ROOT/"results/iris_phase20_calibration_grid.json").read_text());assert len(grid)==135
    assert len({(r["split_seed"],r["model_seed"],r["config_id"]) for r in grid})==135
    assert {r["scale"] for r in grid}=={5,10,15} and {r["margin"] for r in grid}=={.05,.1,.15}
    summary=json.loads((ROOT/"results/iris_phase20_clean_summary.json").read_text());assert sum(r["summary_level"]=="split" for r in summary)==45 and sum(r["summary_level"]=="aggregate" for r in summary)==9
    aggregates=[r for r in summary if r["summary_level"]=="aggregate"]
    for metric in ("accuracy","macro_f1","minimum_class_accuracy","class1_accuracy","class2_accuracy","class12_binary_accuracy","cross_entropy","class12_mean_margin"):
        assert all(f"{metric}_delta_ci95_low" in r and f"{metric}_positive_splits" in r for r in aggregates)
    g=json.loads((ROOT/"results/iris_phase20_gate.json").read_text());attack=json.loads((ROOT/"results/iris_phase20_attack_validation.json").read_text())
    if not g["clean_gate"]:assert attack["status"]=="not_run" and not (ROOT/"results/plots/phase20_pgd_net_gain_by_split.png").exists()
    for n in ("phase20_accuracy_by_scale_margin.png","phase20_class2_accuracy.png","phase20_margin_stability.png"):assert (ROOT/"results/plots"/n).is_file()
    report=(ROOT/"results/phase20_results.md").read_text();assert all(f"Q{i}:" in report for i in range(1,13)) and "no Phase21 candidate" in report
    assert "canonical membership" in report and "first observed checkpoint only" in report and "post-output audit correction" in report
    section29=report.split("## 29. Recommended Phase 21",1)[1].strip()
    assert section29=="clean gate failed, narrow calibration insufficient, do not broaden classifier search, no Phase21 candidate, return to representation robustness only as next research direction."
    unexpected={"phase20_clean_delta_by_config.png","phase20_class12_margin_by_config.png"}
    assert not any((ROOT/"results/plots"/x).exists() for x in unexpected)

def test_selection_ignores_sample_metadata_and_diagnostics():
    rows=[{"config_id":"x","eligible":True,"mean_accuracy_delta":0.1,"minimum_class_delta_nonnegative_splits":4,"class12_margin_delta_mean":0.2,"scale":5,"margin":.1},{"config_id":"y","eligible":True,"mean_accuracy_delta":0.0,"minimum_class_delta_nonnegative_splits":5,"class12_margin_delta_mean":1.0,"scale":10,"margin":.1}]
    expected=select_config(rows)["config_id"]
    diagnostics=[{"sample_id":119,"margin":-1},{"sample_id":1,"margin":1}]
    assert select_config(rows)["config_id"]==expected
    diagnostics=[] # removing all diagnostic records cannot affect aggregate-only selection
    assert select_config(rows)["config_id"]==expected
    assert "sample_id" not in inspect.getsource(select_config).lower() and list(inspect.signature(select_config).parameters)==["split_summaries"]

def test_event_log_ledger_and_manifest_helper_contract():
    events=[json.loads(x) for x in (ROOT/"results/iris_phase20_run_events.jsonl").read_text().splitlines()]
    assert events[0]["event"]=="protocol_written" and events[1]["event"]=="first_output_checkpoint_observed"
    assert events[0]["timestamp_utc"]<events[1]["timestamp_utc"] and "not independently" in events[0]["claim_limit"]
    ledger=json.loads((ROOT/"results/iris_phase20_historical_split_seed_ledger.json").read_text());assert ledger["preferred_absent"] and ledger["scanned_path_count"]==len(ledger["scanned_paths"])
    manifests=json.loads((ROOT/"results/iris_phase20_split_manifest.json").read_text())["splits"]
    assert all(x["manifest_helper_contract"]=="IDs and labels only" and not x["hidden_features_received_by_phase20"] for x in manifests)

def test_exhaustive_artifact_manifest_and_hashes():
    import hashlib
    protocol=json.loads((ROOT/"results/iris_phase20_protocol.json").read_text());declared={x["path"]:x for x in protocol["artifact_manifest"]["artifacts"]}
    actual={p.name for p in (ROOT/"results").glob("iris_phase20_*")}|{p.name for p in (ROOT/"results/plots").glob("phase20_*")}
    assert actual<=set(declared),f"undeclared={actual-set(declared)}"
    assert (ROOT/"results/phase20_results.md").is_file()
    for x in ("iris_phase20_scientific_audit.md","iris_phase20_beginner_summary.md"):
        assert declared[x]["role"]=="agent_owned" and declared[x]["status"]=="pending_at_generation"
    manifest_path=ROOT/"results/iris_phase20_artifact_manifest.json";manifest=json.loads(manifest_path.read_text());gate=json.loads((ROOT/"results/iris_phase20_gate.json").read_text())
    assert gate["artifact_manifest_sha256"]==hashlib.sha256(manifest_path.read_bytes()).hexdigest() and not manifest["self_hash_claimed"]
    for e in manifest["generator_owned_outputs"]:
        p=(ROOT/"results/plots"/e["path"]) if e["path"].endswith(".png") else (ROOT/"results"/e["path"])
        if e["hash_mode"]=="file_bytes":observed=hashlib.sha256(p.read_bytes()).hexdigest()
        else:
            d=json.loads(p.read_text());d.pop("artifact_manifest_sha256",None);observed=hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        assert observed==e["sha256"]
    assert all(x["excluded_from_generator_complete_hashes"] for x in manifest["agent_owned"])
    events=[json.loads(x) for x in (ROOT/"results/iris_phase20_run_events.jsonl").read_text().splitlines()];assert events[-1]["event"]=="generator_outputs_completed_final"
