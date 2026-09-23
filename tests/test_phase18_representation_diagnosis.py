from pathlib import Path
import hashlib
import inspect
import csv
import json
import numpy as np

from experiments.iris.phase18 import (assert_aligned, collision_analysis, fit_linear_diagnostic,
    geometry, local_neighbors, nearest_centroid, ordering_preservation)

ROOT = Path(__file__).resolve().parents[1]


def test_geometry_and_finite_safe_zero_spread():
    result = geometry([[0], [2], [10], [14]], [1, 1, 2, 2])
    assert result["centroid_distance_1_2"] == 11
    assert result["spread_1"] == 1 and result["spread_2"] == 2
    assert geometry([[0], [0]], [1, 2])["separation_1_2"] is None


def test_alignment_neighbors_collisions_and_ordering_are_deterministic():
    assert assert_aligned([2, 1], [1, 2], x=[[0], [1]])
    pred, _ = nearest_centroid([[0], [2]], [1, 2], [[1]])
    assert pred.tolist() == [1]
    neighbors = local_neighbors([[0], [2], [2]], [1, 2, 1], [3, 2, 1], [[2]], [9], 2)
    assert [x["sample_id"] for x in neighbors[0]] == [1, 2]
    collisions = collision_analysis([[0, 0], [0, 0], [.4, 0]], [1, 2, 2], [1, 2, 3])
    assert collisions["exact_count"] == 1 and collisions["near_count"] == 1
    assert ordering_preservation([[0, .5, 1]], [[100, 50, 0]]) == 0


def test_linear_diagnostic_declares_train_fit_validation_evaluation():
    result = fit_linear_diagnostic([[0], [1], [9], [10]], [1, 1, 2, 2], [[.5], [9.5]], [1, 2])
    assert result["fit_split"] == "train" and result["evaluation_split"] == "validation"
    assert result["n_fit"] == 4 and result["n_evaluated"] == 2


def test_phase18_module_cannot_load_test_data_and_phase14_source_is_frozen():
    import experiments.iris.phase18 as phase18
    source = inspect.getsource(phase18)
    assert "load_iris_splits" not in source and "X_test" not in source and "Xtest" not in source
    attack_hash = hashlib.sha256((ROOT / "attacks/classical_timing.py").read_bytes()).hexdigest()
    assert attack_hash == "08b9b4669fa19a12e826c228b7f2d4712895aab13aed475df3d027fe3369ec65"


def test_phase18_is_registered():
    from phase_runner import PHASE_TESTS
    assert PHASE_TESTS[18].endswith("test_phase18_representation_diagnosis.py")


def test_canonical_retained_split_identity_balance_and_controls():
    from experiments.iris.data import load_iris_train_validation
    xtr, xva, ytr, yva, _, train_ids, val_ids = load_iris_train_validation(42, .2, .2)
    assert xtr.shape == (90, 4) and xva.shape == (30, 4)
    assert len(set(train_ids) & set(val_ids)) == 0
    assert [int(np.sum(ytr == c)) for c in range(3)] == [30, 30, 30]
    assert [int(np.sum(yva == c)) for c in range(3)] == [10, 10, 10]
    assert {119, 122, 142}.issubset(set(val_ids))


def test_generated_gate_and_sample_artifact_schemas():
    gate = json.loads((ROOT / "results/iris_phase18_gate.json").read_text(encoding="utf-8"))
    assert gate["test_set_accessed"] is False and gate["test_loader_invoked"] is False
    assert gate["model_seeds"] == [42, 777, 2026] and len(gate["checkpoints"]) == 6
    assert gate["counts"] == {"train": 90, "validation": 30, "models": 6,
                              "representation_rows": 720, "attack_sample_rows": 360}
    with (ROOT / "results/iris_phase18_representations.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 720
    assert {row["split"] for row in rows} == {"train", "validation"}
    assert all("test" not in row["split"].lower() for row in rows)
    assert {"raw_0", "normalized_0", "ttfs_0", "quantum_0", "logits_0", "probabilities_0"} <= set(rows[0])
    with (ROOT / "results/iris_phase18_attack_paired.csv").open(newline="", encoding="utf-8") as handle:
        paired = list(csv.DictReader(handle))
    assert len(paired) == 6
    assert {"common_clean_correct_denominator", "rescued", "broken", "both_fail", "both_robust"} <= set(paired[0])


def test_runner_uses_no_example_csv_or_test_loader_and_report_has_25_numbered_sections():
    source = (ROOT / "scripts/run_phase18_representation_diagnosis.py").read_text(encoding="utf-8")
    assert "example.csv" not in source.lower() and "load_iris_splits" not in source
    report = (ROOT / "results/phase18_results.md").read_text(encoding="utf-8")
    assert sum(f"## {number}. " in report for number in range(1, 30)) == 29
    assert report.count("Phase 19 recommendation") == 1


def test_exact_required_artifact_contract_exists():
    names = ["iris_phase18_dataset_sanity.csv", "iris_phase18_dataset_sanity.json",
        "iris_phase18_representation_separability.csv", "iris_phase18_representation_separability.json",
        "iris_phase18_sample119_pipeline.csv", "iris_phase18_sample119_pipeline.json",
        "iris_phase18_ttfs_information_loss.csv", "iris_phase18_ttfs_collisions.csv",
        "iris_phase18_linear_separability.csv", "iris_phase18_attack_amplification.csv",
        "iris_phase18_attack_amplification.json", "iris_phase18_seed_comparison.csv",
        "iris_phase18_margin_analysis.csv", "iris_phase18_scientific_audit.md",
        "iris_phase18_beginner_summary.md", "phase18_results.md"]
    assert all((ROOT / "results" / name).is_file() for name in names)
    plots = ["phase18_raw_pca.png", "phase18_ttfs_pca.png", "phase18_quantum_pca.png",
             "phase18_margin_histogram.png"]
    assert all((ROOT / "results/plots" / name).is_file() for name in plots)


def test_dataset_sanity_source_identity_and_sample119_contract():
    payload = json.loads((ROOT / "results/iris_phase18_dataset_sanity.json").read_text(encoding="utf-8"))
    rows = {row["check"]: row for row in payload["rows"]}
    assert all(row["passed"] for row in rows.values())
    assert rows["source"]["observed"] == "sklearn.datasets.load_iris"
    assert rows["canonical_shape"]["observed"] == "150x4"
    assert rows["class_sizes"]["observed"] == "50,50,50"
    assert rows["split_sizes"]["observed"] == "90,30,30"
    assert rows["sample119_raw"]["observed"] == "[6.0, 2.2, 5.0, 1.5]"
    assert rows["sample119_label_split"]["observed"] == "label=2;split=validation"
    assert rows["example_csv_role"]["observed"] == "excluded"


def test_raw_rows_are_exact_canonical_rows_for_stable_ids_including_controls():
    from sklearn.datasets import load_iris
    canonical = load_iris().data
    with (ROOT/"results/iris_phase18_representations.csv").open(newline="",encoding="utf-8") as handle:
        rows=list(csv.DictReader(handle))
    selected={(int(r["sample_id"]),r["split"]):r for r in rows if r["seed"]=="42" and r["model"]=="baseline"}
    for sample_id in (33,15,32,119):
        matches=[key for key in selected if key[0]==sample_id]
        assert len(matches)==1
        row=selected[matches[0]]
        observed=np.array([float(row[f"raw_{i}"]) for i in range(4)])
        assert np.array_equal(observed,canonical[sample_id])


def test_exact_separability_and_attack_columns():
    with (ROOT/"results/iris_phase18_representation_separability.csv").open(newline="",encoding="utf-8") as handle:
        fields=next(csv.DictReader(handle)).keys()
    assert {"class1_mean","class1_sd","class2_mean","class2_sd","centroid_1","centroid_2",
        "spread_1","spread_2","centroid_distance_1_2","separation_1_2",
        "nearest_centroid_accuracy","k3_mean_local_purity"} <= set(fields)
    with (ROOT/"results/iris_phase18_attack_amplification.csv").open(newline="",encoding="utf-8") as handle:
        fields=next(csv.DictReader(handle)).keys()
    assert {"mean_probability_js","prediction_js_per_timing_shift",
            "prediction_js_per_timing_shift_status", "prediction_JS/timing_shift_ratio",
            "prediction_JS/timing_shift_denominator_status"} <= set(fields)


def test_margin_schema_includes_0p20_count_for_every_seed_model_class():
    with (ROOT/"results/iris_phase18_margin_analysis.csv").open(newline="",encoding="utf-8") as handle:
        rows=list(csv.DictReader(handle))
    assert len(rows)==3*2*3
    assert all("below_0p20_count" in row and row["below_0p20_count"] != "" for row in rows)
    assert {(int(r["seed"]),r["model"],int(r["class"])) for r in rows} == {
        (seed,model,cls) for seed in (42,777,2026) for model in ("baseline","defense") for cls in (0,1,2)}


def test_report_uses_only_prefixed_phase18_table_references_and_mentions_0p20():
    report=(ROOT/"results/phase18_results.md").read_text(encoding="utf-8")
    for obsolete in ("`representation_separability", "`sample119_pipeline", "`ttfs_information_loss",
                     "`ttfs_collisions", "`linear_separability", "`attack_amplification",
                     "`seed_comparison", "`margin_analysis"):
        assert obsolete not in report
    assert "<0.20" in report
