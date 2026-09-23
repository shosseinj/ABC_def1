from experiments.iris.paired_analysis import (
    CATEGORIES,
    membership_category,
    paired_category,
    summarize_paired_outcomes,
)
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_membership_intersection_and_stable_ids():
    rows = [
        (101, True, True), (102, True, False),
        (103, False, True), (104, False, False),
    ]
    categories = {sample_id: membership_category(base, defense) for sample_id, base, defense in rows}
    common = {sample_id for sample_id, category in categories.items() if category == "common_correct"}
    assert common == {101}
    assert categories[102] == "baseline_only_correct"
    assert categories[103] == "defense_only_correct"
    assert categories[104] == "both_wrong"


def test_paired_categories_are_exclusive_exhaustive_and_consistent():
    outcomes = [(False, False), (True, True), (True, False), (False, True)]
    rows = [{"paired_category": paired_category(*outcome)} for outcome in outcomes]
    assert {row["paired_category"] for row in rows} == set(CATEGORIES)
    summary = summarize_paired_outcomes(rows)
    assert sum(summary[category] for category in CATEGORIES) == summary["N_common"]
    assert summary["baseline_failures"] == summary["both_fail"] + summary["rescued"]
    assert summary["defense_failures"] == summary["both_fail"] + summary["broken"]
    assert summary["net_gain"] == summary["rescued"] - summary["broken"]


def test_paired_asr_uses_only_supplied_common_rows():
    rows = [
        {"paired_category": "rescued_by_defense"},
        {"paired_category": "both_robust"},
    ]
    summary = summarize_paired_outcomes(rows)
    assert summary["N_common"] == 2
    assert summary["baseline_paired_ASR"] == 0.5
    assert summary["defense_paired_ASR"] == 0.0


def test_phase152_artifacts_preserve_paired_invariants_and_ids():
    membership = json.loads((ROOT / "results" / "iris_phase152_clean_membership.json").read_text())
    paired = json.loads((ROOT / "results" / "iris_phase152_paired_attack_summary.json").read_text())
    all_ids = [sample["sample_id"] for sample in membership["samples"]]
    assert len(all_ids) == len(set(all_ids)) == 30
    common = set(membership["sets"]["common_correct"])
    baseline = common | set(membership["sets"]["baseline_only_correct"])
    defense = common | set(membership["sets"]["defense_only_correct"])
    assert common == baseline & defense
    for row in paired["summaries"]:
        assert row["N_common"] == len(common)
        assert row["both_robust"] + row["both_fail"] + row["rescued"] + row["broken"] == row["N_common"]
        assert row["baseline_failures"] == row["both_fail"] + row["rescued"]
        assert row["defense_failures"] == row["both_fail"] + row["broken"]
        assert row["net_gain"] == row["rescued"] - row["broken"]
    phase14 = list(csv.DictReader((ROOT / "results" / "iris_attack_comparison_phase14.csv").open()))
    assert len(phase14) == 12
