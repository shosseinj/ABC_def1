import json
from pathlib import Path

import numpy as np

from experiments.iris.multiseed import (
    FROZEN_DEFENSE,
    PHASE153_SEEDS,
    aggregate_seed_summaries,
    apply_frozen_defense,
    sample_mean_sd,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_protocol_and_seed_set():
    assert PHASE153_SEEDS == (42, 123, 777, 2026, 6543)
    result = apply_frozen_defense({"lambda_q": 999, "consistency_enabled": True})
    assert all(result[key] == value for key, value in FROZEN_DEFENSE.items())


def test_aggregation_uses_every_seed_and_sample_sd():
    rows = []
    for seed, net in zip(PHASE153_SEEDS, (1, 0, -1, 2, 0)):
        rescued, broken = max(net, 0), max(-net, 0)
        rows.append({
            "seed": seed, "attack": "x", "epsilon": 0.1, "tau": 0.1,
            "N_common": 10, "baseline_paired_ASR": 0.2,
            "defense_paired_ASR": 0.2 - net / 10, "Delta_ASR": -net / 10,
            "net_gain": net, "rescued": rescued, "broken": broken,
        })
    result = aggregate_seed_summaries(rows)[0]
    assert result["num_seeds"] == 5
    assert result["total_common_observations"] == 50
    assert result["positive_seeds"] == 2 and result["negative_seeds"] == 1
    assert np.isclose(result["baseline_ASR_SD"], 0.0)
    assert sample_mean_sd([1, 2, 3])[1] == 1.0


def test_phase153_artifacts_include_all_seeds_without_dropping():
    clean = json.loads((ROOT / "results" / "iris_phase153_clean_per_seed.json").read_text())
    paired = json.loads((ROOT / "results" / "iris_phase153_paired_per_seed.json").read_text())
    assert clean["seeds"] == list(PHASE153_SEEDS)
    assert {row["seed"] for row in clean["runs"]} == set(PHASE153_SEEDS)
    assert {row["seed"] for row in paired["runs"]} == set(PHASE153_SEEDS)
    assert len(paired["runs"]) == 5 * 24
    assert all(row["N_common"] == row["both_robust"] + row["both_fail"] + row["rescued"] + row["broken"] for row in paired["runs"])
    assert all(row["baseline_failures"] == row["both_fail"] + row["rescued"] for row in paired["runs"])
    assert all(row["defense_failures"] == row["both_fail"] + row["broken"] for row in paired["runs"])
    assert clean["split_seed"] == 42
    for seed in PHASE153_SEEDS:
        baseline = json.loads((ROOT / "results" / f"iris_phase153_seed_{seed}_baseline_training.json").read_text())
        defense = json.loads((ROOT / "results" / f"iris_phase153_seed_{seed}_quantum_temp_training.json").read_text())
        assert baseline["seed"] == defense["seed"] == seed
        assert baseline["split_seed"] == defense["split_seed"] == 42
        assert all(defense["config"][key] == value for key, value in FROZEN_DEFENSE.items())
