import numpy as np


PHASE153_SEEDS = (42, 123, 777, 2026, 6543)
FROZEN_DEFENSE = {
    "quantum_temp_enabled": True,
    "quantum_temp_epsilon": 0.02,
    "lambda_q": 5.0,
    "lambda_pred": 0.0,
    "consistency_enabled": False,
    "normalize_quantum_loss": False,
}


def apply_frozen_defense(config):
    result = {**config, **FROZEN_DEFENSE}
    if any(result[key] != value for key, value in FROZEN_DEFENSE.items()):
        raise RuntimeError("Frozen defense configuration was overridden.")
    return result


def sample_mean_sd(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None, None
    return float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else 0.0


def aggregate_seed_summaries(rows):
    groups = {}
    for row in rows:
        key = (row["attack"], row["epsilon"], row["tau"])
        groups.setdefault(key, []).append(row)
    output = []
    for (attack, epsilon, tau), members in groups.items():
        baseline_mean, baseline_sd = sample_mean_sd([row["baseline_paired_ASR"] for row in members])
        defense_mean, defense_sd = sample_mean_sd([row["defense_paired_ASR"] for row in members])
        delta_mean, delta_sd = sample_mean_sd([row["Delta_ASR"] for row in members])
        net = [row["net_gain"] for row in members]
        output.append({
            "attack": attack,
            "epsilon": epsilon,
            "tau": tau,
            "num_seeds": len(members),
            "mean_N_common": float(np.mean([row["N_common"] for row in members])),
            "total_common_observations": int(sum(row["N_common"] for row in members)),
            "baseline_ASR_mean": baseline_mean,
            "baseline_ASR_SD": baseline_sd,
            "defense_ASR_mean": defense_mean,
            "defense_ASR_SD": defense_sd,
            "mean_Delta_ASR": delta_mean,
            "Delta_ASR_SD": delta_sd,
            "positive_seeds": sum(value > 0 for value in net),
            "zero_seeds": sum(value == 0 for value in net),
            "negative_seeds": sum(value < 0 for value in net),
            "total_rescued": int(sum(row["rescued"] for row in members)),
            "total_broken": int(sum(row["broken"] for row in members)),
            "mean_net_gain": float(np.mean(net)),
            "median_net_gain": float(np.median(net)),
        })
    return output
