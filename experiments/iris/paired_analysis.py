import math


CATEGORIES = ("both_robust", "both_fail", "rescued_by_defense", "broken_by_defense")


def membership_category(baseline_correct, defense_correct):
    if baseline_correct and defense_correct:
        return "common_correct"
    if baseline_correct:
        return "baseline_only_correct"
    if defense_correct:
        return "defense_only_correct"
    return "both_wrong"


def paired_category(baseline_success, defense_success):
    if baseline_success and defense_success:
        return "both_fail"
    if baseline_success:
        return "rescued_by_defense"
    if defense_success:
        return "broken_by_defense"
    return "both_robust"


def exact_mcnemar_pvalue(rescued, broken):
    discordant = rescued + broken
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(rescued, broken) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def summarize_paired_outcomes(rows):
    counts = {category: 0 for category in CATEGORIES}
    for row in rows:
        counts[row["paired_category"]] += 1
    baseline_failures = counts["both_fail"] + counts["rescued_by_defense"]
    defense_failures = counts["both_fail"] + counts["broken_by_defense"]
    n_common = len(rows)
    rescued = counts["rescued_by_defense"]
    broken = counts["broken_by_defense"]
    return {
        "N_common": n_common,
        "baseline_failures": baseline_failures,
        "defense_failures": defense_failures,
        **counts,
        "rescued": rescued,
        "broken": broken,
        "net_gain": rescued - broken,
        "baseline_paired_ASR": baseline_failures / n_common if n_common else 0.0,
        "defense_paired_ASR": defense_failures / n_common if n_common else 0.0,
        "Delta_ASR": (defense_failures - baseline_failures) / n_common if n_common else 0.0,
        "mcnemar_exact_pvalue": exact_mcnemar_pvalue(rescued, broken),
    }
