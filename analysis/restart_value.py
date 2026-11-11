"""Is coding-agent failure a property of the issue or of the attempt?

Three seeded rollouts per task (Qwen3.8-27B + OpenHands on SWE-rebench-V2).
Estimates p, q = P(another attempt succeeds | an attempt failed), the
between-task share of variance, restart versus fresh-task yield per unit
compute, and q by failure group. Writes evidence/results/restart_value.json.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, gammaln

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evidence" / "external" / "qwen38_27b_bo3" / "data"
OUT = ROOT / "evidence" / "results" / "restart_value.json"
N_BOOT = 1000
SEED = 20260929


def load():
    tasks = pd.read_parquet(DATA / "tasks-00000-of-00001.parquet")
    rolls = pd.read_parquet(DATA / "rollouts-00000-of-00001.parquet")
    rolls = rolls.merge(tasks[["instance_id", "language", "repo", "source_llm_difficulty",
                               "source_num_modified_files", "bo3_complete"]], on="instance_id")
    return tasks, rolls


def pair_table(valid):
    """All ordered pairs (i, j), i != j, of valid attempts on the same task."""
    v = valid[["instance_id", "rollout_index", "success", "main_model_calls", "wall_time_seconds",
               "timed_out", "recursive", "language", "repo", "source_llm_difficulty",
               "source_num_modified_files"]]
    pairs = v.merge(v, on="instance_id", suffixes=("_i", "_j"))
    return pairs[pairs.rollout_index_i != pairs.rollout_index_j].reset_index(drop=True)


def stats_from(att, pairs):
    fail_pairs = pairs[~pairs.success_i]
    p = att.success.mean()
    q = fail_pairs.success_j.mean()
    fresh_calls = 1000 * att.success.sum() / att.main_model_calls.sum()
    fresh_hour = 3600 * att.success.sum() / att.wall_time_seconds.sum()
    restart_calls = 1000 * fail_pairs.success_j.sum() / fail_pairs.main_model_calls_j.sum()
    restart_hour = 3600 * fail_pairs.success_j.sum() / fail_pairs.wall_time_seconds_j.sum()
    return {
        "p": p, "q": q, "q_over_p": q / p, "one_minus_q_over_p": 1 - q / p,
        "fresh_per_1k_calls": fresh_calls, "restart_per_1k_calls": restart_calls,
        "restart_over_fresh_calls": restart_calls / fresh_calls,
        "fresh_per_hour": fresh_hour, "restart_per_hour": restart_hour,
        "restart_over_fresh_hours": restart_hour / fresh_hour,
        "mean_calls_success": att.main_model_calls[att.success].mean(),
        "mean_calls_failure": att.main_model_calls[~att.success].mean(),
        "mean_seconds_success": att.wall_time_seconds[att.success].mean(),
        "mean_seconds_failure": att.wall_time_seconds[~att.success].mean(),
        "mean_calls_restart_of_failure": fail_pairs.main_model_calls_j.mean(),
    }


def icc_anova(k_success, n=3):
    """One-way ANOVA intraclass correlation for binary outcomes, equal group size n."""
    y = k_success / n
    grand = y.mean()
    g = len(y)
    msb = n * ((y - grand) ** 2).sum() / (g - 1)
    within = (k_success * (1 - y) ** 2 + (n - k_success) * y ** 2).sum()
    msw = within / (g * (n - 1))
    return (msb - msw) / (msb + (n - 1) * msw)


def betabinom_fit(k, n=3):
    def nll(theta):
        a, b = np.exp(theta)
        ll = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + betaln(k + a, n - k + b) - betaln(a, b)
        return -ll.sum()
    res = minimize(nll, x0=[0.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 4000})
    a, b = np.exp(res.x)
    return float(a), float(b)


def pass_at_k_betabinom(a, b, k):
    # 1 - E[(1-theta)^k] = 1 - B(a, b+k)/B(a, b)
    return 1 - math.exp(betaln(a, b + k) - betaln(a, b))


def bootstrap(att, pairs, fn, rng, n=N_BOOT):
    ids = att.instance_id.unique()
    att_groups = {i: g for i, g in att.groupby("instance_id")}
    pair_groups = {i: g for i, g in pairs.groupby("instance_id")}
    empty = pairs.iloc[:0]
    out = []
    for _ in range(n):
        draw = rng.choice(ids, size=len(ids), replace=True)
        a = pd.concat([att_groups[i] for i in draw], ignore_index=True)
        p = pd.concat([pair_groups.get(i, empty) for i in draw], ignore_index=True)
        out.append(fn(a, p))
    return out


def interval(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]


def fast_boot_stats(att, pairs, rng, n=N_BOOT):
    """Vectorized task bootstrap of the headline ratios via per-task sums."""
    t = att.groupby("instance_id").agg(s=("success", "sum"), c=("main_model_calls", "sum"),
                                         w=("wall_time_seconds", "sum"), n=("success", "size"))
    fp = pairs[~pairs.success_i].groupby("instance_id").agg(
        fs=("success_j", "sum"), fc=("main_model_calls_j", "sum"),
        fw=("wall_time_seconds_j", "sum"), fn=("success_j", "size"))
    t = t.join(fp, how="left").fillna(0)
    m = t.to_numpy(dtype=float)
    cols = {c: i for i, c in enumerate(t.columns)}
    idx = rng.integers(0, len(t), size=(n, len(t)))
    sums = np.stack([m[row].sum(axis=0) for row in idx])
    s, c, w, na = (sums[:, cols[x]] for x in ("s", "c", "w", "n"))
    fs, fc, fw, fn = (sums[:, cols[x]] for x in ("fs", "fc", "fw", "fn"))
    p = s / na
    q = fs / fn
    fresh_c, restart_c = 1000 * s / c, 1000 * fs / fc
    fresh_h, restart_h = 3600 * s / w, 3600 * fs / fw
    return {
        "p": interval(p), "q": interval(q), "q_over_p": interval(q / p),
        "one_minus_q_over_p": interval(1 - q / p),
        "fresh_per_1k_calls": interval(fresh_c), "restart_per_1k_calls": interval(restart_c),
        "restart_over_fresh_calls": interval(restart_c / fresh_c),
        "fresh_per_hour": interval(fresh_h), "restart_per_hour": interval(restart_h),
        "restart_over_fresh_hours": interval(restart_h / fresh_h),
    }


def group_table(att, pairs, rng, fresh_yield, p):
    fail = pairs[~pairs.success_i].copy()
    fail_att = att[~att.success]
    edges = np.quantile(fail_att.main_model_calls, [1 / 3, 2 / 3])
    fail["compute_tercile"] = np.where(fail.main_model_calls_i <= edges[0], "low",
                                       np.where(fail.main_model_calls_i <= edges[1], "mid", "high"))
    fail["timed_out_g"] = fail.timed_out_i.map({True: "timed_out", False: "not_timed_out"})
    fail["recursive_g"] = fail.recursive_i.map({True: "recursive", False: "not_recursive"})
    nmf = fail.source_num_modified_files_i
    fail["files_g"] = np.where(nmf <= 1, "1", np.where(nmf <= 3, "2-3", "4+"))
    groups = {
        "timed_out": "timed_out_g", "recursive": "recursive_g", "compute_tercile": "compute_tercile",
        "language": "language_i", "source_llm_difficulty": "source_llm_difficulty_i",
        "modified_files": "files_g", "repo": "repo_i",
    }
    rows = []
    for family, col in groups.items():
        for level, g in fail.groupby(col):
            if len(g) < 30:
                continue
            per = g.groupby("instance_id").agg(s=("success_j", "sum"), c=("main_model_calls_j", "sum"),
                                               n=("success_j", "size"))
            m = per.to_numpy(dtype=float)
            idx = rng.integers(0, len(per), size=(N_BOOT, len(per)))
            sums = m[idx].sum(axis=1)
            qs = sums[:, 0] / sums[:, 2]
            ys = 1000 * sums[:, 0] / sums[:, 1]
            q = g.success_j.mean()
            y = 1000 * g.success_j.sum() / g.main_model_calls_j.sum()
            q_ci, y_ci = interval(qs), interval(ys)
            label = ("restart-worthy" if y_ci[0] > fresh_yield
                     else "issue-level" if q_ci[1] < p / 2 else "unresolved")
            rows.append({"family": family, "group": str(level), "failure_anchored_pairs": int(len(g)),
                         "tasks": int(per.shape[0]), "q": q, "q_ci": q_ci,
                         "restart_per_1k_calls": y, "restart_per_1k_calls_ci": y_ci, "call": label})
    return rows, [float(e) for e in edges]


def main():
    rng = np.random.default_rng(SEED)
    tasks, rolls = load()
    exclusions = {
        "tasks_total": int(len(tasks)),
        "tasks_all_three_valid": int(tasks.bo3_complete.sum()),
        "tasks_excluded": int((~tasks.bo3_complete).sum()),
        "slots_by_status": rolls.status.value_counts().to_dict(),
        "slots_by_invalid_reason": rolls.invalid_reason.fillna("valid").value_counts().to_dict(),
        "excluded_tasks_by_valid_slots": tasks[~tasks.bo3_complete].valid_rollouts.value_counts().sort_index().to_dict(),
        "success_rate_of_valid_slots_in_excluded_tasks": float(
            rolls[rolls.valid_scored & ~rolls.bo3_complete].success.astype(bool).mean()),
    }
    valid = rolls[rolls.valid_scored].copy()
    valid["success"] = valid.success.astype(bool)
    valid["recursive"] = valid.recursive.astype(bool)
    primary = valid[valid.bo3_complete].reset_index(drop=True)
    pairs = pair_table(primary)
    head = stats_from(primary, pairs)
    ci = fast_boot_stats(primary, pairs, rng)

    k = primary.groupby("instance_id").success.sum().to_numpy()
    counts = {str(i): int((k == i).sum()) for i in range(4)}
    icc = icc_anova(k)
    a, b = betabinom_fit(k)
    rho_bb = 1 / (a + b + 1)
    n_fail = (3 - k).sum()
    boot_icc = []
    for _ in range(N_BOOT):
        kk = k[rng.integers(0, len(k), len(k))]
        boot_icc.append(icc_anova(kk))
    by_index = primary.groupby("rollout_index").success.mean().to_dict()

    # Sensitivity: every task with >= 2 valid slots, all ordered pairs.
    counts_valid = valid.groupby("instance_id").success.transform("size")
    sens_att = valid[counts_valid >= 2].reset_index(drop=True)
    sens_pairs = pair_table(sens_att)
    sens = stats_from(sens_att, sens_pairs)

    groups, edges = group_table(primary, pairs, rng, head["fresh_per_1k_calls"], head["p"])

    result = {
        "data": "Qwen3.8-27B + OpenHands, SWE-rebench-V2, three seeded rollouts per task (overlay snapshot, incomplete)",
        "exclusions": exclusions,
        "primary": {
            "tasks": int(primary.instance_id.nunique()), "attempts": int(len(primary)),
            "tasks_by_successes_of_3": counts,
            **head, "bootstrap_95": ci,
            "success_by_rollout_index": {str(key): float(v) for key, v in by_index.items()},
            "icc_anova": float(icc), "icc_anova_95": interval(boot_icc),
            "betabinomial": {"alpha": a, "beta": b, "rho": rho_bb,
                             "pass_at_k_fit": {str(kk): pass_at_k_betabinom(a, b, kk) for kk in (1, 2, 3, 5, 10)},
                             "pass_at_k_observed": {"1": float(primary.success.mean()),
                                                    "3": float((k > 0).mean())}},
            "share_of_failures_on_tasks_with_0_of_3": float(3 * counts["0"] / n_fail),
            "share_of_tasks_mixed_outcome": float(((k == 1) | (k == 2)).mean()),
        },
        "sensitivity_tasks_with_2plus_valid": {"tasks": int(sens_att.instance_id.nunique()),
                                               "attempts": int(len(sens_att)), **sens},
        "failure_groups": {"compute_tercile_edges_calls": edges, "fresh_per_1k_calls": head["fresh_per_1k_calls"],
                           "p": head["p"], "rows": groups},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=float) + "\n")
    brief = {k2: result["primary"][k2] for k2 in ("tasks", "tasks_by_successes_of_3", "p", "q", "q_over_p",
                                                  "fresh_per_1k_calls", "restart_per_1k_calls",
                                                  "restart_over_fresh_calls", "fresh_per_hour", "restart_per_hour",
                                                  "icc_anova", "betabinomial", "share_of_failures_on_tasks_with_0_of_3")}
    brief["ci"] = ci
    print(json.dumps(brief, indent=2, default=float))
    print(json.dumps(exclusions, indent=2, default=str))
    for r in groups:
        print(f"{r['family']:22s} {r['group']:28s} n={r['failure_anchored_pairs']:5d} q={r['q']:.3f} "
              f"[{r['q_ci'][0]:.3f},{r['q_ci'][1]:.3f}] y={r['restart_per_1k_calls']:.2f} "
              f"[{r['restart_per_1k_calls_ci'][0]:.2f},{r['restart_per_1k_calls_ci'][1]:.2f}] {r['call']}")


if __name__ == "__main__":
    main()
