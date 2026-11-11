"""Fixed-budget schedule frontier over a queue of tasks with three exchangeable rollouts.

Policies: move-on only, depth-first retry, breadth-first retry, and breadth-first
with restarts ordered by a group score learned on a training half of tasks.
Writes evidence/results/schedule_frontier.json.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evidence" / "external" / "qwen38_27b_bo3" / "data"
OUT = ROOT / "evidence" / "results" / "schedule_frontier.json"
FRACTIONS = np.round(np.arange(0.1, 1.01, 0.1), 2)
N_REP = 1000
SEED = 20260929


def load():
    tasks = pd.read_parquet(DATA / "tasks-00000-of-00001.parquet")
    rolls = pd.read_parquet(DATA / "rollouts-00000-of-00001.parquet")
    keep = set(tasks.instance_id[tasks.bo3_complete])
    rolls = rolls[rolls.instance_id.isin(keep)].sort_values(["instance_id", "rollout_index"])
    ids = rolls.instance_id.unique()
    s = rolls.success.astype(bool).to_numpy().reshape(len(ids), 3)
    calls = rolls.main_model_calls.to_numpy(dtype=float).reshape(len(ids), 3)
    secs = rolls.wall_time_seconds.to_numpy(dtype=float).reshape(len(ids), 3)
    lang = tasks.set_index("instance_id").loc[ids, "language"].to_numpy()
    return ids, s, calls, secs, lang


def half(instance_id):
    return hashlib.sha256((instance_id + ":schedule").encode()).digest()[0] < 128


def solved_at(costs, gains, budgets):
    """Tasks solved when the queue stops at the first attempt that does not fit."""
    cum = np.cumsum(costs)
    solved = np.concatenate([[0], np.cumsum(gains)])
    return solved[np.searchsorted(cum, budgets, side="right")]


def sequences(s, c, order, perm, score=None):
    """Attempt sequences (cost, gain) for each policy on one replicate.

    s, c: (n, 3) outcomes and costs; order: task order; perm: (n, 3) rollout order per task.
    score: optional (n, 3) restart score for the attempt that just failed (higher first).
    """
    so = np.take_along_axis(s[order], perm, axis=1)
    co = np.take_along_axis(c[order], perm, axis=1)
    out = {"move_on": (co[:, 0], so[:, 0])}
    # depth-first: attempts 0..first success
    first = np.where(so.any(axis=1), so.argmax(axis=1), 2)
    mask = np.arange(3)[None, :] <= first[:, None]
    out["depth_first"] = (co[mask], so[mask])
    # breadth-first
    need2 = ~so[:, 0]
    need3 = need2 & ~so[:, 1]
    out["breadth_first"] = (np.concatenate([co[:, 0], co[need2, 1], co[need3, 2]]),
                            np.concatenate([so[:, 0], so[need2, 1], so[need3, 2]]))
    if score is not None:
        sc = np.take_along_axis(score[order], perm, axis=1)
        i2 = np.where(need2)[0][np.argsort(-sc[need2, 0], kind="stable")]
        i3 = np.where(need3)[0][np.argsort(-sc[need3, 1], kind="stable")]
        out["breadth_targeted"] = (np.concatenate([co[:, 0], co[i2, 1], co[i3, 2]]),
                                   np.concatenate([so[:, 0], so[i2, 1], so[i3, 2]]))
    return out


def restart_scores(s, calls, lang, train):
    """Restart success per call for (language, compute tercile of the failed attempt), from training tasks.

    Exchangeable pairs: failed attempt i, other attempt j on the same task.
    """
    fail_calls = calls[train][~s[train]]
    edges = np.quantile(fail_calls, [1 / 3, 2 / 3])
    terc = np.digitize(calls, edges, right=True)
    num, den = {}, {}
    for t in np.where(train)[0]:
        for i in range(3):
            if s[t, i]:
                continue
            for j in range(3):
                if j == i:
                    continue
                for key in ((lang[t], terc[t, i]), (lang[t], None), (None, terc[t, i])):
                    num[key] = num.get(key, 0) + s[t, j]
                    den[key] = den.get(key, 0) + calls[t, j]
    def shrunk(key, base, prior=2000.0):
        return (num.get(key, 0) + prior * base) / (den.get(key, 0) + prior)

    score = np.zeros(calls.shape)
    for t in range(len(lang)):
        for i in range(3):
            tercile_rate = num[(None, terc[t, i])] / den[(None, terc[t, i])]
            lang_rate = shrunk((lang[t], None), tercile_rate)
            score[t, i] = shrunk((lang[t], terc[t, i]), lang_rate)
    table = {f"{k[0]}|{k[1]}": {"restart_successes": float(num[k]), "restart_calls": float(den[k]),
                                "per_1k_calls": 1000 * num[k] / den[k]}
             for k in num if k[0] is not None and k[1] is not None and den[k] > 0}
    return score, edges.tolist(), table


def frontier(s, cost, rng, score=None, n_rep=N_REP):
    n = len(s)
    res = {}
    for _ in range(n_rep):
        boot = rng.integers(0, n, n)
        bs, bc = s[boot], cost[boot]
        bscore = score[boot] if score is not None else None
        order = rng.permutation(n)
        perm = np.argsort(rng.random((n, 3)), axis=1)
        budgets = FRACTIONS * bc.sum()
        for name, (c, g) in sequences(bs, bc, order, perm, bscore).items():
            res.setdefault(name, []).append(solved_at(c, g, budgets))
    return {k: np.array(v) for k, v in res.items()}


def summarize(res, n_tasks):
    out = {"fractions": FRACTIONS.tolist(), "tasks": n_tasks, "policies": {}, "differences": {}}
    for name, arr in res.items():
        out["policies"][name] = {"mean_solved": arr.mean(axis=0).round(2).tolist(),
                                 "mean_share_solved": (arr.mean(axis=0) / n_tasks).round(4).tolist()}
    pairs = [("breadth_first", "depth_first"), ("move_on", "depth_first")]
    if "breadth_targeted" in res:
        pairs.append(("breadth_targeted", "breadth_first"))
    for a, b in pairs:
        d = res[a] - res[b]
        out["differences"][f"{a}_minus_{b}"] = {
            "mean": d.mean(axis=0).round(2).tolist(),
            "ci95": [np.percentile(d, 2.5, axis=0).round(2).tolist(), np.percentile(d, 97.5, axis=0).round(2).tolist()],
            "relative_to_b": (d.mean(axis=0) / res[b].mean(axis=0)).round(4).tolist(),
        }
    return out


def main():
    rng = np.random.default_rng(SEED)
    ids, s, calls, secs, lang = load()
    one_pass_share = {"calls": float(calls[:, 0].sum() / calls.sum()), "seconds": float(secs[:, 0].sum() / secs.sum())}
    result = {"note": "budget = fraction of the cost of all three rollouts on every task; 1,000 replicates, each a task bootstrap and a random task and rollout order shared by all policies",
              "expected_one_pass_share_of_total_cost": one_pass_share}
    result["calls"] = summarize(frontier(s, calls, rng), len(ids))
    result["wall_time"] = summarize(frontier(s, secs, rng), len(ids))
    train = np.array([half(i) for i in ids])
    score, edges, table = restart_scores(s, calls, lang, train)
    test = ~train
    held = frontier(s[test], calls[test], rng, score[test])
    result["targeted_restarts_heldout_half"] = summarize(held, int(test.sum()))
    result["targeted_restarts_heldout_half"]["train_tasks"] = int(train.sum())
    result["targeted_restarts_heldout_half"]["compute_tercile_edges_calls"] = edges
    result["targeted_restarts_heldout_half"]["training_cells"] = table
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    for key in ("calls", "wall_time"):
        r = result[key]
        print(key, "fractions", r["fractions"])
        for name, v in r["policies"].items():
            print(f"  {name:18s}", v["mean_solved"])
        for name, v in r["differences"].items():
            print(f"  {name:32s}", v["mean"], "lo", v["ci95"][0], "hi", v["ci95"][1])
    r = result["targeted_restarts_heldout_half"]
    print("held-out half", r["tasks"])
    for name, v in r["policies"].items():
        print(f"  {name:18s}", v["mean_solved"])
    for name, v in r["differences"].items():
        print(f"  {name:32s}", v["mean"], "lo", v["ci95"][0], "hi", v["ci95"][1])
    print(one_pass_share)


if __name__ == "__main__":
    main()
