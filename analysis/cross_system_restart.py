"""Cross-system analogue of restart value on the 465 matched Verified issues.

Four different agents ran once on the same issues. They are not exchangeable,
so nothing is pooled into a same-agent q/p. We ask whether failures, and prefixes
flagged by each system's frozen gate or by a loop signature, sit on issues that
other agents solve. Writes evidence/results/cross_system_restart.json.
"""
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
from decision_value import FEATURES, ROWS, bucket, fit_threshold, stop_flags  # noqa: E402

OUT = REPO / "evidence" / "results" / "cross_system_restart.json"
SYSTEMS = ("sweagent", "agentless", "openhands", "mini")
GATES = {"sweagent": (10, "source"), "agentless": (5, "footprint"), "openhands": (5, "source"), "mini": (5, "source")}
N_BOOT = 2000


def interval(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main():
    rng = np.random.default_rng(3)
    rows10 = {s: {r["instance_id"]: r for r in ROWS[10] if r["system"] == s and r["split"] == "verified"} for s in SYSTEMS}
    ids = sorted(set.intersection(*[set(v) for v in rows10.values()]))
    n = len(ids)
    y = np.array([[rows10[s][i]["resolved"] for s in SYSTEMS] for i in ids], dtype=bool)
    k = y.sum(axis=1)
    result = {"issues": n, "resolved_rate": {s: float(y[:, j].mean()) for j, s in enumerate(SYSTEMS)},
              "issues_by_systems_resolving": {str(v): int((k == v).sum()) for v in range(5)}}

    # Conditional rates and odds ratios for each ordered pair.
    pairs = {}
    boot = rng.integers(0, n, (N_BOOT, n))
    for a, b in permutations(range(4), 2):
        fa = ~y[:, a]
        cond = y[fa, b].mean()
        marg = y[:, b].mean()
        yb = y[boot]
        cond_b = np.array([yb[r][~yb[r][:, a], b].mean() for r in range(N_BOOT)])
        marg_b = yb[:, :, b].mean(axis=1)
        pairs[f"{SYSTEMS[a]}->{SYSTEMS[b]}"] = {"p_b": float(marg), "p_b_given_a_failed": float(cond),
                                                "ratio": float(cond / marg), "ratio_95": interval(cond_b / marg_b)}
    ratios = [v["ratio"] for v in pairs.values()]
    result["pairs"] = pairs
    result["pair_ratio_range"] = [min(ratios), max(ratios)]
    result["pair_ratio_mean"] = float(np.mean(ratios))

    # Flags: each system's frozen runtime gate at its selected checkpoint, and a loop signature at 10 actions.
    flagged = {}
    for j, s in enumerate(SYSTEMS):
        cp, fam = GATES[s]
        rows_cp = {r["instance_id"]: r for r in ROWS[cp] if r["system"] == s and r["split"] == "verified"}
        train = [rows_cp[i] for i in ids if bucket(i) < 153]
        model, threshold = fit_threshold(train, FEATURES[fam])
        gate, _ = stop_flags([rows_cp[i] for i in ids], model, FEATURES[fam], threshold)
        rep = np.array([rows10[s][i]["repeat_share"] for i in ids])
        loop_cut = np.quantile([rows10[s][i]["repeat_share"] for i in ids if bucket(i) < 153], .8)
        loop = (rep >= loop_cut) & (rep > 0) & np.array([rows10[s][i]["prefix"] < rows10[s][i]["full"] for i in ids])
        others = [m for m in range(4) if m != j]
        other_rate = y[:, others].mean(axis=1)
        any_other = y[:, others].any(axis=1)
        entry = {}
        for name, flag in (("gate", gate), ("loop_top_quintile_repeat_share", loop)):
            groups = {"all": np.ones(n, bool), "flagged": flag, "flagged_failed": flag & ~y[:, j],
                      "failed_not_flagged": (~y[:, j]) & ~flag, "resolved": y[:, j]}
            g = {}
            for label, mask in groups.items():
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    continue
                bo = [other_rate[rng.choice(idx, len(idx))].mean() for _ in range(N_BOOT)]
                g[label] = {"n": int(len(idx)), "own_success": float(y[idx, j].mean()),
                            "mean_other_system_success": float(other_rate[idx].mean()), "ci95": interval(bo),
                            "any_other_system_solves": float(any_other[idx].mean())}
            entry[name] = g
        flagged[s] = entry
    result["flags"] = flagged
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k2: v for k2, v in result.items() if k2 not in ("pairs", "flags")}, indent=1))
    for s, entry in flagged.items():
        for name, g in entry.items():
            print(s, name, {lab: (v["n"], round(v["own_success"], 2), round(v["mean_other_system_success"], 3),
                                  [round(c, 3) for c in v["ci95"]], round(v["any_other_system_solves"], 2)) for lab, v in g.items()})
    for key, v in pairs.items():
        print(key, round(v["p_b"], 3), round(v["p_b_given_a_failed"], 3), round(v["ratio"], 3), [round(x, 3) for x in v["ratio_95"]])


if __name__ == "__main__":
    main()
