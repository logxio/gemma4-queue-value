"""Restart value on a third public source: nebius/SWE-agent-trajectories (CC BY 4.0).

Reads only three columns (instance_id, model_name, target) of the 12 remote parquet
shards with DuckDB over HTTP, stores per model-and-issue attempt counts in
evidence/results/nebius_instance_counts.csv, and reports for each model with repeated
attempts: p, q = P(another attempt on the same issue succeeds | this one failed), q/p,
and the between-issue variance share, with issue-bootstrap intervals.

Run with the analysis venv (needs duckdb): .ai/swebench/venv/bin/python analysis/nebius_restart.py [--refetch]
Attempt counts per issue vary with outcome here, so the headline estimate draws exactly three
attempts per issue (equal issue weight), matching the other two repeated-rollout sources.
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import duckdb

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "evidence" / "results"
BASE = "https://huggingface.co/datasets/nebius/SWE-agent-trajectories/resolve/main/data/train-{:05d}-of-00012.parquet"
SEED = 20260929


def fetch() -> dict:
    con = duckdb.connect()
    con.sql("INSTALL httpfs; LOAD httpfs;")
    counts = defaultdict(lambda: [0, 0])
    for i in range(12):
        rows = con.sql(f"select model_name, instance_id, count(*), sum(target::int) "
                       f"from read_parquet('{BASE.format(i)}') group by 1, 2").fetchall()
        for model, inst, n, s in rows:
            counts[(model, inst)][0] += n
            counts[(model, inst)][1] += s
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "nebius_instance_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_name", "instance_id", "attempts", "successes"])
        for (model, inst), (n, s) in sorted(counts.items()):
            w.writerow([model, inst, n, s])
    return counts


def load() -> dict:
    counts = {}
    with open(OUT / "nebius_instance_counts.csv") as f:
        for r in csv.DictReader(f):
            counts[(r["model_name"], r["instance_id"])] = [int(r["attempts"]), int(r["successes"])]
    return counts


def stats(groups):
    """groups: list of (n, s). Pair estimator for q; one-way ANOVA ICC for unequal group sizes."""
    N = sum(n for n, _ in groups)
    S = sum(s for _, s in groups)
    p = S / N
    num = sum((n - s) * s for n, s in groups)
    den = sum((n - s) * (n - 1) for n, s in groups)
    q = num / den if den else float("nan")
    k = len(groups)
    msb = sum(n * (s / n - p) ** 2 for n, s in groups) / (k - 1)
    msw = sum(s * (1 - s / n) ** 2 + (n - s) * (s / n) ** 2 for n, s in groups) / (N - k)
    n0 = (N - sum(n * n for n, _ in groups) / N) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw) if msb + (n0 - 1) * msw else float("nan")
    return {"p": p, "q": q, "q_over_p": q / p if p else float("nan"), "icc": icc}


def pct(xs):
    xs = sorted(x for x in xs if x == x)
    return [round(xs[int(0.025 * len(xs))], 4), round(xs[min(len(xs) - 1, int(0.975 * len(xs)))], 4)]


def balanced(groups, k, rng, draws=50):
    """Attempt counts per issue depend on outcomes in this dataset, so the pooled pair estimator
    over-weights issues that were sampled more. Draw exactly k attempts per issue without
    replacement (hypergeometric on the issue's counts), estimate p and q with equal issue weight,
    and average over draws."""
    ps, qs = [], []
    for _ in range(draws):
        sampled = []
        for n, s in groups:
            pool = [1] * s + [0] * (n - s)
            sampled.append(sum(rng.sample(pool, k)))
        p = sum(sampled) / (k * len(sampled))
        num = sum((k - x) * x for x in sampled)
        den = sum((k - x) * (k - 1) for x in sampled)
        ps.append(p); qs.append(num / den if den else float("nan"))
    p = sum(ps) / len(ps); q = sum(qs) / len(qs)
    return {"p": p, "q": q, "q_over_p": q / p if p else float("nan")}


def main() -> None:
    import sys
    counts = fetch() if "--refetch" in sys.argv or not (OUT / "nebius_instance_counts.csv").exists() else load()
    by_model = defaultdict(list)
    for (model, _), (n, s) in counts.items():
        if n >= 2:
            by_model[model].append((n, s))
    rng = random.Random(SEED)
    result = {"source": "nebius/SWE-agent-trajectories, columns instance_id, model_name, target", "models": {}}
    for model, groups in sorted(by_model.items(), key=lambda kv: -sum(n for n, _ in kv[1])):
        if len(groups) < 20:
            continue
        point = stats(groups)
        boots = [stats([groups[rng.randrange(len(groups))] for _ in groups]) for _ in range(1000)]
        dist = [sum(1 for n, s in groups if s == 0), sum(1 for n, s in groups if 0 < s < n), sum(1 for n, s in groups if s == n)]
        result["models"][model] = {
            "issues": len(groups), "attempts": sum(n for n, _ in groups),
            "median_attempts_per_issue": sorted(n for n, _ in groups)[len(groups) // 2],
            "issues_never_all_mixed": dist,
            **{k: round(v, 4) for k, v in point.items()},
            "interval_95": {k: pct([b[k] for b in boots]) for k in point},
        }
        eligible = [g for g in groups if g[0] >= 3]
        bal = balanced(eligible, 3, rng)
        bal_boot = [balanced([eligible[rng.randrange(len(eligible))] for _ in eligible], 3, rng, draws=5) for _ in range(300)]
        result["models"][model]["three_per_issue"] = {
            "issues": len(eligible), **{k: round(v, 4) for k, v in bal.items()},
            "interval_95": {k: pct([b[k] for b in bal_boot]) for k in bal}}
    (OUT / "nebius_restart.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
