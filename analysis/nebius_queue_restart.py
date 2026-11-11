"""Separate-model checks of the queue-crossing decision on public Nebius trajectories.

Two sources, one procedure. llama70b (default): the 70B SWE-agent trajectories,
with trajectory message count as a cost proxy. coder480b: Qwen3-Coder-480B-A35B
with OpenHands, with assistant turns (model calls) as cost; its split, categories,
budgets and primary endpoint were frozen before its attempt outcomes and costs
were extracted.

Attempts per issue vary, so each replicate draws three uniformly without
replacement. The issue split and category map come from training issues only.
Repository is an imperfect substitute for Qwen's language. Policies, the
queue-crossing/restart-order split, the budget curve (SEED + 1) and the
training-half curve (SEED + 2) are those of queue_restart_decision.py. The
primary budget is the grid point where the training half's crossing gain peaks;
the held-out gain at that budget is read from the held-out curve.

Run: .ai/swebench/venv/bin/python analysis/nebius_queue_restart.py [llama70b|coder480b]
"""
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from queue_restart_decision import CURVE, MODES, budget_curves, one_schedule, paired, train_rates

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "llama70b": {"csv": "evidence/results/nebius_attempt_costs.csv",
                 "out": "evidence/results/nebius_queue_restart.json",
                 "source": "nebius/SWE-agent-trajectories, swe-agent-llama-70b",
                 "cost": "trajectory message count (proxy, not calls/tokens/seconds)",
                 "salt": ":nebius-queue"},
    "coder480b": {"csv": "evidence/results/openhands_attempt_costs.csv",
                  "out": "evidence/results/openhands_queue_restart.json",
                  "source": "nebius/SWE-rebench-openhands-trajectories, Qwen3-Coder-480B-A35B-Instruct, OpenHands 0.54.0",
                  "cost": "assistant turns (main-model calls)",
                  "salt": ":openhands-queue"},
}
SEED = 20260930
REPS = 150
FRACTIONS = (0.3, 0.4, 0.5)


def load(name):
    """issue -> (n, 2) array of (resolved, cost), and issue -> repository."""
    issues, repos = defaultdict(list), {}
    with (ROOT / SOURCES[name]["csv"]).open(newline="") as f:
        for row in csv.DictReader(f):
            if name == "llama70b":
                if row["model_name"] != "swe-agent-llama-70b":
                    continue
                issue, outcome, cost = row["instance_id"], row["target"].lower() == "true", int(row["trajectory_messages"])
                repos[issue] = issue.rsplit("-", 1)[0]
            else:
                issue, outcome, cost = row["instance_id"], row["resolved"] == "1", int(row["assistant_calls"])
                repos[issue] = row["repo"]
            issues[issue].append((outcome, cost))
    return ({key: np.asarray(values, dtype=np.int32) for key, values in sorted(issues.items())
             if len(values) >= 3}, repos)


def draw_three(issues, ids, rng):
    selected = [issues[ids[i]][rng.choice(len(issues[ids[i]]), 3, replace=False)]
                for i in range(len(ids))]
    array = np.asarray(selected)
    return array[:, :, 0].astype(bool), array[:, :, 1].astype(float)


def main(name="llama70b"):
    spec = SOURCES[name]
    issues, repos = load(name)
    train_ids = [i for i in issues if hashlib.sha256((i + spec["salt"]).encode()).digest()[0] < 128]
    train_set = set(train_ids)
    test_ids = [i for i in issues if i not in train_set]
    # A repository needs at least ten *training* issues to get its own category.
    repositories = Counter(repos[i] for i in train_ids)
    categories = {repo for repo, count in repositories.items() if count >= 10}
    def category(issue):
        return repos[issue] if repos[issue] in categories else "other"

    rng = np.random.default_rng(SEED)
    train_success, train_cost = draw_three(issues, train_ids, rng)
    train_cat = np.asarray([category(i) for i in train_ids])
    fresh, retry, edges, base = train_rates(train_success, train_cost, train_cat)
    fresh["__global__"] = base
    records = {str(f): {m: [] for m in MODES} for f in FRACTIONS}
    for rep in range(REPS):
        sampled = rng.integers(0, len(test_ids), len(test_ids))
        ids = [test_ids[i] for i in sampled]
        ss, cc = draw_three(issues, ids, rng)
        cats = np.asarray([category(i) for i in ids])
        full_cost = float(cc.sum())
        for fraction in FRACTIONS:
            state = rng.bit_generator.state
            for mode in MODES:
                rng.bit_generator.state = state
                records[str(fraction)][mode].append(one_schedule(
                    ss, cc, cats, fresh, retry, edges, fraction * full_cost, mode, rng))
        if (rep + 1) % 25 == 0:
            print(f"replicates: {rep + 1}/{REPS}", flush=True)

    result = {"source": spec["source"], "cost": spec["cost"],
              "split": f"SHA-256(instance_id + '{spec['salt']}') < 128 for training",
              "attempts_per_issue": 3, "train_issues": len(train_ids),
              "heldout_issues": len(test_ids), "category_min_train_issues": 10,
              "categories": len(categories) + 1, "replicates": REPS,
              "train_fresh_per_1000_cost": {key: round(value * 1000, 3) for key, value in fresh.items()},
              "train_retry_global_per_1000_cost": {str(key[2]): round(value * 1000, 3)
                                                   for key, value in retry.items() if key[0] is None},
              "budgets": {}}
    for fraction, modes in records.items():
        summary = {}
        for mode, triples in modes.items():
            array = np.asarray(triples)
            summary[mode] = {"solved_mean": round(float(array[:, 0].mean()), 2),
                             "spent_cost_mean": round(float(array[:, 1].mean()), 1),
                             "early_restarts_mean": round(float(array[:, 2].mean()), 2)}
        solved = {mode: np.asarray(triples)[:, 0] for mode, triples in modes.items()}
        for mode in ("plain_breadth", "ranked_breadth", "deferred_rate"):
            summary["adaptive"]["minus_" + mode] = paired(solved["adaptive"], solved[mode])
        summary["deferred_rate"]["minus_ranked_breadth"] = paired(solved["deferred_rate"], solved["ranked_breadth"])
        result["budgets"][fraction] = summary

    def draw(generator):
        ids = [test_ids[i] for i in generator.integers(0, len(test_ids), len(test_ids))]
        ss, cc = draw_three(issues, ids, generator)
        return ss, cc, np.asarray([category(i) for i in ids])

    def draw_train(generator):
        ids = [train_ids[i] for i in generator.integers(0, len(train_ids), len(train_ids))]
        ss, cc = draw_three(issues, ids, generator)
        return ss, cc, np.asarray([category(i) for i in ids])

    result["curve"], result["decomposition"] = budget_curves(
        fresh, retry, edges, np.random.default_rng(SEED + 1), REPS, CURVE, draw,
        groups=tuple(float(f) for f in FRACTIONS))
    result["train_curve"], _ = budget_curves(fresh, retry, edges, np.random.default_rng(SEED + 2),
                                             REPS, CURVE, draw_train)
    gains = result["train_curve"]["adaptive_minus_deferred_rate"]["mean"]
    k = max(range(len(CURVE)), key=lambda i: gains[i])
    held = result["curve"]["adaptive_minus_deferred_rate"]
    result["primary"] = {"fraction": CURVE[k], "chosen_by": "argmax of the training half's crossing gain",
                         "heldout_crossing": {"mean": held["mean"][k], "paired_95": [held["lo"][k], held["hi"][k]]}}
    (ROOT / spec["out"]).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"train_issues": len(train_ids), "heldout_issues": len(test_ids),
                      "budgets": result["budgets"], "primary": result["primary"]}, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:2])
