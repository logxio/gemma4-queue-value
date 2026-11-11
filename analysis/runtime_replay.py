"""Train on disjoint issue prefixes, then replay runtime-only stopping on held-out issues."""
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / ".ai" / "swebench"
CHECKPOINTS = (5, 10, 15)
ROWS = {cp: json.loads((ROOT / f"runtime_rows_{cp}.json").read_text()) for cp in CHECKPOINTS}
FEATURES = {
    "mentions": ("path_mentions",),
    "focus": ("focus",),
    "source": ("source_paths",),
    "edit_agreement": ("edit_agreement", "has_source_edit"),
    "footprint": ("path_mentions", "source_paths", "focus", "repeat_share", "test_path_share", "edit_early", "edit_agreement", "has_source_edit"),
}


def bucket(issue, salt="train"):
    return hashlib.sha256((issue + ":" + salt).encode()).digest()[0]


def vector(row, names):
    return [math.log1p(row[n]) if n in ("path_mentions", "source_paths") else row[n] for n in names]


def fit(rows, names):
    x = np.array([vector(r, names) for r in rows], dtype=float)
    y = np.array([r["resolved"] for r in rows], dtype=float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1
    x = np.clip((x - mean) / scale, -5, 5)
    w = np.zeros(x.shape[1])
    bias = math.log(max(.01, y.mean()) / max(.01, 1-y.mean()))
    for _ in range(600):
        z = np.clip(x @ w + bias, -20, 20)
        p = 1 / (1 + np.exp(-z))
        w -= .1 * ((x.T @ (p-y))/len(y) + .03*w)
        bias -= .1 * (p-y).mean()
    return mean, scale, w, bias


def predict(model, rows, names):
    mean, scale, w, bias = model
    x = np.array([vector(r, names) for r in rows], dtype=float)
    z = np.clip(np.clip((x-mean)/scale, -5, 5) @ w + bias, -20, 20)
    return 1 / (1 + np.exp(-z))


def auc(y, score):
    pos = [v for v, label in zip(score, y) if label]
    neg = [v for v, label in zip(score, y) if not label]
    if not pos or not neg:
        return .5
    return sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))


def eligible(row):
    return row["prefix"] < row["full"]


def replay(rows, budget_fraction, seed):
    order = list(enumerate(rows))
    random.Random(seed).shuffle(order)
    budget = budget_fraction * sum(r["full"] for _, r in order)
    stop_n = sum(r.get("stop", False) and eligible(r) for r in rows)
    random_set = set(random.Random(seed+100000).sample([i for i, r in enumerate(rows) if eligible(r)], stop_n))
    outcomes = []
    for policy in ("baseline", "signal", "random"):
        used = fixed = 0
        for i, r in order:
            stop = policy == "signal" and r.get("stop", False) and eligible(r) or policy == "random" and i in random_set
            cost = r["prefix"] if stop else r["full"]
            if used + cost > budget:
                break
            used += cost
            if not stop:
                fixed += r["resolved"]
        outcomes.append(fixed)
    return outcomes


def evaluate(rows, threshold, names, model, n_orders=1000):
    probs = predict(model, rows, names)
    prepared = []
    for r, prob in zip(rows, probs):
        item = dict(r)
        item["runtime_score"] = float(prob)
        item["stop"] = bool(prob + bucket(r["instance_id"], "tie")*1e-9 < threshold and eligible(r))
        prepared.append(item)
    samples = [replay(prepared, .5, seed) for seed in range(n_orders)]
    diffs = sorted(g-b for b,g,_ in samples)
    random_diffs = sorted(g-r for _,g,r in samples)
    result = {
        "n": len(rows),
        "resolved": sum(r["resolved"] for r in rows),
        "stop_count": sum(r["stop"] for r in prepared),
        "stopped_would_resolve": sum(r["resolved"] for r in prepared if r["stop"]),
        "auc": float(round(auc([r["resolved"] for r in rows], probs), 3)),
        "baseline_mean": round(sum(a for a,_,_ in samples)/n_orders, 2),
        "gate_mean": round(sum(b for _,b,_ in samples)/n_orders, 2),
        "random_stop_mean": round(sum(c for _,_,c in samples)/n_orders, 2),
        "gate_minus_baseline": round(sum(diffs)/n_orders, 2),
        "order_interval_95": [diffs[int(.025*n_orders)], diffs[int(.975*n_orders)-1]],
        "gate_minus_random": round(sum(random_diffs)/n_orders, 2),
    }
    if n_orders == 1000:
        bootstrap = []
        auc_bootstrap = []
        for seed in range(1000):
            sample = random.Random(seed+500000).choices(prepared, k=len(prepared))
            baseline, gate, _ = replay(sample, .5, seed)
            bootstrap.append(gate-baseline)
            if seed < 500:
                auc_bootstrap.append(auc([r["resolved"] for r in sample], [r["runtime_score"] for r in sample]))
        bootstrap.sort()
        auc_bootstrap.sort()
        result["issue_bootstrap_95"] = [bootstrap[25], bootstrap[974]]
        result["auc_bootstrap_95"] = [round(auc_bootstrap[12],3), round(auc_bootstrap[487],3)]
    return result


def select(system):
    split = "verified"
    choices = []
    for cp in CHECKPOINTS:
        rows = [r for r in ROWS[cp] if r["system"] == system and r["split"] == split and bucket(r["instance_id"]) < 153]
        for name, names in FEATURES.items():
            scored = []
            fold_gains = []
            for fold in range(3):
                train = [r for r in rows if bucket(r["instance_id"], "fold") % 3 != fold]
                valid = [r for r in rows if bucket(r["instance_id"], "fold") % 3 == fold]
                model = fit(train, names)
                scored.extend((r["resolved"], float(p)) for r,p in zip(valid, predict(model, valid, names)))
                probs = predict(model, train, names)
                scores = sorted(float(p)+bucket(r["instance_id"], "tie")*1e-9 for r,p in zip(train, probs) if eligible(r))
                threshold = scores[max(0,int(.2*len(scores))-1)] + 1e-12
                fold_gains.append(evaluate(valid, threshold, names, model, n_orders=100)["gate_minus_baseline"])
            cv_auc = auc([y for y,p in scored], [p for y,p in scored])
            cv_gain = sum(fold_gains)/len(fold_gains)
            choices.append((cv_gain, cv_auc, -cp, name, cp))
    cv_gain, cv_auc, _, name, cp = max(choices)
    names = FEATURES[name]
    train = [r for r in ROWS[cp] if r["system"] == system and r["split"] == split and bucket(r["instance_id"]) < 153]
    test = [r for r in ROWS[cp] if r["system"] == system and r["split"] == split and bucket(r["instance_id"]) >= 153]
    model = fit(train, names)
    probs = predict(model, train, names)
    scores = sorted(float(p)+bucket(r["instance_id"], "tie")*1e-9 for r,p in zip(train, probs) if eligible(r))
    threshold = scores[max(0,int(.2*len(scores))-1)] + 1e-12
    result = {"checkpoint": cp, "feature_set": name, "features": names, "cv_auc": float(round(cv_auc,3)),
              "cv_gate_gain": round(cv_gain,2),
              "train_n": len(train), "train": evaluate(train, threshold, names, model),
              "verified_or_multilingual_holdout": evaluate(test, threshold, names, model)}
    if system in ("sweagent", "openhands"):
        lite = [r for r in ROWS[cp] if r["system"] == system and r["split"] == "lite"]
        result["lite_external"] = evaluate(lite, threshold, names, model)
    if system == "mini":
        multi = [r for r in ROWS[cp] if r["system"] == system and r["split"] == "multilingual"]
        result["multilingual_external"] = evaluate(multi, threshold, names, model)
    return result


if __name__ == "__main__":
    results = {system: select(system) for system in ("sweagent", "agentless", "openhands", "mini")}
    (ROOT / "runtime_replay_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
