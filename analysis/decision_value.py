"""Monitor accuracy is not decision value: value identity, AUC misranking,
break-even screen, pilot size, and a pooled held-out estimate.

Reads the committed runtime rows (copied to .ai/swebench/ by the caller or by
this script) and reuses the fitting code of analysis/runtime_replay.py.
Writes evidence/results/decision_value.json and decision_value_grid.csv.
"""
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / ".ai" / "swebench"
CACHE.mkdir(parents=True, exist_ok=True)
for cp in (5, 10, 15):
    target = CACHE / f"runtime_rows_{cp}.json"
    if not target.exists():
        shutil.copy(REPO / "evidence" / "data" / f"runtime_rows_{cp}.json", target)
sys.path.insert(0, str(REPO / "analysis"))
from runtime_replay import CHECKPOINTS, FEATURES, ROWS, auc, bucket, eligible, fit, predict  # noqa: E402

OUT = REPO / "evidence" / "results"
SYSTEMS = ("sweagent", "agentless", "openhands", "mini")
EXTERNAL = {"sweagent": "lite", "openhands": "lite", "mini": "multilingual"}
_PERMS = {}


def perms(n, n_orders):
    """Issue orders identical to runtime_replay.replay (random.Random(seed).shuffle)."""
    key = (n, n_orders)
    if key not in _PERMS:
        out = []
        for seed in range(n_orders):
            order = list(range(n))
            random.Random(seed).shuffle(order)
            out.append(order)
        _PERMS[key] = np.array(out)
    return _PERMS[key]


def arrays(rows):
    full = np.array([r["full"] for r in rows], dtype=float)
    prefix = np.array([r["prefix"] for r in rows], dtype=float)
    y = np.array([r["resolved"] for r in rows], dtype=float)
    return full, prefix, y


def replay_mean(rows, stop, n_orders=1000, fraction=.5):
    """Mean baseline and gate repairs over paired orders; same semantics as runtime_replay.replay."""
    full, prefix, y = arrays(rows)
    budget = fraction * full.sum()
    order = perms(len(rows), n_orders)
    gate_cost = np.where(stop, prefix, full)
    gate_gain = np.where(stop, 0, y)
    out = []
    for cost, gain in ((full, y), (gate_cost, gate_gain)):
        c = np.cumsum(cost[order], axis=1)
        g = np.cumsum(gain[order], axis=1)
        k = (c <= budget).sum(axis=1)
        solved = np.where(k > 0, g[np.arange(len(k)), np.maximum(k - 1, 0)], 0)
        out.append(solved)
    return out[0], out[1]


def identity(rows, stop, fraction=.5):
    full, prefix, y = arrays(rows)
    C, Y = full.sum(), y.sum()
    R = (full - prefix)[stop].sum()
    YS = y[stop].sum()
    B = fraction * C
    lam = Y / C
    return {"saved_calls": float(R), "sacrificed": float(YS), "lambda": float(lam),
            "sacrificed_per_saved_call": float(YS / R) if R else None,
            "ratio_form": float(B * ((Y - YS) / (C - R) - Y / C)),
            "linear_form": float(fraction * (lam * R - YS)),
            "screen_enable": bool(R > 0 and YS / R < lam)}


def stop_flags(rows, model, names, threshold):
    probs = predict(model, rows, names)
    stop = np.array([bool(float(p) + bucket(r["instance_id"], "tie") * 1e-9 < threshold and eligible(r))
                     for r, p in zip(rows, probs)])
    return stop, probs


def fit_threshold(train, names):
    model = fit(train, names)
    probs = predict(model, train, names)
    scores = sorted(float(p) + bucket(r["instance_id"], "tie") * 1e-9 for r, p in zip(train, probs) if eligible(r))
    return model, scores[max(0, int(.2 * len(scores)) - 1)] + 1e-12


def training_cv(rows, names):
    """Three-fold training criterion exactly as runtime_replay.select (100 orders per fold)."""
    scored, gains = [], []
    for fold in range(3):
        train = [r for r in rows if bucket(r["instance_id"], "fold") % 3 != fold]
        valid = [r for r in rows if bucket(r["instance_id"], "fold") % 3 == fold]
        model, threshold = fit_threshold(train, names)
        stop, probs = stop_flags(valid, model, names, threshold)
        scored.extend((r["resolved"], float(p)) for r, p in zip(valid, probs))
        base, gate = replay_mean(valid, stop, n_orders=100)
        gains.append(round(float((gate - base).mean()), 2))
    return auc([a for a, _ in scored], [b for _, b in scored]), sum(gains) / 3


def split_rows(system, cp):
    rows = ROWS[cp]
    train = [r for r in rows if r["system"] == system and r["split"] == "verified" and bucket(r["instance_id"]) < 153]
    test = [r for r in rows if r["system"] == system and r["split"] == "verified" and bucket(r["instance_id"]) >= 153]
    ext = [r for r in rows if r["system"] == system and r["split"] == EXTERNAL.get(system)]
    return train, test, ext


def evaluate_candidate(system, cp, family):
    names = FEATURES[family]
    train, test, ext = split_rows(system, cp)
    cv_auc, cv_gain = training_cv(train, names)
    model, threshold = fit_threshold(train, names)
    out = {"system": system, "checkpoint": cp, "family": family, "cv_auc": cv_auc, "cv_gain": cv_gain}
    tr_stop, _ = stop_flags(train, model, names, threshold)
    tr_id = identity(train, tr_stop)
    out["train_screen_enable"] = tr_id["screen_enable"]
    out["train_sacrificed_per_saved_call"] = tr_id["sacrificed_per_saved_call"]
    out["train_lambda"] = tr_id["lambda"]
    sets = {"heldout": test}
    if ext:
        sets[EXTERNAL[system]] = ext
    for label, rows in sets.items():
        stop, probs = stop_flags(rows, model, names, threshold)
        base, gate = replay_mean(rows, stop)
        ident = identity(rows, stop)
        out[label] = {"n": len(rows), "auc": auc([r["resolved"] for r in rows], probs),
                      "stops": int(stop.sum()), "baseline_mean": float(base.mean()), "gate_mean": float(gate.mean()),
                      "replay_gain": float((gate - base).mean()), **ident}
    return out


def pilot_power(rows, stop, sizes=(20, 40, 80), draws=2000, seed=7):
    rng = np.random.default_rng(seed)
    full, prefix, y = arrays(rows)
    saved = full - prefix
    out = {}
    for n in sizes:
        disable = 0
        for _ in range(draws):
            idx = rng.choice(len(rows), size=n, replace=False)
            s = stop[idx]
            R, YS = saved[idx][s].sum(), y[idx][s].sum()
            lam = y[idx].sum() / full[idx].sum()
            if R > 0 and not YS / R < lam:
                disable += 1
        out[str(n)] = disable / draws
    return out


def pooled(selected, reps=2000, seed=11):
    rng = np.random.default_rng(seed)
    data = {}
    ids = None
    for system, cand in selected.items():
        names = FEATURES[cand["family"]]
        train, test, _ = split_rows(system, cand["checkpoint"])
        model, threshold = fit_threshold(train, names)
        stop, _ = stop_flags(test, model, names, threshold)
        test_ids = [r["instance_id"] for r in test]
        ids = test_ids if ids is None else ids
        assert sorted(ids) == sorted(test_ids)
        pos = {i: k for k, i in enumerate(test_ids)}
        order = [pos[i] for i in ids]
        full, prefix, y = arrays(test)
        data[system] = (full[order], prefix[order], y[order], stop[order])
    n = len(ids)
    replay_sum, ident_sum, averaged_sum = [], [], []
    for _ in range(reps):
        draw = rng.integers(0, n, n)
        total_r = total_i = total_a = 0.0
        for full, prefix, y, stop in data.values():
            f, p, yy, s = full[draw], prefix[draw], y[draw], stop[draw]
            budget = .5 * f.sum()
            perm = rng.permutation(n)
            for cost, gain, sign in ((f, yy, -1), (np.where(s, p, f), np.where(s, 0, yy), 1)):
                c = np.cumsum(cost[perm])
                k = int((c <= budget).sum())
                total_r += sign * (np.cumsum(gain[perm])[k - 1] if k else 0)
            C, Y = f.sum(), yy.sum()
            R, YS = (f - p)[s].sum(), yy[s].sum()
            total_i += .5 * C * ((Y - YS) / (C - R) - Y / C)
            # Added after the plan was committed: mean over 100 random orders per replicate.
            orders = np.argsort(rng.random((100, n)), axis=1)
            for cost, gain, sign in ((f, yy, -1), (np.where(s, p, f), np.where(s, 0, yy), 1)):
                c = np.cumsum(cost[orders], axis=1)
                g = np.cumsum(gain[orders], axis=1)
                k = (c <= budget).sum(axis=1)
                total_a += sign * np.where(k > 0, g[np.arange(100), np.maximum(k - 1, 0)], 0).mean()
        averaged_sum.append(total_a)
        replay_sum.append(total_r)
        ident_sum.append(total_i)
    return {"issues": n, "replicates": reps,
            "replay_bootstrap_mean": float(np.mean(replay_sum)),
            "replay_bootstrap_95": [float(np.percentile(replay_sum, 2.5)), float(np.percentile(replay_sum, 97.5))],
            "replay_share_above_zero": float(np.mean(np.array(replay_sum) > 0)),
            "identity_bootstrap_mean": float(np.mean(ident_sum)),
            "identity_bootstrap_95": [float(np.percentile(ident_sum, 2.5)), float(np.percentile(ident_sum, 97.5))],
            "identity_share_above_zero": float(np.mean(np.array(ident_sum) > 0)),
            "order_averaged_replay_bootstrap_mean": float(np.mean(averaged_sum)),
            "order_averaged_replay_bootstrap_95": [float(np.percentile(averaged_sum, 2.5)), float(np.percentile(averaged_sum, 97.5))],
            "order_averaged_replay_share_above_zero": float(np.mean(np.array(averaged_sum) > 0)),
            "note": "replay_* uses one random order per replicate (pre-registered); identity_* is the order-free ratio form (pre-registered); order_averaged_* averages 100 orders per replicate (added after the plan)"}


def fit_line(pred, obs):
    pred, obs = np.asarray(pred), np.asarray(obs)
    slope, intercept = np.polyfit(pred, obs, 1)
    resid = obs - pred
    r2 = 1 - (resid ** 2).sum() / ((obs - obs.mean()) ** 2).sum()
    return {"n": int(len(pred)), "r2_identity_as_prediction": float(r2), "slope": float(slope), "intercept": float(intercept),
            "mean_abs_error": float(np.abs(resid).mean()), "max_abs_error": float(np.abs(resid).max()),
            "sign_agreement": float(np.mean(np.sign(pred) == np.sign(obs)))}


def main():
    grid = [evaluate_candidate(s, cp, fam) for s in SYSTEMS for cp in CHECKPOINTS for fam in FEATURES]

    # Reproduce the frozen selection of runtime_replay.select.
    selected = {}
    for s in SYSTEMS:
        cands = [g for g in grid if g["system"] == s]
        best = max(cands, key=lambda g: (g["cv_gain"], g["cv_auc"], -g["checkpoint"], g["family"]))
        selected[s] = best
    reference = json.loads((REPO / "evidence" / "data" / "runtime_replay_summary.json").read_text())
    check = {}
    for s, cand in selected.items():
        ref = reference[s]
        mine = cand["heldout"]
        check[s] = {"checkpoint_match": cand["checkpoint"] == ref["checkpoint"], "family_match": cand["family"] == ref["feature_set"],
                    "baseline_diff": round(mine["baseline_mean"] - ref["verified_or_multilingual_holdout"]["baseline_mean"], 3),
                    "gate_diff": round(mine["gate_mean"] - ref["verified_or_multilingual_holdout"]["gate_mean"], 3)}
    assert all(v["checkpoint_match"] and v["family_match"] and abs(v["baseline_diff"]) <= .01 and abs(v["gate_diff"]) <= .01
               for v in check.values()), check

    # C1: identity versus replay across all evaluated sets.
    evals = [(g, label) for g in grid for label in ("heldout", "lite", "multilingual") if label in g]
    ident_ratio = fit_line([g[l]["ratio_form"] for g, l in evals], [g[l]["replay_gain"] for g, l in evals])
    ident_linear = fit_line([g[l]["linear_form"] for g, l in evals], [g[l]["replay_gain"] for g, l in evals])
    screen_vs_sign = float(np.mean([(g[l]["screen_enable"]) == (g[l]["replay_gain"] > 0) for g, l in evals]))

    # C2: AUC versus value.
    per_system = {}
    forfeited = 0.0
    for s in SYSTEMS:
        cands = [g for g in grid if g["system"] == s]
        gain = [g["heldout"]["replay_gain"] for g in cands]
        by_cv_auc = max(cands, key=lambda g: g["cv_auc"])
        by_cv_val = selected[s]
        by_ho_auc = max(cands, key=lambda g: g["heldout"]["auc"])
        per_system[s] = {
            "spearman_heldout_auc_vs_heldout_gain": float(spearmanr([g["heldout"]["auc"] for g in cands], gain).statistic),
            "spearman_cv_auc_vs_heldout_gain": float(spearmanr([g["cv_auc"] for g in cands], gain).statistic),
            "spearman_cv_value_vs_heldout_gain": float(spearmanr([g["cv_gain"] for g in cands], gain).statistic),
            "spearman_heldout_identity_vs_heldout_gain": float(spearmanr([g["heldout"]["ratio_form"] for g in cands], gain).statistic),
            "selected_by_cv_auc": [by_cv_auc["checkpoint"], by_cv_auc["family"], by_cv_auc["heldout"]["replay_gain"]],
            "selected_by_cv_value": [by_cv_val["checkpoint"], by_cv_val["family"], by_cv_val["heldout"]["replay_gain"]],
            "selected_by_heldout_auc_oracle": [by_ho_auc["checkpoint"], by_ho_auc["family"], by_ho_auc["heldout"]["replay_gain"]],
            "best_heldout_gain_oracle": max(gain),
        }
        forfeited += by_cv_val["heldout"]["replay_gain"] - by_cv_auc["heldout"]["replay_gain"]
    pooled_rank = {
        "spearman_heldout_auc_vs_heldout_gain_all60": float(spearmanr([g["heldout"]["auc"] for g in grid], [g["heldout"]["replay_gain"] for g in grid]).statistic),
        "spearman_heldout_stop_precision_vs_gain_all60": float(spearmanr(
            [g["heldout"]["sacrificed"] / max(1, g["heldout"]["stops"]) for g in grid], [g["heldout"]["replay_gain"] for g in grid]).statistic),
        "mean_within_system_spearman_heldout_auc": float(np.mean([v["spearman_heldout_auc_vs_heldout_gain"] for v in per_system.values()])),
        "repairs_forfeited_selecting_by_cv_auc_instead_of_cv_value_sum4": round(forfeited, 2),
    }

    # C3: break-even screen from training issues only.
    screen_grid = {"train_screen_predicts_heldout_sign": float(np.mean([g["train_screen_enable"] == (g["heldout"]["replay_gain"] > 0) for g in grid])),
                   "always_enable_accuracy_heldout": float(np.mean([g["heldout"]["replay_gain"] > 0 for g in grid])),
                   "train_screen_enabled_count": int(sum(g["train_screen_enable"] for g in grid)),
                   "heldout_gain_when_enabled_mean": float(np.mean([g["heldout"]["replay_gain"] for g in grid if g["train_screen_enable"]] or [np.nan])),
                   "heldout_gain_when_disabled_mean": float(np.mean([g["heldout"]["replay_gain"] for g in grid if not g["train_screen_enable"]] or [np.nan]))}
    selected_rows = {}
    for s, g in selected.items():
        row = {"checkpoint": g["checkpoint"], "family": g["family"], "train_screen_enable": g["train_screen_enable"],
               "train_sacrificed_per_saved_call": g["train_sacrificed_per_saved_call"], "train_lambda": g["train_lambda"]}
        for label in ("heldout", "lite", "multilingual"):
            if label in g:
                row[label] = {k: g[label][k] for k in ("n", "auc", "stops", "sacrificed", "saved_calls", "lambda",
                                                      "sacrificed_per_saved_call", "screen_enable", "replay_gain",
                                                      "ratio_form", "linear_form", "baseline_mean", "gate_mean")}
        selected_rows[s] = row

    # Pilot size to detect a harmful transfer (Multilingual) and false disables (Verified held-out).
    pilots = {}
    for s, g in selected.items():
        names = FEATURES[g["family"]]
        train, test, ext = split_rows(s, g["checkpoint"])
        model, threshold = fit_threshold(train, names)
        stop, _ = stop_flags(test, model, names, threshold)
        pilots[f"verified_heldout_{s}"] = pilot_power(test, stop)
        if s == "mini":
            stop_e, _ = stop_flags(ext, model, names, threshold)
            pilots["multilingual_mini"] = pilot_power(ext, stop_e)

    pool = pooled(selected)
    result = {
        "reproduction_check_vs_committed_summary": check,
        "C1_value_identity": {"ratio_form": ident_ratio, "linear_form": ident_linear,
                              "breakeven_sign_agreement_same_set": screen_vs_sign,
                              "sets": "60 Verified held-out candidates + 30 Lite + 15 Multilingual"},
        "C2_auc_vs_value": {"per_system": per_system, "pooled": pooled_rank},
        "C3_breakeven_screen": {"grid": screen_grid, "selected": selected_rows,
                                "pilot_probability_of_disabling": pilots},
        "C4_pooled_verified_heldout": pool,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "decision_value.json").write_text(json.dumps(result, indent=2) + "\n")
    with open(OUT / "decision_value_grid.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "checkpoint", "family", "cv_auc", "cv_gain", "train_screen_enable", "set", "n", "auc", "stops",
                    "sacrificed", "saved_calls", "replay_gain", "identity_ratio", "identity_linear"])
        for g, l in evals:
            e = g[l]
            w.writerow([g["system"], g["checkpoint"], g["family"], round(g["cv_auc"], 4), round(g["cv_gain"], 3), g["train_screen_enable"],
                        l, e["n"], round(e["auc"], 4), e["stops"], e["sacrificed"], e["saved_calls"], round(e["replay_gain"], 3),
                        round(e["ratio_form"], 3), round(e["linear_form"], 3)])
    print(json.dumps({k: v for k, v in result.items() if k != "C3_breakeven_screen"}, indent=2))
    print(json.dumps(result["C3_breakeven_screen"], indent=2))


if __name__ == "__main__":
    main()
