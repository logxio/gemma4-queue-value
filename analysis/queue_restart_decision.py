"""Frozen, availability-aware restart decision on the Qwen repeated-rollout overlay.

Training and held-out issues use the existing :schedule SHA-256 split. At each
decision the policy sees issue language, prior outcomes, and calls already spent.
It never sees another rollout's outcome or cost before scheduling it. Every policy
stops at the first attempt whose recorded complete-run cost does not fit, which
scores that attempt like a run cut off at the budget with no repair.

Four policies share each replicate's issue order and rollout order:
plain_breadth (random fresh order, restarts after the queue is empty, oldest first),
ranked_breadth (fresh by language yield, restarts after the queue is empty, all
first restarts before any second one, each ranked by yield), deferred_rate (as
ranked_breadth, but restarts ranked by yield alone, the adaptive rule's order) and
adaptive (the same restart order, but a restart may go ahead of fresh issues when
its yield is higher). adaptive minus deferred_rate isolates crossing the queue;
deferred_rate minus ranked_breadth isolates the restart order.

No policy reads the remaining budget, so each replicate's attempt sequence is
budget-free: the budget curve takes prefixes of one unbounded run per policy, on
a separate stream (SEED + 1) so the pre-specified 30/40/50% rows keep their draws.
train_curve replays the training half itself (SEED + 2): it shows where a pilot
alone would place the gain, not how large the held-out gain is.

Run: .ai/swebench/venv/bin/python analysis/queue_restart_decision.py
"""
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np

from schedule_frontier import load

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/results/queue_restart_decision.json"
SEED = 20260930
REPS = 250
FRACTIONS = (0.3, 0.4, 0.5)
CURVE = tuple(round(0.05 * k, 2) for k in range(1, 16))  # 5% ... 75% of all-rollout cost
MODES = ("plain_breadth", "ranked_breadth", "deferred_rate", "adaptive")
PRIOR_CALLS = 2000.0  # fixed before held-out evaluation, same as prior schedule analysis


def train_rates(s, calls, langs):
    all_success = float(s.sum())
    all_calls = float(calls.sum())
    base = all_success / all_calls
    fresh = {}
    for lang in set(langs):
        mask = langs == lang
        fresh[str(lang)] = (float(s[mask].sum()) + PRIOR_CALLS * base) / (float(calls[mask].sum()) + PRIOR_CALLS)

    fail_calls = calls[~s]
    edges = np.quantile(fail_calls, [1 / 3, 2 / 3])
    success = {}
    cost = {}
    def add(key, y, c):
        success[key] = success.get(key, 0.0) + float(y)
        cost[key] = cost.get(key, 0.0) + float(c)

    for row in range(len(s)):
        lang = str(langs[row])
        for a in range(3):
            if s[row, a]:
                continue
            bucket = int(np.digitize(calls[row, a], edges, right=True))
            for b in range(3):
                if b == a:
                    continue
                add((lang, bucket, 1), s[row, b], calls[row, b])
                add((None, None, 1), s[row, b], calls[row, b])
        # With two failures, the third exchangeable rollout is the remaining option.
        for a in range(3):
            if s[row, a]:
                continue
            for b in range(a + 1, 3):
                if s[row, b]:
                    continue
                third = 3 - a - b
                add((lang, None, 2), s[row, third], calls[row, third])
                add((None, None, 2), s[row, third], calls[row, third])

    rates = {}
    for key in success:
        if key[0] is None:
            rates[key] = success[key] / cost[key]
    for key in success:
        if key[0] is not None:
            fallback = rates[(None, None, key[2])]
            rates[key] = (success[key] + PRIOR_CALLS * fallback) / (cost[key] + PRIOR_CALLS)
    return fresh, rates, edges.tolist(), base


def one_schedule(s, calls, langs, fresh_rate, retry_rate, edges, budget, mode, rng, trace=None):
    """Solved, spent and early restarts for one policy; `trace` collects every attempt run."""
    n = len(s)
    permutations = np.argsort(rng.random((n, 3)), axis=1)
    random_order = rng.permutation(n)
    fresh = []
    for rank, issue in enumerate(random_order):
        rate = fresh_rate.get(str(langs[issue]), fresh_rate["__global__"])
        priority = -rate if mode != "plain_breadth" else 0.0
        heapq.heappush(fresh, (priority, rank, issue))
    retries = []
    next_slot = np.zeros(n, dtype=np.int8)
    solved = 0
    spent = 0.0
    retry_rank = 0
    early_restarts = 0
    while fresh or retries:
        use_retry = bool(retries) and (not fresh or (mode == "adaptive" and retries[0][0][0] < fresh[0][0]))
        if use_retry:
            _, _, issue = heapq.heappop(retries)
            early = bool(fresh)
            early_restarts += early
        else:
            _, _, issue = heapq.heappop(fresh)
            early = False
        slot = int(next_slot[issue])
        attempt = permutations[issue, slot]
        actual_cost = float(calls[issue, attempt])
        if spent + actual_cost > budget:
            break
        spent += actual_cost
        next_slot[issue] += 1
        if trace is not None:
            trace.append((issue, slot, actual_cost, bool(s[issue, attempt]), early))
        if s[issue, attempt]:
            solved += 1
        elif next_slot[issue] < 3:
            failed = permutations[issue, :next_slot[issue]]
            bucket = int(np.digitize(float(calls[issue, failed].mean()), edges, right=True))
            count = int(next_slot[issue])
            key = (str(langs[issue]), bucket if count == 1 else None, count)
            rate = retry_rate.get(key, retry_rate[(None, None, count)])
            priority = ((-rate, count) if mode in ("adaptive", "deferred_rate") else
                        (count, -rate) if mode == "ranked_breadth" else (count, 0.0))
            heapq.heappush(retries, (priority, retry_rank, issue))
            retry_rank += 1
    return solved, spent, early_restarts


def paired(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return {"mean": round(float(d.mean()), 2), "paired_95": np.quantile(d, [0.025, 0.975]).round(2).tolist()}


def budget_curves(fresh, retry, edges, rng, reps, fractions, draw, groups=None):
    """Solved counts at every budget share from one unbounded run per policy and replicate.

    draw(rng) returns one replicate's (success, cost, categories) arrays. At each
    pre-specified share in `groups`, it also splits adaptive minus deferred_rate into
    the attempts only one of the two policies ran, by category and attempt kind.
    """
    solved = {m: [] for m in MODES}
    early = {f: [] for f in fractions}
    split = {f: [] for f in (groups or ())}
    for _ in range(reps):
        ss, cc, ll = draw(rng)
        total = float(cc.sum())
        state = rng.bit_generator.state
        traces = {}
        for mode in MODES:
            rng.bit_generator.state = state
            traces[mode] = []
            one_schedule(ss, cc, ll, fresh, retry, edges, float("inf"), mode, rng, traces[mode])
        for mode, trace in traces.items():
            cost = np.cumsum([t[2] for t in trace])
            gain = np.concatenate([[0], np.cumsum([t[3] for t in trace])])
            ran = np.searchsorted(cost, np.asarray(fractions) * total, side="right")
            solved[mode].append(gain[ran])
            if mode == "adaptive":
                flags = np.concatenate([[0], np.cumsum([t[4] for t in trace])])
                for f, k in zip(fractions, ran):
                    early[f].append(flags[k])
        for f in split:
            prefix = {m: traces[m][:int(np.searchsorted(np.cumsum([t[2] for t in traces[m]]), f * total, side="right"))]
                      for m in ("adaptive", "deferred_rate")}
            keys = {m: {(t[0], t[1]): t for t in p} for m, p in prefix.items()}
            row = {}
            for side, other in (("adaptive", "deferred_rate"), ("deferred_rate", "adaptive")):
                for (issue, slot), t in keys[side].items():
                    if (issue, slot) in keys[other]:
                        continue
                    kind = "fresh" if slot == 0 else "restart"
                    cell = row.setdefault((side, kind, str(ll[issue])), [0, 0])
                    cell[0] += 1
                    cell[1] += int(t[3])
            split[f].append(row)
    curve = {"fractions": list(fractions), "replicates": reps,
             "solved_mean": {m: np.mean(v, axis=0).round(2).tolist() for m, v in solved.items()},
             "early_restarts_mean": [round(float(np.mean(early[f])), 2) for f in fractions]}
    for name, (a, b) in {"adaptive_minus_deferred_rate": ("adaptive", "deferred_rate"),
                         "deferred_rate_minus_ranked_breadth": ("deferred_rate", "ranked_breadth"),
                         "adaptive_minus_ranked_breadth": ("adaptive", "ranked_breadth")}.items():
        d = np.asarray(solved[a], dtype=float) - np.asarray(solved[b], dtype=float)
        curve[name] = {"mean": d.mean(axis=0).round(2).tolist(),
                       "lo": np.quantile(d, 0.025, axis=0).round(2).tolist(),
                       "hi": np.quantile(d, 0.975, axis=0).round(2).tolist()}
    decomposition = {}
    for f, rows in split.items():
        cells = {}
        for row in rows:
            for key, (n_att, n_ok) in row.items():
                cell = cells.setdefault(key, [0, 0])
                cell[0] += n_att
                cell[1] += n_ok
        per = {}
        for (side, kind, cat), (n_att, n_ok) in sorted(cells.items()):
            per.setdefault(f"{side}_only_{kind}", {})[cat] = {
                "attempts": round(n_att / reps, 2), "solved": round(n_ok / reps, 2)}
        for label, cats_ in list(per.items()):
            per[label] = {"attempts": round(sum(v["attempts"] for v in cats_.values()), 2),
                          "solved": round(sum(v["solved"] for v in cats_.values()), 2),
                          "by_category": cats_}
        decomposition[str(f)] = per
    return curve, decomposition


def main():
    ids, success, calls, seconds, langs = load()
    train = np.array([hashlib.sha256((i + ":schedule").encode()).digest()[0] < 128 for i in ids])
    fresh, retry, edges, base = train_rates(success[train], calls[train], langs[train])
    fresh["__global__"] = base
    held_s, held_c, held_lang = success[~train], calls[~train], langs[~train]
    rng = np.random.default_rng(SEED)
    records = {f: {m: [] for m in MODES} for f in FRACTIONS}
    for _ in range(REPS):
        sampled = rng.integers(0, len(held_s), len(held_s))
        ss, cc, ll = held_s[sampled], held_c[sampled], held_lang[sampled]
        budget_base = float(cc.sum())
        for fraction in FRACTIONS:
            # Reuse the same random task and rollout order in all three policies.
            state = rng.bit_generator.state
            for mode in records[fraction]:
                rng.bit_generator.state = state
                records[fraction][mode].append(one_schedule(ss, cc, ll, fresh, retry, edges,
                                                            fraction * budget_base, mode, rng))
            # Advance one stream only after all policies shared its starting state.

    result = {"source": "Qwen3.8-27B OpenHands, 3 complete rollouts per issue",
              "split": "SHA-256(instance_id + ':schedule') < 128 for training",
              "train_issues": int(train.sum()), "heldout_issues": int((~train).sum()),
              "replicates": REPS, "fixed_prior_calls": PRIOR_CALLS,
              "train_fresh_per_1000_calls": {k: round(v * 1000, 3) for k, v in sorted(fresh.items())},
              "train_restart_global_per_1000_calls": {str(k[2]): round(v * 1000, 3) for k, v in retry.items() if k[0] is None},
              "failure_call_tercile_edges": edges, "budgets": {}}
    for fraction, modes in records.items():
        summary = {}
        for mode, triples in modes.items():
            a = np.array(triples)
            summary[mode] = {"solved_mean": round(float(a[:, 0].mean()), 2),
                             "spent_calls_mean": round(float(a[:, 1].mean()), 1),
                             "early_restarts_mean": round(float(a[:, 2].mean()), 2)}
        solved = {mode: np.array(triples)[:, 0] for mode, triples in modes.items()}
        for mode in ("ranked_breadth", "deferred_rate", "adaptive"):
            summary[mode]["minus_plain"] = paired(solved[mode], solved["plain_breadth"])
        summary["deferred_rate"]["minus_ranked_breadth"] = paired(solved["deferred_rate"], solved["ranked_breadth"])
        summary["adaptive"]["minus_ranked_breadth"] = paired(solved["adaptive"], solved["ranked_breadth"])
        summary["adaptive"]["minus_deferred_rate"] = paired(solved["adaptive"], solved["deferred_rate"])
        result["budgets"][str(fraction)] = summary

    def draw(generator):
        sampled = generator.integers(0, len(held_s), len(held_s))
        return held_s[sampled], held_c[sampled], held_lang[sampled]

    train_s, train_c, train_lang = success[train], calls[train], langs[train]

    def draw_train(generator):
        sampled = generator.integers(0, len(train_s), len(train_s))
        return train_s[sampled], train_c[sampled], train_lang[sampled]

    result["curve"], result["decomposition"] = budget_curves(
        fresh, retry, edges, np.random.default_rng(SEED + 1), REPS, CURVE, draw, groups=FRACTIONS)
    # Where the gain falls when the rates are replayed on the issues that fitted them.
    result["train_curve"], _ = budget_curves(fresh, retry, edges, np.random.default_rng(SEED + 2),
                                             REPS, CURVE, draw_train)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"train_issues": result["train_issues"], "heldout_issues": result["heldout_issues"],
                      "budgets": result["budgets"], "curve": result["curve"]}, indent=2))


if __name__ == "__main__":
    main()
