"""Offline benchmark-budget replay. Gold-file gating is evaluator-only, not a live agent policy."""

import json
import posixpath
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / ".ai" / "swebench"
SYSTEMS = ("sweagent", "agentless", "openhands")


def auxiliary(path):
    path = path.replace("/testbed/", "").replace("/workspace/", "")
    base = posixpath.basename(path).lower()
    parts = path.lower().split("/")
    return base.startswith(("reproduce", "repro_", "test_")) or base in {"repro.py", "test.py"} or "test" in parts or "tests" in parts


def costs(split, row):
    base = ROOT if split == "verified" else ROOT / "lite"
    issue, system = row["instance_id"], row["system"]
    ext = {"sweagent": ".traj", "agentless": ".log", "openhands": ".json"}[system]
    path = base / "trajs" / system / (issue + ext)
    if system == "sweagent":
        trace = json.loads(path.read_text())
        full = int(trace["info"]["model_stats"]["api_calls"])
        prefix = full
        for i, turn in enumerate(trace["trajectory"], 1):
            action = turn.get("action", "")
            state = json.loads(turn.get("state", "{}"))
            edited = None
            if action.startswith("edit "):
                edited = state.get("open_file", "")
            elif action.startswith("create "):
                edited = action.splitlines()[0][7:]
            if edited and not auxiliary(edited):
                prefix = min(full, i)
                break
    elif system == "agentless":
        trace = path.read_text(errors="replace")
        full = len(re.findall(r"INFO - API response ChatCompletion", trace))
        marker = trace.find("### Model Predicted Suspicious Files")
        prefix = len(re.findall(r"INFO - API response ChatCompletion", trace[:marker])) + 1
    else:
        trace = json.loads(path.read_text())
        assistant = [m for m in trace if m.get("role") == "assistant"]
        full = len(assistant)
        prefix = full
        for i, msg in enumerate(assistant, 1):
            edits = False
            for call in msg.get("tool_calls", []):
                fn = call.get("function", {})
                try:
                    arg = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    continue
                if fn.get("name") == "str_replace_editor" and arg.get("command") in ("create", "str_replace", "insert") and not auxiliary(arg.get("path", "")):
                    edits = True
                if fn.get("name") == "execute_bash" and re.search(r"\b(?:apply_patch|git apply)\b", arg.get("command", "")):
                    edits = True
            if edits:
                prefix = i
                break
    if not full:
        raise ValueError((split, system, issue, "no model calls"))
    return full, max(1, min(full, prefix))


def replay(rows, budget_fraction, seed):
    order = list(rows)
    random.Random(seed).shuffle(order)
    budget = budget_fraction * sum(r["full"] for r in order)
    used = 0
    result = [0, 0, 0]
    for r in order:
        full, prefix = r["full"], r["prefix"]
        if used + full > budget:
            break
        used += full
        result[0] += r["resolved"]
    used = 0
    for r in order:
        stop = r["f1"] == 0 and r["prefix"] < r["full"]
        cost = r["prefix"] if stop else r["full"]
        if used + cost > budget:
            break
        used += cost
        if not stop:
            result[1] += r["resolved"]
    eligible = [r for r in order if r["prefix"] < r["full"]]
    random_stop = {r["instance_id"] for r in random.Random(seed + 100000).sample(eligible, sum(r["f1"] == 0 for r in eligible))}
    used = 0
    for r in order:
        stop = r["instance_id"] in random_stop
        cost = r["prefix"] if stop else r["full"]
        if used + cost > budget:
            break
        used += cost
        if not stop:
            result[2] += r["resolved"]
    return result


def summarize(split):
    base = ROOT if split == "verified" else ROOT / "lite"
    rows = json.loads((base / "audit_rows.json").read_text())
    if split == "lite":
        verified = {r["instance_id"] for r in json.loads((ROOT / "audit_rows.json").read_text())}
        rows = [r for r in rows if r["instance_id"] not in verified]
    for row in rows:
        row["full"], row["prefix"] = costs(split, row)
    groups = defaultdict(list)
    for row in rows:
        groups[row["system"]].append(row)
    out = {}
    for system, part in groups.items():
        # Intention-to-schedule: all tasks remain eligible, regardless of outcome.
        results = {}
        for frac in (.25, .50, .75):
            samples = [replay(part, frac, seed) for seed in range(1000)]
            diffs = sorted(g - b for b, g, _ in samples)
            random_diffs = sorted(g - random_result for _, g, random_result in samples)
            results[str(frac)] = {
                "baseline_mean": round(sum(b for b, g, _ in samples) / len(samples), 2),
                "gate_mean": round(sum(g for b, g, _ in samples) / len(samples), 2),
                "random_stop_mean": round(sum(r for _, _, r in samples) / len(samples), 2),
                "difference_mean": round(sum(diffs) / len(diffs), 2),
                "difference_seed_interval_95": [diffs[25], diffs[974]],
                "win_seed_fraction": round(sum(d > 0 for d in diffs) / len(diffs), 3),
                "gate_minus_random_mean": round(sum(random_diffs) / len(random_diffs), 2),
                "gate_minus_random_seed_interval_95": [random_diffs[25], random_diffs[974]],
            }
            if frac == .50:
                boots = []
                for seed in range(1000):
                    sample = random.Random(seed + 200000).choices(part, k=len(part))
                    baseline, gate, _ = replay(sample, frac, seed)
                    boots.append(gate - baseline)
                boots.sort()
                results[str(frac)]["difference_issue_bootstrap_95"] = [boots[25], boots[974]]
        out[system] = {
            "n": len(part), "full_calls": sum(r["full"] for r in part),
        "zero_f1": sum(r["f1"] == 0 and r["prefix"] < r["full"] for r in part),
        "zero_f1_resolved": sum(r["resolved"] for r in part if r["f1"] == 0 and r["prefix"] < r["full"]),
        "full_calls_zero_f1": sum(r["full"] for r in part if r["f1"] == 0 and r["prefix"] < r["full"]),
        "prefix_calls_zero_f1": sum(r["prefix"] for r in part if r["f1"] == 0 and r["prefix"] < r["full"]),
            "results": results,
        }
    (base / "replay_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({split: out}, indent=2))


if __name__ == "__main__":
    summarize("verified")
    summarize("lite")
