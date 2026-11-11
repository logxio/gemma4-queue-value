"""Independent language/task check for the evaluator-side file gate."""
import json
import random
import re
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1] / ".ai" / "swebench"
DETAILS = json.loads((ROOT / "multilingual_per_instance_details.json").read_text())
PATCH_RE = re.compile(r"(?m)^diff --git a/(.*?) b/")
PATH_RE = re.compile(r"(?:/testbed/)?((?:[\w.@+-]+/)+[\w.@+-]+\.(?:py|js|jsx|ts|tsx|go|rs|java|c|cc|cpp|h|cs|rb|php|kt|scala|vue|sh|xml|yaml|yml|json|md))\b")


def file_paths(value):
    return [m.group(1).lstrip("/") for m in PATH_RE.finditer(value)]


def gold_files(patch):
    return {p for p in PATCH_RE.findall(patch) if not any(x in p.lower().split("/") for x in ("test", "tests", "spec", "specs")) and not Path(p).name.lower().startswith(("test_", "spec_"))}


def row(issue, repo, patch):
    path = ROOT / "multilingual" / "trajs" / f"{issue}.traj.json"
    if not path.exists():
        return None
    trace = json.loads(path.read_text())
    calls = DETAILS[issue]["api_calls"]
    candidates = []
    turns = 0
    # The checkpoint is fixed before seeing outcomes, 15 model turns or earlier termination.
    for msg in trace["messages"]:
        if msg.get("role") == "assistant":
            turns += 1
            for call in msg.get("tool_calls", []):
                args = call.get("function", {}).get("arguments", "")
                try:
                    command = json.loads(args).get("command", "")
                except (ValueError, TypeError):
                    command = ""
                candidates.extend(file_paths(command))
        if turns >= 15 and msg.get("role") == "assistant":
            break
    seen = []
    for p in candidates:
        if p not in seen and not any(x in p.lower().split("/") for x in ("test", "tests", "spec", "specs", "node_modules")):
            seen.append(p)
    seen = seen[:5]
    gold = gold_files(patch)
    if not gold:
        return None
    f1 = 2 * len(gold.intersection(seen)) / (len(gold) + len(seen)) if seen else 0
    return {"instance_id": issue, "repo": repo, "gold": sorted(gold), "candidates": seen,
            "f1": f1, "resolved": int(DETAILS[issue]["resolved"]),
            "full": calls, "prefix": min(calls, 15), "cost_usd": DETAILS[issue]["cost"]}


def replay(rows, fraction, seed):
    order = list(rows)
    random.Random(seed).shuffle(order)
    budget = fraction * sum(r["full"] for r in order)
    eligible = [r for r in order if r["prefix"] < r["full"]]
    stops = {r["instance_id"] for r in random.Random(seed + 100000).sample(eligible, sum(r["f1"] == 0 for r in eligible))}
    out = []
    for policy in ("baseline", "gate", "random"):
        used = resolved = 0
        for r in order:
            stop = (policy == "gate" and r["f1"] == 0 and r["prefix"] < r["full"]) or (policy == "random" and r["instance_id"] in stops)
            cost = r["prefix"] if stop else r["full"]
            if used + cost > budget:
                break
            used += cost
            if not stop:
                resolved += r["resolved"]
        out.append(resolved)
    return out


records = duckdb.sql(f"select instance_id, repo, patch from read_parquet('{ROOT / 'multilingual.parquet'}')").fetchall()
rows = [row(*record) for record in records]
missing = [record[0] for record, value in zip(records, rows) if value is None]
rows = [r for r in rows if r]
output = {"n": len(rows), "missing_or_no_gold": missing,
          "repos": len({r["repo"] for r in rows}),
          "published_full_run_cost_usd": round(sum(r["cost_usd"] for r in rows), 2),
          "zero_f1": sum(r["f1"] == 0 for r in rows),
          "zero_f1_resolved": sum(r["resolved"] for r in rows if r["f1"] == 0),
          "nonzero_f1": sum(r["f1"] > 0 for r in rows),
          "nonzero_f1_resolved": sum(r["resolved"] for r in rows if r["f1"] > 0),
          "results": {}}
for fraction in (.25, .50, .75):
    samples = [replay(rows, fraction, seed) for seed in range(1000)]
    diffs = sorted(g-b for b,g,_ in samples)
    random_diffs = sorted(g-r for _,g,r in samples)
    output["results"][str(fraction)] = {
        "baseline_mean": round(sum(x[0] for x in samples)/1000,2),
        "gate_mean": round(sum(x[1] for x in samples)/1000,2),
        "random_stop_mean": round(sum(x[2] for x in samples)/1000,2),
        "difference_mean": round(sum(diffs)/1000,2),
        "difference_seed_interval_95": [diffs[25], diffs[974]],
        "gate_minus_random_mean": round(sum(random_diffs)/1000,2),
    }
    if fraction == .50:
        boots = []
        for seed in range(1000):
            sample = random.Random(seed + 200000).choices(rows, k=len(rows))
            baseline, gate, _ = replay(sample, fraction, seed)
            boots.append(gate - baseline)
        boots.sort()
        output["results"][str(fraction)]["difference_issue_bootstrap_95"] = [boots[25], boots[974]]
(ROOT / "multilingual" / "audit_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
(ROOT / "multilingual" / "replay_summary.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
