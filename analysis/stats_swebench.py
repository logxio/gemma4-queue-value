import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / ".ai" / "swebench"
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "verified"
if SPLIT == "lite":
    ROOT = ROOT / "lite"
elif SPLIT != "verified":
    raise ValueError("split must be verified or lite")
SYSTEMS = ("sweagent", "agentless", "openhands") if SPLIT == "verified" else ("sweagent", "openhands")
records = json.loads((ROOT / "audit_rows.json").read_text())
if SPLIT == "lite":
    verified_ids = {r["instance_id"] for r in json.loads((ROOT.parent / "audit_manifest.json").read_text())}
    records = [r for r in records if r["instance_id"] not in verified_ids]
by_issue = defaultdict(dict)
repos = {}
for row in records:
    by_issue[row["instance_id"]][row["system"]] = row
    repos[row["instance_id"]] = row["repo"]
issues = sorted(by_issue)
assert issues and all(len(by_issue[i]) == len(SYSTEMS) for i in issues)


def interval(values, rng, boot=10000):
    values = np.array(values, dtype=float)
    indexes = rng.integers(0, len(values), size=(boot, len(values)))
    samples = values[indexes].mean(axis=1)
    return [round(float(np.quantile(samples, q)), 4) for q in [0.025, 0.975]]


rng = np.random.default_rng(20260927)
summary = {"split": SPLIT, "n_issues": len(issues), "systems": {}, "contrasts": {}, "repo_breakdown": {}}
for system in SYSTEMS:
    data = [by_issue[i][system] for i in issues]
    summary["systems"][system] = {metric: {
        "mean": round(float(np.mean([r[metric] for r in data])), 4),
        "ci95": interval([r[metric] for r in data], rng),
        "count": int(sum(r[metric] for r in data)) if metric != "f1" else None,
    } for metric in ("f1", "hit1", "hit5", "resolved")}
    for metric in ("hit1", "hit5"):
        for value in (0, 1):
            subset = [r for r in data if r[metric] == value]
            summary["systems"][system][f"resolved_given_{metric}_{value}"] = [sum(r["resolved"] for r in subset), len(subset)]

contrasts = (("sweagent", "agentless"), ("openhands", "agentless"), ("openhands", "sweagent")) if SPLIT == "verified" else (("openhands", "sweagent"),)
for a, b in contrasts:
    key = f"{a}_minus_{b}"
    summary["contrasts"][key] = {}
    for metric in ("f1", "hit1", "hit5", "resolved"):
        delta = [by_issue[i][a][metric] - by_issue[i][b][metric] for i in issues]
        summary["contrasts"][key][metric] = {"mean": round(float(np.mean(delta)), 4), "ci95": interval(delta, rng)}
    informative = [i for i in issues if by_issue[i][a]["f1"] != by_issue[i][b]["f1"] and by_issue[i][a]["resolved"] != by_issue[i][b]["resolved"]]
    concordant = sum((by_issue[i][a]["f1"] - by_issue[i][b]["f1"]) * (by_issue[i][a]["resolved"] - by_issue[i][b]["resolved"]) > 0 for i in informative)
    summary["contrasts"][key]["issue_level_concordance"] = [concordant, len(informative)]

for repo in sorted(set(repos.values())):
    subset = [i for i in issues if repos[i] == repo]
    summary["repo_breakdown"][repo] = {system: {metric: round(float(np.mean([by_issue[i][system][metric] for i in subset])), 4) for metric in ("f1", "resolved")} for system in SYSTEMS}
    summary["repo_breakdown"][repo]["n"] = len(subset)

(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"n": summary["n_issues"], "systems": summary["systems"], "contrasts": summary["contrasts"]}, indent=2))
