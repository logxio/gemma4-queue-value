"""Break-even screen for an evaluator-side stop gate; no outcome is used at serving time."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / ".ai" / "swebench"

def summarize(split):
    base = ROOT if split == "verified" else ROOT / split
    rows = json.loads((base / "audit_rows.json").read_text())
    if split == "lite":
        verified = {r["instance_id"] for r in json.loads((ROOT / "audit_rows.json").read_text())}
        rows = [r for r in rows if r["instance_id"] not in verified]
    if split != "multilingual":
        from replay_budget import costs
        for r in rows:
            r["full"], r["prefix"] = costs(split, r)
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("system", "mini")].append(r)
    output = {}
    for name, part in groups.items():
        stopped = [r for r in part if r["f1"] == 0 and r["prefix"] < r["full"]]
        saved = sum(r["full"] - r["prefix"] for r in stopped)
        lost = sum(r["resolved"] for r in stopped)
        total = sum(r["full"] for r in part)
        fixed = sum(r["resolved"] for r in part)
        output[name] = {
            "n": len(part), "stopped": len(stopped),
            "saved_calls_if_all_tasks_available": saved,
            "lost_solutions_if_all_tasks_available": lost,
            "lost_solutions_per_saved_call": round(lost / saved, 5),
            "queue_solutions_per_full_call": round(fixed / total, 5),
            "enable_gate_on_future_issues": lost / saved < fixed / total,
        }
    return output

if __name__ == "__main__":
    results = {split: summarize(split) for split in ("verified", "lite", "multilingual")}
    (ROOT / "calibration_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
