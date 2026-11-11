"""Fetch the fourth Verified system on the same 465 common issues."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / ".ai" / "swebench"
ENTRY = "20260217_mini-v2.0.0_claude-4-5-haiku-high"
SHARD = int(sys.argv[1])
assert SHARD in range(4)
DETAILS = json.loads((ROOT / "experiments" / "evaluation" / "verified" / ENTRY / "per_instance_details.json").read_text())
BASE = ROOT / "trajs" / "mini"
BASE.mkdir(parents=True, exist_ok=True)
issues = sorted(set(p.stem for p in (ROOT / "trajs" / "sweagent").glob("*.traj"))
                & set(p.stem for p in (ROOT / "trajs" / "agentless").glob("*.log"))
                & set(p.stem for p in (ROOT / "trajs" / "openhands").glob("*.json")))
issues = [issue for issue in issues if issue in DETAILS]

def fetch(issue):
    target = BASE / f"{issue}.traj.json"
    if target.exists() and target.stat().st_size:
        return issue, "cached"
    url = f"s3://swe-bench-submissions/bash-only/{ENTRY}/trajs/{issue}/{issue}.traj.json"
    done = subprocess.run(["aws", "s3", "cp", url, str(target), "--no-sign-request", "--only-show-errors"],
                          capture_output=True, text=True, timeout=60)
    return issue, "ok" if done.returncode == 0 else done.stderr[:120]

with ThreadPoolExecutor(max_workers=8) as pool:
    output = [future.result() for future in as_completed([pool.submit(fetch, issue) for issue in issues[SHARD::4]])]
print(json.dumps({"shard": SHARD, "total": len(output), "ok": sum(s in ("ok", "cached") for _,s in output),
                  "errors": [(i,s) for i,s in output if s not in ("ok", "cached")]}, indent=2))
