"""Fetch the official multilingual submission's public per-issue trajectories."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / ".ai" / "swebench"
ENTRY = "20260213_mini-v2.0.0a0_claude-4-5-sonnet"
DETAILS = json.loads((ROOT / "multilingual_per_instance_details.json").read_text())
SHARD = int(sys.argv[1])
assert SHARD in range(4)
DEST = ROOT / "multilingual" / "trajs"
DEST.mkdir(parents=True, exist_ok=True)

def fetch(issue):
    target = DEST / f"{issue}.traj.json"
    if target.exists() and target.stat().st_size:
        return issue, "cached"
    url = f"s3://swe-bench-submissions/multilingual/{ENTRY}/trajs/{issue}/{issue}.traj.json"
    done = subprocess.run(["aws", "s3", "cp", url, str(target), "--no-sign-request", "--only-show-errors"],
                          capture_output=True, text=True, timeout=60)
    return issue, "ok" if done.returncode == 0 else done.stderr[:120]

issues = sorted(DETAILS)[SHARD::4]
with ThreadPoolExecutor(max_workers=8) as pool:
    output = [future.result() for future in as_completed([pool.submit(fetch, issue) for issue in issues])]
print(json.dumps({"shard": SHARD, "total": len(output), "ok": sum(status in ("ok", "cached") for _, status in output),
                  "errors": [(issue, status) for issue, status in output if status not in ("ok", "cached")]}, indent=2))
