"""Extract a compact action-count proxy from public SWE-agent trajectories.

The source has no token/call/wall-time column. Trajectory message count is an
observable cost proxy, not a calibrated compute measure. Each shard is processed
once, with a local checkpoint. A network failure exits 75 so comp job can retry.

Run: .ai/swebench/venv/bin/python analysis/nebius_attempt_costs.py
"""
import csv
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / ".ai" / "nebius_cost_shards"
OUT = ROOT / "evidence" / "results" / "nebius_attempt_costs.csv"
BASE = "https://huggingface.co/datasets/nebius/SWE-agent-trajectories/resolve/main/data/train-{:05d}-of-00012.parquet"
HEADER = ["model_name", "instance_id", "target", "trajectory_messages"]


def main():
    CHECK.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(config={"memory_limit": "2GB", "threads": "2"})
    try:
        con.sql("INSTALL httpfs; LOAD httpfs")
        for shard in range(12):
            dest = CHECK / f"{shard:02d}.csv"
            if dest.exists():
                print(f"shard {shard:02d}: checkpoint", flush=True)
                continue
            url = BASE.format(shard)
            rows = con.execute(
                "SELECT model_name, instance_id, target, array_length(trajectory) "
                "FROM read_parquet(?)", [url]
            ).fetchall()
            tmp = dest.with_suffix(".tmp")
            with tmp.open("w", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(HEADER)
                writer.writerows(rows)
            tmp.replace(dest)
            print(f"shard {shard:02d}: {len(rows)} attempts", flush=True)
    except Exception as exc:
        print(f"remote parquet extraction interrupted: {exc}", file=sys.stderr)
        network_error = isinstance(exc, (duckdb.IOException, TimeoutError, ConnectionError)) or any(
            word in str(exc).lower() for word in ("http", "timeout", "connection", "network", "ssl"))
        return 75 if network_error else 1

    tmp = OUT.with_suffix(".tmp")
    total = 0
    with tmp.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(HEADER)
        for shard in range(12):
            with (CHECK / f"{shard:02d}.csv").open(newline="") as part:
                reader = csv.reader(part)
                assert next(reader) == HEADER
                for row in reader:
                    writer.writerow(row)
                    total += 1
    tmp.replace(OUT)
    print(f"combined {total} attempts -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
