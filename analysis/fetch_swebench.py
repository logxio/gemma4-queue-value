import json
import random
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import duckdb

ROOT = Path(__file__).resolve().parents[1] / ".ai" / "swebench"
SPLIT = "verified"
DATASET_REPO = "SWE-bench_Verified"
DATASET_FILE = "verified.parquet"
EXPERIMENTS_COMMIT = "40f164d5b8f1d249bf95a6df8b74b577fd8e519d"
DATASET_COMMIT = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
RUNS = {
    "sweagent": ("20240728_sweagent_gpt4o", ".traj"),
    "agentless": ("20241028_agentless-1.5_gpt4o", ".log"),
    "openhands": ("20241029_OpenHands-CodeAct-2.1-sonnet-20241022", ".json"),
}


def configure(split):
    global ROOT, SPLIT, DATASET_REPO, DATASET_FILE, DATASET_COMMIT, RUNS
    if split == "verified":
        return
    if split != "lite":
        raise ValueError("split must be verified or lite")
    SPLIT = "lite"
    ROOT = ROOT / "lite"
    DATASET_REPO = "SWE-bench_Lite"
    DATASET_FILE = "lite.parquet"
    DATASET_COMMIT = "6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2"
    RUNS = {
        "sweagent": ("20240728_sweagent_gpt4o", ".traj"),
        "openhands": ("20241025_OpenHands-CodeAct-2.1-sonnet-20241022", ".json"),
    }


def ensure_sources():
    ROOT.mkdir(parents=True, exist_ok=True)
    dataset = ROOT / DATASET_FILE
    if not dataset.exists():
        urllib.request.urlretrieve(
            f"https://huggingface.co/datasets/princeton-nlp/{DATASET_REPO}/resolve/{DATASET_COMMIT}/data/test-00000-of-00001.parquet",
            dataset,
        )
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    for name, (run, _) in RUNS.items():
        dest = results / f"{name}.json"
        if not dest.exists():
            urllib.request.urlretrieve(
                f"https://raw.githubusercontent.com/SWE-bench/experiments/{EXPERIMENTS_COMMIT}/evaluation/{SPLIT}/{run}/results/results.json",
                dest,
            )


def available(run):
    cmd = ["aws", "s3", "ls", "--no-sign-request", "--recursive",
           f"s3://swe-bench-submissions/{SPLIT}/{run}/trajs/"]
    output = subprocess.check_output(cmd, text=True, timeout=25)
    return {Path(line.split()[-1]).stem for line in output.splitlines() if line}


def make_manifest():
    rows = duckdb.query(f"SELECT instance_id, patch, repo FROM read_parquet('{ROOT / DATASET_FILE}')").fetchall()
    sets = [available(run) for run, _ in RUNS.values()]
    common = set.intersection(*sets)
    eligible = [(i, p, r) for i, p, r in rows if i in common]
    rng = random.Random(20260927)
    rng.shuffle(eligible)
    picked = eligible
    manifest = []
    for instance_id, patch, repo in picked:
        files = sorted(set(re.findall(r"^diff --git a/(.*?) b/", patch, re.M)))
        manifest.append({"instance_id": instance_id, "repo": repo, "gold_files": files})
    (ROOT / "audit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"eligible={len(eligible)} selected={len(manifest)} repos={len({x['repo'] for x in manifest})}", flush=True)
    return manifest


def download(task):
    name, run, ext, instance_id = task
    dest = ROOT / "trajs" / name / f"{instance_id}{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size:
        return True
    cmd = ["aws", "s3", "cp", "--no-sign-request", "--only-show-errors",
           f"s3://swe-bench-submissions/{SPLIT}/{run}/trajs/{instance_id}{ext}", str(dest)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=45)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    configure(sys.argv[2] if len(sys.argv) > 2 else "verified")
    ensure_sources()
    path = ROOT / "audit_manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else make_manifest()
    tasks = [(name, run, ext, row["instance_id"]) for name, (run, ext) in RUNS.items() for row in manifest]
    shard = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    tasks = [task for i, task in enumerate(tasks) if i % 4 == shard]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(download, tasks))
    print(f"shard={shard} downloaded={sum(results)}/{len(tasks)}", flush=True)
